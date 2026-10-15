from functools import partial

# No Hydra: tweak behavior here. When True, disp tower input is [value_heads | xy from identity grid].
CONCAT_IDENTITY_COORDS_FOR_DISP: bool = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from benchmark.models.common import BaseModel
from benchmark.models.deform_bounds import max_displacement_normalized_one_cell
from torch.utils.checkpoint import checkpoint


class _PassThroughACT(nn.Module):
    """No ACT displacement block; keeps ``(tensor, disp)`` API with ``disp`` optional for the backbone."""

    def forward(self, x: torch.Tensor):
        return x, None


class MultiHeadACTBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=16,
        hidden_dim=16,
        max_disp=0.5,
        use_norm=True,
        disp_net_branched: bool = True,
        skip_output_norm: bool = False,
        hidden_channels=16,
        boundary_mode: str = "periodic",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_disp = max_disp
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm
        self.disp_net_branched = disp_net_branched
        self.skip_output_norm = skip_output_norm
        self.hidden_channels = hidden_channels
        if boundary_mode not in ("periodic", "zeros"):
            raise ValueError(
                f"boundary_mode must be 'periodic' or 'zeros', got {boundary_mode!r}"
            )
        self.boundary_mode = boundary_mode
        self.value_proj = nn.Conv2d(in_channels, num_heads * self.hidden_channels, 1)
        extra_disp_c = 2 if CONCAT_IDENTITY_COORDS_FOR_DISP else 0
        disp_in = self.hidden_channels + extra_disp_c
        if disp_net_branched:
            h = max(hidden_dim // 3, 8)
            self.disp_net = nn.ModuleDict(
                {
                    "branch_1x1": nn.Conv2d(disp_in, h, 1),
                    "branch_3x3": nn.Conv2d(disp_in, h, 3, padding=1),
                    "branch_3x3_d2": nn.Conv2d(disp_in, h, 3, padding=2, dilation=2),
                }
            )
            fuse_in = h * 3
        else:
            h = max(hidden_dim, 8)
            self.disp_net = nn.ModuleDict(
                {
                    "branch_3x3": nn.Sequential(
                        nn.Conv2d(disp_in, h, 3, padding=1),
                        nn.GroupNorm(1, h),
                        nn.GELU(),
                        nn.Conv2d(h, h, 3, padding=1),
                        nn.GroupNorm(1, h),
                        nn.GELU(),
                    )
                }
            )
            fuse_in = h
        self.disp_fuse = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(fuse_in, 2, 1),
        )
        nn.init.zeros_(self.disp_fuse[-1].weight)
        nn.init.zeros_(self.disp_fuse[-1].bias)
        # self.disp_net = DispNet(in_channels, num_heads, hidden_dim)

        
        self.out_proj = nn.Conv2d(num_heads * self.hidden_channels, out_channels, 1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self._id_grid_key: tuple[int, int, torch.device] | None = None
        self._id_grid: torch.Tensor | None = None

    def _identity_grid_hw2(self, x: torch.Tensor) -> torch.Tensor:
        """``(1, H, W, 2)`` ``grid_sample`` coords (x, y); rebuilt when ``H``, ``W``, or device changes."""
        key = (x.size(-2), x.size(-1), x.device)
        if self._id_grid_key != key or self._id_grid is None:
            H, W, dev = key
            gy, gx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=dev),
                torch.linspace(-1, 1, W, device=dev),
                indexing="ij",
            )
            self._id_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            self._id_grid_key = key
        return self._id_grid

    def identity_coord_channels_for_disp(self, x: torch.Tensor, n_pack: int) -> torch.Tensor:
        """``(n_pack, 2, H, W)`` xy channels from the cached identity grid."""
        xy = self._identity_grid_hw2(x)[0].permute(2, 0, 1).contiguous()
        return xy.unsqueeze(0).expand(n_pack, -1, -1, -1)

    def _disp_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.disp_net_branched:
            return torch.cat(
                [
                    self.disp_net["branch_1x1"](x),
                    self.disp_net["branch_3x3"](x),
                    self.disp_net["branch_3x3_d2"](x),
                ],
                dim=1,
            )
        return self.disp_net["branch_3x3"](x)

    def _predict_displacement(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head offsets in (x, y), shape (B, num_heads, 2, H, W)."""
        disp = self.disp_fuse(self._disp_features(x))
        return torch.tanh(disp) * self.max_disp

    def _sampling_grid(self, x: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """grid_sample layout: (n_pack, H, W, 2), last dim (x, y) for ``F.grid_sample``.

        ``boundary_mode == "periodic"``: toroidal fold with period 2 per coordinate,
        but **only where** ``grid`` is strictly outside ``[-1, 1]``. Values in the closed
        interval (including ``±1`` corners from ``linspace``) are unchanged so zero
        displacement matches the identity grid.

        ``boundary_mode == "zeros"``: no fold. Out-of-range coordinates are left outside
        ``[-1, 1]`` so ``grid_sample(padding_mode="zeros")`` returns 0 there — the
        homogeneous-Dirichlet inductive bias (feature decays to 0 across one cell toward
        the geometric boundary).
        """
        delta = disp.movedim(1, -1)
        grid = self._identity_grid_hw2(x) + delta
        if self.boundary_mode == "zeros":
            return grid
        wrapped = torch.remainder(grid + 1.0, 2.0) - 1.0
        outside = (grid > 1.0) | (grid < -1.0)
        return torch.where(outside, wrapped, grid)

    def _grid_sample_heads(
        self,
        vals: torch.Tensor,
        grid: torch.Tensor,
        *,
        B: int,
        nh: int,
        H: int,
        W: int,
        C: int,
    ) -> torch.Tensor:
        """values: (B, nh, C, H, W); sample_grid: (B, nh, H, W, 2) -> (B, nh*C, H, W)."""
        if not grid.is_contiguous():
            grid = grid.contiguous()
        if not vals.is_contiguous():
            vals = vals.contiguous()
        deformed = F.grid_sample(
            vals,
            grid,
            mode="bilinear",
            padding_mode="zeros" if self.boundary_mode == "zeros" else "border",
            align_corners=True,
        )
        # breakpoint()
        return deformed.view(B, nh * C, H, W)

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        nh = self.num_heads
        C = self.hidden_channels

        n_pack = B * nh
        value_heads = self.value_proj(x).reshape(B, nh, C, H, W).reshape(n_pack, C, H, W)
        if CONCAT_IDENTITY_COORDS_FOR_DISP:
            coords = self.identity_coord_channels_for_disp(x, n_pack)
            disp_in = torch.cat([value_heads, coords], dim=1)
        else:
            disp_in = value_heads
        disp = self._predict_displacement(disp_in).view(n_pack, 2, H, W)

       
        sample_grid = self._sampling_grid(x, disp)
        deformed = self._grid_sample_heads(
            value_heads, sample_grid, B=B, nh=nh, H=H, W=W, C=C
        )
        out = self.out_proj(deformed)
        residual = self.residual_proj(x) if self.residual_proj is not None else x
        merged = residual + out
        if self.use_norm and not self.skip_output_norm:
            out = self.norm(merged)
        else:
            out = merged
        # Do not return displacement tensors while training: avoids holding them in
        # parent ``disps`` lists (lower peak memory). Val/test use ``model.eval()``.
        if self.training:
            return out, None
        return out, disp


def set_warp_boundary_mode(module: nn.Module, boundary_mode: str) -> int:
    """Set ``boundary_mode`` on every ``MultiHeadACTBlock`` under ``module``.

    Returns the number of blocks updated.
    """
    if boundary_mode not in ("periodic", "zeros"):
        raise ValueError(
            f"boundary_mode must be 'periodic' or 'zeros', got {boundary_mode!r}"
        )
    count = 0
    for m in module.modules():
        if isinstance(m, MultiHeadACTBlock):
            m.boundary_mode = boundary_mode
            count += 1
    return count


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_height, modes_width):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.half_n_modes_height = modes_height // 2
        self.half_n_modes_width = modes_width // 2
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        self.weight0 = nn.Parameter(
            init_std * torch.randn(in_channels, out_channels,
                self.half_n_modes_height, self.half_n_modes_width, 2)
        )
        self.weight1 = nn.Parameter(
            init_std * torch.randn(in_channels, out_channels,
                self.half_n_modes_height, self.half_n_modes_width, 2)
        )

    def compl_mul2d(self, input, weights):
        w_real, w_imag = weights[..., 0], weights[..., 1]
        input_real, input_imag = input.real, input.imag
        out_real = torch.einsum("bihw,iohw->bohw", input_real, w_real) - torch.einsum("bihw,iohw->bohw", input_imag, w_imag)
        out_imag = torch.einsum("bihw,iohw->bohw", input_real, w_imag) + torch.einsum("bihw,iohw->bohw", input_imag, w_real)
        return torch.complex(out_real, out_imag)

    def forward(self, x):
        batchsize, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm="forward")
        out_ft = torch.zeros(batchsize, self.out_channels, height, width // 2 + 1, device=x.device, dtype=torch.cfloat)
        h_modes = min(self.half_n_modes_height, height)
        w_modes = min(self.half_n_modes_width, width // 2 + 1)
        out_ft[..., :h_modes, :w_modes] = self.compl_mul2d(x_ft[..., :h_modes, :w_modes], self.weight0[..., :h_modes, :w_modes, :])
        if h_modes > 0:
            out_ft[..., -h_modes:, :w_modes] = self.compl_mul2d(x_ft[..., -h_modes:, :w_modes], self.weight1[..., :h_modes, :w_modes, :])
        return torch.fft.irfft2(out_ft, s=(height, width), norm="forward")


class SoftGating(nn.Module):
    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, channels, 1, 1)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias if self.bias is not None else self.weight * x


def make_skip_connection(in_channels: int, out_channels: int, skip_type: str = "soft-gating") -> nn.Module:
    skip_type = skip_type.lower()
    if skip_type == "soft-gating":
        if in_channels != out_channels:
            raise ValueError(f"Soft-gating requires in_channels==out_channels, got {in_channels} vs {out_channels}")
        return SoftGating(in_channels)
    if skip_type == "linear":
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    if skip_type == "identity":
        if in_channels != out_channels:
            raise ValueError(f"Identity skip requires in_channels==out_channels, got {in_channels} vs {out_channels}")
        return nn.Identity()
    raise ValueError(f"Unsupported skip_type={skip_type}")


class FNOBlock2d(nn.Module):
    def __init__(self, channels, modes_height, modes_width, use_mlp=False, mlp_expansion=0.5, mlp_dropout=0.0,
                 norm=None, non_linearity=F.gelu, fno_skip="linear", mlp_skip="soft-gating"):
        super().__init__()
        self.non_linearity = non_linearity
        self.use_mlp = use_mlp
        self.spectral_conv = SpectralConv2d(channels, channels, modes_height, modes_width)
        self.fno_skip = make_skip_connection(channels, channels, skip_type=fno_skip)
        self.norm_fno = nn.GroupNorm(1, channels) if norm == "group_norm" else (nn.BatchNorm2d(channels) if norm == "batch_norm" else None)
        self.norm_mlp = nn.GroupNorm(1, channels) if use_mlp and norm == "group_norm" else (nn.BatchNorm2d(channels) if use_mlp and norm == "batch_norm" else None)
        if use_mlp:
            hidden_mlp = round(channels * mlp_expansion)
            self.mlp = nn.Sequential(
                nn.Conv2d(channels, hidden_mlp, 1), nn.GELU(), nn.Dropout(mlp_dropout), nn.Conv2d(hidden_mlp, channels, 1)
            )
            self.mlp_skip = make_skip_connection(channels, channels, skip_type=mlp_skip)
        else:
            self.mlp = self.mlp_skip = None

    def forward(self, x, is_last_layer: bool = False):
        x_in = x
        x = self.spectral_conv(x) + self.fno_skip(x_in)
        if self.norm_fno is not None:
            x = self.norm_fno(x)
        if (self.mlp is not None) or (not is_last_layer):
            x = self.non_linearity(x)
        if self.mlp is not None:
            x = self.mlp(x) + self.mlp_skip(x_in)
            if self.norm_mlp is not None:
                x = self.norm_mlp(x)
            if not is_last_layer:
                x = self.non_linearity(x)
        return x



class ACTFnoBackbone(nn.Module):
    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        in_channels=3,
        lifting_channels=128,
        hidden_channels=64,
        out_channels=1,
        projection_channels=128,
        n_blocks=4,
        num_heads=4,
        norm="group_norm",
        skip="linear",
        mlp_skip="soft-gating",
        use_mlp=True,
        mlp_dropout=0.0,
        mlp_expansion=0.5,
        disp_net_branched: bool = True,
        checkpoint_fno: bool = False,
        spatial_resolution: tuple[int, int] | None = None,
        max_disp: float | None = None,
        use_act: bool = True,
        boundary_mode: str = "periodic",
    ):
        super().__init__()
        self.checkpoint_fno = checkpoint_fno
        self.use_act = bool(use_act)
        if max_disp is not None:
            deform_max_disp = float(max_disp)
        elif spatial_resolution is not None:
            deform_max_disp = max_displacement_normalized_one_cell(spatial_resolution)
        else:
            deform_max_disp = 0.5
        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels, lifting_channels, 1),
            nn.GELU(),
            nn.Conv2d(lifting_channels, hidden_channels, 1),
        )
        self.blocks = nn.ModuleList()
        self.act_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(
                FNOBlock2d(
                    hidden_channels,
                    n_modes_height,
                    n_modes_width,
                    use_mlp,
                    mlp_expansion,
                    mlp_dropout,
                    norm,
                    F.gelu,
                    skip,
                    mlp_skip,
                )
            )
            if self.use_act:
                self.act_blocks.append(
                    MultiHeadACTBlock(
                        hidden_channels,
                        hidden_channels,
                        num_heads= num_heads,
                        use_norm=True,
                        disp_net_branched=disp_net_branched,
                        max_disp=deform_max_disp,
                        skip_output_norm=(i == n_blocks - 1),
                        boundary_mode=boundary_mode,
                    )
                )
            else:
                self.act_blocks.append(_PassThroughACT())
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_channels, projection_channels, 1),
            nn.GELU(),
            nn.Conv2d(projection_channels, out_channels, 1),
        )

    def forward(self, x):
        x = self.lifting(x)
        disps = []
        for block, act_block in zip(self.blocks, self.act_blocks):
            is_last = act_block is self.act_blocks[-1]
            if self.checkpoint_fno:
                x = checkpoint(
                    partial(block.forward, is_last_layer=is_last),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, is_last_layer=is_last)
            x, disp = act_block(x)
            if disp is not None:
                disps.append(disp)
        x = self.projection(x)
        return x, disps
    


class ACTFNO(BaseModel):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        modes1: int,
        modes2: int,
        modes3: int = 16,
        hidden_channels: int = 64,
        gradient_checkpointing: bool = False,
        disp_net_branched: bool = True,
        max_disp: float | None = None,
        use_act: bool = True,
        boundary_mode: str = "periodic",
    ):
        super().__init__(n_spatial_dims, spatial_resolution)

        if n_spatial_dims != 2:
            raise ValueError(f"ACTFNO expects n_spatial_dims==2, got {n_spatial_dims}")
        self.model = ACTFnoBackbone(
            n_modes_height=modes1,
            n_modes_width=modes2,
            in_channels=dim_in,
            out_channels=dim_out,
            hidden_channels=hidden_channels,
            disp_net_branched=disp_net_branched,
            checkpoint_fno=gradient_checkpointing,
            spatial_resolution=(spatial_resolution[0], spatial_resolution[1]),
            max_disp=max_disp,
            use_act=use_act,
            boundary_mode=boundary_mode,
        )
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, input):
        """Returns ``(out, disps)``; ``disps`` is empty while ``self.training`` (see :class:`MultiHeadACTBlock`)."""
        return self.model(input)



# class FNO2d(nn.Module):
#     def __init__(self, n_modes_height, n_modes_width, in_channels=3, lifting_channels=256, hidden_channels=64,
#                  out_channels=1, projection_channels=256, n_blocks=4, norm="group_norm", skip="linear",
#                  mlp_skip="soft-gating", use_mlp=True, mlp_dropout=0.0, mlp_expansion=0.5, non_linearity=F.gelu):
#         super().__init__()
#         self.lifting = nn.Sequential(
#             nn.Conv2d(in_channels, lifting_channels, 1), nn.GELU(),
#             nn.Conv2d(lifting_channels, hidden_channels, 1),
#         )
#         self.blocks = nn.ModuleList([
#             FNOBlock2d(hidden_channels, n_modes_height, n_modes_width, use_mlp, mlp_expansion, mlp_dropout,
#                        norm, non_linearity, skip, mlp_skip) for _ in range(n_blocks)
#         ])
#         self.projection = nn.Sequential(
#             nn.Conv2d(hidden_channels, projection_channels, 1), nn.GELU(),
#             nn.Conv2d(projection_channels, out_channels, 1),
#         )

#     def forward(self, x):
#         x = self.lifting(x)
#         for i, block in enumerate(self.blocks):
#             x = block(x, is_last_layer=(i == len(self.blocks) - 1))
#         return self.projection(x)

