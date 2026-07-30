from functools import partial

from huggingface_hub import pause_space

from benchmark.models.act_fno import CONCAT_IDENTITY_COORDS_FOR_DISP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from benchmark.models.common import BaseModel
from benchmark.models.deform_bounds import max_displacement_normalized_one_cell


class _PassThroughACT3d(nn.Module):
    """No 3D ACT displacement block; ``forward`` returns ``(x, None)`` like :class:`MultiHeadACTBlock3d`."""

    def forward(self, x: torch.Tensor):
        return x, None


class MultiHeadACTBlock3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=4,
        hidden_dim=64,
        max_disp=0.5,
        use_norm=True,
        disp_net_branched: bool = True,
        skip_output_norm: bool = False,
        hidden_channels=16,
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
        extra_disp_c = 3 if CONCAT_IDENTITY_COORDS_FOR_DISP else 0
        disp_in = self.hidden_channels + extra_disp_c
        if disp_net_branched:
            h = max(hidden_dim // 3, 8)
            self.disp_net = nn.ModuleDict(
                {
                    "branch_1x1": nn.Conv3d(disp_in, h, 1),
                    "branch_3x3": nn.Conv3d(disp_in, h, 3, padding=1),
                    "branch_3x3_d2": nn.Conv3d(disp_in, h, 3, padding=2, dilation=2),
                }
            )
            fuse_in = h * 3
        else:
            raise ValueError("disp_net_branched must be True")
            
        self.disp_fuse = nn.Sequential(
            nn.GELU(),
            nn.Conv3d(fuse_in, 3, 1),
        )
        nn.init.zeros_(self.disp_fuse[-1].weight)
        nn.init.zeros_(self.disp_fuse[-1].bias)

        self.value_proj = nn.Conv3d(in_channels, num_heads * self.hidden_channels, 1)
        self.out_proj = nn.Conv3d(num_heads * self.hidden_channels, out_channels, 1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.residual_proj = (
            nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self._identity_grid_dhw: tuple[int, int, int, torch.device] | None = None
        self._identity_grid_template: torch.Tensor | None = None

    def create_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized sampling grid: last dim (x, y, z) -> (W, H, D) for grid_sample."""
        _, _, D, H, W = x.shape
        device = x.device
        cache_key = (D, H, W, device)
        tpl = self._identity_grid_template
        if (
            self._identity_grid_dhw != cache_key
            or tpl is None
            or tpl.device != device
        ):
            zs = torch.linspace(-1, 1, D, device=device)
            ys = torch.linspace(-1, 1, H, device=device)
            xs = torch.linspace(-1, 1, W, device=device)
            grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
            # (1, 1, D, H, W, 3): broadcast with delta (B, nh, D, H, W, 3) — no expand/unsqueeze per forward.
            self._identity_grid_template = (
                torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0).unsqueeze(0)
            )
            self._identity_grid_dhw = cache_key
        return self._identity_grid_template

    def cached_identity_grid_dhw3(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(1, 1, D, H, W, 3)`` normalized (x, y, z), updating the cache if needed."""
        return self.create_grid(x)

    def identity_coord_channels_for_disp(self, x: torch.Tensor, n_pack: int) -> torch.Tensor:
        """``(n_pack, 3, D, H, W)`` xyz from :attr:`_identity_grid_template` (via :meth:`cached_identity_grid_dhw3`)."""
        self.cached_identity_grid_dhw3(x)
        tpl = self._identity_grid_template
        assert tpl is not None
        xyz = tpl[0, 0].permute(3, 0, 1, 2).contiguous()
        return xyz.unsqueeze(0).expand(n_pack, -1, -1, -1, -1)

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
        """Per-head offsets in (x,y,z), shape (B, num_heads, 3, D, H, W)."""
        B, _, D, H, W = x.shape
        disp = self.disp_fuse(self._disp_features(x))
        return torch.tanh(disp) * self.max_disp

    def _sampling_grid(self, x: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """grid_sample layout: (n_pack, D, H, W, 3), last dim (x, y, z) for ``F.grid_sample``.

        Same toroidal rule as :meth:`benchmark.models.act_fno.MultiHeadACTBlock._sampling_grid`:
        fold with ``remainder`` only where coordinates are strictly outside ``[-1, 1]``.
        """
        # disp (B, nh, 3, D, H, W) -> (B, nh, D, H, W, 3); broadcast with (1,1,D,H,W,3) identity grid
        delta = disp.movedim(2, -1)
        grid = self.create_grid(x) + delta
        wrapped = torch.remainder(grid + 1.0, 2.0) - 1.0
        outside = (grid > 1.0) | (grid < -1.0)
        return torch.where(outside, wrapped, grid)

    def _grid_sample_heads(
        self,
        values: torch.Tensor,
        sample_grid: torch.Tensor,
        *,
        B: int,
        nh: int,
        D: int,
        H: int,
        W: int,
        C: int,
    ) -> torch.Tensor:
        """values: (B, nh, C, D, H, W); sample_grid: (B, nh, D, H, W, 3) -> (B, nh*C, D, H, W)."""
        n_pack = B * nh
        grid = sample_grid.reshape(n_pack, D, H, W, 3)
        vals = values.reshape(n_pack, C, D, H, W)
        if not grid.is_contiguous():
            grid = grid.contiguous()
        if not vals.is_contiguous():
            vals = vals.contiguous()
        deformed = F.grid_sample(
            vals,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return deformed.view(B, nh * C, D, H, W)

    def forward(self, x: torch.Tensor):
        B, _, D, H, W = x.shape
        nh = self.num_heads
        C = self.hidden_channels
        n_pack = B * nh
        
        value_heads = self.value_proj(x).reshape(n_pack, C, D, H, W)
        if CONCAT_IDENTITY_COORDS_FOR_DISP:
            coords = self.identity_coord_channels_for_disp(x, n_pack)
            disp_in = torch.cat([value_heads, coords], dim=1)
        else:
            disp_in = value_heads
        disp = self._predict_displacement(disp_in).view(B, nh, 3, D, H, W)

        sample_grid = self._sampling_grid(x, disp)
        deformed = self._grid_sample_heads(
            value_heads, sample_grid, B=B, nh=nh, D=D, H=H, W=W, C=C
        )
        out = self.out_proj(deformed)
        residual = self.residual_proj(x) if self.residual_proj is not None else x
        merged = residual + out
        if self.use_norm and not self.skip_output_norm:
            out = self.norm(merged)
        else:
            out = merged
        if self.training:
            return out, None
        return out, disp


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_d, modes_h, modes_w):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.half_n_modes_d = modes_d // 2
        self.half_n_modes_h = modes_h // 2
        self.half_n_modes_w = modes_w // 2
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        self.weight00 = nn.Parameter(
            init_std
            * torch.randn(
                in_channels,
                out_channels,
                self.half_n_modes_d,
                self.half_n_modes_h,
                self.half_n_modes_w,
                2,
            )
        )
        self.weight01 = nn.Parameter(
            init_std
            * torch.randn(
                in_channels,
                out_channels,
                self.half_n_modes_d,
                self.half_n_modes_h,
                self.half_n_modes_w,
                2,
            )
        )
        self.weight10 = nn.Parameter(
            init_std
            * torch.randn(
                in_channels,
                out_channels,
                self.half_n_modes_d,
                self.half_n_modes_h,
                self.half_n_modes_w,
                2,
            )
        )
        self.weight11 = nn.Parameter(
            init_std
            * torch.randn(
                in_channels,
                out_channels,
                self.half_n_modes_d,
                self.half_n_modes_h,
                self.half_n_modes_w,
                2,
            )
        )

    def compl_mul3d(self, input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w_real, w_imag = weights[..., 0], weights[..., 1]
        in_r, in_i = input_ft.real, input_ft.imag
        out_r = (
            torch.einsum("bidhw,iodhw->bodhw", in_r, w_real)
            - torch.einsum("bidhw,iodhw->bodhw", in_i, w_imag)
        )
        out_i = (
            torch.einsum("bidhw,iodhw->bodhw", in_r, w_imag)
            + torch.einsum("bidhw,iodhw->bodhw", in_i, w_real)
        )
        return torch.complex(out_r, out_i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize, _, depth, height, width = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="forward")
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            depth,
            height,
            width // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        d_modes = min(self.half_n_modes_d, depth)
        h_modes = min(self.half_n_modes_h, height)
        w_modes = min(self.half_n_modes_w, width // 2 + 1)

        if d_modes > 0 and h_modes > 0 and w_modes > 0:
            out_ft[..., :d_modes, :h_modes, :w_modes] = self.compl_mul3d(
                x_ft[..., :d_modes, :h_modes, :w_modes],
                self.weight00[..., :d_modes, :h_modes, :w_modes, :],
            )
            out_ft[..., :d_modes, -h_modes:, :w_modes] = self.compl_mul3d(
                x_ft[..., :d_modes, -h_modes:, :w_modes],
                self.weight01[..., :d_modes, :h_modes, :w_modes, :],
            )
            out_ft[..., -d_modes:, :h_modes, :w_modes] = self.compl_mul3d(
                x_ft[..., -d_modes:, :h_modes, :w_modes],
                self.weight10[..., :d_modes, :h_modes, :w_modes, :],
            )
            out_ft[..., -d_modes:, -h_modes:, :w_modes] = self.compl_mul3d(
                x_ft[..., -d_modes:, -h_modes:, :w_modes],
                self.weight11[..., :d_modes, :h_modes, :w_modes, :],
            )

        return torch.fft.irfftn(out_ft, s=(depth, height, width), dim=(-3, -2, -1), norm="forward")


class SoftGating3d(nn.Module):
    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, channels, 1, 1, 1)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias if self.bias is not None else self.weight * x


def make_skip_connection_3d(
    in_channels: int, out_channels: int, skip_type: str = "soft-gating"
) -> nn.Module:
    skip_type = skip_type.lower()
    if skip_type == "soft-gating":
        if in_channels != out_channels:
            raise ValueError(
                f"Soft-gating requires in_channels==out_channels, got {in_channels} vs {out_channels}"
            )
        return SoftGating3d(in_channels)
    if skip_type == "linear":
        return nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
    if skip_type == "identity":
        if in_channels != out_channels:
            raise ValueError(
                f"Identity skip requires in_channels==out_channels, got {in_channels} vs {out_channels}"
            )
        return nn.Identity()
    raise ValueError(f"Unsupported skip_type={skip_type}")


class FNOBlock3d(nn.Module):
    def __init__(
        self,
        channels,
        modes_d,
        modes_h,
        modes_w,
        use_mlp=False,
        mlp_expansion=0.5,
        mlp_dropout=0.0,
        norm=None,
        non_linearity=F.gelu,
        fno_skip="linear",
        mlp_skip="soft-gating",
    ):
        super().__init__()
        self.non_linearity = non_linearity
        self.use_mlp = use_mlp
        self.spectral_conv = SpectralConv3d(channels, channels, modes_d, modes_h, modes_w)
        self.fno_skip = make_skip_connection_3d(channels, channels, skip_type=fno_skip)
        self.norm_fno = (
            nn.GroupNorm(1, channels)
            if norm == "group_norm"
            else (nn.BatchNorm3d(channels) if norm == "batch_norm" else None)
        )
        self.norm_mlp = (
            nn.GroupNorm(1, channels)
            if use_mlp and norm == "group_norm"
            else (nn.BatchNorm3d(channels) if use_mlp and norm == "batch_norm" else None)
        )
        if use_mlp:
            hidden_mlp = round(channels * mlp_expansion)
            self.mlp = nn.Sequential(
                nn.Conv3d(channels, hidden_mlp, 1),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Conv3d(hidden_mlp, channels, 1),
            )
            self.mlp_skip = make_skip_connection_3d(channels, channels, skip_type=mlp_skip)
        else:
            self.mlp = None
            self.mlp_skip = None

    def forward(self, x: torch.Tensor, is_last_layer: bool = False):
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


class ACTFnoBackbone3d(nn.Module):
    def __init__(
        self,
        n_modes_d,
        n_modes_h,
        n_modes_w,
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
        spatial_resolution: tuple[int, int, int] | None = None,
        max_disp: float | None = None,
        use_act: bool = True,
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
            nn.Conv3d(in_channels, lifting_channels, 1),
            nn.GELU(),
            nn.Conv3d(lifting_channels, hidden_channels, 1),
        )
        self.blocks = nn.ModuleList()
        self.act_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(
                FNOBlock3d(
                    hidden_channels,
                    n_modes_d,
                    n_modes_h,
                    n_modes_w,
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
                    MultiHeadACTBlock3d(
                        hidden_channels,
                        hidden_channels,
                        num_heads,
                        use_norm=True,
                        disp_net_branched=disp_net_branched,
                        max_disp=deform_max_disp,
                        skip_output_norm=(i == n_blocks - 1),
                    )
                )
            else:
                self.act_blocks.append(_PassThroughACT3d())
        self.projection = nn.Sequential(
            nn.Conv3d(hidden_channels, projection_channels, 1),
            nn.GELU(),
            nn.Conv3d(projection_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor):
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


class ACTFNO3D(BaseModel):
    """FNO + optional multi-head ACT blocks for volumetric (D, H, W) fields.

    Set ``use_act=False`` for a pure FNO stack (see :class:`ACTFnoBackbone3d`).
    """

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
    ):
        super().__init__(n_spatial_dims, spatial_resolution)
        if n_spatial_dims != 3:
            raise ValueError(
                f"ACTFNO3D expects n_spatial_dims==3, got {n_spatial_dims}"
            )
        self.gradient_checkpointing = gradient_checkpointing
        self.model = ACTFnoBackbone3d(
            n_modes_d=modes1,
            n_modes_h=modes2,
            n_modes_w=modes3,
            in_channels=dim_in,
            out_channels=dim_out,
            hidden_channels=hidden_channels,
            disp_net_branched=disp_net_branched,
            checkpoint_fno=gradient_checkpointing,
            spatial_resolution=(
                spatial_resolution[0],
                spatial_resolution[1],
                spatial_resolution[2],
            ),
            max_disp=max_disp,
            use_act=use_act,
        )

    def forward(self, input: torch.Tensor):
        """Returns ``(out, disps)``; ``disps`` is empty while training."""
        return self.model(input)
