"""
Transolver-style physics attention on flattened spatial tokens.

Adapted from Transolver_plus-style architectures (physics attention with learnable
slice routing). Grid fields ``(B, C, *)`` are viewed as token sequences ``(B, N, C)``
with optional normalized coordinate features, matching :class:`BaseModel` contracts.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.checkpoint import checkpoint

from benchmark.models.common import BaseModel
from benchmark.models.act_fno import MultiHeadACTBlock, set_warp_boundary_mode
from benchmark.models.act_fno3d import MultiHeadACTBlock3d
from benchmark.models.deform_bounds import max_displacement_normalized_one_cell

ACTIVATION: dict[str, Callable[[], nn.Module]] = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


def gumbel_softmax(
    logits: torch.Tensor, tau: torch.Tensor | float = 1.0, hard: bool = False
) -> torch.Tensor:
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = logits + gumbel_noise
    if not torch.is_tensor(tau):
        tau = logits.new_tensor(float(tau))
    y = y / tau
    y = F.softmax(y, dim=-1)
    if hard:
        y_hard = torch.argmax(y, dim=-1, keepdim=True)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard, 1.0)
        y = (y_one_hot - y).detach() + y
    return y


def _maybe_all_reduce_sum_(t: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


class PhysicsAttention1DEidetic(nn.Module):
    """Slice-mixing attention over spatial tokens (channels-last ``B, N, C``)."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU(),
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, N, C
        b, n, _ = x.shape
        x_mid = (
            self.in_project_x(x)
            .reshape(b, n, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
        slice_norm = slice_weights.sum(2)
        _maybe_all_reduce_sum_(slice_norm)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        _maybe_all_reduce_sum_(slice_token)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].expand(-1, -1, -1, self.dim_head)
        )

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(
            q_slice_token, k_slice_token, v_slice_token
        )

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverMLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 1,
        act: str = "gelu",
        res: bool = True,
    ):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(f"Unknown activation: {act}")
        act_cls = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_cls())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_cls()) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        return self.linear_post(x)


def _tokens_to_channel_grid(
    fx: torch.Tensor, n_spatial_dims: int, spatial_shape: tuple[int, ...]
) -> torch.Tensor:
    """Point-cloud tokens ``(B, N, C)`` -> channel-first grid ``(B, C, *)`` for conv / ACT."""
    b, n, c = fx.shape
    if n != int(math.prod(spatial_shape)):
        raise ValueError(
            f"Token count N={n} does not match spatial_shape {spatial_shape} (prod={math.prod(spatial_shape)})"
        )
    if n_spatial_dims == 1:
        (l,) = spatial_shape
        return rearrange(fx, "b (l) c -> b c l", l=l)
    if n_spatial_dims == 2:
        h, w = spatial_shape
        return rearrange(fx, "b (h w) c -> b c h w", h=h, w=w)
    if n_spatial_dims == 3:
        d, h, w = spatial_shape
        return rearrange(fx, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
    raise ValueError(f"Unsupported n_spatial_dims={n_spatial_dims}")


def _channel_grid_to_tokens(
    grid: torch.Tensor, n_spatial_dims: int, spatial_shape: tuple[int, ...]
) -> torch.Tensor:
    """Channel-first grid ``(B, C, *)`` -> tokens ``(B, N, C)``."""
    if n_spatial_dims == 1:
        (l,) = spatial_shape
        return rearrange(grid, "b c l -> b (l) c", l=l)
    if n_spatial_dims == 2:
        h, w = spatial_shape
        return rearrange(grid, "b c h w -> b (h w) c", h=h, w=w)
    if n_spatial_dims == 3:
        d, h, w = spatial_shape
        return rearrange(grid, "b c d h w -> b (d h w) c", d=d, h=h, w=w)
    raise ValueError(f"Unsupported n_spatial_dims={n_spatial_dims}")


def _make_deform_block_transolver(
    n_spatial_dims: int,
    channels: int,
    spatial_resolution: tuple[int, ...],
    num_heads: int,
    disp_net_branched: bool,
    max_disp: float | None,
    deform_hidden_dim: int,
    skip_output_norm: bool = False,
) -> nn.Module | None:
    if n_spatial_dims not in (2, 3):
        return None
    deform_max = float(max_disp) if max_disp is not None else max_displacement_normalized_one_cell(
        spatial_resolution
    )
    if n_spatial_dims == 2:
        return MultiHeadACTBlock(
            channels,
            channels,
            num_heads=num_heads,
            hidden_dim=deform_hidden_dim,
            max_disp=deform_max,
            use_norm=True,
            disp_net_branched=disp_net_branched,
            skip_output_norm=skip_output_norm,
        )
    return MultiHeadACTBlock3d(
        channels,
        channels,
        num_heads=num_heads,
        hidden_dim=deform_hidden_dim,
        max_disp=deform_max,
        use_norm=True,
        disp_net_branched=disp_net_branched,
        skip_output_norm=skip_output_norm,
    )


class TransolverPlusBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: float = 4.0,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
        gradient_checkpointing: bool = False,
        head_on_grid: bool = False,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.head_on_grid = bool(head_on_grid)
        self.gradient_checkpointing = gradient_checkpointing
        dim_head = hidden_dim // num_heads
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention1DEidetic(
            hidden_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = TransolverMLP(
            hidden_dim,
            int(hidden_dim * mlp_ratio),
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer and not self.head_on_grid:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
        elif self.last_layer:
            self.ln_3 = None
            self.mlp2 = None

    def _sub_forward_attn_mlp(self, fx: torch.Tensor) -> torch.Tensor:
        def _attn_branch(z: torch.Tensor) -> torch.Tensor:
            return self.attn(self.ln_1(z))

        def _mlp_branch(z: torch.Tensor) -> torch.Tensor:
            return self.mlp(self.ln_2(z))

        if self.training and self.gradient_checkpointing:
            fx = fx + checkpoint(_attn_branch, fx, use_reentrant=False)
            fx = fx + checkpoint(_mlp_branch, fx, use_reentrant=False)
        else:
            fx = fx + _attn_branch(fx)
            fx = fx + _mlp_branch(fx)
        return fx

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        fx = self._sub_forward_attn_mlp(fx)
        if self.last_layer and self.mlp2 is not None:
            return self.mlp2(self.ln_3(fx))
        return fx


class TransolverPlusBlockWithACT(TransolverPlusBlock):
    """Same as :class:`TransolverPlusBlock`, then optional multi-head ACT on a channel-first grid.

    Tokens ``(B, N, C)`` are reshaped to ``(B, C, H, W)`` or ``(B, C, D, H, W)``, transformed, and
    reshaped back. For ``n_spatial_dims == 1`` there is no 1D ACT implementation; the ACT block is
    skipped and ``disp`` is ``None``.

    Returns ``(fx, disp)`` where ``disp`` may be ``None`` when ACT is disabled.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: float = 4.0,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
        gradient_checkpointing: bool = False,
        *,
        n_spatial_dims: int,
        spatial_shape: tuple[int, ...],
        num_deform_heads: int = 4,
        disp_net_branched: bool = True,
        max_disp: float | None = None,
        deform_hidden_dim: int = 64,
        skip_output_norm: bool = False,
        head_on_grid: bool = False,
    ):
        super().__init__(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            act=act,
            mlp_ratio=mlp_ratio,
            last_layer=last_layer,
            out_dim=out_dim,
            slice_num=slice_num,
            gradient_checkpointing=gradient_checkpointing,
            head_on_grid=head_on_grid,
        )
        self.spatial_shape = spatial_shape
        self.n_spatial_dims = n_spatial_dims
        self.deform = _make_deform_block_transolver(
            n_spatial_dims,
            hidden_dim,
            spatial_shape,
            num_deform_heads,
            disp_net_branched,
            max_disp,
            deform_hidden_dim,
            skip_output_norm=skip_output_norm,
        )

    def forward(self, fx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        fx = self._sub_forward_attn_mlp(fx)
        disp: torch.Tensor | None = None
        if self.deform is not None:
            grid = _tokens_to_channel_grid(fx, self.n_spatial_dims, self.spatial_shape)
            grid, disp = self.deform(grid)
            fx = _channel_grid_to_tokens(grid, self.n_spatial_dims, self.spatial_shape)
        if self.last_layer and self.mlp2 is not None:
            return self.mlp2(self.ln_3(fx)), disp
        return fx, disp


def _reference_distance_features(
    my_pos: torch.Tensor, ref: int, bounds: tuple[float, float, float, float, float, float]
) -> torch.Tensor:
    """``my_pos`` (B, N, 3), returns (B, N, ref**3) distances to a fixed reference grid."""
    batchsize, n_tokens, _ = my_pos.shape
    device, dtype = my_pos.device, my_pos.dtype
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    gridx = torch.linspace(xmin, xmax, ref, device=device, dtype=dtype)
    gridy = torch.linspace(ymin, ymax, ref, device=device, dtype=dtype)
    gridz = torch.linspace(zmin, zmax, ref, device=device, dtype=dtype)
    gx = gridx.reshape(1, ref, 1, 1, 1).expand(batchsize, ref, ref, ref, 1)
    gy = gridy.reshape(1, 1, ref, 1, 1).expand(batchsize, ref, ref, ref, 1)
    gz = gridz.reshape(1, 1, 1, ref, 1).expand(batchsize, ref, ref, ref, 1)
    grid_ref = torch.cat((gx, gy, gz), dim=-1).reshape(batchsize, ref**3, 3)
    dist = torch.sqrt(
        torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)
    )
    return dist.contiguous()


class TransolverPlus(BaseModel):
    """
    Tokenized Transolver-style stack for Well grid data.

    Flattens ``(B, dim_in, *)`` to ``(B, N, dim_in)``, optionally appends per-token
    coordinates and/or reference-grid distance features, runs stacked
    :class:`TransolverPlusBlock`, and reshapes to ``(B, dim_out, *)``.

    Args:
        dim_in: Input channels (after formatter).
        dim_out: Output channels.
        n_spatial_dims: 1, 2, or 3.
        spatial_resolution: Grid shape (length ``n_spatial_dims``).
        n_layers: Number of transformer blocks.
        n_hidden: Hidden width on tokens.
        dropout: Dropout inside attention / output projection.
        n_head: Number of attention heads (``n_hidden`` must be divisible by ``n_head``).
        act: Activation name for MLP branches.
        mlp_ratio: Expansion ratio for the token MLP.
        slice_num: Number of physics-attention slices.
        ref: Reference grid side length when ``unified_pos`` is True (``ref**3`` extra features).
        unified_pos: If True, concatenate distance features to a ``ref``³ reference grid.
        gradient_checkpointing: Checkpoint attention+MLP inside each block when training.
        use_act: If True, use :class:`TransolverPlusBlockWithACT` (2D/3D; 1D leaves ACT off).
        num_deform_heads: ACT-style deform heads (separate from attention ``n_head``).
        deform_disp_net_branched: Branching displacement tower (see :class:`MultiHeadACTBlock`).
        deform_max_disp: If set, caps deform magnitude; else uses :func:`max_displacement_normalized_one_cell`.
        deform_hidden_dim: Hidden width inside the displacement tower.
        projection_channels: When ``use_act`` is True, middle width of the final
            ``ConvNd(1x1) -> GELU -> ConvNd(1x1)`` head on the channel-first grid
            (same pattern as :class:`ACTFnoBackbone` projection). Ignored when
            ``use_act`` is False.
        boundary_mode: Warp sampling outside the domain for every 2D
            :class:`MultiHeadACTBlock`. ``"periodic"`` folds coordinates;
            ``"zeros"`` leaves them out of range so samples decay to 0.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: float = 4.0,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = False,
        unified_pos_bounds: tuple[float, float, float, float, float, float] = (
            -1.5,
            1.5,
            0.0,
            2.0,
            -4.0,
            4.0,
        ),
        gradient_checkpointing: bool = False,
        use_act: bool = False,
        num_deform_heads: int = 4,
        deform_disp_net_branched: bool = True,
        deform_max_disp: float | None = None,
        deform_hidden_dim: int = 64,
        projection_channels: int = 128,
        boundary_mode: str = "periodic",
    ):
        spatial_resolution = tuple(int(s) for s in spatial_resolution)
        super().__init__(n_spatial_dims, spatial_resolution)
        if n_hidden % n_head != 0:
            raise ValueError(f"n_hidden ({n_hidden}) must be divisible by n_head ({n_head})")
        self.n_spatial_dims = n_spatial_dims
        self.spatial_shape = tuple(int(s) for s in spatial_resolution)
        self.num_tokens = int(math.prod(self.spatial_shape))
        self.ref = ref
        self.unified_pos = unified_pos
        self.unified_pos_bounds = unified_pos_bounds
        self.gradient_checkpointing = gradient_checkpointing
        self.use_act = use_act
        self.projection_channels = int(projection_channels)

        if self.unified_pos:
            pre_in = dim_in + ref * ref * ref
        else:
            pre_in = dim_in + n_spatial_dims

        self.preprocess = TransolverMLP(
            pre_in, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act
        )
        blocks: list[nn.Module] = []
        head_on_grid = bool(use_act)
        for i in range(n_layers):
            last = i == n_layers - 1
            if use_act:
                blocks.append(
                    TransolverPlusBlockWithACT(
                        num_heads=n_head,
                        hidden_dim=n_hidden,
                        dropout=dropout,
                        act=act,
                        mlp_ratio=mlp_ratio,
                        out_dim=dim_out,
                        slice_num=slice_num,
                        last_layer=last,
                        gradient_checkpointing=gradient_checkpointing,
                        n_spatial_dims=n_spatial_dims,
                        spatial_shape=self.spatial_shape,
                        num_deform_heads=num_deform_heads,
                        disp_net_branched=deform_disp_net_branched,
                        max_disp=deform_max_disp,
                        deform_hidden_dim=deform_hidden_dim,
                        skip_output_norm=last,
                        head_on_grid=head_on_grid and last,
                    )
                )
            else:
                blocks.append(
                    TransolverPlusBlock(
                        num_heads=n_head,
                        hidden_dim=n_hidden,
                        dropout=dropout,
                        act=act,
                        mlp_ratio=mlp_ratio,
                        out_dim=dim_out,
                        slice_num=slice_num,
                        last_layer=last,
                        gradient_checkpointing=gradient_checkpointing,
                        head_on_grid=False,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        if use_act:
            pc = self.projection_channels
            if n_spatial_dims == 1:
                self.projection = nn.Sequential(
                    nn.Conv1d(n_hidden, pc, 1),
                    nn.GELU(),
                    nn.Conv1d(pc, dim_out, 1),
                )
            elif n_spatial_dims == 2:
                self.projection = nn.Sequential(
                    nn.Conv2d(n_hidden, pc, 1),
                    nn.GELU(),
                    nn.Conv2d(pc, dim_out, 1),
                )
            else:
                self.projection = nn.Sequential(
                    nn.Conv3d(n_hidden, pc, 1),
                    nn.GELU(),
                    nn.Conv3d(pc, dim_out, 1),
                )
        else:
            self.projection = None
        self.placeholder = nn.Parameter((1.0 / n_hidden) * torch.randn(n_hidden))
        self.apply(self._init_weights)
        if boundary_mode not in ("periodic", "zeros"):
            raise ValueError(
                f"boundary_mode must be 'periodic' or 'zeros', got {boundary_mode!r}"
            )
        self.boundary_mode = boundary_mode
        set_warp_boundary_mode(self, boundary_mode)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
            if getattr(m, "weight", None) is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _tokens_from_grid(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """``x`` (B, C, *spatial) -> (B, N, C), spatial_shape."""
        b, c, *spatial = x.shape
        spatial_tuple = tuple(int(s) for s in spatial)
        if spatial_tuple != self.spatial_shape:
            raise ValueError(
                f"Input spatial {spatial_tuple} does not match model spatial_resolution {self.spatial_shape}"
            )
        if self.n_spatial_dims == 1:
            x_seq = rearrange(x, "b c l -> b l c")
        elif self.n_spatial_dims == 2:
            x_seq = rearrange(x, "b c h w -> b (h w) c")
        elif self.n_spatial_dims == 3:
            x_seq = rearrange(x, "b c d h w -> b (d h w) c")
        else:
            raise ValueError(f"Unsupported n_spatial_dims={self.n_spatial_dims}")
        return x_seq, spatial_tuple

    def _normalized_coords(self, b: int, spatial_tuple: tuple[int, ...], ref: torch.Tensor) -> torch.Tensor:
        """Per-token coordinates in ``[-1, 1]^{n_spatial}`` on ``ref``'s device/dtype."""
        device, dtype = ref.device, ref.dtype
        axes = [torch.linspace(-1, 1, s, device=device, dtype=dtype) for s in spatial_tuple]
        grids = torch.meshgrid(*axes, indexing="ij")
        pos = torch.stack(grids, dim=-1)
        pos = pos.reshape(-1, len(spatial_tuple))
        return pos.unsqueeze(0).expand(b, -1, -1)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        collect_disps = not self.training
        x_seq, spatial_tuple = self._tokens_from_grid(x)
        b, n, _ = x_seq.shape
        if self.unified_pos:
            pos_xyz = self._normalized_coords(b, spatial_tuple, x_seq)
            if pos_xyz.shape[-1] != 3:
                if pos_xyz.shape[-1] == 2:
                    pos_xyz = F.pad(pos_xyz, (0, 1))
                elif pos_xyz.shape[-1] == 1:
                    pos_xyz = F.pad(pos_xyz, (0, 2))
                else:
                    raise ValueError("unified_pos reference grid expects positions with 3 components")
            new_pos = _reference_distance_features(pos_xyz, self.ref, self.unified_pos_bounds)
            feats = torch.cat((x_seq, new_pos), dim=-1)
        else:
            pos = self._normalized_coords(b, spatial_tuple, x_seq)
            feats = torch.cat((x_seq, pos), dim=-1)

        fx = self.preprocess(feats)
        fx = fx + self.placeholder[None, None, :]

        all_disps: list[torch.Tensor] = []
        for block in self.blocks:
            if self.use_act:
                fx, disp = block(fx)
                if collect_disps and disp is not None:
                    all_disps.append(disp)
            else:
                fx = block(fx)

        if self.n_spatial_dims == 1:
            grid = rearrange(fx, "b (l) c -> b c l", l=spatial_tuple[0])
        elif self.n_spatial_dims == 2:
            h, w = spatial_tuple
            grid = rearrange(fx, "b (h w) c -> b c h w", h=h, w=w)
        else:
            d, h, w = spatial_tuple
            grid = rearrange(fx, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
        if self.projection is not None:
            out = self.projection(grid)
        else:
            out = grid
        if self.use_act and collect_disps:
            return out, all_disps
        return out
