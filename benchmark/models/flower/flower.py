"""Flower backbone (vendored from ``flowers.models.flower``) + ACT-NeuralOP wrapper.

``FlowerNet`` is the U-Net-style deform-based operator from the ``flowers`` project.
``Flower`` adapts it to the ACT-NeuralOP :class:`~benchmark.models.common.BaseModel`
interface: constructor ``(dim_in, dim_out, n_spatial_dims, spatial_resolution, ...)``
and ``forward(x) -> tensor``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from benchmark.models.common import BaseModel
from benchmark.models.flower._layers import FlowerBlock
from benchmark.models.flower._utils import create_grid, derive_padding_strategies

conv_modules = {2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}


class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_resolution,
        num_heads,
        padding_modes,
        use_meta_conditioning=False,
        meta_dim=1,
        groups=32,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution

        self.shift_layer = FlowerBlock(
            in_channels,
            in_channels,
            spatial_resolution=spatial_resolution,
            num_heads=num_heads,
            padding_modes=padding_modes,
            use_meta_conditioning=use_meta_conditioning,
            dropout_rate=dropout_rate,
            meta_dim=meta_dim,
            groups=groups,
        )
        self.downsample = conv_modules[self.n_spatial_dims](
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, meta=None):
        x_down = self.shift_layer(x, meta)
        x_down = self.act(self.downsample(x_down))
        return x, x_down


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        spatial_resolution,
        num_heads,
        padding_modes,
        use_meta_conditioning=False,
        meta_dim=1,
        groups=32,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution

        self.fio = FlowerBlock(
            in_channels,
            out_channels - skip_channels,
            spatial_resolution=spatial_resolution,
            num_heads=num_heads,
            padding_modes=padding_modes,
            use_meta_conditioning=use_meta_conditioning,
            dropout_rate=dropout_rate,
            meta_dim=meta_dim,
            groups=groups,
        )
        self.upsample = conv_transpose_modules[self.n_spatial_dims](
            out_channels - skip_channels,
            out_channels - skip_channels,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, meta=None):
        x = self.fio(x, meta)
        x = self.act(self.upsample(x))
        return x


class FlowerNet(nn.Module):
    """U-Net-style deform-based neural operator (the ``flowers`` ``Flower`` model)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution,
        lifting_dim=96,
        n_levels=4,
        num_heads=32,
        boundary_condition_types=["PERIODIC"],
        dim_meta=0,
        groups=32,
        dropout_rate=0.00,
    ):
        super().__init__()
        # Copy so we never mutate a caller-owned (or metadata) resolution container.
        spatial_resolution = list(spatial_resolution)

        self.input_dim = dim_in
        self.output_dim = dim_out
        self.lifting_dim = lifting_dim
        self.n_levels = n_levels
        self.dim_meta = dim_meta
        self.use_meta_conditioning = dim_meta > 0
        self.num_heads = num_heads
        self.boundary_condition_types = boundary_condition_types
        self.padding_type = derive_padding_strategies(
            self.boundary_condition_types, n_spatial_dims=n_spatial_dims
        )
        self.n_spatial_dims = n_spatial_dims
        self.spatial_resolution = spatial_resolution
        self.groups = groups

        min_divisor = 2 ** (n_levels - 1)
        for i, d_i in enumerate(spatial_resolution):
            if d_i % min_divisor != 0:
                if d_i == 66:  # post_neutron_star_merger
                    spatial_resolution[i] = 64
                    continue
                else:
                    raise ValueError(
                        f"Dimension {i} must be divisible by 2^(n_levels-1) = {min_divisor}, "
                        f"or n_levels={n_levels}, but is {d_i}."
                    )

        coord_dim = self.n_spatial_dims
        if self.use_meta_conditioning:
            coord_dim += dim_meta

        self.lift = conv_modules[self.n_spatial_dims](
            dim_in + coord_dim, lifting_dim, kernel_size=1
        )

        channel_multipliers = [2.0**i for i in range(n_levels)]
        self.encoder_channels = [int(lifting_dim * mult) for mult in channel_multipliers]

        self.encoder_blocks = nn.ModuleList()
        current_spatial_resolution = self.spatial_resolution

        for i in range(n_levels - 1):
            in_ch = self.encoder_channels[i]
            out_ch = self.encoder_channels[i + 1]
            self.encoder_blocks.append(
                DownsampleBlock(
                    in_ch,
                    out_ch,
                    current_spatial_resolution,
                    num_heads,
                    self.padding_type,
                    self.use_meta_conditioning,
                    meta_dim=dim_meta,
                    groups=self.groups,
                    dropout_rate=dropout_rate,
                )
            )
            current_spatial_resolution = [d_i // 2 for d_i in current_spatial_resolution]

        bottleneck_channels = self.encoder_channels[-1]
        self.bottleneck = FlowerBlock(
            bottleneck_channels,
            bottleneck_channels,
            spatial_resolution=current_spatial_resolution,
            num_heads=num_heads,
            padding_modes=self.padding_type,
            use_meta_conditioning=self.use_meta_conditioning,
            meta_dim=dim_meta,
            groups=groups,
            dropout_rate=dropout_rate,
        )

        self.decoder_blocks = nn.ModuleList()
        decoder_in_ch = bottleneck_channels

        for i in range(n_levels - 1, 0, -1):
            skip_ch = self.encoder_channels[i - 1]
            out_ch = skip_ch * 2
            self.decoder_blocks.append(
                UpsampleBlock(
                    decoder_in_ch,
                    skip_ch,
                    out_ch,
                    current_spatial_resolution,
                    num_heads,
                    self.padding_type,
                    self.use_meta_conditioning,
                    meta_dim=dim_meta,
                    groups=self.groups,
                    dropout_rate=dropout_rate,
                )
            )
            decoder_in_ch = out_ch
            current_spatial_resolution = [2 * d_i for d_i in current_spatial_resolution]

        self.project = nn.Sequential(
            conv_modules[self.n_spatial_dims](out_ch, lifting_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_modules[self.n_spatial_dims](lifting_dim, dim_out, kernel_size=1),
        )
        self.dropout = (
            nn.Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)
        )

        coord_grid = rearrange(
            create_grid(spatial_resolution),
            "... xy -> 1 xy ...",
        ).contiguous()
        self.register_buffer("coord_grid", coord_grid)

    def forward(self, x, meta=None):
        batchsize = x.shape[0]

        needs_66_interpolation = self.n_spatial_dims == 3 and x.shape[-1] == 66
        if needs_66_interpolation:
            x = F.interpolate(
                x,
                size=(x.shape[2], x.shape[3], 64),
                mode="trilinear",
                align_corners=False,
            )

        expand_shape = (batchsize, -1) + (-1,) * self.n_spatial_dims
        grid = self.coord_grid.expand(*expand_shape)

        if self.use_meta_conditioning and meta is not None:
            view_shape = (batchsize, self.dim_meta) + (1,) * self.n_spatial_dims
            expand_shape = (-1, -1) + tuple(self.spatial_resolution)
            meta_broadcast = meta.view(*view_shape).expand(*expand_shape)
            grid = torch.cat((grid, meta_broadcast), dim=1)

        x = torch.cat((x, grid), dim=1)
        x = self.lift(x)

        skip_connections = []
        for encoder_block in self.encoder_blocks:
            skip, x = encoder_block(x, meta)
            skip_connections.append(skip)

        x = self.bottleneck(x, meta)

        for decoder_block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder_block(x, meta)
            x = torch.cat([x, skip], dim=1)

        x = self.dropout(x)
        prediction = self.project(x)

        if needs_66_interpolation:
            prediction = F.interpolate(
                prediction,
                size=(prediction.shape[2], prediction.shape[3], 66),
                mode="trilinear",
                align_corners=False,
            )

        return prediction


class Flower(BaseModel):
    """ACT-NeuralOP wrapper around :class:`FlowerNet`.

    Matches the model-instantiation contract in ``benchmark/train.py``
    (``dim_in``, ``dim_out``, ``n_spatial_dims``, ``spatial_resolution`` are
    injected) and returns a plain tensor prediction (channels-first).
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        lifting_dim: int = 96,
        n_levels: int = 4,
        num_heads: int = 32,
        groups: int = 32,
        dropout_rate: float = 0.0,
        boundary_condition_types: list[str] | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(n_spatial_dims, spatial_resolution)
        if boundary_condition_types is None:
            boundary_condition_types = ["PERIODIC"]
        self.net = FlowerNet(
            dim_in=dim_in,
            dim_out=dim_out,
            n_spatial_dims=n_spatial_dims,
            spatial_resolution=spatial_resolution,
            lifting_dim=lifting_dim,
            n_levels=n_levels,
            num_heads=num_heads,
            boundary_condition_types=boundary_condition_types,
            dim_meta=0,
            groups=groups,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x, meta=None)
