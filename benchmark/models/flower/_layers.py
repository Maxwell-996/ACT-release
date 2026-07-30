"""Flower layers (vendored from ``flowers.models.flower_layers``)."""

import torch.nn as nn
from einops import rearrange

from benchmark.models.flower._utils import (
    FiLM,
    create_grid,
    custom_grid_sample2d,
    custom_grid_sample3d,
)

conv_modules = {2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}

flow_fold_heads = {
    2: "B (heads dir) H W -> (B heads) H W dir",
    3: "B (heads dir) D H W -> (B heads) D H W dir",
}
value_fold_heads = {
    2: "B (heads C_i) H W -> (B heads) C_i H W",
    3: "B (heads C_i) D H W -> (B heads) C_i D H W",
}
value_unfold_heads = {
    2: "(B heads) C_i H W -> B (heads C_i) H W",
    3: "(B heads) C_i D H W -> B (heads C_i) D H W",
}
grid_samples = {2: custom_grid_sample2d, 3: custom_grid_sample3d}


class SelfDeform(nn.Module):
    """u(x) -> u(x + delta(x)), powered by grid_sample."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_resolution,
        num_heads=32,
        padding_modes=["zeros", "zeros"],
        meta_dim=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution
        self.padding_modes = padding_modes
        self.num_heads = num_heads

        self.flow_head = nn.Sequential(
            conv_modules[self.n_spatial_dims](in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_modules[self.n_spatial_dims](
                out_channels, self.n_spatial_dims * self.num_heads, kernel_size=1
            ),
        )
        self.value_head = conv_modules[self.n_spatial_dims](
            in_channels, out_channels, kernel_size=1
        )

        base_grid = create_grid(self.spatial_resolution)
        self.register_buffer("base_grid", base_grid)

    def flow(self, u, return_value=False):
        flow = self.flow_head(u)
        if return_value:
            value = self.value_head(u)
            return flow, value
        return flow

    def forward(self, u, meta=None):
        flow = self.flow_head(u)
        value = self.value_head(u)

        flow = rearrange(
            flow,
            flow_fold_heads[self.n_spatial_dims],
            dir=self.n_spatial_dims,
            heads=self.num_heads,
        )
        base_grid_batch = self.base_grid.unsqueeze(0)
        grid = base_grid_batch + flow

        value = rearrange(
            value, value_fold_heads[self.n_spatial_dims], heads=self.num_heads
        )
        u_deform = grid_samples[self.n_spatial_dims](
            value,
            grid,
            mode="bilinear",
            padding_modes=self.padding_modes,
            align_corners=True,
        )
        u_deform = rearrange(
            u_deform, value_unfold_heads[self.n_spatial_dims], heads=self.num_heads
        )
        return u_deform


class FlowerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_resolution,
        num_heads,
        padding_modes,
        use_meta_conditioning,
        dropout_rate=0.0,
        groups=32,
        meta_dim=1,
        film_norm_type="group",
    ):
        super().__init__()
        self.self_deform = SelfDeform(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_resolution=spatial_resolution,
            num_heads=num_heads,
            padding_modes=padding_modes,
            meta_dim=meta_dim,
        )
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution
        self.id = conv_modules[self.n_spatial_dims](in_channels, out_channels, 1)
        self.use_meta_conditioning = use_meta_conditioning

        if use_meta_conditioning:
            self.norm = FiLM(
                num_channels=out_channels,
                meta_dim=meta_dim,
                norm_type=film_norm_type,
                num_groups=groups,
            )
        else:
            self.norm = nn.GroupNorm(
                num_groups=groups, num_channels=out_channels, affine=True
            )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, meta=None):
        id_ = self.id(x)
        out = self.self_deform(x, meta) + id_

        if self.use_meta_conditioning and meta is not None:
            out = self.norm(out, meta)
        else:
            out = self.norm(out)

        return self.dropout(self.act(out))
