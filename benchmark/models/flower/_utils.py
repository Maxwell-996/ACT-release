"""Utilities for the Flower backbone (vendored from the ``flowers`` project).

Combines ``flowers.models.utils.common`` (grid, padding, FiLM) and
``flowers.models.utils.grid_samples`` (periodic-aware grid_sample). See the
original ``flowers`` repository for the source these are adapted from.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def create_grid(spatial_resolution):
    """Create a coordinate grid in [-1, 1] range."""
    meshgrid = torch.meshgrid(
        *[torch.linspace(-1.0, 1.0, d_i) for d_i in spatial_resolution]
    )
    grid = torch.stack(meshgrid[::-1], dim=-1)  # reverse the meshgrid list
    return grid


def derive_padding_strategies(bc_list, n_spatial_dims):
    """Convert boundary condition list to padding strategies."""
    if not bc_list:
        return ["zeros" for _ in range(n_spatial_dims)]

    padding = ["periodic" if bc.upper() == "PERIODIC" else "zeros" for bc in bc_list]

    # If only one element, duplicate it
    if len(padding) == 1:
        padding = [padding[0] for _ in range(n_spatial_dims)]

    return padding


class InstanceNorm(nn.Module):
    """Dimension-agnostic instance normalization layer for neural operators."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer with flexible normalization."""

    def __init__(
        self,
        num_channels,
        meta_dim=1,
        norm_type="layer",
        num_groups=32,
        eps=1e-6,
        data_format="channels_first",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.meta_dim = meta_dim
        self.norm_type = norm_type.lower()
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (num_channels,)

        if self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=False)
        elif self.norm_type == "instance":
            self.norm = InstanceNorm()
        elif self.norm_type == "identity":
            self.norm = nn.Identity()
        else:
            raise ValueError(
                f"norm_type must be 'group', 'layer', 'instance', or 'identity', got {norm_type}"
            )

        self.weight = nn.Linear(meta_dim, num_channels)
        self.bias = nn.Linear(meta_dim, num_channels)

    def forward(self, x, meta=None):
        if self.norm_type in ["group", "instance"]:
            if self.data_format == "channels_last":
                x_for_norm = x.permute(0, -1, *range(1, x.dim() - 1))
                x_for_norm = self.norm(x_for_norm)
                x = x_for_norm.permute(0, *range(2, x.dim()), 1)
            else:
                x = self.norm(x)
        elif self.norm_type == "layer":
            if self.data_format == "channels_last":
                x = self.norm(x)
            else:
                x_for_norm = x.permute(0, *range(2, x.dim()), 1)
                x_for_norm = self.norm(x_for_norm)
                x = x_for_norm.permute(0, -1, *range(1, x.dim() - 1))
        elif self.norm_type == "identity":
            x = self.norm(x)
        else:
            raise ValueError()

        if meta is None:
            return x

        if meta.dim() == 1:
            meta = meta.unsqueeze(-1)

        meta = meta.type_as(x)
        weight = self.weight(meta)
        bias = self.bias(meta)

        if self.data_format == "channels_last":
            while weight.dim() < x.dim():
                weight = weight.unsqueeze(1)
                bias = bias.unsqueeze(1)
            return weight * x + bias
        else:
            while weight.dim() < x.dim():
                weight = weight.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
            return weight * x + bias


def custom_grid_sample2d(
    input, grid, mode="bilinear", padding_modes=("zeros", "zeros"), align_corners=False
):
    """grid_sample with per-dimension periodic boundary support (2D).

    ``padding_modes`` is a ``(padding_h, padding_w)`` tuple, each one of
    ``{'zeros', 'border', 'reflection', 'periodic'}``. Periodic dims are handled by
    minimal wrap-around padding plus modulo coordinate mapping.
    """
    padding_h, padding_w = padding_modes

    if padding_w.lower() != "periodic" and padding_h.lower() != "periodic":
        if padding_w != padding_h:
            msg = "When using different paddings for x and y dimensions, one should be `periodic`."
            raise ValueError(msg)
        return F.grid_sample(
            input, grid, mode=mode, padding_mode=padding_w, align_corners=align_corners
        )

    periodic_w = padding_w == "periodic"
    periodic_h = padding_h == "periodic"
    final_padding_w = "border" if periodic_w else padding_w
    final_padding_h = "border" if periodic_h else padding_h

    N, C, H, W = input.shape

    if periodic_h and periodic_w:
        top_row = input[:, :, 0:1, :]
        left_col = input[:, :, :, 0:1]
        top_left_corner = input[:, :, 0:1, 0:1]
        padded_h = torch.cat([input, top_row], dim=2)
        left_col_extended = torch.cat([left_col, top_left_corner], dim=2)
        padded_input = torch.cat([padded_h, left_col_extended], dim=3)
        H_padded = H + 1
        W_padded = W + 1
    elif periodic_h:
        top_row = input[:, :, 0:1, :]
        padded_input = torch.cat([input, top_row], dim=2)
        H_padded = H + 1
        W_padded = W
    elif periodic_w:
        left_col = input[:, :, :, 0:1]
        padded_input = torch.cat([input, left_col], dim=3)
        H_padded = H
        W_padded = W + 1

    gx, gy = grid[..., 0], grid[..., 1]

    if align_corners:
        px = (gx + 1) * (W - 1) / 2
        py = (gy + 1) * (H - 1) / 2
        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py
        if periodic_w:
            gx_new = 2 * px_wrapped / W - 1
        else:
            gx_new = 2 * px_wrapped / (W_padded - 1) - 1 if W_padded > 1 else gx
        if periodic_h:
            gy_new = 2 * py_wrapped / H - 1
        else:
            gy_new = 2 * py_wrapped / (H_padded - 1) - 1 if H_padded > 1 else gy
    else:
        px = ((gx + 1) * W - 1) / 2
        py = ((gy + 1) * H - 1) / 2
        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py
        if periodic_w:
            gx_new = 2 * (px_wrapped + 0.5) / (W + 1) - 1
        else:
            gx_new = 2 * (px_wrapped + 0.5) / W_padded - 1 if W_padded > 0 else gx
        if periodic_h:
            gy_new = 2 * (py_wrapped + 0.5) / (H + 1) - 1
        else:
            gy_new = 2 * (py_wrapped + 0.5) / H_padded - 1 if H_padded > 0 else gy

    grid_new = torch.stack([gx_new, gy_new], dim=-1)

    if periodic_w and periodic_h:
        final_padding = "border"
    elif periodic_w:
        final_padding = final_padding_h
    elif periodic_h:
        final_padding = final_padding_w

    return F.grid_sample(
        padded_input,
        grid_new,
        mode=mode,
        padding_mode=final_padding,
        align_corners=align_corners,
    )


def custom_grid_sample3d(
    input,
    grid,
    mode="bilinear",
    padding_modes=("periodic", "periodic", "periodic"),
    align_corners=True,
):
    """grid_sample with per-dimension periodic boundary support (3D)."""
    padding_d, padding_h, padding_w = padding_modes
    periodic_d = padding_d == "periodic"
    periodic_h = padding_h == "periodic"
    periodic_w = padding_w == "periodic"

    if not (periodic_d or periodic_h or periodic_w):
        if not (padding_d == padding_h == padding_w):
            raise ValueError(
                "When using different paddings for D, H, W and none is 'periodic', "
                "they must all be equal."
            )
        return F.grid_sample(
            input, grid, mode=mode, padding_mode=padding_d, align_corners=align_corners
        )

    non_periodic_paddings = []
    if not periodic_d:
        non_periodic_paddings.append(padding_d)
    if not periodic_h:
        non_periodic_paddings.append(padding_h)
    if not periodic_w:
        non_periodic_paddings.append(padding_w)

    if len(non_periodic_paddings) == 0:
        final_padding = "border"
    else:
        if len(set(non_periodic_paddings)) > 1:
            raise ValueError(
                "Non-periodic dimensions must use the same padding_mode, "
                f"got: {set(non_periodic_paddings)}"
            )
        final_padding = non_periodic_paddings[0]

    N, C, D, H, W = input.shape
    padded_input = input
    D_padded, H_padded, W_padded = D, H, W

    if periodic_d:
        front_slice = padded_input[:, :, 0:1, :, :]
        padded_input = torch.cat([padded_input, front_slice], dim=2)
        D_padded += 1
    if periodic_h:
        top_slice = padded_input[:, :, :, 0:1, :]
        padded_input = torch.cat([padded_input, top_slice], dim=3)
        H_padded += 1
    if periodic_w:
        left_slice = padded_input[:, :, :, :, 0:1]
        padded_input = torch.cat([padded_input, left_slice], dim=4)
        W_padded += 1

    gx = grid[..., 0]
    gy = grid[..., 1]
    gz = grid[..., 2]

    if align_corners:
        px = (gx + 1.0) * (W - 1.0) / 2.0
        py = (gy + 1.0) * (H - 1.0) / 2.0
        pz = (gz + 1.0) * (D - 1.0) / 2.0
        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py
        pz_wrapped = pz % D if periodic_d else pz
        gx_new = (
            2.0 * px_wrapped / W - 1.0
            if periodic_w
            else (2.0 * px_wrapped / (W_padded - 1.0) - 1.0 if W_padded > 1 else gx)
        )
        gy_new = (
            2.0 * py_wrapped / H - 1.0
            if periodic_h
            else (2.0 * py_wrapped / (H_padded - 1.0) - 1.0 if H_padded > 1 else gy)
        )
        gz_new = (
            2.0 * pz_wrapped / D - 1.0
            if periodic_d
            else (2.0 * pz_wrapped / (D_padded - 1.0) - 1.0 if D_padded > 1 else gz)
        )
    else:
        px = ((gx + 1.0) * W - 1.0) / 2.0
        py = ((gy + 1.0) * H - 1.0) / 2.0
        pz = ((gz + 1.0) * D - 1.0) / 2.0
        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py
        pz_wrapped = pz % D if periodic_d else pz
        gx_new = (
            2.0 * (px_wrapped + 0.5) / (W + 1.0) - 1.0
            if periodic_w
            else (2.0 * (px_wrapped + 0.5) / W_padded - 1.0 if W_padded > 0 else gx)
        )
        gy_new = (
            2.0 * (py_wrapped + 0.5) / (H + 1.0) - 1.0
            if periodic_h
            else (2.0 * (py_wrapped + 0.5) / H_padded - 1.0 if H_padded > 0 else gy)
        )
        gz_new = (
            2.0 * (pz_wrapped + 0.5) / (D + 1.0) - 1.0
            if periodic_d
            else (2.0 * (pz_wrapped + 0.5) / D_padded - 1.0 if D_padded > 0 else gz)
        )

    grid_new = torch.stack([gx_new, gy_new, gz_new], dim=-1)
    return F.grid_sample(
        padded_input,
        grid_new,
        mode=mode,
        padding_mode=final_padding,
        align_corners=align_corners,
    )
