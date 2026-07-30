"""Relative H^1 (Sobolev) seminorm loss for channels-last fields (..., spatial..., C)."""

from __future__ import annotations

import math
from typing import List, Sequence

import torch

from benchmark.metrics.common import Metric
from data.datasets import WellMetadata


def central_diff_2d(
    x: torch.Tensor,
    h: Sequence[float],
    fix_x_bnd: bool = False,
    fix_y_bnd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    hh, hw = float(h[0]), float(h[1])
    
    # Along H (dim -3)
    center_h = (x[..., 2:, :, :] - x[..., :-2, :, :]) / (2.0 * hh)
    if fix_x_bnd:
        bnd_h_0 = torch.zeros_like(x[..., 0:1, :, :])
        bnd_h_1 = torch.zeros_like(x[..., -1:, :, :])
    else:
        bnd_h_0 = (x[..., 1:2, :, :] - x[..., 0:1, :, :]) / hh
        bnd_h_1 = (x[..., -1:, :, :] - x[..., -2:-1, :, :]) / hh
    x_h = torch.cat([bnd_h_0, center_h, bnd_h_1], dim=-3)

    # Along W (dim -2)
    center_w = (x[..., :, 2:, :] - x[..., :, :-2, :]) / (2.0 * hw)
    if fix_y_bnd:
        bnd_w_0 = torch.zeros_like(x[..., :, 0:1, :])
        bnd_w_1 = torch.zeros_like(x[..., :, -1:, :])
    else:
        bnd_w_0 = (x[..., :, 1:2, :] - x[..., :, 0:1, :]) / hw
        bnd_w_1 = (x[..., :, -1:, :] - x[..., :, -2:-1, :]) / hw
    x_w = torch.cat([bnd_w_0, center_w, bnd_w_1], dim=-2)

    return x_h, x_w


def central_diff_3d(
    x: torch.Tensor,
    h: Sequence[float],
    fix_x_bnd: bool = False,
    fix_y_bnd: bool = False,
    fix_z_bnd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hd, hh, hw = float(h[0]), float(h[1]), float(h[2])

    # Along D (dim -4)
    center_d = (x[..., 2:, :, :, :] - x[..., :-2, :, :, :]) / (2.0 * hd)
    if fix_x_bnd:
        bnd_d_0 = torch.zeros_like(x[..., 0:1, :, :, :])
        bnd_d_1 = torch.zeros_like(x[..., -1:, :, :, :])
    else:
        bnd_d_0 = (x[..., 1:2, :, :, :] - x[..., 0:1, :, :, :]) / hd
        bnd_d_1 = (x[..., -1:, :, :, :] - x[..., -2:-1, :, :, :]) / hd
    x_d = torch.cat([bnd_d_0, center_d, bnd_d_1], dim=-4)

    # Along H (dim -3)
    center_h = (x[..., :, 2:, :, :] - x[..., :, :-2, :, :]) / (2.0 * hh)
    if fix_y_bnd:
        bnd_h_0 = torch.zeros_like(x[..., :, 0:1, :, :])
        bnd_h_1 = torch.zeros_like(x[..., :, -1:, :, :])
    else:
        bnd_h_0 = (x[..., :, 1:2, :, :] - x[..., :, 0:1, :, :]) / hh
        bnd_h_1 = (x[..., :, -1:, :, :] - x[..., :, -2:-1, :, :]) / hh
    x_h = torch.cat([bnd_h_0, center_h, bnd_h_1], dim=-3)

    # Along W (dim -2)
    center_w = (x[..., :, :, 2:, :] - x[..., :, :, :-2, :]) / (2.0 * hw)
    if fix_z_bnd:
        bnd_w_0 = torch.zeros_like(x[..., :, :, 0:1, :])
        bnd_w_1 = torch.zeros_like(x[..., :, :, -1:, :])
    else:
        bnd_w_0 = (x[..., :, :, 1:2, :] - x[..., :, :, 0:1, :]) / hw
        bnd_w_1 = (x[..., :, :, -1:, :] - x[..., :, :, -2:-1, :]) / hw
    x_w = torch.cat([bnd_w_0, center_w, bnd_w_1], dim=-2)

    return x_d, x_h, x_w


def _L_vector(n_spatial: int, L: float | Sequence[float] | None) -> tuple[float, ...]:
    """Domain length per axis; default isotropic ``2π``."""
    twopi = 2.0 * math.pi
    if L is None:
        return (twopi,) * n_spatial
    if isinstance(L, (float, int)):
        v = float(L)
        return (v,) * n_spatial
    t = tuple(float(x) for x in L)
    if len(t) == n_spatial:
        return t
    raise ValueError(
        f"L must be a float (isotropic), None (2π isotropic), or a sequence of length "
        f"{n_spatial} matching n_spatial_dims; got length {len(t)}."
    )


def _uniform_h_from_meta(meta: WellMetadata, L: tuple[float, ...]) -> list[float]:
    """Cell size ``L_i / N_i`` for each spatial axis (tensor order ``...`` then spatial then ``C``)."""
    n = meta.n_spatial_dims
    if len(L) != n:
        raise ValueError(f"L has length {len(L)} but n_spatial_dims={n}.")
    if len(meta.spatial_resolution) != n:
        raise ValueError("meta.spatial_resolution length must match n_spatial_dims.")
    return [L[i] / meta.spatial_resolution[i] for i in range(n)]


def _flatten_spatial_and_channel(x: torch.Tensor, n_spatial: int) -> torch.Tensor:
    """Merge ``(..., *spatial, C)`` -> ``(..., prod_spatial * C)``."""
    return x.flatten(start_dim=-n_spatial - 1, end_dim=-1)


class H1RelLoss(Metric):
    """Relative discrete H^1 seminorm (value + spatial central gradients), channels-last.

    **Per variable channel** (same layout contract as :class:`MSE`): returns a tensor of
    shape ``(B, T, C)`` so validation can ``mean(0)`` to ``(T, C)`` and split by field.

    For each channel ``c`` alone, computes the relative Sobolev-style ratio (value + central
    differences over spatial grid) vs the reference channel ``y[..., c]``.

    Layout: ``(..., *spatial, C)`` with ``len(spatial) == meta.n_spatial_dims`` (2 or 3).
    Boundary flags: 2D uses ``fix_x_bnd`` / ``fix_y_bnd`` on the two spatial axes; 3D adds
    ``fix_z_bnd`` on the third axis (``D, H, W`` order matching ``spatial_resolution``).
    """

    def __init__(
        self,
        L: float | Sequence[float] | None = None,
        fix_x_bnd: bool = False,
        fix_y_bnd: bool = False,
        fix_z_bnd: bool = False,
        reduce_dims: int | Sequence[int] | None = None,
        reductions: str | Sequence[str] = "sum",
        eps: float = 1e-12,
    ):
        super().__init__()
        self._L_config: float | Sequence[float] | None
        if L is None:
            self._L_config = None
        elif isinstance(L, (float, int)):
            self._L_config = float(L)
        else:
            self._L_config = tuple(float(x) for x in L)
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd
        self.reduce_dims: List[int] | None
        if reduce_dims is None:
            self.reduce_dims = None
        elif isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = list(reduce_dims)
        if self.reduce_dims is None:
            self.reductions: List[str] = []
        elif isinstance(reductions, str):
            self.reductions = [reductions] * len(self.reduce_dims)
        else:
            self.reductions = list(reductions)

        self.eps = eps

    def _compute_terms(
        self, x: torch.Tensor, y: torch.Tensor, h: Sequence[float], n_spatial: int
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        dict_x = {0: _flatten_spatial_and_channel(x, n_spatial)}
        dict_y = {0: _flatten_spatial_and_channel(y, n_spatial)}
        if n_spatial == 2:
            x_a, x_b = central_diff_2d(x, h, self.fix_x_bnd, self.fix_y_bnd)
            y_a, y_b = central_diff_2d(y, h, self.fix_x_bnd, self.fix_y_bnd)
            grads_x = (x_a, x_b)
            grads_y = (y_a, y_b)
        elif n_spatial == 3:
            x_a, x_b, x_c = central_diff_3d(
                x, h, self.fix_x_bnd, self.fix_y_bnd, self.fix_z_bnd
            )
            y_a, y_b, y_c = central_diff_3d(
                y, h, self.fix_x_bnd, self.fix_y_bnd, self.fix_z_bnd
            )
            grads_x = (x_a, x_b, x_c)
            grads_y = (y_a, y_b, y_c)
        else:
            raise ValueError(f"n_spatial must be 2 or 3, got {n_spatial}")

        for j, (gx, gy) in enumerate(zip(grads_x, grads_y, strict=True), start=1):
            dict_x[j] = _flatten_spatial_and_channel(gx, n_spatial)
            dict_y[j] = _flatten_spatial_and_channel(gy, n_spatial)
        return dict_x, dict_y

    def _reduce_all(self, t: torch.Tensor) -> torch.Tensor:
        if self.reduce_dims is None:
            return t
        out = t
        for j, dim in enumerate(self.reduce_dims):
            if self.reductions[j] == "sum":
                out = torch.sum(out, dim=dim, keepdim=True)
            else:
                out = torch.mean(out, dim=dim, keepdim=True)
        return out

    def eval(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        meta: WellMetadata,
        h: float | Sequence[float] | None = None,
    ) -> torch.Tensor:
        n_spatial = meta.n_spatial_dims
        if n_spatial not in (2, 3):
            raise NotImplementedError(
                f"H1RelLoss: n_spatial_dims={n_spatial} not supported (only 2D and 3D)."
            )

        L_vec = _L_vector(n_spatial, self._L_config)
        h_list: list[float]
        if h is None:
            h_list = _uniform_h_from_meta(meta, L_vec)
        elif isinstance(h, (float, int)):
            hf = float(h)
            h_list = [hf] * n_spatial
        else:
            h_list = [float(h[i]) for i in range(n_spatial)]
            if len(h_list) != n_spatial:
                raise ValueError(f"h must have length {n_spatial}, got {len(h_list)}.")

        # Vectorize over channels: (*lead, *spatial, C) -> (n_lead*C, *spatial, 1), one batched H1 per row.
        C = x.shape[-1]
        lead_shape = x.shape[: -(n_spatial + 1)]
        n_lead = math.prod(lead_shape) if lead_shape else 1
        spatial_sizes = x.shape[-(n_spatial + 1) : -1]
        perm = (0, -1) + tuple(range(1, n_spatial + 1))
        x_b = (
            x.reshape(n_lead, *spatial_sizes, C)
            .permute(*perm)
            .reshape(n_lead * C, *spatial_sizes, 1)
            .contiguous()
        )
        y_b = (
            y.reshape(n_lead, *spatial_sizes, C)
            .permute(*perm)
            .reshape(n_lead * C, *spatial_sizes, 1)
            .contiguous()
        )

        dict_x, dict_y = self._compute_terms(x_b, y_b, h_list, n_spatial)
        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False) ** 2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False) ** 2
        for j in range(1, n_spatial + 1):
            diff = diff + torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False) ** 2
            ynorm = ynorm + torch.norm(dict_y[j], p=2, dim=-1, keepdim=False) ** 2
        rel_flat = (diff ** 0.5) / ((ynorm ** 0.5) + self.eps)
        out = rel_flat.reshape(n_lead, C).reshape(*lead_shape, C)
        if self.reduce_dims is not None:
            out = self._reduce_all(out).squeeze()
        return out
