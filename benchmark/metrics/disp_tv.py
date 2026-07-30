"""Total-variation-style smoothness penalty on displacement fields."""

from __future__ import annotations

import torch


def tv_loss_disp(disp: torch.Tensor) -> torch.Tensor:
    """Squared finite-difference energy of ``disp``, averaged over all elements.

    Args:
        disp: ACT displacement offsets. 2D: ``(B, num_heads, 2, H, W)``;
            3D: ``(B, num_heads, 3, D, H, W)`` (6 dimensions).

    Returns:
        Scalar tensor: sum of ``mean((diff_axis)**2)`` over spatial axes.
    """
    if disp.ndim == 5:
        # (B, nh, 2, H, W) — differences along H and W.
        diff_h = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
        diff_w = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
        return diff_h.pow(2).mean() + diff_w.pow(2).mean()
    if disp.ndim == 6:
        # (B, nh, 3, D, H, W) — differences along D, H, W.
        diff_d = disp[:, :, :, 1:, :, :] - disp[:, :, :, :-1, :, :]
        diff_h = disp[:, :, :, :, 1:, :] - disp[:, :, :, :, :-1, :]
        diff_w = disp[:, :, :, :, :, 1:] - disp[:, :, :, :, :, :-1]
        return diff_d.pow(2).mean() + diff_h.pow(2).mean() + diff_w.pow(2).mean()
    raise ValueError(f"tv_loss_disp: expected 5D or 6D disp, got shape {tuple(disp.shape)}")


def tv_loss_disps(disps: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Sum of :func:`tv_loss_disp` over a list of per-block displacement tensors."""
    if not disps:
        raise ValueError("tv_loss_disps: disps must be non-empty")
    total = tv_loss_disp(disps[0])
    for d in disps[1:]:
        total = total + tv_loss_disp(d)
    return total
