"""Bounds for ACT displacements in normalized ``grid_sample`` coordinates."""


def max_displacement_normalized_one_cell(spatial_resolution: tuple[int, ...]) -> float:
    """Approximate one grid-cell step along each axis in ``[-1, 1]`` normalized coords.

    Uses the same spacing as ``torch.linspace(-1, 1, N)`` between adjacent indices:
    ``Δ = 2 / (N - 1)`` for ``N >= 2``. Returns the minimum over spatial axes so a
    single scalar cap keeps both (or all) components within about one cell.

    For ``N == 1``, that axis is skipped; if no valid axis, falls back to ``0.5``.
    """
    steps: list[float] = []
    for n in spatial_resolution:
        if n >= 2:
            steps.append(2.0 / float(n - 1))
    return min(steps) if steps else 0.5
