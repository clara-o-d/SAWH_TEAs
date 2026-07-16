"""Shared plotting helpers for the comparison scripts.

Mirrors the conventions already established in the per-package
``npv_heatmap.py`` / ``tornado_plot.py`` scripts (e.g.
``solar_lumped/scripts/npv_heatmap.py``): ``RdYlGn`` + ``TwoSlopeNorm`` for
diverging NPV colormaps, ``viridis_r`` + gray ``///``-hatched rectangles for
infeasible payback, plus (new here) a categorical colormap helper for the
4-config "winner" maps used by ``grid_heatmap.py``.
"""

from __future__ import annotations

import numpy as np

NPV_CMAP = "RdYlGn"
MARGIN_CMAP = "viridis"
PAYBACK_CMAP = "viridis_r"

INFEASIBLE_FACECOLOR = "0.75"
INFEASIBLE_EDGECOLOR = "0.4"
INFEASIBLE_HATCH = "///"

COMPARABLE_HATCH_COLOR = "0.3"
COMPARABLE_HATCH = "///"


def diverging_norm_centered_zero(values: np.ndarray):
    """``TwoSlopeNorm(vcenter=0)`` sized to ``values``, guarding all-positive/all-negative grids.

    Same edge-case handling as ``solar_lumped/scripts/npv_heatmap.py``'s
    ``plot_npv_heatmap``: without the ``eps`` pad, an all-positive or
    all-negative grid makes ``vmin == vcenter`` or ``vcenter == vmax``, which
    ``TwoSlopeNorm`` rejects.
    """
    import matplotlib.colors as mcolors

    finite = values[np.isfinite(values)]
    eps = 1e-6
    if finite.size:
        vmin = min(0.0, float(finite.min())) - eps
        vmax = max(0.0, float(finite.max())) + eps
    else:
        vmin, vmax = -1.0, 1.0
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def cell_edges(vals: np.ndarray) -> np.ndarray:
    """Bin edges for ``pcolormesh`` given cell-center coordinates (linear or log-spaced)."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 1:
        v = float(vals[0])
        half = 0.5 if v == 0.0 else abs(v) * 0.5
        return np.array([v - half, v + half])
    mid = (vals[:-1] + vals[1:]) / 2.0
    first = vals[0] - (mid[0] - vals[0])
    last = vals[-1] + (vals[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def add_infeasible_hatch(ax, x_edges: np.ndarray, y_edges: np.ndarray, infeasible: np.ndarray) -> None:
    """Overlay gray ``///``-hatched rectangles wherever ``infeasible`` is True."""
    from matplotlib.patches import Rectangle

    ny, nx = infeasible.shape
    for iy in range(ny):
        for ix in range(nx):
            if infeasible[iy, ix]:
                ax.add_patch(
                    Rectangle(
                        (x_edges[ix], y_edges[iy]),
                        x_edges[ix + 1] - x_edges[ix],
                        y_edges[iy + 1] - y_edges[iy],
                        facecolor=INFEASIBLE_FACECOLOR,
                        edgecolor=INFEASIBLE_EDGECOLOR,
                        linewidth=0.2,
                        hatch=INFEASIBLE_HATCH,
                        zorder=4,
                    )
                )


def add_comparable_hatch(ax, x_edges: np.ndarray, y_edges: np.ndarray, comparable: np.ndarray) -> None:
    """Overlay semi-transparent gray ``///`` hatch wherever ``comparable`` (margin below threshold) is True."""
    from matplotlib.patches import Rectangle

    ny, nx = comparable.shape
    for iy in range(ny):
        for ix in range(nx):
            if comparable[iy, ix]:
                ax.add_patch(
                    Rectangle(
                        (x_edges[ix], y_edges[iy]),
                        x_edges[ix + 1] - x_edges[ix],
                        y_edges[iy + 1] - y_edges[iy],
                        facecolor="none",
                        edgecolor=COMPARABLE_HATCH_COLOR,
                        linewidth=0.0,
                        hatch=COMPARABLE_HATCH,
                        alpha=0.55,
                        zorder=6,
                    )
                )


def categorical_colormap(config_ids: list[str], colors_by_id: dict[str, str]):
    """``ListedColormap``/``BoundaryNorm`` for an integer-coded winner map.

    Returns ``(cmap, norm, code_by_config_id)`` where ``code_by_config_id``
    maps each ``config_id`` to the integer code used in the winner grid
    (codes are ``0..len(config_ids)-1`` in ``config_ids`` order).
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    codes = {cid: i for i, cid in enumerate(config_ids)}
    colors = [colors_by_id[cid] for cid in config_ids]
    cmap = ListedColormap(colors)
    boundaries = np.arange(-0.5, len(config_ids) + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, codes


def baseline_marker(ax, x: float, y: float, label: str = "Baseline") -> None:
    """Star marker for a baseline point, matching the per-package heatmap convention."""
    ax.plot(
        x,
        y,
        marker="*",
        markersize=18,
        markerfacecolor="black",
        markeredgecolor="white",
        markeredgewidth=1.0,
        linestyle="none",
        zorder=5,
        label=label,
    )
