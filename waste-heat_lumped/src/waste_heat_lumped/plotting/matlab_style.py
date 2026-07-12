"""
Matplotlib styling aligned with PlotDefaults_Slides.m and PrintFigure.m.

PlotDefaults_Slides sets:
  LineWidth 1.5, MarkerSize 6, Axes LineWidth 1.5, FontSize 14, box on.

PrintFigure exports a white-background figure at 5*alpha × 4*alpha inches, 600 dpi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_ALPHA = 0.7
BASE_WIDTH_IN = 5.0
BASE_HEIGHT_IN = 4.0
DEFAULT_DPI = 600


def plot_defaults_slides() -> None:
    """Apply global rcParams matching PlotDefaults_Slides.m."""
    plt.rcParams.update(
        {
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "axes.linewidth": 1.5,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "mathtext.default": "regular",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def panel_size_inches(*, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """Single-panel size from PrintFigure.m (5×alpha by 4×alpha inches)."""
    return BASE_WIDTH_IN * alpha, BASE_HEIGHT_IN * alpha


def figure_size_inches(
    ncols: int,
    nrows: int = 1,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Composite figure size as ncols × nrows MATLAB slide panels."""
    w, h = panel_size_inches(alpha=alpha)
    return ncols * w, nrows * h


def scaled_fontsize(rc_key: str, scale: float) -> float:
    """Scale an rcParam font size (e.g. for crowded multi-series legends)."""
    return float(plt.rcParams[rc_key]) * scale


def style_axes(ax, *, tick_in: bool = True) -> None:
    """Axis box and tick styling consistent with PlotDefaults_Slides."""
    ax.tick_params(
        direction="in" if tick_in else "out",
        which="both",
        top=True,
        right=True,
    )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(float(plt.rcParams["axes.linewidth"]))


def ref_marker_kwargs(*, color: str) -> dict:
    """Open-circle markers for digitized reference data."""
    ms = float(plt.rcParams["lines.markersize"])
    return {
        "s": ms**2,
        "marker": "o",
        "facecolors": "white",
        "edgecolors": color,
        "linewidths": float(plt.rcParams["lines.linewidth"]),
        "zorder": 6,
    }


def print_figure(
    fig,
    path: Path | str,
    *,
    alpha: float = DEFAULT_ALPHA,
    dpi: int = DEFAULT_DPI,
    bbox_inches: str | None = "tight",
) -> Path:
    """Save figure with PrintFigure.m conventions (white, high-res)."""
    out = Path(path)
    fig.patch.set_facecolor("white")
    fig.savefig(
        out,
        dpi=dpi,
        facecolor="white",
        edgecolor="none",
        bbox_inches=bbox_inches,
    )
    return out
