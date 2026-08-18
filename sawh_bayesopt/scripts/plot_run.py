#!/usr/bin/env python3
"""Two plots describing a finished BayesOpt run against its base case.

Reads only a completed run directory (``report.json`` + ``history.csv`` +
``config.json``), so it works for a single-site run, one site of the global
complex sweep, or any archived run -- nothing needs re-simulating.

The base case is Wilson Table S3's system as ``write_final_report`` already
evaluated it: same LCOW model, same JAX physics path, same site and weather year
as the optimization. That is what makes the comparison meaningful rather than a
number from a paper table -- see ``reporting.evaluate_baseline``.

Examples::

  python plot_run.py --run-dir outputs/runs/complex_gpu_run_1
  python plot_run.py --run-dir outputs/gpu_bayesopt_sweep_complex/task_0/-23.6000_-70.4000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator, ScalarFormatter  # noqa: E402

# Validated default data-viz palette (light surface). Categorical slots 1-2 for the two
# series; the sequential blue ramp's steps 250/550 for the before/after dumbbell (one hue,
# two shades); everything textual in ink tokens rather than a series color.
_SURFACE = "#fcfcfb"
_SERIES_1 = "#2a78d6"   # evaluated designs
_SERIES_2 = "#eb6834"   # best so far
# Slot 3 sits just under 3:1 on this surface, so it ships with relief: a diamond marker
# (shape, not hue) plus the bold direct label already beside it.
_SERIES_3 = "#1baf7a"   # recommendation promoted from verification
_SHADE_BASE = "#86b6ef"  # base case (light step -- no lighter than 250 on a light surface)
_SHADE_OPT = "#1c5cab"   # optimized design
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"

# Anything at/above this is evaluator.PENALTY_LCOW_USD_PER_M3 (1e4), an infeasible design
# rather than a cost -- plotting it would flatten the whole y range into one line.
_PENALTY_FLOOR = 1e3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="A finished run dir (holds report.json).")
    p.add_argument("--output-dir", type=Path, default=None, help="Defaults to --run-dir.")
    return p.parse_args(argv)


def _style(ax) -> None:
    """Recessive chrome: hairline grid behind the marks, no top/right spines, muted ticks."""
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_MUTED, labelsize=9)


def plot_convergence(history: pd.DataFrame, baseline_lcow: float, recommended: float, path: Path) -> None:
    """Evaluated designs and the best-so-far trace against the base case's LCOW.

    The base case is a horizontal reference rule, not a third series: it is one number
    that does not vary along x, and drawing it as a line with its own legend entry would
    imply it was evaluated per design point.
    """
    lcow = history["combined_lcow"].to_numpy(float)
    feasible = lcow < _PENALTY_FLOOR
    idx = np.arange(len(lcow))
    best_so_far = np.minimum.accumulate(np.where(feasible, lcow, np.inf))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.patch.set_facecolor(_SURFACE)
    _style(ax)

    ax.scatter(
        idx[feasible], lcow[feasible], s=44, color=_SERIES_1, alpha=0.55,
        edgecolor=_SURFACE, linewidth=1.0, zorder=3, label="evaluated design",
    )
    # steps-post, because best-so-far only changes when a new incumbent lands -- a straight
    # interpolation between two evaluations implies progress that never happened.
    ax.plot(idx, best_so_far, color=_SERIES_2, linewidth=2.0, zorder=4,
            drawstyle="steps-post", label="best so far")
    ax.axhline(baseline_lcow, color=_INK_MUTED, linewidth=1.5, linestyle=(0, (5, 4)), zorder=2)

    # Log y: the early random designs run to ~40 USD/m³ while the comparison that matters
    # happens between 2 and 5, which linear scaling squeezes into the bottom eighth of the
    # axis. LCOW is compared as a ratio ("x% below the base case") anyway.
    ax.set_yscale("log")
    ax.set_yticks([2, 3, 5, 10, 20, 40])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    # Evaluation counts are integers; a 3-point run otherwise gets ticks at 0.25, 0.75, ...
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # A recommendation below every point in history came from verification's perturbed
    # neighbors, which are evaluated after the loop -- without its own mark the reader sees
    # a headline number that sits nowhere on the best-so-far trace.
    if recommended < best_so_far[-1] - 1e-9:
        ax.scatter(
            [len(lcow) - 1], [recommended], s=90, marker="D", color=_SERIES_3,
            edgecolor=_SURFACE, linewidth=1.5, zorder=6, label="verified neighbor (recommended)",
        )

    # Direct labels instead of legend entries for the reference rule and the final value:
    # two numbers the reader actually wants, not a number on every point. Both sit above
    # their line and left of the axis edge, clear of the tick labels.
    # Both labels get a surface-colored backing plate: they span the width of the plot and
    # would otherwise read through the best-so-far line where it risers past them.
    plate = {"facecolor": _SURFACE, "edgecolor": "none", "pad": 2.0}
    ax.annotate(
        f"Wilson Table S3 base case — {baseline_lcow:.2f} USD/m³",
        xy=(len(lcow) - 1, baseline_lcow), xytext=(-4, 7), textcoords="offset points",
        ha="right", va="bottom", fontsize=9, color=_INK_SECONDARY, bbox=plate, zorder=5,
    )
    # An under-budgeted run can finish worse than the base case (a few LHS points and no
    # infill rounds will), so the direction has to come from the sign, not be assumed.
    improvement = (baseline_lcow - recommended) / baseline_lcow
    versus = (
        f"{improvement:.0%} below base case" if improvement >= 0
        else f"{-improvement:.0%} above base case"
    )
    ax.annotate(
        f"{recommended:.2f} USD/m³ — {versus}",
        xy=(len(lcow) - 1, recommended), xytext=(-4, 8), textcoords="offset points",
        ha="right", va="bottom", fontsize=10, color=_INK, fontweight="semibold",
        bbox=plate, zorder=5,
    )

    n_infeasible = int((~feasible).sum())
    subtitle = f"{len(lcow)} designs evaluated"
    if n_infeasible:
        subtitle += f" ({n_infeasible} infeasible, off scale)"
    ax.set_xlabel("design evaluation", color=_INK_SECONDARY, fontsize=10)
    ax.set_ylabel("combined LCOW (USD/m³)", color=_INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Bayesian optimization vs. base case, same LCOW model",
        color=_INK, fontsize=12, loc="left", pad=26,
    )
    ax.text(
        0.0, 1.015, subtitle, transform=ax.transAxes,
        color=_INK_MUTED, fontsize=9, ha="left", va="bottom",
    )
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(_INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def plot_design_vs_baseline(
    baseline: dict, optimized: dict, bounds: dict, n_base_dims: int, path: Path
) -> None:
    """Dumbbell: base case -> optimized value for every design variable.

    Each variable is drawn as its position within its own optimizer bound, because the 13
    variables span metres, degrees, hours and dimensionless ratios -- a shared axis in real
    units would be meaningless and a second axis is never the answer. Real values are
    direct-labelled per row so the normalization stays readable.
    """
    names = list(optimized)
    y = np.arange(len(names))[::-1]  # first variable at the top

    def _frac(name: str, value: float) -> float:
        lo, hi = bounds[name]
        return 0.5 if hi <= lo else (value - lo) / (hi - lo)

    base_frac = np.array([_frac(n, baseline[n]) for n in names])
    opt_frac = np.array([_frac(n, optimized[n]) for n in names])

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.patch.set_facecolor(_SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=_GRID, linewidth=0.8)
    ax.grid(False, axis="y")

    ax.hlines(y, base_frac, opt_frac, color=_SHADE_BASE, linewidth=2.0, zorder=2)
    ax.scatter(base_frac, y, s=64, color=_SHADE_BASE, edgecolor=_SURFACE, linewidth=1.5,
               zorder=3, label="base case (Wilson Table S3)")
    ax.scatter(opt_frac, y, s=64, color=_SHADE_OPT, edgecolor=_SURFACE, linewidth=1.5,
               zorder=4, label="optimized")

    for yi, name in zip(y, names):
        ax.annotate(
            f"{baseline[name]:.3g} → {optimized[name]:.3g}",
            xy=(1.02, yi), xycoords=("axes fraction", "data"),
            va="center", ha="left", fontsize=8.5, color=_INK_MUTED,
        )

    # In a complex run the 6 geometry variables and the 7 complex-fidelity ones are
    # different families, so mark the split rather than letting the reader count rows
    # against COMPLEX_VAR_ORDER. A simple run has one family and draws no divider.
    if 0 < n_base_dims < len(names):
        ax.axhline(y[n_base_dims] + 0.5, color=_GRID, linewidth=1.2)
        ax.annotate(
            "complex-fidelity variables", xy=(0.012, y[n_base_dims] + 0.5),
            xycoords=("axes fraction", "data"), fontsize=8.5, color=_INK_MUTED,
            va="center", bbox={"facecolor": _SURFACE, "edgecolor": "none", "pad": 2.0}, zorder=5,
        )

    ax.set_yticks(y, names, fontsize=9)
    ax.tick_params(axis="y", colors=_INK_SECONDARY)
    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["lower\nbound", "", "mid", "", "upper\nbound"])
    ax.set_xlabel("position within the optimizer's bound for that variable",
                  color=_INK_SECONDARY, fontsize=10)
    ax.set_title("What the optimizer changed", color=_INK, fontsize=12, loc="left", pad=26)

    # Bound-pinning count in the subtitle, because it is the first thing to check on this
    # figure: an optimum sitting on many box edges means the bounds are setting the answer.
    n_pinned = int(np.sum((opt_frac <= 1e-6) | (opt_frac >= 1 - 1e-6)))
    ax.text(
        0.0, 1.015, f"{n_pinned} of {len(names)} variables land on a bound",
        transform=ax.transAxes, color=_INK_MUTED, fontsize=9, ha="left", va="bottom",
    )
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, -0.22), ncol=2)
    for text in leg.get_texts():
        text.set_color(_INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir
    out_dir = args.output_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads((run_dir / "report.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    history = pd.read_csv(run_dir / "history.csv")

    baseline_lcow = report["baseline_wilson_table_s3"]["combined_lcow_usd_per_m3"]
    recommended = report["recommended_combined_lcow_usd_per_m3"]

    plot_convergence(history, baseline_lcow, recommended, out_dir / "convergence_vs_baseline.png")
    plot_design_vs_baseline(
        report["baseline_wilson_table_s3"]["design"],
        report["recommended_design"],
        config["bounds"],
        # Everything past design_space.BASE_VAR_ORDER's 6 names is complex-fidelity. A
        # simple-mode run has 5 dims total, so the guard below skips the divider.
        n_base_dims=6,
        path=out_dir / "design_vs_baseline.png",
    )

    print(
        f"base case {baseline_lcow:.4f} -> optimized {recommended:.4f} USD/m³ "
        f"({(baseline_lcow - recommended) / baseline_lcow:+.1%}), "
        # Absent in runs made before write_final_report started recording it.
        f"recommendation from {report.get('recommended_from', 'unknown')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
