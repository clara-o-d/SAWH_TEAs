#!/usr/bin/env python3
"""Box plot of the optimized LCOW at every land point, one box per scenario.

    python3 analysis/performance/optimization/plot_lcow_by_scenario.py \
        --sweep-dir solar_lumped/outputs/bayesopt_global_12deg

Log y-axis, not truncated linear: the distributions span 1.6-24.3 USD/m3 (15x), so on a
shared linear axis the three instant-kinetics scenarios collapse into ~4% of the plot
height. Quartiles are order statistics, so the box geometry is unaffected by the transform.

Colour encodes the OPTICS FAMILY (Wilson / improved / optical limits), which is the
categorical identity; the relaxed limits are read from the x labels. Colour is deliberately
not mapped to LCOW -- that is what the y-axis already does.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dataviz reference palette, categorical slots 1-3. These three validate on the ALL-PAIRS
# pairlist in both modes (worst CVD dE 9.2, worst normal-vision dE 24.0), which is the
# gate that matters when same-coloured boxes are non-adjacent.
FAMILY_COLOR = {
    "wilson": "#2a78d6",           # slot 1, blue
    "improved": "#eb6834",         # slot 2, orange
    "optical_limits": "#1baf7a",   # slot 3, aqua
}
# Slot 3 sits at 2.74:1 on the light surface, under the 3:1 gate. The validator's relief
# rule applies: every box carries a visible median label, so identity never rests on hue.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"

LABELS = {
    "wilson": "Wilson",
    "improved": "Improved",
    "improved_perfect_cond": "Improved\n+ perfect cond.",
    "improved_instant_g": "Improved\n+ instant $g$",
    "optical_limits": "Optical\nlimits",
    "optical_limits_perfect_cond": "Optical limits\n+ perfect cond.",
    "optical_limits_instant_g": "Optical limits\n+ instant $g$",
    "optical_limits_instant_g_perfect_cond": "Optical limits\n+ instant $g$\n+ perfect cond.",
}


def family(scenario: str) -> str:
    """Which optics base a scenario is built on. Longest prefix wins, so
    optical_limits_instant_g maps to optical_limits rather than to a new family."""
    for f in ("optical_limits", "improved", "wilson"):
        if scenario.startswith(f):
            return f
    raise ValueError(f"Unrecognized scenario {scenario!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    files = sorted(glob.glob(str(args.sweep_dir / "*" / "summary.csv")))
    if not files:
        raise SystemExit(f"No */summary.csv under {args.sweep_dir}")
    df = pd.concat(
        [pd.read_csv(f).assign(scenario=os.path.basename(os.path.dirname(f))) for f in files],
        ignore_index=True,
    )
    col = "best_combined_lcow_usd_m3"
    order = df.groupby("scenario")[col].median().sort_values().index.tolist()
    groups = [df.loc[df.scenario == s, col].to_numpy() for s in order]
    n_per = {s: len(g) for s, g in zip(order, groups)}

    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bp = ax.boxplot(
        groups, widths=0.6, patch_artist=True, showfliers=False,
        medianprops=dict(color=SURFACE, linewidth=2.0),      # 2px, reads on any fill
        whiskerprops=dict(color=INK_MUTED, linewidth=1.0),
        capprops=dict(color=INK_MUTED, linewidth=1.0),
        boxprops=dict(linewidth=0),
    )
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(FAMILY_COLOR[family(s)])
        patch.set_alpha(0.85)
        patch.set_edgecolor(SURFACE)          # 2px surface gap between adjacent fills
        patch.set_linewidth(2.0)

    # Every site as a faint jittered dot: n=87 per box is small enough that the raw spread
    # is worth showing, and it reveals any clumping the quartiles would hide.
    rng = np.random.default_rng(0)
    for i, (s, g) in enumerate(zip(order, groups), start=1):
        ax.scatter(
            i + rng.uniform(-0.19, 0.19, size=len(g)), g,
            s=7, color=INK, alpha=0.16, linewidths=0, zorder=3,
        )

    # Direct median labels: the relief rule for slot 3, and the number readers want.
    for i, g in enumerate(groups, start=1):
        med = float(np.median(g))
        ax.annotate(
            f"{med:.2f}", xy=(i, med), xytext=(0, 0), textcoords="offset points",
            ha="center", va="center", fontsize=8.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=SURFACE, edgecolor="none", alpha=0.92),
            zorder=6,
        )

    ax.set_yscale("log")
    ax.set_yticks([2, 3, 4, 6, 9, 14, 20])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(1.4, 28)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([LABELS[s] for s in order], fontsize=8.5, color=INK)
    ax.set_ylabel("Optimized LCOW  (USD m$^{-3}$, log scale)", fontsize=10, color=INK)
    ax.set_title(
        "Optimized cost of water across 87 land points, by scenario",
        fontsize=12.5, color=INK, pad=30, loc="left",
    )
    ax.text(
        0.0, 1.012,
        f"Bayesian optimization per site, 12° land grid, day-stride 10 · "
        f"n = {n_per[order[0]]} sites per scenario · box = IQR, whiskers = 1.5×IQR, label = median",
        transform=ax.transAxes, fontsize=8.5, color=INK_MUTED, ha="left", va="bottom",
    )

    ax.yaxis.grid(True, color="#d8d7d2", linewidth=0.6, alpha=0.9)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.tick_params(axis="y", length=0, labelsize=9, colors=INK_MUTED)
    ax.tick_params(axis="x", length=0)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=8,
                   markerfacecolor=FAMILY_COLOR[f], markeredgecolor="none",
                   label={"wilson": "Wilson optics",
                          "improved": "Improved optics",
                          "optical_limits": "Optical material limits"}[f])
        for f in ("wilson", "improved", "optical_limits")
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9,
                    ncols=3, bbox_to_anchor=(0.0, -0.16))
    for t in leg.get_texts():
        t.set_color(INK_MUTED)

    fig.tight_layout()
    out = args.out or (args.sweep_dir / "figures" / "lcow_by_scenario.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")

    # The table view the accessibility pass requires, and a check on the ordering claim.
    tbl = df.groupby("scenario")[col].describe()[["count", "min", "25%", "50%", "75%", "max"]]
    tbl = tbl.loc[order].round(2)
    tbl.to_csv(out.with_suffix(".csv"))
    print(tbl.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
