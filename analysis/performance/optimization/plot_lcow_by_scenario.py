#!/usr/bin/env python3
"""Box plot of the optimized LCOW at every land point, one box per scenario.

    python3 analysis/performance/optimization/plot_lcow_by_scenario.py \
        --sweep-dir solar_lumped/outputs/bayesopt_global_12deg

Log y-axis, not truncated linear: the distributions span 1.6-24.3 USD/m3 (15x), so on a
shared linear axis the three instant-kinetics scenarios collapse into ~4% of the plot
height. Quartiles are order statistics, so the box geometry is unaffected by the transform.

One series, one neutral fill: the scenario names on the x axis carry identity, so there is
nothing for hue to encode and no legend to read. Distributions are shown as boxes only --
no per-site dots, no median labels.
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

# Single-series chart, so no categorical palette and no contrast/CVD gate to clear: hue is
# not carrying identity here. Neutral fill, dark median, recessive everything else.
BOX_FILL = "#c2c1ba"
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
        medianprops=dict(color=INK, linewidth=2.0),
        whiskerprops=dict(color=INK_MUTED, linewidth=1.0),
        capprops=dict(color=INK_MUTED, linewidth=1.0),
        boxprops=dict(linewidth=0),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_FILL)
        patch.set_edgecolor(SURFACE)          # 2px surface gap between adjacent fills
        patch.set_linewidth(2.0)

    ax.set_yscale("log")
    ax.set_yticks([2, 3, 4, 6, 9, 14, 20])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(1.4, 22)
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
        f"n = {n_per[order[0]]} sites per scenario · box = IQR, whiskers = 1.5×IQR",
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
