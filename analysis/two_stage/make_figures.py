#!/usr/bin/env python3
"""Figures for analysis/two_stage/README.md: the two-stage pipeline's results so far.

Reads the run directory synced from Sherlock (sawh_bayesopt/outputs/two_stage/main) and
writes PNGs to analysis/two_stage/figures/. Rerun after syncing newer results:

    python analysis/two_stage/make_figures.py
    python analysis/two_stage/make_figures.py --run-dir sawh_bayesopt/outputs/two_stage/main --mode daily_hindsight
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "analysis" / "global"))
from plot_map import world_ax  # noqa: E402

# Palette: the dataviz reference instance. The four tail regimes take categorical slots
# 1-4 (validated: worst adjacent CVD dE 9.1, normal-vision 22.9; aqua and yellow sit below
# 3:1 on white, so every line carrying them is direct-labelled). "other" is muted grey so
# it reads as the background climate it is. Colour follows the regime in every figure.
REGIME_COLOR = {"hyper_arid": "#2a78d6", "high_altitude": "#eb6834", "monsoonal": "#1baf7a",
                "coastal_humid": "#eda100", "other": "#898781"}
REGIME_LABEL = {"hyper_arid": "hyper-arid", "high_altitude": "high-altitude", "monsoonal": "monsoonal",
                "coastal_humid": "coastal-humid", "other": "other"}
SERIES3 = ("#2a78d6", "#eb6834", "#1baf7a")  # first three slots: validated all-pairs
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=_REPO / "sawh_bayesopt" / "outputs" / "two_stage" / "main")
    ap.add_argument("--mode", default="daily_hindsight", help="opt/<mode>/anchors.* to plot")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run, out = args.run_dir, args.out_dir
    cfg = json.loads((run / "config.json").read_text())

    fig_weather_pca(np.load(run / "pca.npz"), cfg["n_components"], out / "fig1_weather_pca.png")
    fig_featurization_check(pd.read_csv(run / "featval_summary.csv"), cfg["n_components"],
                            out / "fig2_featurization_check.png")
    fig_surrogate_holdout(json.loads((run / "holdout_report.json").read_text()), out / "fig3_surrogate_holdout.png")
    anchors = pd.read_csv(run / "opt" / args.mode / "anchors.csv")
    fig_anchor_map(anchors, "lcow_usd_m3", "predicted LCOW (USD/m³), darker = more expensive",
                   f"Fig. 4  Best predicted cost at the {{n}} anchor locations  ({_mode_label(args.mode)})",
                   out / "fig4_anchor_lcow_map.png")
    fig_anchor_map(anchors, "hydrogel_thickness_mm", "optimal hydrogel thickness (mm), darker = thicker",
                   f"Fig. 4b  Optimal hydrogel thickness at the {{n}} anchor locations  ({_mode_label(args.mode)})",
                   out / "fig4b_anchor_thickness_map.png")
    fig_lcow_by_regime(anchors, args.mode, out / "fig5_lcow_by_regime.png")
    fig_daily_tilt(anchors, np.load(run / "opt" / args.mode / "anchors.npz"), args.mode, out / "fig6_daily_tilt.png")
    return 0


def fig_weather_pca(pca, k_used: int, path: Path) -> None:
    """Variance explained vs component count, and the shapes of the first three components."""
    evr = np.cumsum(pca["explained_variance_ratio"])
    comps = pca["components"]
    fig = plt.figure(figsize=(12, 3.6), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1, 1])

    ax = fig.add_subplot(gs[0])
    k = np.arange(1, len(evr) + 1)
    ax.plot(k, 100 * evr, color=SERIES3[0])
    ax.axvline(k_used, color=MUTED, linewidth=0.8)
    ax.annotate(f"k = {k_used} used\n{100 * evr[k_used - 1]:.1f}% of variance", (k_used, 100 * evr[k_used - 1]),
                xytext=(k_used + 6, 100 * evr[k_used - 1] - 9), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
    ax.set_xlim(0, 40)
    ax.set_ylim(min(100 * evr[0], 50) - 5, 100.5)
    ax.set_xlabel("components retained")
    ax.set_ylabel("variance explained (%)")
    ax.set_title("Variance captured")

    hours = np.arange(24)
    for i, (name, unit) in enumerate([("Temperature", "T"), ("Relative humidity", "RH"), ("Solar (GHI)", "GHI")]):
        ax = fig.add_subplot(gs[i + 1])
        for j in range(3):
            ax.plot(hours, comps[j, i * 24:(i + 1) * 24], color=SERIES3[j], label=f"PC{j + 1}")
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("local hour")
        ax.set_title(f"{name}")
        if i == 0:
            ax.set_ylabel("loading (standardized)")
        if i == 2:
            ax.legend(loc="upper right", fontsize=9, handlelength=1.2)
    fig.suptitle("Fig. 1  Weather PCA: each day's 24-h T, RH and GHI cycle as a few numbers",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_featurization_check(summary: pd.DataFrame, k_used: int, path: Path) -> None:
    """Annual-yield error of PCA-rebuilt years vs real ones, by component count and regime."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    for ax, col, title in [(axes[0], "median_abs_annual_rel_err", "Median location"),
                           (axes[1], "p90_abs_annual_rel_err", "90th-percentile location")]:
        for regime, color in REGIME_COLOR.items():
            g = summary[summary.regime == regime].sort_values("k")
            if g.empty:
                continue
            ax.plot(g.k, 100 * g[col], color=color, marker="o", markersize=5, label=REGIME_LABEL[regime])
        _end_labels(ax, [(REGIME_LABEL[r], summary[summary.regime == r].sort_values("k"))
                         for r in REGIME_COLOR if (summary.regime == r).any()], col)
        ax.axvline(k_used, color=MUTED, linewidth=0.8)
        ax.text(k_used, 0.95, f" k = {k_used} used", transform=ax.get_xaxis_transform(), color=INK2, fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ks = sorted(summary.k.unique())
        ax.set_xticks(ks, [str(v) + (" (floor)" if v == 72 else "") for v in ks])
        ax.minorticks_off()
        ax.set_xlim(min(ks) * 0.85, max(ks) * 1.9)
        ax.set_xlabel("PCA components used to rebuild the year")
        ax.set_title(title)
    axes[0].set_ylabel("annual water-yield error vs real year (%)")
    axes[0].set_yticks([0.2, 0.5, 1, 2, 5, 10, 20, 50])
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.suptitle("Fig. 2  Do the retained components keep what matters for yield?\n"
                 "Physics on each real year vs the same year rebuilt from k components (12 locations per climate)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_surrogate_holdout(report: dict, path: Path) -> None:
    """Held-out annual error (median, p90) and within-location design ranking, by regime."""
    regimes = [r for r in REGIME_COLOR if r in report["annual"]]
    y = np.arange(len(regimes))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.4), layout="constrained", sharey=True,
                             gridspec_kw={"width_ratios": [1, 0.75]})

    ax = axes[0]
    for yi, r in zip(y, regimes):
        a = report["annual"][r]
        ax.plot([100 * a["median_rel_err"], 100 * a["p90_rel_err"]], [yi, yi], color=REGIME_COLOR[r], linewidth=2)
        ax.plot(100 * a["median_rel_err"], yi, "o", color=REGIME_COLOR[r], markersize=8,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax.plot(100 * a["p90_rel_err"], yi, "|", color=REGIME_COLOR[r], markersize=12, markeredgewidth=2)
        ax.text(100 * a["p90_rel_err"] + 0.15, yi, f"{100 * a['median_rel_err']:.1f}% / {100 * a['p90_rel_err']:.1f}%",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [REGIME_LABEL[r] for r in regimes])
    ax.set_xlim(0, max(100 * report["annual"][r]["p90_rel_err"] for r in regimes) * 1.45)
    ax.set_xlabel("annual water-yield error (%)")
    ax.set_title("Year-long accuracy: dot = median, tick = 90th pct")

    ax = axes[1]
    for yi, r in zip(y, regimes):
        d = report["design_ranking"][r]
        ax.plot(d["median_spearman"], yi, "o", color=REGIME_COLOR[r], markersize=8,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax.text(1.03, yi, f"{d['median_spearman']:.3f}   thickness sign agrees {100 * d['thickness_sign_agreement']:.0f}%",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0.95, 1.0)
    ax.set_xlabel("rank correlation with physics across 16 designs")
    ax.set_title("Design ranking (median over held-out locations)")
    n_cells = report["design_ranking"]["all"]["n_cells"]
    fig.suptitle(f"Fig. 3  Daily surrogate on {n_cells} held-out locations it never trained on",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_anchor_map(anchors: pd.DataFrame, col: str, cbar_label: str, title: str, path: Path) -> None:
    """One anchor-location column on a world map, sequential blue (2nd-98th percentile)."""
    import cartopy.crs as ccrs

    d = anchors[anchors.feasible]
    lo, hi = d[col].quantile(0.02), d[col].quantile(0.98)
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig = plt.figure(figsize=(12, 5.2), layout="constrained")
    ax = world_ax(fig, (1, 1, 1))
    ax.set_extent([-180, 180, -58, 80], crs=ccrs.PlateCarree())
    sc = ax.scatter(d.lon, d.lat, c=d[col].clip(lo, hi), cmap=cmap, vmin=lo, vmax=hi, s=22,
                    edgecolors="#fcfcfb", linewidths=0.4, transform=ccrs.PlateCarree(), zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01, extend="both")
    cb.set_label(cbar_label, color=INK2)
    cb.outline.set_edgecolor(AXIS)
    ax.set_title(title.format(n=len(d)), loc="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.15)  # set_extent leaves blank figure rows
    plt.close(fig)


def fig_lcow_by_regime(anchors: pd.DataFrame, mode: str, path: Path) -> None:
    """Distribution of predicted LCOW and water per regime (strip + median)."""
    d = anchors[anchors.feasible]
    regimes = [r for r in REGIME_COLOR if (d.regime == r).any()]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained", sharey=True)
    for ax, col, label in [(axes[0], "lcow_usd_m3", "predicted LCOW (USD/m³)"),
                           (axes[1], "water_kg_m2_day", "predicted water (kg/m²/day)")]:
        for yi, r in enumerate(regimes[::-1]):
            v = d.loc[d.regime == r, col].to_numpy()
            ax.scatter(v, yi + rng.uniform(-0.18, 0.18, len(v)), s=9, color=REGIME_COLOR[r], alpha=0.55,
                       linewidths=0)
            med = np.median(v)
            ax.plot([med, med], [yi - 0.32, yi + 0.32], color=INK, linewidth=1.6)
            ax.text(med, yi + 0.36, f"{med:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(regimes)), [f"{REGIME_LABEL[r]}  (n={int((d.regime == r).sum())})"
                                            for r in regimes[::-1]])
        ax.set_xlabel(label + ";  black bar = median")
        ax.grid(axis="y", visible=False)
    if (d.lcow_usd_m3.max() / d.lcow_usd_m3.min()) > 4:
        axes[0].set_xscale("log")
        ticks = [t for t in (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50) if d.lcow_usd_m3.min() * 0.8 <= t <= d.lcow_usd_m3.max() * 1.2]
        axes[0].set_xticks(ticks, [str(t) for t in ticks])
        axes[0].minorticks_off()
    fig.suptitle(f"Fig. 5  Cost and yield of each anchor's best design, by climate  ({_mode_label(mode)})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_daily_tilt(anchors: pd.DataFrame, schedules, mode: str, path: Path) -> None:
    """Chosen tilt through the year at three anchors spanning latitude."""
    controls = dict(zip(schedules["cell"].tolist(), schedules["controls"]))
    picks = []
    for target in (-30.0, 0.0, 40.0):  # southern mid-latitude, equator, northern mid-latitude
        c = anchors.iloc[(anchors.lat - target).abs().argmin()]
        picks.append(c)
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    days = np.arange(366)
    for color, c in zip(SERIES3, picks):
        tilt = pd.Series(controls[int(c.cell)][:, 2]).rolling(7, center=True, min_periods=1).mean()
        ns = "" if round(abs(c.lat)) == 0 else ("N" if c.lat > 0 else "S")
        label = f"{abs(c.lat):.0f}°{ns}, {abs(c.lon):.0f}°{'E' if c.lon >= 0 else 'W'}"
        ax.plot(days, tilt, color=color, label=label)
        ax.annotate(label, (days[-1], tilt.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    month_starts = pd.date_range("2024-01-01", periods=12, freq="MS").dayofyear - 1
    ax.set_xticks(month_starts, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlim(0, 366 * 1.32)
    ax.set_ylim(-2, 62)
    ax.set_ylabel("chosen tilt (deg), 7-day mean")
    ax.legend(loc="upper left", fontsize=9, ncols=3)
    ax.set_title(f"Fig. 6  The controller re-tilts through the year  ({_mode_label(mode)})",
                 fontsize=12, color=INK)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _end_labels(ax, series, col, min_gap_log10=0.12) -> None:
    """Direct labels at each line's right end, nudged apart on the (log) y axis so close
    endings do not overprint."""
    ends = sorted(((np.log10(100 * g[col].iloc[-1]), name, g.k.iloc[-1]) for name, g in series), key=lambda t: t[0])
    placed = []
    for y, name, x in ends:
        y = max(y, placed[-1] + min_gap_log10) if placed else y
        placed.append(y)
        ax.annotate(name, (x, 10 ** y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2)


def _mode_label(mode: str) -> str:
    tilt, schedule = mode.split("_", 1)
    return f"{tilt} tilt, {schedule} schedule"


if __name__ == "__main__":
    raise SystemExit(main())
