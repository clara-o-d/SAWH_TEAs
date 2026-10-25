#!/usr/bin/env python3
"""Summary and figures for run_monte_carlo.py's outputs (samples.csv, timeseries.npz)
-- no solves, so it redraws in seconds.

Writes outputs/summary.txt (per site and eps_Al scenario: yield percentiles against
Wilson's measured yield, Spearman rank correlation of yield with each sampled
parameter, and the parameters left unsampled with why) and outputs/monte_carlo.png
(yield histograms per site with the dry and wet scenarios overlaid, over grouped
rank-correlation bars), and outputs/recreation_bands.png: the Fig. 3B/3C (Cambridge)
and Fig. 4C/4D (Atacama) recreations as our model's median and P10-P90 band per eps_Al
scenario, with Wilson's measurements on top -- the thermocouple records
(*_experiment.csv) and the measured water output (Cambridge graduated-cylinder record;
Atacama endpoint only, its container was read once).

Examples::

  python plot_monte_carlo.py
  python plot_monte_carlo.py --csv other_run/samples.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from run_monte_carlo import EPS_AL_SCENARIOS, MEASURED_YIELD_KG_M2, NOT_SAMPLED, SCENARIOS, SITES  # noqa: E402
from run_monte_carlo import figure3  # noqa: E402  (already on sys.path via run_monte_carlo)
from solar_lumped.weather import ATACAMA_DESORPTION_START_OFFSET_H  # noqa: E402
from solar_lumped.plotting import (  # noqa: E402
    figure_size_inches,
    plot_defaults_slides,
    print_figure,
    scaled_fontsize,
    style_axes,
)

_OUT_DIR = Path(__file__).resolve().parent / "outputs"
# Categorical slots 1-2 of the dataviz reference palette, validated as a pair (light
# surface: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 contrast).
_COLOR = {"dry": "#2a78d6", "wet": "#eb6834"}
_INK = "0.15"
# Temperature components: dataviz reference slots 8/7/6 (red, violet, green), validated
# as a set on the light surface (CVD dE 22.7, normal-vision dE 33.6, all >= 3:1) and
# distinct from the scenario pair above, since a figure can't reuse a hue for two things.
_COMPONENTS = {"t_abs_c": ("absorber", "#e34948"), "t_glass_c": ("glass", "#4a3aa7"),
               "t_cond_c": ("condenser", "#008300")}
_REF = Path(__file__).resolve().parents[1] / "paper_recreation" / "wilson" / "reference"
# Wilson's thermocouple records, digitized: (site, component) -> file.
_MEASURED_T = {
    (site, comp): _REF / fig / f"{name}_{label}_experiment.csv"
    for site, fig, name in (("cambridge", "figure3", "Cambridge"), ("atacama", "figure4", "Atacama"))
    for comp, (label, _) in _COMPONENTS.items()
}
# Hours on each site's time axis, as the paper plots it: Cambridge from the 9:15 a.m.
# desorption start; Atacama from the 8 a.m. install, desorption beginning 0.15 h later.
_AXIS = {"cambridge": ("Hours since 9:15 a.m.", 0.0),
         "atacama": ("Hours since 8 a.m.", ATACAMA_DESORPTION_START_OFFSET_H)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=_OUT_DIR / "samples.csv")
    args = ap.parse_args()
    out = args.csv.parent

    df = pd.read_csv(args.csv, keep_default_na=False, na_values=["nan"])
    nominal = df[df["sample"] == "nominal"].iloc[0]
    df = df[df["sample"] != "nominal"]
    # Parameter columns are everything between "sample" and the first per-run result.
    cols = list(df.columns)
    keys = cols[1 : next(i for i, c in enumerate(cols) if c.startswith("yield_kg_m2_"))]

    # --- summary: distribution vs measured, and which parameters drive it ---
    rho: dict[tuple[str, str], list[float]] = {}
    lines = [f"Monte Carlo, n={len(df)} draws  ({args.csv})", ""]
    for site in SITES:
        meas = MEASURED_YIELD_KG_M2[site]
        lines.append(f"{site}  (measured {meas:.3f} kg/m2/d)")
        for sc in SCENARIOS:
            lo, hi, eps_nom = EPS_AL_SCENARIOS[sc]
            y = df[f"yield_kg_m2_{site}_{sc}"].to_numpy(float)
            ok = np.isfinite(y)
            p5, p50, p95 = np.percentile(y[ok], [5, 50, 95])
            lines.append(
                f"  {sc} (eps_Al {lo:.2f}-{hi:.2f}): median {p50:.3f}  5-95% [{p5:.3f}, {p95:.3f}]  "
                f"nominal {float(nominal[f'yield_kg_m2_{site}_{sc}']):.3f} (eps_Al {eps_nom:.2f})  "
                f"P(yield <= measured) {np.mean(y[ok] <= meas):.2f}  failed {int((~ok).sum())}/{len(y)}"
            )
            rho[site, sc] = [float(spearmanr(df[k].to_numpy(float)[ok], y[ok]).statistic) for k in keys]
            for j in np.argsort(-np.abs(rho[site, sc])):
                lines.append(f"      {keys[j]:16s} Spearman rho = {rho[site, sc][j]:+.2f}")
        lines.append("")
    lines.append("Not sampled:")
    lines += [f"    {name}: {why}" for name, why in NOT_SAMPLED.items()]
    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary + "\n")
    print(summary)

    # --- figure: yield histograms per site (top), rank correlations (bottom) ---
    plot_defaults_slides()
    # Bottom row is 13 parameters x 2 scenarios, so it gets twice the histograms' height.
    w, h = figure_size_inches(2, 3)
    fig, axes = plt.subplots(2, 2, figsize=(w, h), gridspec_kw={"height_ratios": [1, 2]})
    small = scaled_fontsize("font.size", 0.6)
    labels = {sc: f"{sc} (ε_Al {EPS_AL_SCENARIOS[sc][0]:.2f}–{EPS_AL_SCENARIOS[sc][1]:.2f})" for sc in SCENARIOS}
    for col, site in enumerate(SITES):
        ax = axes[0, col]
        ys = {sc: df[f"yield_kg_m2_{site}_{sc}"].to_numpy(float) for sc in SCENARIOS}
        ys = {sc: y[np.isfinite(y)] for sc, y in ys.items()}
        # One shared bin grid so the two scenarios' bars line up.
        edges = np.histogram_bin_edges(np.concatenate(list(ys.values())), bins=40)
        for sc in SCENARIOS:
            ax.hist(ys[sc], bins=edges, color=_COLOR[sc], alpha=0.75, edgecolor="white",
                    linewidth=0.6, label=labels[sc])
        ax.axvline(MEASURED_YIELD_KG_M2[site], color=_INK, linewidth=2, linestyle="--")
        top = ax.get_ylim()[1]
        ax.annotate("measured", (MEASURED_YIELD_KG_M2[site], top * 0.97), xytext=(4, 0),
                    textcoords="offset points", color=_INK, va="top", fontsize=small)
        ax.set_title(site.capitalize())
        ax.set_xlabel("Daily yield (kg/m²/d)")
        ax.set_ylabel("Samples")
        style_axes(ax)

        ax = axes[1, col]
        # Parameter order by the larger |rho| of the two scenarios, strongest on top.
        order = np.argsort(np.max([np.abs(rho[site, sc]) for sc in SCENARIOS], axis=0))
        pos = np.arange(len(keys))
        for i, sc in enumerate(SCENARIOS):
            ax.barh(pos + (0.2 if i == 0 else -0.2), [rho[site, sc][j] for j in order],
                    height=0.38, color=_COLOR[sc], label=sc)
        ax.set_yticks(pos, [keys[j] for j in order])
        ax.axvline(0.0, color=_INK, linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman ρ with yield")
        ax.tick_params(axis="y", labelsize=small)
        style_axes(ax, tick_in=False)
    # One legend for all four panels -- both rows color the scenarios the same way.
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, fontsize=small)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path = print_figure(fig, out / "monte_carlo.png")
    plt.close(fig)

    # --- recreation bands: one row per site; dry temps | wet temps | water ---
    ts = np.load(out / "timeseries.npz")
    names = list(ts["series"])
    w, h = figure_size_inches(3, 2)
    fig, axes = plt.subplots(2, 3, figsize=(w, h * 1.1), sharex="row")
    for row, site in enumerate(("cambridge", "atacama")):
        xlabel, offset_h = _AXIS[site]
        hours = ts[f"{site}_time_s"] / 3600.0 + offset_h
        # P10/P50/P90 over draws; draw 0 is the nominal, not a sample, so it's excluded.
        pct = {sc: np.nanpercentile(ts[f"{site}_{sc}"][1:], [10, 50, 90], axis=0) for sc in SCENARIOS}
        for col, sc in enumerate(SCENARIOS):
            ax = axes[row, col]
            for comp, (label, color) in _COMPONENTS.items():
                p10, p50, p90 = pct[sc][:, names.index(comp)]
                ax.fill_between(hours, p10, p90, color=color, alpha=0.2, linewidth=0)
                ax.plot(hours, p50, color=color, linewidth=2, label=f"{label} (model)")
                meas = np.genfromtxt(_MEASURED_T[site, comp], delimiter=",")
                meas = meas[~np.isnan(meas).any(axis=1)]
                ax.scatter(meas[:, 0], meas[:, 1], s=22, facecolors="white", edgecolors=color,
                           linewidths=1.4, zorder=5, label=f"{label} (Wilson measured)")
            lo, hi, _ = EPS_AL_SCENARIOS[sc]
            ax.set_title(f"{site.capitalize()}: {sc} condenser (ε_Al {lo:.2f}–{hi:.2f})", fontsize=small * 1.3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Temperature (°C)")
            style_axes(ax)
        # Both temperature panels of a row share one y range so dry vs wet reads directly.
        lims = [axes[row, c].get_ylim() for c in range(len(SCENARIOS))]
        for c in range(len(SCENARIOS)):
            axes[row, c].set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))

        ax = axes[row, len(SCENARIOS)]
        for sc in SCENARIOS:
            p10, p50, p90 = pct[sc][:, names.index("water_cumulative_kg_m2")]
            ax.fill_between(hours, p10, p90, color=_COLOR[sc], alpha=0.25, linewidth=0)
            ax.plot(hours, p50, color=_COLOR[sc], linewidth=2, label=f"{sc} (model)")
        if site == "cambridge":
            # Graduated-cylinder record: mL of system total over the 0.078 m2 absorber.
            meas = np.genfromtxt(_REF / "figure3" / "Cambridge_water_output_ml.csv", delimiter=",")
            meas = meas[~np.isnan(meas).any(axis=1)]
            ax.scatter(meas[:, 0], meas[:, 1] / figure3._A_C_M2 / 1000.0, s=22, facecolors="white",
                       edgecolors=_INK, linewidths=1.4, zorder=5, label="Wilson measured")
        else:
            ax.scatter([hours[-1]], [MEASURED_YIELD_KG_M2[site]], marker="*", s=120, color=_INK,
                       zorder=5, label="Wilson measured (endpoint)")
        ax.set_title(f"{site.capitalize()}: cumulative water", fontsize=small * 1.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Water (L/m²)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=small, frameon=False, loc="upper left")
        style_axes(ax)
    # Temperature legend once: the components and markers mean the same in all four panels.
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3,
               frameon=False, fontsize=small)
    fig.text(0.5, 0.005, f"Lines: median of {ts[f'cambridge_{SCENARIOS[0]}'].shape[0] - 1} draws; bands: P10–P90",
             ha="center", fontsize=small, color=_INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    bands_path = print_figure(fig, out / "recreation_bands.png")
    plt.close(fig)

    # --- Cambridge-only 1x2 for slides: dry-condenser temps | dry-condenser water, the
    # scenario left unnamed since it's the only one shown ---
    site, sc = "cambridge", "dry"
    xlabel, offset_h = _AXIS[site]
    hours = ts[f"{site}_time_s"] / 3600.0 + offset_h
    pct = np.nanpercentile(ts[f"{site}_{sc}"][1:], [10, 50, 90], axis=0)
    w, h = figure_size_inches(2, 1)
    fig, (ax_t, ax_w) = plt.subplots(1, 2, figsize=(w, h * 1.15), sharex=True)
    for comp, (label, color) in _COMPONENTS.items():
        p10, p50, p90 = pct[:, names.index(comp)]
        ax_t.fill_between(hours, p10, p90, color=color, alpha=0.2, linewidth=0)
        ax_t.plot(hours, p50, color=color, linewidth=2, label=f"{label} (model)")
        meas = np.genfromtxt(_MEASURED_T[site, comp], delimiter=",")
        meas = meas[~np.isnan(meas).any(axis=1)]
        ax_t.scatter(meas[:, 0], meas[:, 1], s=22, facecolors="white", edgecolors=color,
                     linewidths=1.4, zorder=5, label=f"{label} (Wilson measured)")
    ax_t.set_title("Temperatures", fontsize=small * 1.3)
    ax_t.set_ylabel("Temperature (°C)")
    p10, p50, p90 = pct[:, names.index("water_cumulative_kg_m2")]
    ax_w.fill_between(hours, p10, p90, color=_COLOR[sc], alpha=0.25, linewidth=0)
    ax_w.plot(hours, p50, color=_COLOR[sc], linewidth=2, label="model")
    meas = np.genfromtxt(_REF / "figure3" / "Cambridge_water_output_ml.csv", delimiter=",")
    meas = meas[~np.isnan(meas).any(axis=1)]
    ax_w.scatter(meas[:, 0], meas[:, 1] / figure3._A_C_M2 / 1000.0, s=22, facecolors="white",
                 edgecolors=_INK, linewidths=1.4, zorder=5, label="Wilson measured")
    ax_w.set_title("Cumulative water", fontsize=small * 1.3)
    ax_w.set_ylabel("Water (L/m²)")
    ax_w.set_ylim(bottom=0)
    ax_w.legend(fontsize=small, frameon=False, loc="upper left")
    for ax in (ax_t, ax_w):
        ax.set_xlabel(xlabel)
        style_axes(ax)
    fig.legend(*ax_t.get_legend_handles_labels(), loc="upper center", ncol=3,
               frameon=False, fontsize=small)
    fig.text(0.5, 0.005, f"Lines: median of {ts[f'{site}_{sc}'].shape[0] - 1} draws; bands: P10–P90",
             ha="center", fontsize=small, color=_INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    cambridge_path = print_figure(fig, out / "recreation_bands_cambridge.png")
    plt.close(fig)
    print(f"\nWrote {out / 'summary.txt'}\n      {fig_path}\n      {bands_path}\n      {cambridge_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
