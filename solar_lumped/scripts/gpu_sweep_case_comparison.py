#!/usr/bin/env python3
"""Side-by-side comparison of the 3 GPU-sweep radiative-physics cases (see
docs/gpu_sweep_handoff.md): Case 1 (original blackbody Eqs. 3/4), Case 2
(selective-surface real IR emissivities), Case 3 (idealized optical-material
limits). Reuses gpu_sweep_analysis.py's data loading / LCOW / sensitivity
functions directly rather than reimplementing them.

Produces:

1. Optimal-configuration LCOW maps, all 3 cases side-by-side on one shared
   colorbar scale.
2. Optimal-configuration yield maps, same layout.
3. Delta maps (Case 2 - Case 1, Case 3 - Case 1) for optimal LCOW -- where the
   modified radiative physics helps most/least, geographically.
4. Grouped-bar sensitivity comparison across the 3 cases: exogenous weather
   variables (always comparable, all 3 cases sweep the same site grid) and the
   2 device parameters common to all 3 cases (hydrogel_thickness_mm,
   fin_area_ratio -- eps_abs/tau_glass aren't swept in Case 3, so they're
   compared separately, Case 1 vs Case 2 only).
5. A headline summary table (median/mean optimal LCOW and yield per case).

Usage::

    python scripts/gpu_sweep_case_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from gpu_sweep_analysis import (  # noqa: E402
    _EXOGENOUS_PARAMS,
    _PARAM_LABELS,
    _import_map_stack,
    _world_ax,
    build_optimal_config,
    device_param_tornado,
    exogenous_tornado,
    load_data,
    swept_device_params,
)

_OUT = _REPO / "outputs" / "gpu_grid_sweep_comparison"

_CASES: tuple[tuple[str, Path], ...] = (
    ("Case 1\n(blackbody)", _REPO / "outputs" / "gpu_grid_sweep" / "full_sweep.csv"),
    ("Case 2\n(selective surface)", _REPO / "outputs" / "gpu_grid_sweep_case2" / "full_sweep_case2.csv"),
    ("Case 3\n(optical limits)", _REPO / "outputs" / "gpu_grid_sweep_case3" / "full_sweep_case3.csv"),
)

_COMMON_DEVICE_PARAMS: tuple[str, ...] = ("hydrogel_thickness_mm", "fin_area_ratio")
_CASE_COLORS: tuple[str, ...] = ("#3B4CC0", "#20A387", "#DD8452")


def load_all_cases() -> dict[str, pd.DataFrame]:
    out = {}
    for label, csv_path in _CASES:
        name = label.split("\n")[0]
        print(f"Loading {name}: {csv_path} ...", flush=True)
        df = load_data(csv_path)
        print(f"  {len(df)} rows, swept device params: {swept_device_params(df)}", flush=True)
        out[label] = df
    return out


# --------------------------------------------------------------------------- 1/2. side-by-side optimal maps


def _plot_side_by_side_maps(
    winners_by_case: dict[str, pd.DataFrame], value_col: str, cmap: str, log_scale: bool,
    cbar_label: str, suptitle: str, out_path: Path,
) -> None:
    ccrs, cfeature = _import_map_stack()

    all_vals = np.concatenate([w[value_col].to_numpy() for w in winners_by_case.values()])
    if log_scale:
        all_vals = np.clip(all_vals, 1e-9, None)
        norm = LogNorm(vmin=float(all_vals.min() * 0.9), vmax=float(all_vals.max() * 1.1))
    else:
        norm = plt.Normalize(vmin=0.0, vmax=float(all_vals.max() * 1.02))

    n = len(winners_by_case)
    fig = plt.figure(figsize=(6.5 * n + 1.0, 5.6))
    fig.suptitle(suptitle, fontsize=12, y=1.03)

    sc_last = None
    for i, (label, winners) in enumerate(winners_by_case.items()):
        ax = _world_ax(fig, (1, n, i + 1), ccrs=ccrs, cfeature=cfeature)
        ax.set_title(label.replace("\n", " "), fontsize=10, pad=5)
        sc_last = ax.scatter(
            winners["lon"], winners["lat"], c=winners[value_col], s=13, marker="o",
            transform=ccrs.PlateCarree(), zorder=4, cmap=cmap, norm=norm,
        )
    cbar = fig.colorbar(sc_last, ax=fig.axes, fraction=0.02, pad=0.03, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- 3. delta maps


def _plot_delta_maps(winners_by_case: dict[str, pd.DataFrame], out_path: Path) -> None:
    ccrs, cfeature = _import_map_stack()
    labels = list(winners_by_case.keys())
    base = winners_by_case[labels[0]].sort_values(["lat", "lon"]).reset_index(drop=True)

    deltas = []
    for label in labels[1:]:
        other = winners_by_case[label].sort_values(["lat", "lon"]).reset_index(drop=True)
        assert (other["lat"].to_numpy() == base["lat"].to_numpy()).all(), "site grids must match across cases"
        pct = (other["lcow_usd_per_m3"].to_numpy() - base["lcow_usd_per_m3"].to_numpy()) / base["lcow_usd_per_m3"].to_numpy() * 100.0
        deltas.append((label, pct))

    vmax = max(float(np.abs(pct).max()) for _, pct in deltas)
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    fig = plt.figure(figsize=(6.5 * len(deltas) + 1.0, 5.6))
    fig.suptitle(
        "Change in optimal-configuration LCOW vs. Case 1 (blackbody radiative physics)\n"
        "negative (blue) = cheaper water under the modified physics",
        fontsize=12, y=1.03,
    )
    sc_last = None
    for i, (label, pct) in enumerate(deltas):
        ax = _world_ax(fig, (1, len(deltas), i + 1), ccrs=ccrs, cfeature=cfeature)
        ax.set_title(f"{label.replace(chr(10), ' ')} vs. Case 1", fontsize=10, pad=5)
        sc_last = ax.scatter(
            base["lon"], base["lat"], c=pct, s=13, marker="o",
            transform=ccrs.PlateCarree(), zorder=4, cmap="RdBu_r", norm=norm,
        )
    cbar = fig.colorbar(sc_last, ax=fig.axes, fraction=0.02, pad=0.03, shrink=0.85)
    cbar.set_label("Optimal LCOW change (%)", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- 4. grouped-bar sensitivity comparison


def _grouped_bar_comparison(
    sens_by_case: dict[str, pd.DataFrame], variables: tuple[str, ...], metric_label: str,
    title: str, out_path: Path,
) -> None:
    labels = list(sens_by_case.keys())
    x = np.arange(len(variables))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, label in enumerate(labels):
        sens_df = sens_by_case[label].set_index("variable")
        vals = [float(sens_df.loc[v, "max_abs_effect"]) if v in sens_df.index else np.nan for v in variables]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=label.replace("\n", " "), color=_CASE_COLORS[i])

    ax.set_xticks(x)
    ax.set_xticklabels([_PARAM_LABELS.get(v, v).replace("\n", " ") for v in variables], fontsize=9)
    ax.set_ylabel(f"|Sensitivity| — {metric_label}", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- main


def main() -> int:
    data = load_all_cases()
    winners = {label: build_optimal_config(df) for label, df in data.items()}

    print("\n--- 1. Side-by-side optimal LCOW maps ---", flush=True)
    _plot_side_by_side_maps(
        winners, "lcow_usd_per_m3", cmap="viridis", log_scale=True,
        cbar_label="Optimal-configuration LCOW (USD/m³, log scale)",
        suptitle="Best achievable LCOW per site, by radiative-physics case",
        out_path=_OUT / "lcow_optimal_map_comparison.png",
    )

    print("\n--- 2. Side-by-side optimal yield maps ---", flush=True)
    _plot_side_by_side_maps(
        winners, "mean_yield_kg_m2", cmap="YlGnBu", log_scale=False,
        cbar_label="Yield at the min-LCOW configuration (kg/m²/day)",
        suptitle="Water yield at the best-LCOW configuration per site, by radiative-physics case",
        out_path=_OUT / "yield_at_optimal_map_comparison.png",
    )

    print("\n--- 3. Delta maps (Case 2/3 vs Case 1) ---", flush=True)
    _plot_delta_maps(winners, _OUT / "lcow_delta_map_vs_case1.png")

    print("\n--- 4. Exogenous-weather sensitivity comparison ---", flush=True)
    exo_sens = {
        label: exogenous_tornado(df, "lcow_usd_per_m3", swept_device_params(df))
        for label, df in data.items()
    }
    _grouped_bar_comparison(
        exo_sens, _EXOGENOUS_PARAMS, metric_label="LCOW elasticity",
        title="Exogenous weather sensitivity — LCOW, all 3 cases",
        out_path=_OUT / "sensitivity_comparison_exogenous_lcow.png",
    )

    print("\n--- 5. Common device-parameter sensitivity comparison ---", flush=True)
    device_sens = {
        label: device_param_tornado(df, "lcow_usd_per_m3", swept_device_params(df))
        for label, df in data.items()
    }
    _grouped_bar_comparison(
        device_sens, _COMMON_DEVICE_PARAMS, metric_label="LCOW sensitivity (%/%)",
        title="Device-parameter sensitivity common to all 3 cases — LCOW",
        out_path=_OUT / "sensitivity_comparison_common_device_params_lcow.png",
    )

    print("\n--- 6. Headline summary table ---", flush=True)
    rows = []
    for label, w in winners.items():
        rows.append({
            "case": label.replace("\n", " "),
            "n_sites": len(w),
            "median_lcow_usd_per_m3": float(w["lcow_usd_per_m3"].median()),
            "mean_lcow_usd_per_m3": float(w["lcow_usd_per_m3"].mean()),
            "median_yield_kg_m2": float(w["mean_yield_kg_m2"].median()),
            "mean_yield_kg_m2": float(w["mean_yield_kg_m2"].mean()),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False), flush=True)
    summary_path = _OUT / "headline_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}", flush=True)

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
