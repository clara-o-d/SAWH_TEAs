#!/usr/bin/env python3
"""One-at-a-time tornado sensitivity plot from solar_lumped's parameter-sweep CSV.

Usage: python tornado_plot_oat.py [--metric ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parent.parent.parent

_FAIL_LCO_THRESHOLD = 1e20
_BAR_COLOR = "#20A387"

_METRIC_COLUMNS = frozenset({
    "daily_yield_kg_m2",
    "thermal_efficiency",
    "lcow_usd_per_m3",
    "capex_usd_per_m3",
    "opex_usd_per_m3",
    "npv_usd_per_m2",
    "payback_years_simple",
    "payback_years_discounted",
})

_METRIC_LABELS: dict[str, str] = {
    "lcow_usd_per_m3": "LCOW (USD/m³)",
    "npv_usd_per_m2": "NPV (USD/m²)",
    "payback_years_simple": "Simple payback (years)",
    "payback_years_discounted": "Discounted payback (years)",
    "daily_yield_kg_m2": "Daily yield (kg/m²)",
    "thermal_efficiency": "Thermal efficiency",
    "capex_usd_per_m3": "CAPEX (USD/m³)",
    "opex_usd_per_m3": "OPEX (USD/m³)",
}


_EXCLUDED_PARAMS = frozenset({"humidity_high", "relative_humidity", "swept_param"})

_PARAM_LABELS: dict[str, str] = {
    "h_des_j_per_kg": "h_des\n(J/kg)",
    "salt_weight_factor": "Salt weight\nfactor",
    "hydrogel_lifetime_years": "Hydrogel lifetime\n(yr)",
    "hydrogel_thickness_mm": "Hydrogel thickness\n(mm)",
    "vapor_gap_mm": "Vapor gap\n(mm)",
    "humidity_high": "Uptake RH",
    "solar_irradiance_w_per_m2": "Solar GHI\n(W/m²)",
    "h_amb_w_m2_k": "h_amb\n(W/m²K)",
    "discount_rate": "Discount rate",
    "device_lifetime_years": "Device lifetime\n(yr)",
    "utilization_factor": "Utilization\nfactor",
    "water_price_usd_per_m3": "Water price\n(USD/m³)",
    "salt_to_polymer_ratio": "Salt:polymer\nratio (S/L)",
    "c_acrylamide_usd_per_kg": "Acrylamide price\n(USD/kg)",
    "c_additives_usd_per_kg_composite": "Additives price\n(USD/kg composite)",
    "insulation_gap_mm": "Insulation gap\n(mm)",
    "fin_area_ratio": "Condenser fin\narea ratio",
    "tilt_deg": "Tilt angle\n(deg)",
    "temperature_c": "Ambient temperature\n(°C)",
    "total_investment_factor": "Total investment\nfactor",
    "maintenance_cost_fraction": "Maintenance cost\n(frac CAPEX/yr)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--table-csv", type=Path, default=None)
    ap.add_argument("--metric", default="lcow_usd_per_m3")
    ap.add_argument("--show", action="store_true", help="Display plot interactively")
    args = ap.parse_args()

    repo = _ROOT / "solar_lumped"
    inp = args.input or repo / "outputs" / "parameter_sweeps" / "parameter_sweep.csv"
    out_png = args.output or repo / "outputs" / "tornado_plot" / "tornado_plot.png"
    if not inp.exists():
        sys.exit(f"Missing {inp}; run parameter_sweep.py first.")

    df = pd.read_csv(inp)
    if args.metric not in df.columns:
        sys.exit(f"Metric column {args.metric!r} not found in {inp}")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=[args.metric])
    df = df[df[args.metric] < _FAIL_LCO_THRESHOLD]
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns from {inp}")
    if len(df) < 3:
        sys.exit("Very few valid data points; cannot proceed with sensitivity analysis.")

    input_params = [c for c in df.columns if c not in _METRIC_COLUMNS and c not in _EXCLUDED_PARAMS]
    sensitivity_df = oat_sensitivity(df, args.metric, input_params)
    if sensitivity_df.empty:
        sys.exit("No valid sensitivity data calculated.")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    table_csv = args.table_csv or out_png.with_suffix(".table.csv")
    sensitivity_df.to_csv(table_csv, index=False)

    fig, _ = create_tornado_plot(
        sensitivity_df,
        args.metric,
        param_name_mapping=_PARAM_LABELS,
    )
    if fig is not None:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        if args.show:
            plt.show()
        plt.close(fig)
    print(f"Wrote {table_csv}")
    print(f"Wrote {out_png}")


def oat_sensitivity(df: pd.DataFrame, target_col: str, input_params: list[str]) -> pd.DataFrame:
    """One-at-a-time sensitivity of ``target_col`` to each input parameter."""
    print(f"Analyzing sensitivity of {target_col!r} to {len(input_params)} input parameters:")
    rows: list[dict] = []

    for var in input_params:
        valid = df.loc[df[var].notna() & df[target_col].notna()]
        if valid.empty:
            print(f"  Warning: no valid data for {var}, skipping")
            continue

        # An OAT pair is two rows agreeing on every other parameter, i.e. one groupby bucket.
        others = [p for p in input_params if p != var]
        buckets = [g for _, g in valid.groupby(others, sort=False)] if others else [valid]

        inc: list[float] = []
        dec: list[float] = []
        for g in buckets:
            pts = sorted(zip(g[var].astype(float), g[target_col].astype(float)))
            for a, (x_lo, y_lo) in enumerate(pts):
                for x_hi, y_hi in pts[a + 1:]:
                    inc.extend(_sensitivity(x_lo, y_lo, x_hi, y_hi))
                    dec.extend(_sensitivity(x_hi, y_hi, x_lo, y_lo))

        if not inc and not dec:
            print(f"  Warning: no valid OAT pairs found for {var}")
            continue
        avg_inc = float(np.mean(inc)) if inc else 0.0
        avg_dec = float(np.mean(dec)) if dec else 0.0
        rows.append({
            "variable": var,
            "avg_sensitivity": float(np.mean(inc + dec)),
            "abs_avg_sensitivity": float(np.mean(np.abs(inc + dec))),
            "avg_increase_sensitivity": avg_inc,
            "avg_decrease_sensitivity": avg_dec,
            "num_increase_points": len(inc),
            "num_decrease_points": len(dec),
            "num_point_sensitivities": len(inc) + len(dec),
            "x_min": float(valid[var].min()),
            "x_median": float(valid[var].median()),
            "x_max": float(valid[var].max()),
            "y_min": float(valid[target_col].min()),
            "y_median": float(valid[target_col].median()),
            "y_max": float(valid[target_col].max()),
            "x_mean": float(valid[var].mean()),
            "y_mean": float(valid[target_col].mean()),
            "valid_points": len(valid),
        })
        print(f"  {var}: increase={avg_inc:.3f} (n={len(inc)}), decrease={avg_dec:.3f} (n={len(dec)})")

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["max_abs_effect"] = out[["avg_increase_sensitivity", "avg_decrease_sensitivity"]].abs().max(axis=1)
    return out.sort_values(by="max_abs_effect", ascending=False)


def _sensitivity(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """[% change in y per % change in x], or [] if x barely moved / the ratio blows up."""
    pct_x = (x2 - x1) / x1 * 100 if abs(x1) > 1e-10 else abs(x2 - x1)
    pct_y = (y2 - y1) / y1 * 100 if abs(y1) > 1e-10 else y2 - y1
    if abs(pct_x) <= 0.01:
        return []
    s = pct_y / pct_x
    return [s] if abs(s) < 1000 else []


def create_tornado_plot(
    sensitivity_df: pd.DataFrame,
    target_col: str,
    title: str = "Parameter sensitivity",
    param_name_mapping: dict[str, str] | None = None,
    metric_label: str | None = None,
) -> tuple[plt.Figure | None, plt.Axes | None]:
    """Horizontal +/- sensitivity bars, one row per parameter, sorted by effect size."""
    if sensitivity_df.empty:
        print("No valid sensitivity data to plot")
        return None, None
    metric_label = metric_label or _METRIC_LABELS.get(target_col, target_col)

    plot_df = sensitivity_df.copy()
    plot_df["variable"] = plot_df["variable"].astype(str).str.lstrip("# ").str.strip()
    plot_df = plot_df.sort_values(by="max_abs_effect", ascending=True)
    names = plot_df["variable"]
    if param_name_mapping is not None:
        names = names.map(param_name_mapping).fillna(plot_df["variable"])

    # Shrink labels once there are enough bars that they would collide.
    n_bars = len(plot_df)
    label_fs = 14 if n_bars <= 12 else max(8.0, 14 - 0.3 * (n_bars - 12))
    fig, ax = plt.subplots(figsize=(8, max(5.0, 0.5 * n_bars + 1.5)))
    ax.set_frame_on(False)

    y_pos = np.arange(n_bars)
    hatch = ["" if s >= 0 else "///" for s in plot_df["avg_increase_sensitivity"]]
    for values in (plot_df["avg_increase_sensitivity"], -plot_df["avg_decrease_sensitivity"]):
        ax.barh(y_pos, values, height=0.35, color=_BAR_COLOR, alpha=0.7, hatch=hatch)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=label_fs)
    ax.set_ylim(-0.6, n_bars - 0.4)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel(f"% change in {metric_label}\nper % change in parameter", fontsize=13)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(axis="x", labelsize=12)
    ax.legend(
        handles=[
            Patch(facecolor=_BAR_COLOR, alpha=0.7, label="Positive"),
            Patch(facecolor=_BAR_COLOR, alpha=0.7, hatch="///", label="Negative"),
        ],
        loc="lower right",
        fontsize=12,
    )
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    main()
