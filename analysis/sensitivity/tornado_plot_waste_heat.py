#!/usr/bin/env python3
"""Elasticity tornado table + PNG from the waste-heat parameter-sweep CSV.

Usage: python tornado_plot_waste_heat.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_REPO = Path(__file__).resolve().parent.parent.parent / "waste-heat"

_EXCLUDED_PARAMS = frozenset({"humidity_high", "relative_humidity"})
_FAIL_LCO_THRESHOLD = 1e20
_BAR_COLOR = "#7A9E9E"

_METRIC_LABELS: dict[str, str] = {
    "lcow_usd_per_m3": "LCOW (USD/m³)",
    "npv_usd_per_m2": "NPV (USD/m²)",
    "payback_years_simple": "Simple payback (years)",
    "payback_years_discounted": "Discounted payback (years)",
}

# Runaway-value guard shaped for LCOW (strictly positive, has a FAIL sentinel). Signed
# metrics like NPV/payback would be clipped at legitimate values, so they skip it.
_METRICS_WITH_OUTLIER_CAP = frozenset({"lcow_usd_per_m3"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--table-csv", type=Path, default=None)
    ap.add_argument("--metric", default="lcow_usd_per_m3")
    args = ap.parse_args()

    inp = args.input or _REPO / "parameter_sweeps" / "parameter_sweep.csv"
    out_png = args.output or _REPO / "tornado_plots" / "tornado_plot.png"
    if not inp.exists():
        sys.exit(f"Missing {inp}; run parameter_sweep.py first.")

    sweep, baseline_metric = _load_sweep(inp, args.metric)
    table = _build_table(sweep, args.metric, baseline_metric)
    table_csv = args.table_csv or out_png.with_suffix(".table.csv")
    table_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_csv, index=False)
    _plot_tornado(table, out_png, _METRIC_LABELS.get(args.metric, args.metric))
    print(f"Wrote {table_csv}")
    print(f"Wrote {out_png}")


def _load_sweep(path: Path, metric: str) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(path)
    bl_rows = df[df["sweep_param"] == "baseline"]
    baseline_value = float(bl_rows[metric].iloc[0]) if not bl_rows.empty else float("nan")

    sweep = df[(df["sweep_param"] != "baseline") & ~df["sweep_param"].isin(_EXCLUDED_PARAMS)].copy()
    sweep[metric] = pd.to_numeric(sweep[metric], errors="coerce")
    sweep = sweep.dropna(subset=[metric])
    sweep = sweep[sweep[metric] < _FAIL_LCO_THRESHOLD]
    if not math.isfinite(baseline_value):
        baseline_value = float(sweep[metric].median())
    return sweep, baseline_value


def _build_table(sweep: pd.DataFrame, metric: str, baseline_metric: float) -> pd.DataFrame:
    rows = []
    for key, grp in sweep.groupby("sweep_param"):
        valid = grp[grp[metric] < _FAIL_LCO_THRESHOLD]
        if metric in _METRICS_WITH_OUTLIER_CAP:
            valid = valid[valid[metric] <= max(baseline_metric * 50.0, 1e4)]
        grp = valid.sort_values("param_value")
        if len(grp) < 2:
            continue

        param_vals = grp["param_value"].astype(float)
        metric_vals = grp[metric].astype(float)
        bl_idx = (param_vals - param_vals.median()).abs().idxmin()
        param_base = float(param_vals.loc[bl_idx])
        metric_base = float(metric_vals.loc[bl_idx])
        # Only fall back to the overall baseline at (near-)zero, which would blow up the
        # elasticity denominator -- a legitimately negative NPV base must not be overwritten.
        if not math.isfinite(metric_base) or abs(metric_base) < 1e-9:
            metric_base = baseline_metric

        dec = _elasticity(float(param_vals.iloc[0]), float(metric_vals.iloc[0]), param_base, metric_base)
        inc = _elasticity(float(param_vals.iloc[-1]), float(metric_vals.iloc[-1]), param_base, metric_base)
        if not math.isfinite(dec) or not math.isfinite(inc):
            continue

        rows.append({
            "sweep_param": key,
            "param_label": grp["param_label"].iloc[0],
            "baseline_metric": metric_base,
            "decrease_sensitivity": dec,
            "increase_sensitivity": inc,
            "total_span": abs(dec) + abs(inc),
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table[table["total_span"] > 1e-6].sort_values("total_span", ascending=True)


def _elasticity(param_val: float, metric_val: float, param_base: float, metric_base: float) -> float:
    param_frac = (param_val - param_base) / param_base
    metric_frac = (metric_val - metric_base) / metric_base
    if abs(param_frac) < 1e-15 or not math.isfinite(metric_frac):
        return float("nan")
    return metric_frac / param_frac


def _plot_tornado(table: pd.DataFrame, out_png: Path, metric_label: str) -> None:
    if table.empty:
        print(f"No valid sensitivity rows for {metric_label!r} -- skipping plot.")
        return
    labels = table["param_label"].tolist()
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for yi, dec, inc in zip(y, table["decrease_sensitivity"], table["increase_sensitivity"]):
        for sens in (dec, inc):
            if abs(sens) < 1e-15:
                continue
            # Bar spans [sens, 0] when negative, [0, sens] when positive.
            ax.barh(yi, abs(sens), left=min(sens, 0.0), height=0.65, color=_BAR_COLOR,
                    hatch="///" if sens < 0 else None, edgecolor="white", linewidth=0.5)

    ax.axvline(0, color="black", linewidth=1.0, zorder=3)
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"% change in {metric_label} per % change in parameter")
    ax.set_title("Tornado sensitivity")
    ax.legend(
        handles=[
            Patch(facecolor=_BAR_COLOR, edgecolor="white", label="Positive correlation"),
            Patch(facecolor=_BAR_COLOR, edgecolor="white", hatch="///", label="Negative correlation"),
        ],
        loc="lower right",
        frameon=True,
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
