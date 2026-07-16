#!/usr/bin/env python3
"""Build tornado sensitivity table and PNG from parameter sweep CSV."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DEFAULT_INPUT = _REPO / "parameter_sweeps" / "parameter_sweep.csv"
_DEFAULT_OUTPUT = _REPO / "tornado_plots" / "tornado_plot.png"

_EXCLUDED_PARAMS = frozenset({"humidity_high", "relative_humidity"})
_FAIL_LCO_THRESHOLD = 1e20
_BAR_COLOR = "#7A9E9E"

_METRIC_LABELS: dict[str, str] = {
    "lcow_usd_per_m3": "LCOW (USD/m³)",
    "npv_usd_per_m2": "NPV (USD/m²)",
    "payback_years_simple": "Simple payback (years)",
    "payback_years_discounted": "Discounted payback (years)",
}

# The outlier cap below is shaped for LCOW: it is a strictly-positive cost
# metric with a known FAIL sentinel (FAIL_LCO), so "> 50x baseline" is a
# sensible runaway-value guard. NPV/payback are signed (NPV can be very
# negative "on-target"; payback can legitimately be +inf) so the same
# heuristic would clip real, meaningful values -- it's a no-op for any
# metric other than lcow_usd_per_m3.
_METRICS_WITH_OUTLIER_CAP = frozenset({"lcow_usd_per_m3"})


def _load_sweep(path: Path, metric: str) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(path)
    bl_rows = df[df["sweep_param"] == "baseline"]
    if not bl_rows.empty:
        baseline_value = float(bl_rows[metric].iloc[0])
    else:
        baseline_value = float("nan")
    sweep = df[df["sweep_param"] != "baseline"].copy()
    sweep = sweep[~sweep["sweep_param"].isin(_EXCLUDED_PARAMS)]
    sweep[metric] = pd.to_numeric(sweep[metric], errors="coerce")
    sweep = sweep.dropna(subset=[metric])
    sweep = sweep[sweep[metric] < _FAIL_LCO_THRESHOLD]
    if not math.isfinite(baseline_value):
        baseline_value = float(sweep[metric].median())
    return sweep, baseline_value


def _elasticity(
    param_val: float,
    metric_val: float,
    param_base: float,
    metric_base: float,
) -> float:
    param_frac = (param_val - param_base) / param_base
    metric_frac = (metric_val - metric_base) / metric_base
    if abs(param_frac) < 1e-15 or not math.isfinite(metric_frac):
        return float("nan")
    return metric_frac / param_frac


def _valid_sweep_points(grp: pd.DataFrame, metric: str, baseline_metric: float) -> pd.DataFrame:
    valid = grp[grp[metric] < _FAIL_LCO_THRESHOLD]
    if metric in _METRICS_WITH_OUTLIER_CAP:
        # LCOW-shaped guard against runaway values near the FAIL_LCO
        # sentinel; not meaningful for signed metrics like NPV, so it's
        # skipped entirely for any other metric (see _METRICS_WITH_OUTLIER_CAP).
        cap = max(baseline_metric * 50.0, 1e4)
        valid = valid[valid[metric] <= cap]
    return valid.sort_values("param_value")


def _build_table(sweep: pd.DataFrame, metric: str, baseline_metric: float) -> pd.DataFrame:
    rows = []
    for key, grp in sweep.groupby("sweep_param"):
        grp = _valid_sweep_points(grp, metric, baseline_metric)
        if len(grp) < 2:
            continue

        param_vals = grp["param_value"].astype(float)
        metric_vals = grp[metric].astype(float)
        param_lo = float(param_vals.iloc[0])
        param_hi = float(param_vals.iloc[-1])
        metric_lo = float(metric_vals.iloc[0])
        metric_hi = float(metric_vals.iloc[-1])

        bl_idx = (param_vals - param_vals.median()).abs().idxmin()
        param_base = float(param_vals.loc[bl_idx])
        metric_base = float(metric_vals.loc[bl_idx])
        # Only fall back to the overall baseline when this metric is
        # (near-)zero, which would blow up the elasticity's denominator.
        # For LCOW (always >= 0) this reduces to the old "<= 0" check; for
        # signed metrics like NPV a legitimately negative metric_base must
        # NOT be overwritten.
        if not math.isfinite(metric_base) or abs(metric_base) < 1e-9:
            metric_base = baseline_metric

        dec_sens = _elasticity(param_lo, metric_lo, param_base, metric_base)
        inc_sens = _elasticity(param_hi, metric_hi, param_base, metric_base)
        if not math.isfinite(dec_sens) or not math.isfinite(inc_sens):
            continue

        rows.append(
            {
                "sweep_param": key,
                "param_label": grp["param_label"].iloc[0],
                "baseline_metric": metric_base,
                "decrease_sensitivity": dec_sens,
                "increase_sensitivity": inc_sens,
                "total_span": abs(dec_sens) + abs(inc_sens),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table[table["total_span"] > 1e-6].sort_values("total_span", ascending=True)


def _draw_sensitivity_bar(ax, y: float, sens: float) -> None:
    if abs(sens) < 1e-15:
        return
    hatch = "///" if sens < 0 else None
    if sens <= 0:
        ax.barh(
            y,
            -sens,
            left=sens,
            height=0.65,
            color=_BAR_COLOR,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
        )
    else:
        ax.barh(
            y,
            sens,
            left=0,
            height=0.65,
            color=_BAR_COLOR,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
        )


def _plot_tornado(table: pd.DataFrame, out_png: Path, metric_label: str) -> None:
    if table.empty:
        print(
            f"No valid sensitivity rows for {metric_label!r} "
            "(e.g. every sweep point may be non-finite, such as payback=+inf "
            "for a device that never breaks even) -- skipping plot."
        )
        return
    labels = table["param_label"].tolist()
    dec = table["decrease_sensitivity"].values
    inc = table["increase_sensitivity"].values
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for yi, d, i in zip(y, dec, inc):
        _draw_sensitivity_bar(ax, yi, d)
        _draw_sensitivity_bar(ax, yi, i)

    ax.axvline(0, color="black", linewidth=1.0, zorder=3)
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"% change in {metric_label} per % change in parameter")
    ax.set_title("Tornado sensitivity")

    from matplotlib.patches import Patch

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    ap.add_argument("--table-csv", type=Path, default=None)
    ap.add_argument("--metric", default="lcow_usd_per_m3")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"Missing {args.input}; run parameter_sweep.py first.")

    sweep, baseline_metric = _load_sweep(args.input, args.metric)
    table = _build_table(sweep, args.metric, baseline_metric)
    table_csv = args.table_csv or args.output.with_suffix(".table.csv")
    table.to_csv(table_csv, index=False)
    metric_label = _METRIC_LABELS.get(args.metric, args.metric)
    _plot_tornado(table, args.output, metric_label)
    print(f"Wrote {table_csv}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
