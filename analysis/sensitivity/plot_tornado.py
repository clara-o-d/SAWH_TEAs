#!/usr/bin/env python3
"""Tornado sensitivity plot from any sweep CSV, for any swept parameters and any metric.

Handles both CSV shapes the sweeps produce, autodetected:

* **long**  -- one row per perturbation, with ``sweep_param`` / ``param_value`` columns
  (what an one-at-a-time sweep writes). Sensitivity is the endpoint elasticity about the
  baseline row.
* **wide**  -- one row per full configuration, every parameter its own column (what a
  grid / full-factorial / BayesOpt sweep writes). Sensitivity defaults to true OAT pairs
  (rows agreeing on every other parameter); ``--method regression`` instead fits a
  point elasticity at the sample means, for observational data that was never a
  designed sweep (e.g. real weather covarying across sites).

Examples::

  python plot_tornado.py --csv parameter_sweep.csv --metric lcow_usd_per_m3
  python plot_tornado.py --csv full_sweep.csv --metric mean_eta_thermal --params hydrogel_thickness_mm fin_area_ratio
  python plot_tornado.py --csv full_sweep.csv --metric lcow_usd_per_m3 --method regression
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_BAR_COLOR = "#20A387"
_SENTINEL = 1e20
# Columns that are results, never swept inputs -- excluded from parameter autodetection.
_METRIC_COLUMNS = frozenset({
    "daily_yield_kg_m2", "mean_yield_kg_m2", "yield_kg_m2", "thermal_efficiency",
    "mean_eta_thermal", "eta_thermal", "lcow_usd_per_m3", "capex_usd_per_m3",
    "opex_usd_per_m3", "npv_usd_per_m2", "payback_years_simple", "payback_years_discounted",
    "n_cycles_per_day", "specific_energy_wh_kwh_per_l", "specific_energy_parasitic_kwh_per_l",
    "specific_energy_total_kwh_per_l", "lat", "lon", "latitude", "longitude",
    "sweep_param", "param_value", "param_label", "swept_param",
})

_LABELS: dict[str, str] = {
    "lcow_usd_per_m3": "LCOW (USD/m³)", "npv_usd_per_m2": "NPV (USD/m²)",
    "payback_years_simple": "Simple payback (years)", "payback_years_discounted": "Discounted payback (years)",
    "daily_yield_kg_m2": "Daily yield (kg/m²)", "mean_yield_kg_m2": "Mean yield (kg/m²)",
    "thermal_efficiency": "Thermal efficiency", "mean_eta_thermal": "Thermal efficiency",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--params", nargs="+", default=None,
                    help="Parameters to rank (default: every non-metric column)")
    ap.add_argument("--method", choices=("auto", "oat", "elasticity", "regression"), default="auto")
    ap.add_argument("--top", type=int, default=None, help="Show only the N most sensitive parameters")
    ap.add_argument("--title", default="Parameter sensitivity")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--table-csv", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        sys.exit(f"Metric {args.metric!r} not in {args.csv}. Available: {', '.join(df.columns)}")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=[args.metric])
    df = df[df[args.metric].abs() < _SENTINEL]
    if df.empty:
        sys.exit(f"No finite {args.metric!r} values in {args.csv}.")

    is_long = {"sweep_param", "param_value"}.issubset(df.columns)
    method = args.method
    if method == "auto":
        method = "elasticity" if is_long else "oat"
    if method == "elasticity" and not is_long:
        sys.exit("--method elasticity needs long-format sweep_param/param_value columns.")
    if method in ("oat", "regression") and is_long:
        sys.exit(f"--method {method} needs wide-format data (one row per configuration).")

    if method == "elasticity":
        table = _elasticity_table(df, args.metric)
    else:
        params = args.params or [c for c in df.columns if c not in _METRIC_COLUMNS]
        params = [p for p in params if df[p].nunique() > 1]
        if not params:
            sys.exit("No varying parameter columns found; pass --params explicitly.")
        print(f"{method} sensitivity of {args.metric!r} to {len(params)} parameter(s): {', '.join(params)}")
        table = _oat_table(df, args.metric, params) if method == "oat" \
            else _regression_table(df, args.metric, params)

    if table.empty:
        sys.exit("No parameter produced a finite sensitivity.")
    table = table.sort_values("max_abs_effect", ascending=False)
    if args.top:
        table = table.head(args.top)

    out_png = args.output or args.csv.parent / f"tornado_{args.metric}.png"
    table_csv = args.table_csv or out_png.with_suffix(".table.csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_csv, index=False)
    _plot(table, args.metric, args.title, out_png)
    print(f"Wrote {table_csv}")
    print(f"Wrote {out_png}")
    return 0


def _elasticity_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Long format: endpoint elasticity of each sweep_param group about its median point."""
    rows = []
    for key, grp in df[df["sweep_param"] != "baseline"].groupby("sweep_param"):
        grp = grp.sort_values("param_value")
        if len(grp) < 2:
            continue
        pv = grp["param_value"].astype(float)
        mv = grp[metric].astype(float)
        base_i = (pv - pv.median()).abs().idxmin()
        p0, m0 = float(pv.loc[base_i]), float(mv.loc[base_i])
        if not math.isfinite(m0) or abs(m0) < 1e-9:
            continue
        dec = _elasticity(float(pv.iloc[0]), float(mv.iloc[0]), p0, m0)
        inc = _elasticity(float(pv.iloc[-1]), float(mv.iloc[-1]), p0, m0)
        if not (math.isfinite(dec) and math.isfinite(inc)):
            continue
        label = grp["param_label"].iloc[0] if "param_label" in grp.columns else str(key)
        rows.append({"variable": str(key), "label": label, "increase": inc, "decrease": dec,
                     "max_abs_effect": max(abs(inc), abs(dec)), "n_points": len(grp)})
    return pd.DataFrame(rows)


def _oat_table(df: pd.DataFrame, metric: str, params: list[str]) -> pd.DataFrame:
    """Wide format: average elasticity over row pairs agreeing on every other parameter."""
    rows = []
    for var in params:
        others = [p for p in params if p != var]
        buckets = [g for _, g in df.groupby(others, sort=False)] if others else [df]
        inc: list[float] = []
        dec: list[float] = []
        for g in buckets:
            pts = sorted(zip(g[var].astype(float), g[metric].astype(float)))
            for a, (x_lo, y_lo) in enumerate(pts):
                for x_hi, y_hi in pts[a + 1:]:
                    inc.extend(_pct_ratio(x_lo, y_lo, x_hi, y_hi))
                    dec.extend(_pct_ratio(x_hi, y_hi, x_lo, y_lo))
        if not inc and not dec:
            print(f"  no OAT pairs for {var}")
            continue
        i_avg = float(np.mean(inc)) if inc else 0.0
        d_avg = float(np.mean(dec)) if dec else 0.0
        rows.append({"variable": var, "label": var, "increase": i_avg, "decrease": d_avg,
                     "max_abs_effect": max(abs(i_avg), abs(d_avg)), "n_points": len(inc) + len(dec)})
        print(f"  {var}: increase={i_avg:+.3f} decrease={d_avg:+.3f} (n={len(inc) + len(dec)})")
    return pd.DataFrame(rows)


def _regression_table(df: pd.DataFrame, metric: str, params: list[str]) -> pd.DataFrame:
    """Point elasticity at the sample means from a joint linear fit, controlling for the rest."""
    x = df[params].astype(float).to_numpy()
    y = df[metric].astype(float).to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_mean = float(np.mean(y))
    rows = []
    for k, var in enumerate(params):
        x_mean = float(np.mean(x[:, k]))
        if abs(y_mean) < 1e-12:
            continue
        e = float(coef[k + 1]) * x_mean / y_mean
        rows.append({"variable": var, "label": var, "increase": e, "decrease": -e,
                     "max_abs_effect": abs(e), "n_points": len(df)})
        print(f"  {var}: elasticity={e:+.3f} (n={len(df)})")
    return pd.DataFrame(rows)


def _elasticity(p: float, m: float, p0: float, m0: float) -> float:
    p_frac = (p - p0) / p0 if p0 else float("nan")
    m_frac = (m - m0) / m0
    return m_frac / p_frac if abs(p_frac) > 1e-15 else float("nan")


def _pct_ratio(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """[% change in y per % change in x], or [] if x barely moved / the ratio blows up."""
    pct_x = (x2 - x1) / x1 * 100 if abs(x1) > 1e-10 else abs(x2 - x1)
    pct_y = (y2 - y1) / y1 * 100 if abs(y1) > 1e-10 else y2 - y1
    if abs(pct_x) <= 0.01:
        return []
    s = pct_y / pct_x
    return [s] if abs(s) < 1000 else []


def _plot(table: pd.DataFrame, metric: str, title: str, out_png: Path) -> None:
    t = table.sort_values("max_abs_effect", ascending=True)
    n = len(t)
    y = np.arange(n)
    # Shrink labels once there are enough bars that they would collide.
    label_fs = 14 if n <= 12 else max(8.0, 14 - 0.3 * (n - 12))

    fig, ax = plt.subplots(figsize=(8, max(5.0, 0.5 * n + 1.5)))
    ax.set_frame_on(False)
    hatch = ["" if v >= 0 else "///" for v in t["increase"]]
    for values in (t["increase"], -t["decrease"]):
        ax.barh(y, values, height=0.35, color=_BAR_COLOR, alpha=0.7, hatch=hatch)

    ax.set_yticks(y)
    ax.set_yticklabels(t["label"], fontsize=label_fs)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel(f"% change in {_LABELS.get(metric, metric)}\nper % change in parameter", fontsize=13)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(axis="x", labelsize=12)
    ax.legend(handles=[Patch(facecolor=_BAR_COLOR, alpha=0.7, label="Positive"),
                       Patch(facecolor=_BAR_COLOR, alpha=0.7, hatch="///", label="Negative")],
              loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
