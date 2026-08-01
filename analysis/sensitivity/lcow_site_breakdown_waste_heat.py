#!/usr/bin/env python3
"""Simulate one waste-heat operating point and plot its LCOW cost breakdown.

Overrides are the same ``--key=value`` sweep parameters ``parameter_sweep.py`` accepts
(see ``make_sweep_params``); anything not given stays at the baseline.

Example::

  python lcow_site_breakdown_waste_heat.py
  python lcow_site_breakdown_waste_heat.py --set t_wh_in_c=70 --set hydrogel_thickness_mm=3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent / "waste-heat"
_TEA_ROOT = Path(__file__).resolve().parent.parent / "black_box_tea"
for _p in (_REPO / "scripts", _REPO / "src", _TEA_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from parameter_sweep import BASELINE_ECON, _apply_overrides, make_sweep_params  # noqa: E402
from tea_workbook_plots import LcowBreakdown, plot_lcow_breakdown_stacked  # noqa: E402
from waste_heat.economics import KG_WATER_PER_M3  # noqa: E402
from waste_heat.economics import lcow_cost_breakdown_from_daily_yield  # noqa: E402
from waste_heat.simulation import simulate_daily  # noqa: E402

_DEFAULT_OUT_DIR = _REPO / "outputs" / "lcow" / "site"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a sweep parameter (repeatable)")
    ap.add_argument("--tag", default="baseline", help="Output filename tag")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--table-csv", type=Path, default=None)
    args = ap.parse_args()

    valid = {p.key for p in make_sweep_params()}
    overrides: dict[str, float] = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in valid:
            raise SystemExit(f"Unknown parameter {key!r}. Available: {', '.join(sorted(valid))}")
        overrides[key] = float(value)

    cfg, profile, econ = _apply_overrides(overrides)
    result = simulate_daily(profile, cfg)
    cycles_per_day = float(result.n_cycles_per_day)
    breakdown = lcow_cost_breakdown_from_daily_yield(
        result.mean_daily_yield_kg_m2,
        salt_name=cfg.salt_name,
        salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
        hydrogel_thickness_m=cfg.hydrogel_thickness_m,
        econ=econ,
        cycles_per_day=cycles_per_day,
    )
    if breakdown is None:
        sys.exit("Simulation produced no LCOW breakdown (zero or invalid yield).")

    # Annual water actually sold, used to turn each USD/m³ term back into USD/m²/yr.
    net_water_m3 = econ.utilization_factor * cycles_per_day * 365.0 * (
        result.mean_daily_yield_kg_m2 / KG_WATER_PER_M3
    )
    title = f"Waste-heat SAWH — {args.tag}" + (f" ({', '.join(args.set)})" if args.set else "")
    wb = LcowBreakdown(
        title=title,
        lcow_usd_per_m3=breakdown.total_usd_per_m3,
        segments=tuple((label, v * net_water_m3, v) for label, v in breakdown.items),
    )

    out_png = args.output or _DEFAULT_OUT_DIR / f"lcow_breakdown_{args.tag}.png"
    out_csv = args.table_csv or _DEFAULT_OUT_DIR / f"lcow_breakdown_{args.tag}.csv"
    png_path = plot_lcow_breakdown_stacked(wb, output_path=out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "annual_usd_per_m2", "lcow_usd_per_m3"])
        w.writerows((label, f"{annual:.6f}", f"{per_m3:.6f}") for label, annual, per_m3 in wb.segments)

    print(f"Daily yield: {result.mean_daily_yield_kg_m2:.4f} kg/m²/d over {cycles_per_day:.2f} cycles/day")
    print(f"Net annual water: {net_water_m3:.4f} m³/m²/yr")
    print(f"LCOW = ${wb.lcow_usd_per_m3:.2f}/m³ ({len(wb.segments)} segments)")
    print(f"Wrote {png_path}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
