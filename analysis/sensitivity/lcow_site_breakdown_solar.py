#!/usr/bin/env python3
"""Run ``run_solar_sim`` for one site and plot LCOW cost breakdown (black_box_tea style).

Example::

  python scripts/lcow_site_breakdown.py --lat 25 --lon 65 --year 2024
  python scripts/lcow_site_breakdown.py --weather-mode cambridge-replay
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent / "solar_lumped"
_SCRIPTS = _REPO / "scripts"
_SRC = _REPO / "src"
_TEA_ROOT = _REPO.parent / "analysis" / "black_box_tea"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_TEA_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEA_ROOT))

from run_solar_sim import (  # noqa: E402
    SolarSimResult,
    _lcow_kwargs,
    output_tag,
    register_solar_sim_arguments,
    resolve_solar_sim_arguments,
    run_solar_simulation,
)
from solar_lumped.economics import LcowCostBreakdown  # noqa: E402
from solar_lumped.economics import KG_WATER_PER_M3  # noqa: E402
from solar_lumped.economics import lcow_cost_breakdown_from_daily_yield  # noqa: E402
from solar_lumped.economics import lcow_from_daily_yield  # noqa: E402
from tea_workbook_plots import LcowBreakdown, plot_lcow_breakdown_stacked  # noqa: E402

_DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs" / "lcow" / "site"


def _net_annual_water_m3(result: SolarSimResult, *, cycles_per_day: float = 1.0) -> float:
    gross = cycles_per_day * 365.0 * result.daily_yield_kg_per_m2 / KG_WATER_PER_M3
    return result.econ.utilization_factor * gross


def _breakdown_title(result: SolarSimResult) -> str:
    parts = [f"Solar SAWH — {result.weather_mode}"]
    if result.lat is not None and result.lon is not None:
        parts.append(f"({result.lat:.2f}°, {result.lon:.2f}°)")
    if result.weather_mode == "real":
        parts.append(f"year {result.year}")
    sorbent = result.config.sorbent
    if sorbent == "mof":
        parts.append(result.config.mof_name)
    else:
        parts.append(result.config.salt_name)
    if result.simplified:
        parts.append("(simplified TEA)")
    config = result.config
    design = (
        f"hydrogel thickness {config.hydrogel_thickness_m * 1e3:.2f} mm, "
        f"condenser (fin) area ratio {config.fin_area_ratio:.2f}, "
        f"vapor gap {config.vapor_gap_m * 1e3:.1f} mm"
    )
    return " ".join(parts) + "\n" + design


def _apply_yield_override(result: SolarSimResult, daily_yield_kg_m2: float) -> SolarSimResult:
    """Recompute LCOW/breakdown for an externally supplied yield (e.g. a sweep row)."""
    config = result.config
    lcow_kw = _lcow_kwargs(config)
    common = dict(
        salt_name=config.salt_name,
        salt_to_polymer_ratio=config.salt_to_polymer_ratio,
        hydrogel_thickness_m=config.hydrogel_thickness_m,
        econ=result.econ,
        simplified=result.simplified,
        **lcow_kw,
    )
    lcow = lcow_from_daily_yield(daily_yield_kg_m2, **common)
    breakdown = lcow_cost_breakdown_from_daily_yield(daily_yield_kg_m2, **common)
    return replace(
        result,
        daily_yield_kg_per_m2=daily_yield_kg_m2,
        lcow_usd_per_m3=lcow,
        breakdown=breakdown,
    )


def _to_workbook_breakdown(
    breakdown: LcowCostBreakdown,
    *,
    lcow_usd_per_m3: float,
    title: str,
    net_annual_water_m3: float,
) -> LcowBreakdown:
    segments = tuple(
        (label, usd_per_m3 * net_annual_water_m3, usd_per_m3)
        for label, usd_per_m3 in breakdown.items
    )
    return LcowBreakdown(
        title=title,
        lcow_usd_per_m3=lcow_usd_per_m3,
        segments=segments,
    )


def _write_table_csv(path: Path, breakdown: LcowBreakdown) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "annual_usd_per_m2", "lcow_usd_per_m3"])
        for label, annual, usd_per_m3 in breakdown.segments:
            w.writerow([label, f"{annual:.6f}", f"{usd_per_m3:.6f}"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Single-site solar LCOW simulation + stacked cost breakdown plot",
    )
    register_solar_sim_arguments(ap)
    ap.set_defaults(weather_mode="real")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Stacked LCOW breakdown PNG (default: outputs/lcow/site/lcow_breakdown_<tag>.png)",
    )
    ap.add_argument(
        "--table-csv",
        type=Path,
        default=None,
        help="Breakdown table CSV (default: outputs/lcow/site/lcow_breakdown_<tag>.csv)",
    )
    ap.add_argument(
        "--daily-yield-kg-m2",
        type=float,
        default=None,
        help="Override the simulated daily yield (kg/m²/d) — e.g. a sweep's mean_yield_kg_m2 — "
        "for the LCOW breakdown instead of re-simulating it",
    )
    args = ap.parse_args()

    resolve_solar_sim_arguments(args, ap)
    result = run_solar_simulation(args)
    if args.daily_yield_kg_m2 is not None:
        result = _apply_yield_override(result, args.daily_yield_kg_m2)

    if result.breakdown is None:
        sys.exit("Simulation produced no LCOW breakdown (zero or invalid yield).")

    tag = output_tag(args, result.config)
    if result.simplified:
        tag += "_simplified"
    out_png = args.output or (_DEFAULT_OUT_DIR / f"lcow_breakdown_{tag}.png")
    out_csv = args.table_csv or (_DEFAULT_OUT_DIR / f"lcow_breakdown_{tag}.csv")

    net_water = _net_annual_water_m3(result)
    wb_breakdown = _to_workbook_breakdown(
        result.breakdown,
        lcow_usd_per_m3=result.lcow_usd_per_m3,
        title=_breakdown_title(result),
        net_annual_water_m3=net_water,
    )
    png_path = plot_lcow_breakdown_stacked(wb_breakdown, output_path=out_png)
    _write_table_csv(out_csv, wb_breakdown)

    print(f"Weather mode: {result.weather_mode}")
    if result.lat is not None and result.lon is not None:
        print(f"Site: lat={result.lat:.4f} lon={result.lon:.4f}")
    print(f"Daily yield: {result.daily_yield_kg_per_m2:.4f} kg/m²/d")
    print(f"Net annual water: {net_water:.4f} m³/m²/yr")
    print(f"LCOW = ${result.lcow_usd_per_m3:.2f}/m³ ({len(wb_breakdown.segments)} segments)")
    print(f"Wrote {png_path}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
