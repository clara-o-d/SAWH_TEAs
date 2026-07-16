#!/usr/bin/env python3
"""Compare two ways of estimating daily water yield for a site:

  A. Simulate every day of the year sequentially (gel state carried over from
     one day to the next) and average the daily yields.
  B. Build a single representative mean/diurnal day (averaged across the year)
     and cycle it to a steady periodic state, as the global-map scripts
     (lcow_full_global_map.py, lcow_random_global_map.py) do for speed.

Both approaches use the same fetched weather and the same warmup-cycle count,
so the difference isolates the error introduced by the mean-day shortcut.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_SRC = _REPO / "src"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_annual_simulation import build_device_config  # noqa: E402

from solar_lumped.simulation.annual_yield import simulate_annual_year  # noqa: E402
from solar_lumped.simulation.ode_system import run_daily_cycle  # noqa: E402
from solar_lumped.weather.client import WeatherClient  # noqa: E402
from solar_lumped.weather.climate import representative_mean_day_df  # noqa: E402
from solar_lumped.weather.profiles import (  # noqa: E402
    profile_from_day_df,
    real_weather_days_from_df,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, default=-50.0)
    p.add_argument("--lon", type=float, default=-75.0)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--stride", type=int, default=1, help="Simulate every Nth day for the annual pass (default: 1)")
    p.add_argument("--warmup-cycles", type=int, default=2, help="Warmup cycles, applied identically to both approaches")
    p.add_argument("--salt", type=str, default="LiCl")
    p.add_argument("--salt-loading", type=float, default=4.0)
    p.add_argument("--hydrogel-thickness-mm", type=float, default=4.0)
    p.add_argument("--vapor-gap-mm", type=float, default=40.0)
    p.add_argument("--insulation-gap-mm", type=float, default=5.0)
    p.add_argument("--tilt-deg", type=float, default=35.0)
    p.add_argument("--fin-area-ratio", type=float, default=7.1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_device_config(
        salt=args.salt,
        salt_loading=args.salt_loading,
        hydrogel_thickness_mm=args.hydrogel_thickness_mm,
        vapor_gap_mm=args.vapor_gap_mm,
        insulation_gap_mm=args.insulation_gap_mm,
        tilt_deg=args.tilt_deg,
        fin_area_ratio=args.fin_area_ratio,
    )

    print(
        f"Fetching {args.year} weather for ({args.lat:+.4f}, {args.lon:+.4f}) "
        f"[cache={args.cache_dir}]…",
        flush=True,
    )
    client = WeatherClient(cache_dir=args.cache_dir)
    start = f"{args.year}-01-01"
    end = f"{args.year}-12-31"
    try:
        _, df = client.get_historical_forecast_site_weather(args.lat, args.lon, start, end)
    except Exception:
        df = client.get_historical(args.lat, args.lon, start, end)

    print("--- Approach A: every day of the year, sequential state, averaged ---", flush=True)
    day_items = real_weather_days_from_df(df, stride=args.stride)
    if not day_items:
        print("No valid weather days found.", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    records = simulate_annual_year(day_items, config, warmup_cycles=args.warmup_cycles)
    annual_yields = [r.daily_yield_kg_m2 for r in records]
    annual_mean = sum(annual_yields) / len(annual_yields)
    print(f"  {len(records)} day(s) simulated in {time.perf_counter() - t0:.1f}s", flush=True)
    print(
        f"  Mean daily yield (annual average): {annual_mean:.6f} kg/m²  "
        f"[min={min(annual_yields):.6f}  max={max(annual_yields):.6f}]",
        flush=True,
    )

    print("--- Approach B: single mean/representative day, cycled to steady state ---", flush=True)
    t0 = time.perf_counter()
    mean_day_df = representative_mean_day_df(df, reference_day=date(args.year, 6, 15))
    mean_profile = profile_from_day_df(mean_day_df)
    mean_yield_kg, eta, _, _ = run_daily_cycle(
        mean_profile,
        config,
        cyclic_initial=True,
        cyclic_warmup_cycles=args.warmup_cycles,
    )
    print(f"  Simulated in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"  Mean-day yield: {mean_yield_kg:.6f} kg/m²  (eta_thermal={eta:.4f})", flush=True)

    diff = mean_yield_kg - annual_mean
    pct = (diff / annual_mean * 100.0) if annual_mean != 0 else float("nan")
    print("--- Comparison ---", flush=True)
    print(f"  Annual day-by-day average : {annual_mean:.6f} kg/m²", flush=True)
    print(f"  Single mean-day (cyclic)  : {mean_yield_kg:.6f} kg/m²", flush=True)
    print(f"  Difference                : {diff:+.6f} kg/m²  ({pct:+.2f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
