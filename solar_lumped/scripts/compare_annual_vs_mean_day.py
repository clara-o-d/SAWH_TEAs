#!/usr/bin/env python3
"""Compare four ways of estimating daily water yield for a site:

  A. Simulate every day of the year sequentially (gel state carried over from
     one day to the next) and average the daily yields.
  B. Build a single representative mean/diurnal day (averaged across the year)
     and cycle it to a steady periodic state, as the global-map scripts
     (lcow_full_global_map.py, lcow_random_global_map.py) do for speed.
  C. Build one representative mean/diurnal day per calendar month, cycle each
     to its own steady periodic state, and average the 12 results (weighted
     by days present). ~12x the cost of B, far cheaper than A.
  D. Same idea at ISO-week granularity (~52 periods instead of 12). ~4x the
     cost of C, still far cheaper than A.

B/C/D each need the steady periodic (c_w, H) state for their mean day; that's
found via restarted vector Aitken extrapolation (find_cyclic_state) rather
than brute-force warmup cycling, since plain fixed-point iteration can need
100+ cycles to converge at strongly seasonal sites (Aitken typically needs
~3-6 rounds regardless). Approach A's Jan-1 warmup is a separate, unrelated
use of --warmup-cycles (warming up a real sequential year, not a periodic
fixed point).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_SRC = _REPO / "src"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_annual_simulation import build_device_config  # noqa: E402

from solar_lumped.simulation.annual_yield import simulate_annual_year  # noqa: E402
from solar_lumped.simulation.ode_system import find_cyclic_state, run_daily_cycle  # noqa: E402
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
    p.add_argument(
        "--warmup-cycles",
        type=int,
        default=2,
        help="Warmup cycles for Approach A's Jan-1 sequential warmup only "
        "(B/C/D find their steady state via Aitken extrapolation instead)",
    )
    p.add_argument("--salt", type=str, default="LiCl")
    p.add_argument("--salt-loading", type=float, default=4.0)
    p.add_argument("--hydrogel-thickness-mm", type=float, default=4.0)
    p.add_argument("--vapor-gap-mm", type=float, default=40.0)
    p.add_argument("--insulation-gap-mm", type=float, default=5.0)
    p.add_argument("--tilt-deg", type=float, default=35.0)
    p.add_argument("--fin-area-ratio", type=float, default=7.1)
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Append a summary row here (creates the file with a header if missing).",
    )
    p.add_argument("--label", type=str, default=None, help="Free-text tag for the output row (e.g. site name)")
    p.add_argument(
        "--skip-annual",
        action="store_true",
        help="Skip the expensive Approach A (365-day) pass; requires --annual-mean-kg-m2.",
    )
    p.add_argument(
        "--annual-mean-kg-m2",
        type=float,
        default=None,
        help="Known Approach A result to reuse instead of recomputing it (used with --skip-annual).",
    )
    return p.parse_args(argv)


_CSV_COLUMNS: tuple[str, ...] = (
    "label",
    "lat",
    "lon",
    "year",
    "salt",
    "warmup_cycles",
    "n_days_simulated",
    "annual_mean_yield_kg_m2",
    "annual_min_yield_kg_m2",
    "annual_max_yield_kg_m2",
    "mean_day_yield_kg_m2",
    "mean_day_eta_thermal",
    "diff_kg_m2",
    "diff_pct",
    "monthly_mean_yield_kg_m2",
    "monthly_diff_kg_m2",
    "monthly_diff_pct",
    "weekly_mean_yield_kg_m2",
    "weekly_diff_kg_m2",
    "weekly_diff_pct",
)


def _append_summary_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _steady_state_yield(profile, config) -> tuple[float, float]:
    """Find the steady periodic state for *profile* (Aitken-accelerated) and
    return the (yield_kg_m2, eta_thermal) of one cycle from that state.
    """
    cw, h = find_cyclic_state(profile, config)
    yield_kg, eta, _, _ = run_daily_cycle(profile, config, c_w_initial=cw, h_initial=h)
    return yield_kg, eta


def _binned_mean_day_average(df, config, *, period: str) -> float:
    """Build one representative mean day per bin (month or ISO week), find
    its steady periodic state, and return the days-present-weighted average
    yield across bins.
    """
    if period == "month":
        bin_of = df.index.month
    elif period == "week":
        bin_of = df.index.isocalendar().week.to_numpy()
    else:
        raise ValueError(f"Unknown period: {period!r}")

    yields: list[float] = []
    weights: list[int] = []
    for b in sorted(set(bin_of)):
        bin_df = df[bin_of == b]
        if bin_df.empty:
            continue
        ref_day = bin_df.index[len(bin_df) // 2].date()
        bin_mean_day_df = representative_mean_day_df(bin_df, reference_day=ref_day)
        bin_profile = profile_from_day_df(bin_mean_day_df)
        bin_yield_kg, bin_eta = _steady_state_yield(bin_profile, config)
        n_days = len(pd.unique(bin_df.index.date))
        yields.append(bin_yield_kg)
        weights.append(n_days)
        print(f"  {period} {b:>2}: yield={bin_yield_kg:.6f} kg/m²  eta={bin_eta:.4f}  ({n_days} day(s))", flush=True)
    return sum(y * w for y, w in zip(yields, weights)) / sum(weights)


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

    n_days_simulated: int | None = None
    annual_min = annual_max = None
    if args.skip_annual:
        if args.annual_mean_kg_m2 is None:
            print("--skip-annual requires --annual-mean-kg-m2.", file=sys.stderr)
            return 1
        annual_mean = args.annual_mean_kg_m2
        print(f"--- Approach A: skipped, reusing known result {annual_mean:.6f} kg/m² ---", flush=True)
    else:
        print("--- Approach A: every day of the year, sequential state, averaged ---", flush=True)
        day_items = real_weather_days_from_df(df, stride=args.stride)
        if not day_items:
            print("No valid weather days found.", file=sys.stderr)
            return 1
        t0 = time.perf_counter()
        records = simulate_annual_year(day_items, config, warmup_cycles=args.warmup_cycles)
        annual_yields = [r.daily_yield_kg_m2 for r in records]
        annual_mean = sum(annual_yields) / len(annual_yields)
        n_days_simulated = len(records)
        annual_min, annual_max = min(annual_yields), max(annual_yields)
        print(f"  {n_days_simulated} day(s) simulated in {time.perf_counter() - t0:.1f}s", flush=True)
        print(
            f"  Mean daily yield (annual average): {annual_mean:.6f} kg/m²  "
            f"[min={annual_min:.6f}  max={annual_max:.6f}]",
            flush=True,
        )

    print("--- Approach B: single mean/representative day, Aitken-accelerated steady state ---", flush=True)
    t0 = time.perf_counter()
    mean_day_df = representative_mean_day_df(df, reference_day=date(args.year, 6, 15))
    mean_profile = profile_from_day_df(mean_day_df)
    mean_yield_kg, eta = _steady_state_yield(mean_profile, config)
    print(f"  Simulated in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"  Mean-day yield: {mean_yield_kg:.6f} kg/m²  (eta_thermal={eta:.4f})", flush=True)

    print("--- Approach C: mean day per calendar month, Aitken-accelerated steady state, day-weighted ---", flush=True)
    t0 = time.perf_counter()
    monthly_mean = _binned_mean_day_average(df, config, period="month")
    print(f"  Simulated 12 month mean-days in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"  Monthly mean-day average (day-weighted): {monthly_mean:.6f} kg/m²", flush=True)

    print("--- Approach D: mean day per ISO week, Aitken-accelerated steady state, day-weighted ---", flush=True)
    t0 = time.perf_counter()
    weekly_mean = _binned_mean_day_average(df, config, period="week")
    n_weeks = len(sorted(set(df.index.isocalendar().week.to_numpy())))
    print(f"  Simulated {n_weeks} weekly mean-days in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"  Weekly mean-day average (day-weighted): {weekly_mean:.6f} kg/m²", flush=True)

    diff = mean_yield_kg - annual_mean
    pct = (diff / annual_mean * 100.0) if annual_mean != 0 else float("nan")
    monthly_diff = monthly_mean - annual_mean
    monthly_pct = (monthly_diff / annual_mean * 100.0) if annual_mean != 0 else float("nan")
    weekly_diff = weekly_mean - annual_mean
    weekly_pct = (weekly_diff / annual_mean * 100.0) if annual_mean != 0 else float("nan")
    print("--- Comparison ---", flush=True)
    print(f"  Annual day-by-day average   : {annual_mean:.6f} kg/m²", flush=True)
    print(f"  Single mean-day (cyclic)    : {mean_yield_kg:.6f} kg/m²  ({diff:+.6f} kg/m², {pct:+.2f}%)", flush=True)
    print(f"  Monthly mean-days (cyclic)  : {monthly_mean:.6f} kg/m²  ({monthly_diff:+.6f} kg/m², {monthly_pct:+.2f}%)", flush=True)
    print(f"  Weekly mean-days (cyclic)   : {weekly_mean:.6f} kg/m²  ({weekly_diff:+.6f} kg/m², {weekly_pct:+.2f}%)", flush=True)

    if args.output_csv is not None:
        _append_summary_row(
            args.output_csv,
            {
                "label": args.label or "",
                "lat": args.lat,
                "lon": args.lon,
                "year": args.year,
                "salt": args.salt,
                "warmup_cycles": args.warmup_cycles,
                "n_days_simulated": n_days_simulated if n_days_simulated is not None else "",
                "annual_mean_yield_kg_m2": f"{annual_mean:.6f}",
                "annual_min_yield_kg_m2": f"{annual_min:.6f}" if annual_min is not None else "",
                "annual_max_yield_kg_m2": f"{annual_max:.6f}" if annual_max is not None else "",
                "mean_day_yield_kg_m2": f"{mean_yield_kg:.6f}",
                "mean_day_eta_thermal": f"{eta:.6f}",
                "diff_kg_m2": f"{diff:.6f}",
                "diff_pct": f"{pct:.2f}",
                "monthly_mean_yield_kg_m2": f"{monthly_mean:.6f}",
                "monthly_diff_kg_m2": f"{monthly_diff:.6f}",
                "monthly_diff_pct": f"{monthly_pct:.2f}",
                "weekly_mean_yield_kg_m2": f"{weekly_mean:.6f}",
                "weekly_diff_kg_m2": f"{weekly_diff:.6f}",
                "weekly_diff_pct": f"{weekly_pct:.2f}",
            },
        )
        print(f"Appended summary row to {args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
