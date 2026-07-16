#!/usr/bin/env python3
"""Run a sequential day-by-day SAWH simulation over a full calendar year."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from solar_lumped.physics.adsorbent import DEFAULT_MOF_NAME
from solar_lumped.physics.salt_properties import get_salt
from solar_lumped.simulation.annual_yield import (
    simulate_annual_year,
    write_daily_summary_csv,
)
from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.weather.client import WeatherClient
from solar_lumped.weather.profiles import real_weather_days_from_df


def build_device_config(
    *,
    sorbent: str = "hydrogel",
    mof: str = DEFAULT_MOF_NAME,
    salt: str = "LiCl",
    salt_loading: float = 4.0,
    hydrogel_thickness_mm: float = 4.0,
    vapor_gap_mm: float = 40.0,
    insulation_gap_mm: float = 5.0,
    tilt_deg: float = 35.0,
    fin_area_ratio: float = 7.1,
) -> DeviceConfig:
    get_salt(salt)
    return DeviceConfig(
        sorbent=sorbent,  # type: ignore[arg-type]
        mof_name=mof,
        salt_name=salt,
        salt_to_polymer_ratio=salt_loading,
        hydrogel_thickness_m=hydrogel_thickness_mm * 1e-3,
        vapor_gap_m=vapor_gap_mm * 1e-3,
        insulation_gap_m=insulation_gap_mm * 1e-3,
        tilt_deg=tilt_deg,
        fin_area_ratio=fin_area_ratio,
    )


def default_output_path(lat: float, lon: float, year: int) -> Path:
    tag = f"annual_{lat:+.4f}_{lon:+.4f}_{year}".replace("+", "")
    return _REPO / "outputs" / "annual_sim" / f"{tag}.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--stride", type=int, default=1, help="Simulate every Nth day (default: 1)")
    p.add_argument(
        "--warmup-cycles",
        type=int,
        default=2,
        help="Warmup cycles on Jan 1 weather before recording (default: 2)",
    )
    p.add_argument(
        "--save-daily-timeseries",
        action="store_true",
        help="Write per-day diagnostics and water-inventory CSVs",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--sorbent", choices=("hydrogel", "mof"), default="hydrogel")
    p.add_argument("--mof", default=DEFAULT_MOF_NAME)
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
    output = args.output or default_output_path(args.lat, args.lon, args.year)
    timeseries_dir = output.parent / output.stem / "timeseries"

    config = build_device_config(
        sorbent=args.sorbent,
        mof=args.mof,
        salt=args.salt,
        salt_loading=args.salt_loading,
        hydrogel_thickness_mm=args.hydrogel_thickness_mm,
        vapor_gap_mm=args.vapor_gap_mm,
        insulation_gap_mm=args.insulation_gap_mm,
        tilt_deg=args.tilt_deg,
        fin_area_ratio=args.fin_area_ratio,
    )

    print(
        f"Fetching {args.year} weather for ({args.lat:+.4f}, {args.lon:+.4f})…",
        flush=True,
    )
    client = WeatherClient(cache_dir=args.cache_dir)
    start = f"{args.year}-01-01"
    end = f"{args.year}-12-31"
    try:
        _, df = client.get_historical_forecast_site_weather(args.lat, args.lon, start, end)
    except Exception:
        df = client.get_historical(args.lat, args.lon, start, end)

    day_items = real_weather_days_from_df(df, stride=args.stride)
    if not day_items:
        print("No valid weather days found.", file=sys.stderr)
        return 1

    print(
        f"Simulating {len(day_items)} day(s) "
        f"(stride={args.stride}, warmup_cycles={args.warmup_cycles})…",
        flush=True,
    )
    t0 = time.perf_counter()
    last_logged = 0

    def progress(done: int, total: int, day_key: object) -> None:
        nonlocal last_logged
        if done == total or done - last_logged >= 30:
            print(f"  [{done}/{total}] {day_key}", flush=True)
            last_logged = done

    records = simulate_annual_year(
        day_items,
        config,
        warmup_cycles=args.warmup_cycles,
        save_daily_timeseries=args.save_daily_timeseries,
        timeseries_dir=timeseries_dir if args.save_daily_timeseries else None,
        progress_callback=progress,
    )
    write_daily_summary_csv(output, records)

    elapsed = time.perf_counter() - t0
    mean_yield = sum(r.daily_yield_l_m2 for r in records) / len(records)
    print(f"Wrote {len(records)} rows to {output}", flush=True)
    print(f"Mean daily yield: {mean_yield:.4f} L/m²", flush=True)
    print(f"Elapsed: {elapsed:.1f} s", flush=True)
    if args.save_daily_timeseries:
        print(f"Timeseries: {timeseries_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
