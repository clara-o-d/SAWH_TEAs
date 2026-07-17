#!/usr/bin/env python3
"""Prefetch/cache weather for every site in the land grid, in one process.

Building the land/country-exclusion geometry and importing the physics stack
costs ~1.4s and is identical for every site. Fetching one site per subprocess
(e.g. looping ``grid_param_sweep.py --combo-limit 0`` in a shell ``for``) pays
that cost once per site -- 1400+ times over. This pays it once, then fetches
every site's weather in the same process and connection pool.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_SRC = _REPO / "src"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lcow_full_global_map import grid_land_points  # noqa: E402

from solar_lumped.weather.client import WeatherClient  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", type=float, default=3.0)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--sleep", type=float, default=0.3, help="Seconds between requests")
    p.add_argument("--limit", type=int, default=None, help="Only the first N sites (for testing)")
    args = p.parse_args(argv)

    points = grid_land_points(args.step)
    if args.limit is not None:
        points = points[: args.limit]
    client = WeatherClient(cache_dir=args.cache_dir)
    start = f"{args.year}-01-01"
    end = f"{args.year}-12-31"

    t0 = time.perf_counter()
    n_ok = n_fail = 0
    for i, (lat, lon) in enumerate(points, start=1):
        try:
            try:
                client.get_historical_forecast_site_weather(lat, lon, start, end)
            except Exception:
                client.get_historical(lat, lon, start, end)
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"  [{i}/{len(points)}] ({lat:+.4f}, {lon:+.4f}) FAILED: {exc}", flush=True)
        if i % 50 == 0 or i == len(points):
            print(
                f"  [{i}/{len(points)}] {n_ok} ok, {n_fail} failed ({time.perf_counter() - t0:.0f}s elapsed)",
                flush=True,
            )
        time.sleep(args.sleep)

    print(f"Done: {n_ok} ok, {n_fail} failed, {time.perf_counter() - t0:.0f}s total.", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
