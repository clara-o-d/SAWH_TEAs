#!/usr/bin/env python3
"""Pre-fetch every land point's weather into .weather_cache, from a LOGIN node.

Sherlock's compute nodes have no outbound network, so anything not already cached fails
inside the job -- site by site, hours in, as an opaque fetch error. Run this first and read
the MISSING count: 0 means the campaign can run offline.

Cheap to re-run. Every already-cached site is a sqlite hit, so this doubles as a checker.

    python3 gpu_sweep/warm_weather_cache.py --step 12.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from solar_lumped.weather import fetch_year_weather, grid_land_points  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", type=float, default=12.0, help="Grid spacing in degrees.")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    args = p.parse_args(argv)

    points = grid_land_points(args.step)
    print(f"{len(points)} land point(s) at {args.step} deg -> {args.cache_dir}", flush=True)

    failed: list[tuple[float, float, str]] = []
    for i, (lat, lon) in enumerate(points):
        try:
            df = fetch_year_weather(lat, lon, args.year, cache_dir=args.cache_dir)
            print(f"  [{i + 1}/{len(points)}] {lat:+.1f},{lon:+.1f}  {len(df)} rows", flush=True)
        except Exception as exc:  # noqa: BLE001
            # Keep going: one unreachable site should not hide the other 86's status.
            failed.append((lat, lon, repr(exc)))
            print(f"  [{i + 1}/{len(points)}] {lat:+.1f},{lon:+.1f}  FAILED {exc!r}", flush=True)

    print(f"\nMISSING: {len(failed)} of {len(points)}")
    for lat, lon, err in failed:
        print(f"  {lat:+.4f},{lon:+.4f}  {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
