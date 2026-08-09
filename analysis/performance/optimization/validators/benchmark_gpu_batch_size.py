#!/usr/bin/env python3
"""GPU batch-size scan -- run this on an actual GPU node (see GPU_PRIMER.md /
FINDINGS.md's next steps; no GPU was available when this prototype was built).

Answers the handoff doc's open sizing question directly: how many
(site, combo) instances fit in one batched compiled call on one GPU, and how
does compile time / per-instance throughput scale as batch size grows toward
the real grid's 189,675 total? Tiles real Atacama day profiles and a
handful of system configs up to each target batch size (repetition is fine for
a throughput/memory test -- it doesn't need to be 189,675 *distinct* physical
setups, just the same shapes and arithmetic intensity as the real grid).

Usage:
    python3 analysis/performance/optimization/validators/benchmark_gpu_batch_size.py \
        [--sizes 12 120 1200 12000 60000]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# This benchmark lives under analysis/ but drives solar_lumped's JAX modules, so both
# paths anchor to solar_lumped/ rather than to this file's parent --
# .../SAWH_TEAs/analysis/performance/optimization/validators -> SAWH_TEAs.
_REPO = Path(__file__).resolve().parents[4] / "solar_lumped"
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_REPO / "gpu_sweep"))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solar_lumped.physics import SystemThermalParams  # noqa: E402
from solar_lumped.physics import initial_loading  # noqa: E402
from solar_lumped.simulation import SystemConfig  # noqa: E402
from solar_lumped.weather import WeatherClient  # noqa: E402
from solar_lumped.weather import real_weather_days_from_df  # noqa: E402

from jax_daily_cycle import (  # noqa: E402
    build_day_weather,
    build_system_arrays,
    find_cyclic_state_batched,
    make_year_step_fn,
    year_padding,
)


def _gpu_memory_used_mb() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().replace("\n", " | ")
    except Exception as e:  # noqa: BLE001
        return f"(nvidia-smi unavailable: {e})"


# Base profiles are real days sampled across the year (tiled up to each target batch
# size below), never mean days.
PROFILE_STRIDE = 30


def sampled_real_day_profiles(df):
    """Real day profiles sampled every PROFILE_STRIDE days across the year."""
    return [prof for _day, prof, _group in real_weather_days_from_df(df, stride=PROFILE_STRIDE)]


def build_tiled_batch(target_size: int, base_profiles, base_configs):
    n_base = len(base_profiles)
    n_cfg = len(base_configs)
    profiles = [base_profiles[i % n_base] for i in range(target_size)]
    configs = [base_configs[i % n_cfg] for i in range(target_size)]
    return profiles, configs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[12, 120, 1200, 12000, 60000, 189675])
    parser.add_argument("--lat", type=float, default=-23.6)
    parser.add_argument("--lon", type=float, default=-70.4)
    args = parser.parse_args()

    print(f"jax.devices(): {jax.devices()}", flush=True)
    print(f"GPU memory before start: {_gpu_memory_used_mb()}\n", flush=True)

    print("Fetching weather (cached if available)...", flush=True)
    client = WeatherClient(cache_dir=str(_REPO / ".weather_cache"))
    try:
        _, df = client.get_historical_forecast_site_weather(args.lat, args.lon, "2024-01-01", "2024-12-31")
    except Exception:
        df = client.get_historical(args.lat, args.lon, "2024-01-01", "2024-12-31")
    base_profiles = sampled_real_day_profiles(df)
    print(f"Built {len(base_profiles)} base real-day profiles.\n", flush=True)

    base_configs = [
        SystemConfig(
            tilt_deg=35.0, fin_area_ratio=7.1,
            thermal=SystemThermalParams(insulation_gap_m=0.005, vapor_gap_m=0.04, eps_abs=eps, tau_glass=0.85, tilt_deg=35.0),
        )
        for eps in (0.85, 0.90, 0.95)
    ]

    for size in args.sizes:
        print(f"=== batch size {size} ===", flush=True)
        profiles, configs = build_tiled_batch(size, base_profiles, base_configs)
        try:
            t0 = time.perf_counter()
            # make_year_step_fn takes weather as a runtime argument (one compiled step
            # reused for all 365 days). This scan measures one fixed day's batch, so the
            # weather is bound once here and the closure gives find_cyclic_state_batched
            # the (c_w, h) -> ... signature it expects.
            dt, n_abs_max, n_des_max = year_padding([[p] for p in profiles])
            system = build_system_arrays(configs)
            step_fn = make_year_step_fn(system, dt, n_abs_max, n_des_max)
            weather = build_day_weather(profiles, n_abs_max, n_des_max)

            def batched_fn(cw, h, _step=step_fn, _weather=weather):
                return _step(cw, h, _weather)

            cw0_arr = np.array([initial_loading(c) for c in configs])
            h0_arr = np.array([c.hydrogel_thickness_m for c in configs])

            cw_conv, h_conv = find_cyclic_state_batched(batched_fn, c_w_initial=cw0_arr, h_initial=h0_arr, max_rounds=8)
            water, eta, _, _ = batched_fn(cw_conv, h_conv)
            jax.block_until_ready(water)
            first_elapsed = time.perf_counter() - t0

            t0 = time.perf_counter()
            cw_conv, h_conv = find_cyclic_state_batched(batched_fn, c_w_initial=cw0_arr, h_initial=h0_arr, max_rounds=8)
            water, eta, _, _ = batched_fn(cw_conv, h_conv)
            jax.block_until_ready(water)
            warm_elapsed = time.perf_counter() - t0

            print(f"  padded shapes: n_abs_max={n_abs_max} n_des_max={n_des_max}")
            print(f"  first call (compile+run): {first_elapsed:.2f}s  ({first_elapsed / size * 1000:.4f} ms/instance)")
            print(f"  warm rerun:               {warm_elapsed:.2f}s  ({warm_elapsed / size * 1000:.4f} ms/instance)")
            print(f"  sample yields: {np.asarray(water)[:3]}")
            print(f"  GPU memory: {_gpu_memory_used_mb()}\n", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED at size {size}: {type(e).__name__}: {e}\n", flush=True)
            print("  (stopping the scan here -- this is the practical ceiling for this batch shape)")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
