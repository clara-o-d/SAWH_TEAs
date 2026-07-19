#!/usr/bin/env python3
"""Step 4: validate cross-length batching (Result 7 in FINDINGS.md).

The 12 real monthly mean-day profiles at one site naturally have different
absorption/desorption step counts (day length varies by month) -- a perfect,
already-available test bed for the padding+masking approach needed to batch
across sites/months in one compiled call, without fetching new weather data.

Compares the batched (padded, masked, fixed-round-count Aitken) JAX pipeline
against the serial per-month JAX pipeline (already validated against CPU in
validate_monthly_pipeline.py) on the same 12 profiles.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from solar_lumped.physics.device_balances import DeviceThermalParams  # noqa: E402
from solar_lumped.physics.sorbent import initial_loading  # noqa: E402
from solar_lumped.simulation.device_config import DeviceConfig  # noqa: E402
from solar_lumped.weather.client import WeatherClient  # noqa: E402
from solar_lumped.weather.climate import representative_mean_day_df  # noqa: E402
from solar_lumped.weather.profiles import profile_from_day_df  # noqa: E402

from jax_daily_cycle import (  # noqa: E402
    build_batch_arrays,
    find_cyclic_state_batched,
    find_cyclic_state_jax,
    make_batched_daily_cycle_fn,
    make_daily_cycle_fn,
)


def monthly_mean_profiles(df):
    import pandas as pd

    out = []
    for m in sorted(set(df.index.month)):
        month_df = df[df.index.month == m]
        if month_df.empty:
            continue
        ref_day = month_df.index[len(month_df) // 2].date()
        mean_day_df = representative_mean_day_df(month_df, reference_day=ref_day)
        profile = profile_from_day_df(mean_day_df)
        n_days = len(pd.unique(month_df.index.date))
        out.append((m, profile, n_days))
    return out


def main() -> int:
    print("Fetching Atacama 2024 weather (cached)...", flush=True)
    client = WeatherClient(cache_dir=str(_REPO / ".weather_cache"))
    lat, lon = -23.6, -70.4
    try:
        _, df = client.get_historical_forecast_site_weather(lat, lon, "2024-01-01", "2024-12-31")
    except Exception:
        df = client.get_historical(lat, lon, "2024-01-01", "2024-12-31")
    monthly = monthly_mean_profiles(df)
    months, profiles, n_days = zip(*monthly)
    print(f"Built {len(profiles)} monthly profiles, lengths (desorption steps): "
          f"{[len(p.desorption.temperature_c) for p in profiles]}\n", flush=True)

    config = DeviceConfig(
        tilt_deg=35.0, fin_area_ratio=7.1,
        thermal=DeviceThermalParams(insulation_gap_m=0.005, vapor_gap_m=0.04, eps_abs=0.90, tau_glass=0.85, tilt_deg=35.0),
    )
    cw0 = initial_loading(config)
    h0 = config.hydrogel_thickness_m

    print("=== Serial (one jit-compiled daily_cycle_fn per month, adaptive Aitken) ===", flush=True)
    t0 = time.perf_counter()
    serial_yields = []
    for profile in profiles:
        daily_cycle_fn = make_daily_cycle_fn(profile, config)
        cw, h = find_cyclic_state_jax(daily_cycle_fn, c_w_initial=cw0, h_initial=h0)
        water, _eta, _, _ = daily_cycle_fn(cw, h)
        serial_yields.append(float(water))
    serial_elapsed = time.perf_counter() - t0
    serial_yields = np.array(serial_yields)
    print(f"  per-month yields: {np.round(serial_yields, 6)}")
    print(f"  elapsed: {serial_elapsed:.1f}s\n")

    print("=== Batched (one compiled call for all 12 months, padded+masked, fixed-round Aitken) ===", flush=True)
    configs = [config] * len(profiles)
    batch, dt, n_abs_max, n_des_max = build_batch_arrays(list(profiles), configs)
    print(f"  padded shapes: n_abs_max={n_abs_max} n_des_max={n_des_max}")
    batched_fn = make_batched_daily_cycle_fn(batch, dt, n_abs_max, n_des_max)

    cw0_arr = np.full(len(profiles), cw0)
    h0_arr = np.full(len(profiles), h0)

    t0 = time.perf_counter()
    cw_conv, h_conv = find_cyclic_state_batched(batched_fn, c_w_initial=cw0_arr, h_initial=h0_arr, max_rounds=8)
    water_batch, eta_batch, _, _ = batched_fn(cw_conv, h_conv)
    water_batch = np.asarray(water_batch)
    batched_elapsed = time.perf_counter() - t0
    print(f"  per-month yields: {np.round(water_batch, 6)}")
    print(f"  elapsed (compile + run, first call): {batched_elapsed:.2f}s\n")

    t0 = time.perf_counter()
    cw_conv, h_conv = find_cyclic_state_batched(batched_fn, c_w_initial=cw0_arr, h_initial=h0_arr, max_rounds=8)
    water_batch2, _, _, _ = batched_fn(cw_conv, h_conv)
    warm_elapsed = time.perf_counter() - t0
    print(f"  warm rerun elapsed: {warm_elapsed:.3f}s\n")

    rel_err = np.abs(water_batch - serial_yields) / np.maximum(np.abs(serial_yields), 1e-9)
    print("=== Comparison (batched vs serial, per month) ===")
    for m, s, b, r in zip(months, serial_yields, water_batch, rel_err):
        print(f"  month {m:2d}: serial={s:.6f}  batched={b:.6f}  rel_err={r:.4%}")
    print(f"\n  worst per-month rel_err: {rel_err.max():.4%}")

    total_w = sum(n_days)
    serial_mean = float(np.sum(serial_yields * np.array(n_days)) / total_w)
    batched_mean = float(np.sum(water_batch * np.array(n_days)) / total_w)
    print(f"\n  day-weighted mean yield: serial={serial_mean:.6f}  batched={batched_mean:.6f}  "
          f"rel_diff={abs(batched_mean - serial_mean) / serial_mean:.4%}")
    print(f"  wall-clock: serial={serial_elapsed:.1f}s  batched(first)={batched_elapsed:.2f}s  batched(warm)={warm_elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
