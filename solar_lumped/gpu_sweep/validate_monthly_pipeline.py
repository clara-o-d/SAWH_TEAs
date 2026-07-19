#!/usr/bin/env python3
"""Step 3: full pipeline validation -- monthly-resolution profiles + Aitken
steady-periodic-state search, JAX end to end, compared against both the CPU
pipeline (same profiles/config) and Wilson's published reference yields.

This is the comparison the handoff doc actually asked for
(docs/gpu_sweep_handoff.md's Validation section): Atacama desert site
(-23.6, -70.4), eps_abs=0.90 -> 1.707476 kg/m^2, eps_abs=0.95 -> 1.800478 kg/m^2
(everything else at baseline: hydrogel 4.0mm, tau_glass=0.85, fin_area_ratio=7.1,
monthly resolution).
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

from solar_lumped.physics.device_balances import DeviceThermalParams  # noqa: E402
from solar_lumped.physics.sorbent import initial_loading  # noqa: E402
from solar_lumped.simulation.device_config import DeviceConfig  # noqa: E402
from solar_lumped.simulation.ode_system import find_cyclic_state, run_daily_cycle  # noqa: E402
from solar_lumped.weather.client import WeatherClient  # noqa: E402
from solar_lumped.weather.climate import representative_mean_day_df  # noqa: E402
from solar_lumped.weather.profiles import profile_from_day_df  # noqa: E402

from jax_daily_cycle import find_cyclic_state_jax, make_daily_cycle_fn  # noqa: E402


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


def cpu_combo_yield(profiles, config) -> tuple[float, float]:
    yields, etas, weights = [], [], []
    for _m, profile, n_days in profiles:
        cw, h = find_cyclic_state(profile, config, verbose=False)
        yield_kg, eta, _, _ = run_daily_cycle(profile, config, c_w_initial=cw, h_initial=h)
        yields.append(yield_kg)
        etas.append(eta)
        weights.append(n_days)
    total_w = sum(weights)
    return (
        sum(y * w for y, w in zip(yields, weights)) / total_w,
        sum(e * w for e, w in zip(etas, weights)) / total_w,
    )


def jax_combo_yield(profiles, config) -> tuple[float, float, float]:
    yields, etas, weights = [], [], []
    t0 = time.perf_counter()
    for _m, profile, n_days in profiles:
        daily_cycle_fn = make_daily_cycle_fn(profile, config)
        cw0 = initial_loading(config)
        h0 = config.hydrogel_thickness_m
        cw, h = find_cyclic_state_jax(daily_cycle_fn, c_w_initial=cw0, h_initial=h0)
        water, eta, _, _ = daily_cycle_fn(cw, h)
        yields.append(float(water))
        etas.append(float(eta))
        weights.append(n_days)
    elapsed = time.perf_counter() - t0
    total_w = sum(weights)
    return (
        sum(y * w for y, w in zip(yields, weights)) / total_w,
        sum(e * w for e, w in zip(etas, weights)) / total_w,
        elapsed,
    )


def main() -> int:
    print("Fetching Atacama 2024 weather (cached)...", flush=True)
    client = WeatherClient(cache_dir=str(_REPO / ".weather_cache"))
    lat, lon = -23.6, -70.4
    try:
        _, df = client.get_historical_forecast_site_weather(lat, lon, "2024-01-01", "2024-12-31")
    except Exception:
        df = client.get_historical(lat, lon, "2024-01-01", "2024-12-31")
    profiles = monthly_mean_profiles(df)
    print(f"Built {len(profiles)} monthly mean-day profiles.\n", flush=True)

    for eps_abs, ref_yield in ((0.90, 1.707476), (0.95, 1.800478)):
        config = DeviceConfig(
            tilt_deg=35.0,
            fin_area_ratio=7.1,
            thermal=DeviceThermalParams(
                insulation_gap_m=0.005, vapor_gap_m=0.04, eps_abs=eps_abs, tau_glass=0.85, tilt_deg=35.0,
            ),
        )
        print(f"=== eps_abs={eps_abs} ===", flush=True)

        t0 = time.perf_counter()
        cpu_yield, cpu_eta = cpu_combo_yield(profiles, config)
        cpu_elapsed = time.perf_counter() - t0

        jax_yield, jax_eta, jax_elapsed = jax_combo_yield(profiles, config)

        print(f"  CPU:  mean_yield={cpu_yield:.6f} kg/m^2  eta={cpu_eta:.4f}  ({cpu_elapsed:.1f}s)")
        print(f"  JAX:  mean_yield={jax_yield:.6f} kg/m^2  eta={jax_eta:.4f}  ({jax_elapsed:.1f}s)")
        print(f"  Wilson reference: {ref_yield:.6f} kg/m^2")
        print(f"  JAX vs CPU:      {abs(jax_yield - cpu_yield) / cpu_yield:.4%}")
        print(f"  JAX vs reference: {abs(jax_yield - ref_yield) / ref_yield:.4%}")
        print(f"  CPU vs reference: {abs(cpu_yield - ref_yield) / ref_yield:.4%}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
