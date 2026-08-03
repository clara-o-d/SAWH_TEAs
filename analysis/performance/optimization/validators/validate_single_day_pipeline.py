#!/usr/bin/env python3
"""Step 3: full pipeline validation on one real day -- Aitken steady-periodic-state
search, JAX end to end, compared against the CPU pipeline on the same profile/config.

Single real day (never a mean day) at the Atacama desert site (-23.6, -70.4), baseline
config otherwise: hydrogel 4.0mm, tau_glass=0.85, fin_area_ratio=7.1. CPU-vs-JAX
agreement is the assertion here; absolute annual yields come from the 365-day GPU sweep,
not from this validator.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_lumped.physics import SystemThermalParams  # noqa: E402
from solar_lumped.physics import initial_loading  # noqa: E402
from solar_lumped.simulation import SystemConfig  # noqa: E402
from solar_lumped.simulation import find_cyclic_state, run_daily_cycle  # noqa: E402
from solar_lumped.weather import real_day_profile  # noqa: E402

from jax_daily_cycle import find_cyclic_state_jax, make_daily_cycle_fn  # noqa: E402

VALIDATION_DAY = date(2024, 6, 15)
LAT, LON = -23.6, -70.4

# Agreement we expect between the two backends solving the same equations.
PARITY_TOL = 0.01


def cpu_day_yield(profile, config) -> tuple[float, float]:
    cw, h = find_cyclic_state(profile, config, verbose=False)
    yield_kg, eta, _, _ = run_daily_cycle(profile, config, c_w_initial=cw, h_initial=h)
    return float(yield_kg), float(eta)


def jax_day_yield(profile, config) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    daily_cycle_fn = make_daily_cycle_fn(profile, config)
    cw, h = find_cyclic_state_jax(
        daily_cycle_fn,
        c_w_initial=initial_loading(config),
        h_initial=config.hydrogel_thickness_m,
    )
    water, eta, _, _ = daily_cycle_fn(cw, h)
    return float(water), float(eta), time.perf_counter() - t0


def main() -> int:
    print(f"Fetching Atacama {VALIDATION_DAY.isoformat()} weather (cached)...", flush=True)
    profile = real_day_profile(LAT, LON, VALIDATION_DAY, cache_dir=str(_REPO / ".weather_cache"))

    worst = 0.0
    for eps_abs in (0.90, 0.95):
        config = SystemConfig(
            tilt_deg=35.0,
            fin_area_ratio=7.1,
            thermal=SystemThermalParams(
                insulation_gap_m=0.005, vapor_gap_m=0.04, eps_abs=eps_abs, tau_glass=0.85, tilt_deg=35.0,
            ),
        )
        print(f"=== eps_abs={eps_abs} ===", flush=True)

        t0 = time.perf_counter()
        cpu_yield, cpu_eta = cpu_day_yield(profile, config)
        cpu_elapsed = time.perf_counter() - t0

        jax_yield, jax_eta, jax_elapsed = jax_day_yield(profile, config)

        rel = abs(jax_yield - cpu_yield) / cpu_yield
        worst = max(worst, rel)
        print(f"  CPU:  yield={cpu_yield:.6f} kg/m^2  eta={cpu_eta:.4f}  ({cpu_elapsed:.1f}s)")
        print(f"  JAX:  yield={jax_yield:.6f} kg/m^2  eta={jax_eta:.4f}  ({jax_elapsed:.1f}s)")
        print(f"  JAX vs CPU: {rel:.4%}\n")

    ok = worst <= PARITY_TOL
    print(f"{'PASS' if ok else 'FAIL'}: worst CPU/JAX disagreement {worst:.4%} (tol {PARITY_TOL:.2%})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
