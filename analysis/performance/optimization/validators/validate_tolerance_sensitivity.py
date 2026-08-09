"""Does tightening the ODE tolerance change the answer, and does the solver survive it?

Runs the real GPU sweep shape -- full combo grid x real weather days -- at the production
tolerance and at tighter ones, and reports how much the yield moves against how much wall
clock it costs. The question this answers is not "is tighter more accurate" (it is, by
definition) but "is the production tolerance already converged", which is the only reason
to care.

Two failure modes it is specifically watching for, both silent in the sweep itself:

  1. diffrax runs with ``throw=False`` and ``max_steps=16384``. A tightened tolerance that
     blows the step ceiling does not raise -- it returns whatever state it reached. That is
     a wrong answer wearing a right answer's shape, so the result codes are checked
     explicitly and reported as PASS/FAIL rather than left to the yield delta to imply.
  2. Smooth synthetic days hide the cost. jax_daily_cycle's own tolerance comment notes a
     ~36x wall-clock hit on real weather that the constant-profile day did not show, so
     this defaults to real Open-Meteo weather rather than a baseline profile.

Usage (see gpu_sweep/sbatch_tolerance_check.sh for the Sherlock wrapper):

    python3 analysis/performance/optimization/validators/validate_tolerance_sensitivity.py
    python3 .../validate_tolerance_sensitivity.py --days 0 --complex   # full year
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "solar_lumped" / "gpu_sweep"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import diffrax  # noqa: E402

import jax_daily_cycle as jdc  # noqa: E402
from solar_lumped import site_sweep as gps  # noqa: E402
from solar_lumped.simulation import initial_loading  # noqa: E402
from solar_lumped.weather import WeatherClient, real_weather_days_from_df  # noqa: E402

# (rtol, atol) pairs. The first is production and every later one is compared against it;
# the intermediate 10x point is what distinguishes "already converged" from "converging".
DEFAULT_TOLERANCES = ((1e-4, 1e-7), (1e-5, 1e-8), (1e-6, 1e-9))

# A yield shift below this is noise for a techno-economic answer -- LCOW is not resolvable
# to 0.1% given the cost inputs, so a tolerance change under it buys nothing real.
NEGLIGIBLE_DELTA_PCT = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat-lon", nargs=2, type=float, default=[-23.65, -70.40],
                   metavar=("LAT", "LON"), help="Site to test (default: Atacama field site)")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--days", type=int, default=14,
                   help="Days to simulate; 0 runs the full year (slow at tight tolerance)")
    p.add_argument("--max-rounds", type=int, default=8, help="Aitken warmup rounds")
    p.add_argument("--complex", action="store_true", help="Run the 13-dim complex fidelity")
    # Same default as run_gpu_sweep.py: Sherlock already has a ~21GB Open-Meteo cache
    # there, so this must not depend on the caller's cwd or the job re-fetches the year.
    p.add_argument("--cache-dir", default=str(_REPO / "solar_lumped" / ".weather_cache"),
                   help="WeatherClient cache directory")
    p.add_argument("--tolerances", nargs="+", default=None, metavar="RTOL,ATOL",
                   help='Override the tolerance ladder, e.g. --tolerances 1e-4,1e-7 1e-6,1e-9')
    return p.parse_args()


def load_days(args) -> list:
    lat, lon = args.lat_lon
    client = WeatherClient(cache_dir=args.cache_dir)
    start, end = f"{args.year}-01-01", f"{args.year}-12-31"
    try:
        _, df = client.get_historical_forecast_site_weather(lat, lon, start, end)
    except Exception:
        df = client.get_historical(lat, lon, start, end)
    days = [prof for _d, prof, _g in real_weather_days_from_df(df)]
    if not days:
        raise SystemExit(f"no usable weather days for ({lat:+.4f}, {lon:+.4f})")
    return days if args.days <= 0 else days[: args.days]


def solver_diagnostics(config, profile, dt, n_abs, n_des, *, complex_mode):
    """Step counts and diffrax result codes for one instance, run eagerly.

    The batched path is jitted, which makes sol.stats a traced value and sol.result
    invisible -- so the ceiling check runs unbatched, where both are concrete. One
    instance is enough: max_steps is per-lane, so if the worst combo clears it, all do.
    """
    system = jdc.build_system_arrays([config], complex_mode=complex_mode)
    weather = jdc.build_day_weather([profile], n_abs, n_des)
    single = jdc._make_single(dt, n_abs, n_des, complex_mode=complex_mode)

    seen = []
    real_solve = diffrax.diffeqsolve

    def spy(*a, **kw):
        sol = real_solve(*a, **kw)
        seen.append((int(sol.stats["num_steps"]), int(sol.result._value), kw.get("max_steps")))
        return sol

    diffrax.diffeqsolve = spy
    try:
        single(
            jnp.asarray(initial_loading(config)),
            jnp.asarray(config.hydrogel_thickness_m),
            *[w[0] for w in weather],
            *[v[0] for v in system.values()],
        )
    finally:
        diffrax.diffeqsolve = real_solve
    return seen


def run_at_tolerance(rtol, atol, configs, days, *, complex_mode, max_rounds):
    """Full combo grid over the day list at one tolerance. Returns (yields, etas, seconds).

    _make_single reads _RTOL/_ATOL at trace time, so setting them here and rebuilding the
    step function is what actually changes the compiled kernel -- reusing a step_fn built
    at another tolerance would silently measure the old one.
    """
    jdc._RTOL, jdc._ATOL = rtol, atol

    dt, n_abs, n_des = jdc.year_padding([days])
    system = jdc.build_system_arrays(configs, complex_mode=complex_mode)
    step_fn = jdc.make_year_step_fn(system, dt, n_abs, n_des, complex_mode=complex_mode)
    day_weathers = [jdc.build_day_weather([d] * len(configs), n_abs, n_des) for d in days]

    t0 = time.perf_counter()
    mean_yield, mean_eta = jdc.run_year_batched(
        step_fn, day_weathers,
        c_w_initial=np.array([initial_loading(c) for c in configs]),
        h_initial=np.array([c.hydrogel_thickness_m for c in configs]),
        aitken_max_rounds=max_rounds,
    )
    return np.asarray(mean_yield), np.asarray(mean_eta), time.perf_counter() - t0


def main() -> int:
    args = parse_args()
    if args.tolerances:
        ladder = tuple(tuple(float(v) for v in t.split(",")) for t in args.tolerances)
    else:
        ladder = DEFAULT_TOLERANCES

    print(f"jax.devices(): {jax.devices()}")
    if not any(d.platform == "gpu" for d in jax.devices()):
        print("WARNING: no GPU visible -- wall-clock ratios below are CPU numbers and will "
              "not match A100 behaviour. Correctness results are still valid.")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")

    days = load_days(args)
    # Same grid and same per-combo config the production sweep builds, so a tolerance
    # verdict here transfers to run_gpu_sweep.py without re-deriving anything.
    combos = gps.combo_grid(
        hydrogel_thickness_mm=list(gps.DEFAULT_HYDROGEL_THICKNESS_MM),
        fin_area_ratio=list(gps.DEFAULT_FIN_AREA_RATIO),
        vapor_gap_mm=list(gps.DEFAULT_VAPOR_GAP_MM),
    )
    configs = [
        gps.build_system_config(
            c, salt="LiCl", salt_loading=4.0, insulation_gap_mm=5.0, tilt_deg=gps.TILT_DEG,
            eps_abs=gps.DEFAULT_EPS_ABS, tau_glass=gps.DEFAULT_TAU_GLASS,
        )
        for c in combos
    ]
    if args.complex:
        import dataclasses
        from solar_lumped.complex_model import ComplexOptions
        configs = [dataclasses.replace(c, complex=ComplexOptions()) for c in configs]

    lat, lon = args.lat_lon
    print(f"\nsite ({lat:+.4f}, {lon:+.4f})  {len(configs)} combos x {len(days)} days  "
          f"fidelity={'complex' if args.complex else 'simple'}")

    # --- Step-ceiling check, before any timing: a tolerance that cannot solve at all is
    # --- not worth benchmarking, and throw=False means nothing else would report it.
    print(f"\n{'=' * 78}\nSolver health (eager, worst-case combo)\n{'=' * 78}")
    dt, n_abs, n_des = jdc.year_padding([days])
    worst = max(configs, key=lambda c: c.hydrogel_thickness_m)  # thickest gel = stiffest day
    failures = []
    for rtol, atol in ladder:
        jdc._RTOL, jdc._ATOL = rtol, atol
        stats = solver_diagnostics(worst, days[0], dt, n_abs, n_des, complex_mode=args.complex)
        parts, bad = [], False
        for phase, (n, code, ceiling) in zip(("abs", "des"), stats):
            headroom = ceiling / max(n, 1)
            bad |= code != 0 or headroom < 2.0
            parts.append(f"{phase}={n}/{ceiling} steps ({headroom:.0f}x headroom, result={code})")
        status = "FAIL" if bad else "ok"
        if bad:
            failures.append((rtol, atol))
        print(f"  rtol={rtol:.0e} atol={atol:.0e}: {'  '.join(parts)}  [{status}]")
    if failures:
        print("\n  FAIL means either a nonzero diffrax result code (solve did not succeed) or\n"
              "  under 2x max_steps headroom (one harsher day away from silently truncating).")

    # --- Yield sensitivity and cost at the real batch width. ---
    print(f"\n{'=' * 78}\nYield sensitivity ({len(configs)} combos, {len(days)} days)\n{'=' * 78}")
    base_yield, base_wall = None, 1.0
    rel_max = 0.0
    wall_tight = 0.0
    for rtol, atol in ladder:
        y, _eta, wall = run_at_tolerance(
            rtol, atol, configs, days, complex_mode=args.complex, max_rounds=args.max_rounds,
        )
        line = (f"  rtol={rtol:.0e} atol={atol:.0e}: "
                f"yield mean {y.mean():.6f} range [{y.min():.6f}, {y.max():.6f}] kg/m2/day  "
                f"{wall:7.1f}s")
        if base_yield is None:
            base_yield, base_wall = y, wall
            print(line + "   (reference)")
            continue
        rel = np.abs(y - base_yield) / np.maximum(np.abs(base_yield), 1e-12) * 100.0
        print(line + f"   max delta {rel.max():.4f}%  ({wall / base_wall:.1f}x cost)")
        print(f"      worst combo: index {int(rel.argmax())}, "
              f"{base_yield[rel.argmax()]:.6f} -> {y[rel.argmax()]:.6f} kg/m2/day")
        # The verdict is about the tightest rung, which is the last one the loop sees.
        rel_max, wall_tight = float(rel.max()), wall

    # --- Verdict. ---
    print(f"\n{'=' * 78}\nVerdict\n{'=' * 78}")
    if len(ladder) < 2:
        print("  Only one tolerance given -- nothing to compare against, so no verdict.")
    elif failures:
        print(f"  Tightening BREAKS the solver at {failures} -- see the health table above.")
    elif rel_max < NEGLIGIBLE_DELTA_PCT:
        print(f"  Production tolerance is converged: {rel_max:.4f}% max shift at "
              f"{ladder[-1][0]:.0e}/{ladder[-1][1]:.0e}, under the {NEGLIGIBLE_DELTA_PCT}% "
              f"threshold, for {wall_tight / base_wall:.1f}x the cost. Keep 1e-4/1e-7.")
    else:
        print(f"  Production tolerance is NOT converged: {rel_max:.4f}% max shift exceeds "
              f"{NEGLIGIBLE_DELTA_PCT}%. Tightening changes the answer -- investigate before "
              f"trusting swept yields at 1e-4/1e-7.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
