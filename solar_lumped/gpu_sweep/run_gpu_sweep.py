#!/usr/bin/env python3
"""GPU-driven global scenario sweep: every real land site x the eight scenarios in
``site_sweep.SCENARIOS``, at the parameters.xlsx baseline design.

Nothing else varies. Geometry (hydrogel thickness, fin area ratio, vapor gap,
insulation gap, tilt), salt loading and the baseline optics all come straight from
docs/parameters.xlsx, so a row is identified by (site, scenario) alone. The scenarios
are the three absorber/glass optical cases (Wilson et al., reasonable improvements,
optical material limits), the two improved optical cases with instantaneous sorption
(g -> infinity), the same two with a perfect condenser (T_cond == T_amb), and the
optical limits with both.

``instant_equilibrium`` and ``condenser_ambient`` select code paths in the compiled
step rather than per-instance numbers, so scenarios are run in the four groups those
two booleans define. Within a group the batch axis is **site x scenario**: all sites
in one invocation pad to one weather shape and share one compilation, which is what
keeps the per-site recompile cost of FINDINGS.md Result 11 from dominating now that
there are 8 designs per site instead of 125.

Usage (small subset first -- see GPU_PRIMER.md / SHERLOCK_GPU_RUNBOOK.md):
    python3 gpu_sweep/run_gpu_sweep.py --num-sites 10 --output-csv outputs/gpu_scenario_sweep/smoke.csv
    python3 gpu_sweep/run_gpu_sweep.py --lat-lon -23.6 -70.4 --output-csv outputs/gpu_scenario_sweep/atacama.csv
"""

from __future__ import annotations

import argparse
import sys
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from solar_lumped import site_sweep as gps  # noqa: E402
from solar_lumped.physics import initial_loading  # noqa: E402
from solar_lumped.weather import WeatherClient  # noqa: E402
from solar_lumped.weather import grid_land_points  # noqa: E402

from jax_daily_cycle import (  # noqa: E402
    build_day_weather,
    build_system_arrays,
    make_year_step_fn,
    run_year_batched,
    year_padding,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    site = p.add_mutually_exclusive_group(required=True)
    site.add_argument("--lat-lon", type=float, nargs=2, action="append", metavar=("LAT", "LON"),
                       help="One site; repeat this flag for multiple explicit sites.")
    site.add_argument("--num-sites", type=int, help="First N sites of the --step land grid (index 0..N-1).")
    site.add_argument("--site-indices", type=int, nargs="+", help="Specific indices into the --step land grid.")
    site.add_argument("--site-range", type=int, nargs=2, metavar=("START", "END"),
                       help="Sites [START, END) of the --step land grid -- for splitting the full grid across "
                       "multiple concurrent GPU jobs (see sbatch_gpu_sweep_array.sh).")
    p.add_argument("--step", type=float, default=3.0, help="Grid spacing in degrees, used with --num-sites/--site-indices")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--scenarios", nargs="+", choices=tuple(gps.SCENARIOS), default=list(gps.SCENARIOS),
                   help="Subset of scenarios to run (default: all eight).")
    p.add_argument("--cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--salt", type=str, default="LiCl")
    p.add_argument("--max-rounds", type=int, default=8, help="Fixed Aitken round count (see FINDINGS.md Result 7)")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--resume", action="store_true",
                   help="Skip (site, scenario) pairs already present in --output-csv")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    sites = _site_list(args)
    print(f"{len(sites)} site(s) x {len(args.scenarios)} scenario(s) to run.", flush=True)
    client = WeatherClient(cache_dir=args.cache_dir)

    t0 = time.perf_counter()
    loaded = [w for lat, lon in sites if (w := _load_site(lat, lon, args, client)) is not None]
    if not loaded:
        print("No sites with usable weather -- nothing to do.")
        return 0
    # One padded weather shape and one compilation for the whole batch, so every site
    # must walk the same number of days. Real day counts differ only when Open-Meteo is
    # missing days for a site, so this truncates rather than dropping the site -- said
    # out loud, because a silently shortened year is a silently different mean yield.
    n_days = min(len(w.days) for w in loaded)
    short = [w for w in loaded if len(w.days) > n_days]
    if short:
        print(f"  day counts differ across sites; truncating all to {n_days} day(s) "
              f"({len(short)} site(s) lose up to {max(len(w.days) for w in short) - n_days}).", flush=True)
    print(f"Weather loaded for {len(loaded)} site(s) in {time.perf_counter() - t0:.1f}s.", flush=True)

    done = gps._existing_scenarios(args.output_csv) if args.resume else set()
    total_rows = 0
    for (instant, ambient), names in _scenario_groups(args.scenarios).items():
        instances = [
            (w, name)
            for name in names
            for w in loaded
            if (round(w.lat, 6), round(w.lon, 6), name) not in done
        ]
        total_rows += run_group(instances, n_days, args, instant_equilibrium=instant, condenser_ambient=ambient)
    print(f"Done: {total_rows} row(s) written across {len(loaded)} site(s), "
          f"{time.perf_counter() - t0:.1f}s total.", flush=True)
    return 0


def _site_list(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.lat_lon is not None:
        return [(lat, lon) for lat, lon in args.lat_lon]
    points = grid_land_points(args.step)
    if args.num_sites is not None:
        indices = range(min(args.num_sites, len(points)))
    elif args.site_range is not None:
        start, end = args.site_range
        indices = range(max(0, start), min(end, len(points)))
    else:
        indices = args.site_indices
    out = []
    for i in indices:
        if not (0 <= i < len(points)):
            print(f"index {i} out of range [0, {len(points) - 1}] for --step {args.step}", file=sys.stderr)
            continue
        out.append(points[i])
    return out


@dataclass(frozen=True, slots=True)
class SiteWeather:
    lat: float
    lon: float
    # Straight off the Open-Meteo response. Thinner air at altitude both diffuses vapour
    # faster and insulates the collector better, so this is not a cosmetic correction --
    # see physics.pressure_from_elevation_m.
    elevation_m: float
    days: list
    mean_rh: float
    mean_t_amb: float
    mean_solar: float


def _load_site(lat: float, lon: float, args: argparse.Namespace, client: WeatherClient) -> SiteWeather | None:
    """A year of real daily weather profiles for one site, or None if there are none."""
    start, end = f"{args.year}-01-01", f"{args.year}-12-31"
    try:
        _, df = client.get_historical_forecast_site_weather(lat, lon, start, end)
    except Exception:
        df = client.get_historical(lat, lon, start, end)

    from solar_lumped.weather import real_weather_days_from_df, site_elevation_m

    days = [prof for _d, prof, _g in real_weather_days_from_df(df, poa_tilt_deg=gps.TILT_DEG)]
    if not days:
        print(f"  ({lat:+.4f}, {lon:+.4f}): no usable weather days, skipping.", flush=True)
        return None
    mean_rh, mean_t_amb, mean_solar = gps.mean_weather_stats(days)
    return SiteWeather(
        lat=lat, lon=lon, elevation_m=site_elevation_m(df), days=days,
        mean_rh=mean_rh, mean_t_amb=mean_t_amb, mean_solar=mean_solar,
    )


def _scenario_groups(names: list[str]) -> dict[tuple[bool, bool], list[str]]:
    """Scenario names bucketed by (instant_equilibrium, condenser_ambient) -- the two
    static flags that fix the compiled code path, so one bucket = one compilation."""
    groups: dict[tuple[bool, bool], list[str]] = {}
    for name in names:
        sc = gps.SCENARIOS[name]
        groups.setdefault((sc.instant_equilibrium, sc.condenser_ambient), []).append(name)
    return groups


def run_group(
    instances: list[tuple[SiteWeather, str]],
    n_days: int,
    args: argparse.Namespace,
    *,
    instant_equilibrium: bool,
    condenser_ambient: bool,
) -> int:
    """Run one (site x scenario) batch sharing a compiled step. Returns rows written."""
    if not instances:
        return 0
    configs = [
        gps.build_system_config(
            gps.BASELINE_COMBO,
            salt=args.salt,
            salt_loading=gps.BASELINE_SALT_LOADING,
            insulation_gap_mm=gps.BASELINE_INSULATION_GAP_MM,
            tilt_deg=gps.TILT_DEG,
            eps_abs=gps.SCENARIOS[name].eps_abs,
            tau_glass=gps.SCENARIOS[name].tau_glass,
            eps_abs_ir=gps.SCENARIOS[name].eps_abs_ir,
            eps_glass_ir=gps.SCENARIOS[name].eps_glass_ir,
            site_elevation_m=w.elevation_m,
        )
        for w, name in instances
    ]
    # Uniform across the batch by construction (that is what _scenario_groups buys), but
    # still set on every config: build_system_arrays cross-checks instant_equilibrium
    # against the flag, and the CPU-side config is what any later re-run reads.
    configs = [
        dataclasses.replace(
            cfg, instant_equilibrium=instant_equilibrium,
            condenser_tracks_ambient=condenser_ambient,
        )
        for cfg in configs
    ]

    t0 = time.perf_counter()
    dt, n_abs_max, n_des_max = year_padding([w.days for w, _n in instances])
    system = build_system_arrays(configs, instant_equilibrium=instant_equilibrium)
    step_fn = make_year_step_fn(
        system, dt, n_abs_max, n_des_max,
        condenser_tracks_ambient=condenser_ambient,
        instant_equilibrium=instant_equilibrium,
    )
    day_weathers = [
        build_day_weather([w.days[d] for w, _n in instances], n_abs_max, n_des_max)
        for d in range(n_days)
    ]

    mean_yield, mean_eta = run_year_batched(
        step_fn, day_weathers,
        c_w_initial=np.array([initial_loading(cfg) for cfg in configs]),
        h_initial=np.array([cfg.hydrogel_thickness_m for cfg in configs]),
        aitken_max_rounds=args.max_rounds,
    )
    elapsed = time.perf_counter() - t0

    for i, (w, name) in enumerate(instances):
        sc = gps.SCENARIOS[name]
        gps._append_row(
            args.output_csv,
            {
                "lat": w.lat, "lon": w.lon, "elevation_m": f"{w.elevation_m:.1f}",
                "mean_rh_frac": f"{w.mean_rh:.6f}", "mean_t_amb_c": f"{w.mean_t_amb:.4f}",
                "mean_solar_w_m2": f"{w.mean_solar:.2f}",
                "salt": args.salt,
                "scenario": name,
                "hydrogel_thickness_mm": gps.BASELINE_COMBO.hydrogel_thickness_mm,
                "eps_abs": sc.eps_abs, "tau_glass": sc.tau_glass,
                "eps_abs_ir": sc.eps_abs_ir, "eps_glass_ir": sc.eps_glass_ir,
                "fin_area_ratio": gps.BASELINE_COMBO.fin_area_ratio,
                "vapor_gap_mm": gps.BASELINE_COMBO.vapor_gap_mm,
                "warmup_method": "aitken-gpu-fixed-round", "resolution": "annual",
                "condenser_mode": "ambient" if condenser_ambient else "ode",
                "kinetics": "instant" if instant_equilibrium else "finite_g",
                "mean_yield_kg_m2": f"{mean_yield[i]:.6f}", "mean_eta_thermal": f"{mean_eta[i]:.6f}",
                "n_periods": n_days,
            },
        )
    print(
        f"  [{'instant' if instant_equilibrium else 'finite_g'}/"
        f"{'ambient' if condenser_ambient else 'ode'}] {len(instances)} instance(s) "
        f"x {n_days} day(s) in {elapsed:.1f}s", flush=True,
    )
    return len(instances)


if __name__ == "__main__":
    raise SystemExit(main())
