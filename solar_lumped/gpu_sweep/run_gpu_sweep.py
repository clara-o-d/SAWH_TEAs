#!/usr/bin/env python3
"""GPU-driven device-parameter sweep -- the JAX/diffrax counterpart to
scripts/grid_param_sweep.py (see docs/sherlock_param_sweep.tex for the CPU
version this mirrors). One invocation = one or more sites; each site's full
125-combo grid (hydrogel thickness x fin area ratio x vapor gap, 5 values
each; eps_abs/tau_glass are fixed constants per case, not swept) is batched
across combos and walked through all 365 real days in lockstep -- one
compiled vmapped step per day, each warm-starting from the previous day's
end state, after Aitken-converging day 1 to its steady periodic state.

Deliberately mirrors grid_param_sweep.py's CLI, weather fetch, combo grid, and
CSV schema exactly (imported directly, not reimplemented) so output is
schema-identical to and comparable with the CPU sweep's outputs/grid_sweep/
CSVs -- but writes to a separate --output-dir by default so this doesn't touch
the live CPU sweep's files.

Usage (small subset first -- see GPU_PRIMER.md / SHERLOCK_GPU_RUNBOOK.md):
    python3 gpu_sweep/run_gpu_sweep.py --num-sites 10 --output-csv outputs/gpu_grid_sweep/smoke.csv
    python3 gpu_sweep/run_gpu_sweep.py --lat-lon -23.6 -70.4 --output-csv outputs/gpu_grid_sweep/atacama.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
_SCRIPTS = _REPO / "scripts"
for p in (_SRC, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from solar_lumped import site_sweep as gps  # noqa: E402
from solar_lumped.physics import initial_loading  # noqa: E402
from solar_lumped.weather import WeatherClient  # noqa: E402
from solar_lumped.weather import grid_land_points  # noqa: E402

from jax_daily_cycle import (  # noqa: E402
    build_day_weather,
    build_device_arrays,
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
    p.add_argument("--cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--salt", type=str, default="LiCl")
    p.add_argument("--salt-loading", type=float, default=4.0)
    p.add_argument("--insulation-gap-mm", type=float, default=5.0)
    p.add_argument("--tilt-deg", type=float, default=gps.TILT_DEG)
    p.add_argument("--hydrogel-thickness-mm", type=float, nargs="+", default=list(gps.DEFAULT_HYDROGEL_THICKNESS_MM))
    p.add_argument("--fin-area-ratio", type=float, nargs="+", default=list(gps.DEFAULT_FIN_AREA_RATIO))
    p.add_argument("--vapor-gap-mm", type=float, nargs="+", default=list(gps.DEFAULT_VAPOR_GAP_MM))
    p.add_argument(
        "--eps-abs", type=float, default=gps.DEFAULT_EPS_ABS,
        help="Absorber solar absorptivity (fixed constant, not swept). Case 1/2 baseline: 0.95; "
        "Case 3 'optical material limits': 1.0.",
    )
    p.add_argument(
        "--tau-glass", type=float, default=gps.DEFAULT_TAU_GLASS,
        help="Glass solar transmittance (fixed constant, not swept). Case 1/2 baseline: 0.90; "
        "Case 3: 1.0.",
    )
    p.add_argument(
        "--eps-abs-ir", type=float, default=None,
        help="Absorber IR emissivity for the modified Eqs. 3/4 radiative exchange (fixed constant, "
        "not swept). Default None reproduces the original blackbody/cavity approximation exactly "
        "(Case 1) -- set together with --eps-glass-ir for Case 2 (0.05) or Case 3 (0.0).",
    )
    p.add_argument(
        "--eps-glass-ir", type=float, default=None,
        help="Glass IR emissivity for the modified Eqs. 3/4 radiative exchange. See --eps-abs-ir. "
        "(Case 2: 0.95; Case 3: 0.0.)",
    )
    p.add_argument("--max-rounds", type=int, default=8, help="Fixed Aitken round count (see FINDINGS.md Result 7)")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Skip a site entirely if all its combos are already in --output-csv")
    return p.parse_args(argv)


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


def run_site(lat: float, lon: float, args: argparse.Namespace, client: WeatherClient) -> int:
    """Compute and append all combo rows for one site. Returns rows written."""
    start, end = f"{args.year}-01-01", f"{args.year}-12-31"
    try:
        _, df = client.get_historical_forecast_site_weather(lat, lon, start, end)
    except Exception:
        df = client.get_historical(lat, lon, start, end)

    from solar_lumped.weather import real_weather_days_from_df

    days = [prof for _d, prof, _g in real_weather_days_from_df(df)]
    if not days:
        print(f"  ({lat:+.4f}, {lon:+.4f}): no usable weather days, skipping.", flush=True)
        return 0
    mean_rh, mean_t_amb, mean_solar = gps.mean_weather_stats([(0, p, 1) for p in days])

    all_combos = gps.combo_grid(
        hydrogel_thickness_mm=args.hydrogel_thickness_mm,
        fin_area_ratio=args.fin_area_ratio, vapor_gap_mm=args.vapor_gap_mm,
    )
    if args.resume:
        done = gps._existing_combo_keys(args.output_csv, lat, lon)
        if len(done) >= len(all_combos):
            print(f"  ({lat:+.4f}, {lon:+.4f}): all {len(all_combos)} combos already done, skipping.", flush=True)
            return 0
    else:
        done = set()

    combos = [
        c for c in all_combos
        if (round(c.hydrogel_thickness_mm, 6), round(c.fin_area_ratio, 6), round(c.vapor_gap_mm, 6))
        not in done
    ]
    if not combos:
        return 0

    configs = [
        gps.build_device_config(
            c, salt=args.salt, salt_loading=args.salt_loading, insulation_gap_mm=args.insulation_gap_mm,
            tilt_deg=args.tilt_deg, eps_abs=args.eps_abs, tau_glass=args.tau_glass,
            eps_abs_ir=args.eps_abs_ir, eps_glass_ir=args.eps_glass_ir,
        )
        for c in combos
    ]

    # Batch axis is the combo list; every combo walks the same year in lockstep, one
    # vmapped step per day, so days stay sequential and combos run in parallel.
    t0 = time.perf_counter()
    dt, n_abs_max, n_des_max = year_padding([days])
    device = build_device_arrays(configs)
    step_fn = make_year_step_fn(device, dt, n_abs_max, n_des_max)
    day_weathers = [build_day_weather([d] * len(configs), n_abs_max, n_des_max) for d in days]

    mean_yield, mean_eta = run_year_batched(
        step_fn, day_weathers,
        c_w_initial=np.array([initial_loading(cfg) for cfg in configs]),
        h_initial=np.array([cfg.hydrogel_thickness_m for cfg in configs]),
        aitken_max_rounds=args.max_rounds,
    )
    elapsed = time.perf_counter() - t0

    for ci, combo in enumerate(combos):
        gps._append_row(
            args.output_csv,
            {
                "lat": lat, "lon": lon,
                "mean_rh_frac": f"{mean_rh:.6f}", "mean_t_amb_c": f"{mean_t_amb:.4f}", "mean_solar_w_m2": f"{mean_solar:.2f}",
                "salt": args.salt,
                "hydrogel_thickness_mm": combo.hydrogel_thickness_mm, "eps_abs": args.eps_abs,
                "tau_glass": args.tau_glass,
                "eps_abs_ir": args.eps_abs_ir if args.eps_abs_ir is not None else "",
                "eps_glass_ir": args.eps_glass_ir if args.eps_glass_ir is not None else "",
                "fin_area_ratio": combo.fin_area_ratio,
                "vapor_gap_mm": combo.vapor_gap_mm,
                "warmup_method": "aitken-gpu-fixed-round", "resolution": "annual",
                "mean_yield_kg_m2": f"{mean_yield[ci]:.6f}", "mean_eta_thermal": f"{mean_eta[ci]:.6f}",
                "n_periods": len(days),
            },
        )
    print(
        f"  ({lat:+.4f}, {lon:+.4f}): {len(combos)} combo(s) x {len(days)} day(s) "
        f"in {elapsed:.1f}s", flush=True,
    )
    return len(combos)


def main() -> int:
    args = parse_args()
    sites = _site_list(args)
    print(f"{len(sites)} site(s) to run.", flush=True)
    client = WeatherClient(cache_dir=args.cache_dir)

    t0 = time.perf_counter()
    total_rows = 0
    for lat, lon in sites:
        total_rows += run_site(lat, lon, args, client)
    print(f"Done: {total_rows} row(s) written across {len(sites)} site(s), {time.perf_counter() - t0:.1f}s total.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
