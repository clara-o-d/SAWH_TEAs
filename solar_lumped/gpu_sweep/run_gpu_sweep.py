#!/usr/bin/env python3
"""GPU-driven device-parameter sweep -- the JAX/diffrax counterpart to
scripts/grid_param_sweep.py (see docs/sherlock_param_sweep.tex for the CPU
version this mirrors). One invocation = one or more sites; each site's full
135-combo grid x 12 monthly profiles = up to 1,620 instances is batched into a
single compiled call on the GPU (see FINDINGS.md Results 5/7/8/9 -- batching
combos and batching different-length months are each separately validated;
this is their cross product, not yet validated on real hardware at this
combined size -- that's what this script's first runs are for).

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

import grid_param_sweep as gps  # noqa: E402
from solar_lumped.physics.sorbent import initial_loading  # noqa: E402
from solar_lumped.weather.client import WeatherClient  # noqa: E402
from solar_lumped.weather.land_grid import grid_land_points  # noqa: E402

from jax_daily_cycle import build_batch_arrays, find_cyclic_state_batched, make_batched_daily_cycle_fn  # noqa: E402


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
    p.add_argument("--tilt-deg", type=float, default=35.0)
    p.add_argument("--hydrogel-thickness-mm", type=float, nargs="+", default=list(gps.DEFAULT_HYDROGEL_THICKNESS_MM))
    p.add_argument("--eps-abs", type=float, nargs="+", default=list(gps.DEFAULT_EPS_ABS))
    p.add_argument("--tau-glass", type=float, nargs="+", default=list(gps.DEFAULT_TAU_GLASS))
    p.add_argument("--fin-area-ratio", type=float, nargs="+", default=list(gps.DEFAULT_FIN_AREA_RATIO))
    p.add_argument("--vapor-gap-mm", type=float, default=gps.DEFAULT_VAPOR_GAP_MM)
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

    months = gps.monthly_mean_profiles(df)
    mean_rh, mean_t_amb, mean_solar = gps.mean_weather_stats(months)

    all_combos = gps.combo_grid(
        hydrogel_thickness_mm=args.hydrogel_thickness_mm, eps_abs=args.eps_abs,
        tau_glass=args.tau_glass, fin_area_ratio=args.fin_area_ratio,
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
        if (round(c.hydrogel_thickness_mm, 6), round(c.eps_abs, 6), round(c.tau_glass, 6), round(c.fin_area_ratio, 6))
        not in done
    ]
    if not combos:
        return 0

    configs = [
        gps.build_device_config(
            c, salt=args.salt, salt_loading=args.salt_loading, insulation_gap_mm=args.insulation_gap_mm,
            tilt_deg=args.tilt_deg, vapor_gap_mm=args.vapor_gap_mm,
            eps_abs_ir=args.eps_abs_ir, eps_glass_ir=args.eps_glass_ir,
        )
        for c in combos
    ]

    # Cross product: every combo x every month, all batched into one compiled call.
    profiles_list, configs_list = [], []
    combo_of, month_of = [], []
    for ci, cfg in enumerate(configs):
        for mi, (_month, profile, _n_days) in enumerate(months):
            profiles_list.append(profile)
            configs_list.append(cfg)
            combo_of.append(ci)
            month_of.append(mi)

    t0 = time.perf_counter()
    batch, dt, n_abs_max, n_des_max = build_batch_arrays(profiles_list, configs_list)
    batched_fn = make_batched_daily_cycle_fn(batch, dt, n_abs_max, n_des_max)

    cw0_arr = np.array([initial_loading(cfg) for cfg in configs_list])
    h0_arr = np.array([cfg.hydrogel_thickness_m for cfg in configs_list])
    cw_conv, h_conv = find_cyclic_state_batched(batched_fn, c_w_initial=cw0_arr, h_initial=h0_arr, max_rounds=args.max_rounds)
    water, eta, _, _ = batched_fn(cw_conv, h_conv)
    water = np.asarray(water)
    eta = np.asarray(eta)
    elapsed = time.perf_counter() - t0

    weights = np.array([n_days for _, _, n_days in months], dtype=float)
    n_combos, n_months = len(combos), len(months)
    water_grid = water.reshape(n_combos, n_months)
    eta_grid = eta.reshape(n_combos, n_months)
    mean_yield = (water_grid * weights).sum(axis=1) / weights.sum()
    mean_eta = (eta_grid * weights).sum(axis=1) / weights.sum()

    for ci, combo in enumerate(combos):
        gps._append_row(
            args.output_csv,
            {
                "lat": lat, "lon": lon,
                "mean_rh_frac": f"{mean_rh:.6f}", "mean_t_amb_c": f"{mean_t_amb:.4f}", "mean_solar_w_m2": f"{mean_solar:.2f}",
                "salt": args.salt,
                "hydrogel_thickness_mm": combo.hydrogel_thickness_mm, "eps_abs": combo.eps_abs,
                "tau_glass": combo.tau_glass,
                "eps_abs_ir": args.eps_abs_ir if args.eps_abs_ir is not None else "",
                "eps_glass_ir": args.eps_glass_ir if args.eps_glass_ir is not None else "",
                "fin_area_ratio": combo.fin_area_ratio,
                "vapor_gap_mm": args.vapor_gap_mm,
                "warmup_method": "aitken-gpu-fixed-round", "resolution": "monthly",
                "mean_yield_kg_m2": f"{mean_yield[ci]:.6f}", "mean_eta_thermal": f"{mean_eta[ci]:.6f}",
                "n_periods": len(months),
            },
        )
    print(
        f"  ({lat:+.4f}, {lon:+.4f}): {len(combos)} combo(s) x {n_months} month(s) = {len(combos) * n_months} "
        f"instances in {elapsed:.1f}s", flush=True,
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
