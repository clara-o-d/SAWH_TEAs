#!/usr/bin/env python3
"""At every map point run_gpu_sweep.py would brute-force sweep, run BayesOpt
instead: same site selection CLI, same 12-month Aitken JAX fast path
(evaluator.py's JAX_AITKEN_MAX_ROUNDS=8, resolution="monthly"), optimizing
over the same 3 combo variables (hydrogel_thickness, fin_area_ratio,
vapor_gap) with bounds taken from the same --hydrogel-thickness-mm/
--fin-area-ratio/--vapor-gap-mm lists the brute-force sweep grids over
(min/max instead of the 5 discrete grid values). insulation_gap_mm/tilt_deg/
salt-loading are fixed constants (bounds collapsed to a point), same as the
brute-force sweep's non-swept args.

Reuses run_gpu_sweep.py's site selection (so --site-range/--site-indices
split across concurrent GPU jobs the same way, see sbatch_gpu_sweep_array.sh)
and sawh_bayesopt's run_bayesopt loop (each site's BayesOpt evaluations are
still one batched jax.vmap call per round via evaluator.py, so this stays
GPU-parallel the same way the brute-force sweep is).

Usage:
    python3 gpu_sweep/run_bayesopt_sweep.py --num-sites 10 --output-dir outputs/gpu_bayesopt_sweep/smoke
    python3 gpu_sweep/run_bayesopt_sweep.py --lat-lon -23.6 -70.4 --output-dir outputs/gpu_bayesopt_sweep/atacama
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
_SCRIPTS = _REPO / "scripts"
_BAYESOPT_SRC = _REPO.parent / "sawh_bayesopt" / "src"
for p in (_SRC, _SCRIPTS, _BAYESOPT_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import grid_param_sweep as gps  # noqa: E402

from run_gpu_sweep import _site_list  # noqa: E402

from sawh_bayesopt.bayesopt import BayesOptConfig, run_bayesopt  # noqa: E402
from sawh_bayesopt.design_space import CASE_EPS_IR, DesignBounds  # noqa: E402
from sawh_bayesopt.reporting import write_history_csv, write_run_config  # noqa: E402
from sawh_bayesopt.sites import SiteSpec  # noqa: E402
from sawh_bayesopt.surrogate import save_state  # noqa: E402
from sawh_bayesopt.verification import verify_optimum  # noqa: E402


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

    # Fixed (non-optimized) device constants -- same role as run_gpu_sweep.py's
    # matching flags, just not swept here.
    p.add_argument("--salt", type=str, default="LiCl")
    p.add_argument("--salt-loading", type=float, default=4.0, help="salt_to_polymer_ratio, held fixed.")
    p.add_argument("--insulation-gap-mm", type=float, default=5.0, help="insulation_gap_m, held fixed.")
    p.add_argument("--tilt-deg", type=float, default=gps.TILT_DEG, help="tilt_deg, held fixed.")
    p.add_argument("--case", choices=tuple(CASE_EPS_IR), default="case2")

    # Same 3 combo variables the brute-force sweep grids over -- min/max of
    # these lists become the BayesOpt box bounds instead of 5 discrete values.
    p.add_argument("--hydrogel-thickness-mm", type=float, nargs="+", default=list(gps.DEFAULT_HYDROGEL_THICKNESS_MM))
    p.add_argument("--fin-area-ratio", type=float, nargs="+", default=list(gps.DEFAULT_FIN_AREA_RATIO))
    p.add_argument("--vapor-gap-mm", type=float, nargs="+", default=list(gps.DEFAULT_VAPOR_GAP_MM))

    # BayesOpt loop params -- same defaults as sawh_bayesopt/scripts/run_bayesopt.py.
    p.add_argument("--n-init", type=int, default=24)
    p.add_argument("--n-total", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ei-xi", type=float, default=0.01)
    p.add_argument("--stall-rel-tol", type=float, default=0.005)
    p.add_argument("--stall-rounds", type=int, default=3)
    p.add_argument("--de-maxiter", type=int, default=1000)
    p.add_argument("--de-popsize", type=int, default=40)

    # Per-site verification -- re-evaluates the reported optimum (+ perturbed
    # neighbors) against the true model, same check run_bayesopt.py does, to
    # flag GP surrogate artifacts rather than trusting the EI loop blindly.
    p.add_argument("--n-verify-neighbors", type=int, default=5)
    p.add_argument("--verify-perturbation-frac", type=float, default=0.10)

    p.add_argument("--output-dir", type=Path, required=True, help="Per-site run dirs (history/config/gp_state) go here.")
    p.add_argument("--resume", action="store_true", help="Skip a site entirely if it's already in --output-dir/summary.csv")
    return p.parse_args(argv)


def _bounds(args: argparse.Namespace) -> DesignBounds:
    """Box bounds for the 3 optimized dims from the sweep's own combo lists
    (min/max, mm -> m); the other 3 DesignBounds dims collapse to a point at
    the fixed CLI value -- from_unit_cube handles low==high with no special
    case needed.
    """
    salt_ratio = args.salt_loading
    insulation_m = args.insulation_gap_mm / 1000.0
    tilt = args.tilt_deg
    return DesignBounds(
        hydrogel_thickness_m=(min(args.hydrogel_thickness_mm) / 1000.0, max(args.hydrogel_thickness_mm) / 1000.0),
        vapor_gap_m=(min(args.vapor_gap_mm) / 1000.0, max(args.vapor_gap_mm) / 1000.0),
        insulation_gap_m=(insulation_m, insulation_m),
        fin_area_ratio=(min(args.fin_area_ratio), max(args.fin_area_ratio)),
        tilt_deg=(tilt, tilt),
        salt_to_polymer_ratio=(salt_ratio, salt_ratio),
    )


def _existing_sites(path: Path) -> set[tuple[float, float]]:
    if not path.is_file():
        return set()
    with path.open() as f:
        return {(round(float(r["lat"]), 6), round(float(r["lon"]), 6)) for r in csv.DictReader(f)}


def _append_summary_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.is_file()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            w.writeheader()
        w.writerow(row)


def run_site(lat: float, lon: float, args: argparse.Namespace, bounds: DesignBounds, summary_csv: Path) -> None:
    site = SiteSpec(name=f"{lat:+.4f}_{lon:+.4f}", lat=lat, lon=lon, year=args.year)
    run_dir = args.output_dir / site.name

    cfg = BayesOptConfig(
        bounds=bounds,
        sites=(site,),
        combine_rule="mean",
        n_init=args.n_init,
        n_total=args.n_total,
        batch_size=args.batch_size,
        seed=args.seed,
        ei_xi=args.ei_xi,
        stall_rel_tol=args.stall_rel_tol,
        stall_rounds=args.stall_rounds,
        resolution="monthly",
        weather_cache_dir=args.cache_dir,
        case=args.case,
        de_maxiter=args.de_maxiter,
        de_popsize=args.de_popsize,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(cfg, run_dir / "config.json")

    t0 = time.perf_counter()
    result = run_bayesopt(cfg, run_dir)
    elapsed = time.perf_counter() - t0

    write_history_csv(result.history, run_dir / "history.csv")
    save_state(result.surrogate, run_dir / "gp_state.joblib")

    verification = verify_optimum(
        result, cfg, run_dir,
        n_neighbors=args.n_verify_neighbors,
        perturbation_frac=args.verify_perturbation_frac,
        seed=args.seed,
    )
    if verification.flagged_as_surrogate_artifact:
        print(
            f"  ({lat:+.4f}, {lon:+.4f}): WARNING a perturbed neighbor beat the reported optimum by "
            f"{verification.max_neighbor_improvement_frac:.2%} -- possible surrogate artifact.", flush=True,
        )

    best = result.best
    _append_summary_row(summary_csv, {
        "lat": lat, "lon": lon,
        "hydrogel_thickness_mm": best.design_vector[0] * 1000.0,
        "vapor_gap_mm": best.design_vector[1] * 1000.0,
        "fin_area_ratio": best.design_vector[3],
        "insulation_gap_mm": args.insulation_gap_mm,
        "tilt_deg": args.tilt_deg,
        "salt": args.salt,
        "salt_loading": args.salt_loading,
        "case": args.case,
        "warmup_method": "aitken-gpu-fixed-round", "resolution": "monthly",
        "best_combined_lcow_usd_m3": f"{best.combined_lcow:.6f}",
        "n_evals": len(result.history),
        "stopped_reason": result.stopped_reason,
        "flagged_as_surrogate_artifact": verification.flagged_as_surrogate_artifact,
        "max_neighbor_improvement_frac": f"{verification.max_neighbor_improvement_frac:.6f}",
        "wall_time_s": f"{elapsed:.1f}",
    })
    print(
        f"  ({lat:+.4f}, {lon:+.4f}): {len(result.history)} eval(s), stopped={result.stopped_reason}, "
        f"best_lcow={best.combined_lcow:.4f} USD/m3, {elapsed:.1f}s", flush=True,
    )


def main() -> int:
    args = parse_args()
    sites = _site_list(args)
    print(f"{len(sites)} site(s) to run.", flush=True)

    summary_csv = args.output_dir / "summary.csv"
    if args.resume:
        done = _existing_sites(summary_csv)
        sites = [(lat, lon) for lat, lon in sites if (round(lat, 6), round(lon, 6)) not in done]
        print(f"{len(sites)} site(s) remaining after --resume.", flush=True)

    bounds = _bounds(args)
    t0 = time.perf_counter()
    for lat, lon in sites:
        run_site(lat, lon, args, bounds, summary_csv)
    print(f"Done: {len(sites)} site(s) in {time.perf_counter() - t0:.1f}s total.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
