#!/usr/bin/env python3
"""BayesOpt at every map point run_gpu_sweep.py would brute-force sweep: same site
selection CLI and 12-month Aitken JAX fast path, optimizing hydrogel_thickness /
fin_area_ratio / vapor_gap over the min/max of the sweep's own combo lists.

Each site writes the same artifacts hp_sweep.py records per combination (history.csv,
convergence.png, gp_state.joblib, de_diagnostics.json, report.json, gp_regression_report
.json, gp_slices.png). A failing stage lands in an "error"/"verify_error"/
"diagnostics_error" field rather than killing this task's remaining sites.

Usage:
    python3 gpu_sweep/run_bayesopt_sweep.py --num-sites 10 --output-dir outputs/gpu_bayesopt_sweep/smoke
    python3 gpu_sweep/run_bayesopt_sweep.py --lat-lon -23.6 -70.4 --output-dir outputs/gpu_bayesopt_sweep/atacama
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
_SCRIPTS = _REPO / "scripts"
_BAYESOPT_REPO = _REPO.parent / "sawh_bayesopt"
_BAYESOPT_SRC = _BAYESOPT_REPO / "src"
_BAYESOPT_DIAG = _BAYESOPT_REPO / "scripts" / "diagnostics"
for p in (_SRC, _SCRIPTS, _BAYESOPT_SRC, _BAYESOPT_DIAG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_lumped import site_sweep as gps  # noqa: E402

from run_gpu_sweep import _site_list  # noqa: E402

import gp_diagnostics  # noqa: E402
from sawh_bayesopt.bayesopt import BayesOptConfig, run_bayesopt  # noqa: E402
from sawh_bayesopt.design_space import CASE_EPS_IR, DesignBounds  # noqa: E402
from sawh_bayesopt.reporting import (  # noqa: E402
    write_convergence_plot,
    write_de_diagnostics,
    write_final_report,
    write_history_csv,
    write_run_config,
)
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

    # Re-evaluates the reported optimum and perturbed neighbors against the true model
    # (same check as run_bayesopt.py) to flag GP surrogate artifacts.
    p.add_argument("--n-verify-neighbors", type=int, default=5)
    p.add_argument("--verify-perturbation-frac", type=float, default=0.10)

    p.add_argument("--output-dir", type=Path, required=True, help="Per-site run dirs (history/config/gp_state) go here.")
    p.add_argument("--resume", action="store_true", help="Skip a site entirely if it's already in --output-dir/summary.csv")
    return p.parse_args(argv)


# to_unit_cube divides by (hi - lo), so an exact lo == hi fixed dim would NaN the whole
# normalization. This span is indistinguishable from the fixed value at any real bound.
_FIXED_DIM_EPS = 1e-9


def _bounds(args: argparse.Namespace) -> DesignBounds:
    """Box bounds for the 3 optimized dims from the sweep's combo lists (min/max, mm -> m);
    the other 3 collapse to a tiny span around the fixed CLI value (see _FIXED_DIM_EPS)."""
    salt_ratio = args.salt_loading
    insulation_m = args.insulation_gap_mm / 1000.0
    tilt = args.tilt_deg
    return DesignBounds(
        hydrogel_thickness_m=(min(args.hydrogel_thickness_mm) / 1000.0, max(args.hydrogel_thickness_mm) / 1000.0),
        vapor_gap_m=(min(args.vapor_gap_mm) / 1000.0, max(args.vapor_gap_mm) / 1000.0),
        insulation_gap_m=(insulation_m, insulation_m + _FIXED_DIM_EPS),
        fin_area_ratio=(min(args.fin_area_ratio), max(args.fin_area_ratio)),
        tilt_deg=(tilt, tilt + _FIXED_DIM_EPS),
        salt_to_polymer_ratio=(salt_ratio, salt_ratio + _FIXED_DIM_EPS),
    )


def _load_summary_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _write_summary_rows(path: Path, rows: list[dict]) -> None:
    """Rewrite the whole file from *rows* each call rather than appending: error rows have
    far fewer columns than full ones, so a locked-in header would crash. At most a few
    thousand rows, so the rewrite is cheap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_site(lat: float, lon: float, args: argparse.Namespace, bounds: DesignBounds) -> dict:
    """One site's full BayesOpt loop plus every diagnostic hp_sweep.py records. Each stage
    past the optimization loop is isolated, so a diagnostics failure neither discards a
    good result nor kills this task's remaining sites."""
    site = SiteSpec(name=f"{lat:+.4f}_{lon:+.4f}", lat=lat, lon=lon, year=args.year)
    run_dir = args.output_dir / site.name
    run_dir.mkdir(parents=True, exist_ok=True)

    row: dict = {"lat": lat, "lon": lon}

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
    write_run_config(cfg, run_dir / "config.json")

    t0 = time.perf_counter()
    try:
        result = run_bayesopt(cfg, run_dir)
    except Exception as exc:  # noqa: BLE001 -- isolate one site's failure from the rest of this task's sites
        row["error"] = f"run_bayesopt: {exc!r}"
        print(f"  ({lat:+.4f}, {lon:+.4f}): ERROR {row['error']}", flush=True)
        return row
    elapsed = time.perf_counter() - t0

    best = result.best
    row.update({
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
        "wall_time_s": f"{elapsed:.1f}",
    })

    write_history_csv(result.history, run_dir / "history.csv")
    write_convergence_plot(result.history, run_dir / "convergence.png")
    save_state(result.surrogate, run_dir / "gp_state.joblib")
    write_de_diagnostics(result.de_diagnostics, run_dir / "diagnostics" / "de_diagnostics.json")

    try:
        verification = verify_optimum(
            result, cfg, run_dir,
            n_neighbors=args.n_verify_neighbors,
            perturbation_frac=args.verify_perturbation_frac,
            seed=args.seed,
        )
        report = write_final_report(result, cfg, run_dir, verification, run_dir / "report.json")
        row["improvement_vs_baseline_frac"] = report["improvement_vs_baseline_frac"]
        row["flagged_as_surrogate_artifact"] = verification.flagged_as_surrogate_artifact
        row["max_neighbor_improvement_frac"] = f"{verification.max_neighbor_improvement_frac:.6f}"
        if verification.flagged_as_surrogate_artifact:
            print(
                f"  ({lat:+.4f}, {lon:+.4f}): WARNING a perturbed neighbor beat the reported optimum by "
                f"{verification.max_neighbor_improvement_frac:.2%} -- possible surrogate artifact.", flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        row["verify_error"] = f"verify_optimum/write_final_report: {exc!r}"

    try:
        gp_diagnostics.main(["--run-dir", str(run_dir), "--seed", str(args.seed)])
        gp_report = json.loads((run_dir / "diagnostics" / "gp_regression_report.json").read_text())
        cv = gp_report["cross_validation"]
        row["cv_rmse"] = cv["cv_rmse"]
        row["standardized_residual_mean"] = cv["standardized_residual_mean"]
        row["standardized_residual_std"] = cv["standardized_residual_std"]
        row["msll_gp_minus_trivial"] = cv["msll_gp_minus_trivial"]
        row["n_hyperparameter_warnings"] = len(gp_report["hyperparameter_convergence_warnings"])
        de_summary = gp_report.get("de_diagnostics_summary")
        if de_summary and de_summary.get("n_de_calls"):
            row["n_de_calls"] = de_summary["n_de_calls"]
            row["frac_de_hit_maxiter"] = de_summary["frac_hit_maxiter"]
            row["frac_de_not_success"] = de_summary["frac_not_success"]
    except Exception as exc:  # noqa: BLE001
        row["diagnostics_error"] = f"gp_diagnostics: {exc!r}"

    print(
        f"  ({lat:+.4f}, {lon:+.4f}): {len(result.history)} eval(s), stopped={result.stopped_reason}, "
        f"best_lcow={best.combined_lcow:.4f} USD/m3, {elapsed:.1f}s", flush=True,
    )
    return row


def main() -> int:
    args = parse_args()
    sites = _site_list(args)
    print(f"{len(sites)} site(s) to run.", flush=True)

    summary_csv = args.output_dir / "summary.csv"
    rows = _load_summary_rows(summary_csv)
    if args.resume:
        done = {(round(float(r["lat"]), 6), round(float(r["lon"]), 6)) for r in rows}
        sites = [(lat, lon) for lat, lon in sites if (round(lat, 6), round(lon, 6)) not in done]
        print(f"{len(sites)} site(s) remaining after --resume.", flush=True)

    bounds = _bounds(args)
    t0 = time.perf_counter()
    for lat, lon in sites:
        rows.append(run_site(lat, lon, args, bounds))
        _write_summary_rows(summary_csv, rows)  # rewritten after every site -- see _write_summary_rows
    print(f"Done: {len(sites)} site(s) in {time.perf_counter() - t0:.1f}s total.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
