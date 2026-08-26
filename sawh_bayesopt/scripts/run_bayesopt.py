#!/usr/bin/env python3
"""CLI entry point for the sawh_bayesopt EGO loop: LHS init -> batched
Expected-Improvement infill against solar_lumped's true model -> verification
-> report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sawh_bayesopt.bayesopt import BayesOptConfig, run_bayesopt  # noqa: E402
from sawh_bayesopt.design_space import CASE_EPS_IR, DesignBounds  # noqa: E402
from sawh_bayesopt.reporting import (  # noqa: E402
    write_convergence_plot,
    write_de_diagnostics,
    write_final_report,
    write_history_csv,
    write_run_config,
)
from sawh_bayesopt.sites import ATACAMA, STANFORD, site_from_lat_lon  # noqa: E402
from sawh_bayesopt.surrogate import save_state  # noqa: E402
from sawh_bayesopt.verification import verify_optimum  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-init", type=int, default=24)
    p.add_argument("--n-total", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ei-xi", type=float, default=0.01)
    p.add_argument("--stall-rel-tol", type=float, default=0.005)
    p.add_argument("--stall-rounds", type=int, default=3)

    # Site selection: one validated field site by name, or one arbitrary coordinate.
    # Optimization is single-site (evaluator.site_lcow_or_penalty), so there is no
    # multi-site or land-grid option here -- sweeping the grid means one optimization
    # per site, which is gpu_sweep/run_bayesopt_sweep.py's job.
    site = p.add_mutually_exclusive_group()
    site.add_argument("--site", choices=("atacama", "stanford"), default="atacama")
    site.add_argument(
        "--lat-lon", type=float, nargs=2, metavar=("LAT", "LON"),
        help="Optimize at these coordinates instead of a named site.",
    )
    p.add_argument("--year", type=int, default=2024)

    p.add_argument(
        "--complex", action="store_true",
        help="Run the complex-fidelity model (A1/B1/B2/B3/B4/B8): 13 design dims on "
             "solar_lumped's CPU path instead of 5 on the JAX fast path.",
    )
    p.add_argument(
        "--backend", choices=("jax", "cpu"), default="jax",
        help="jax: vmapped/jitted, batches designs x sites -- use for global sweeps. "
             "cpu: sequential solar_lumped ODE path, no GPU stack, single site only.",
    )
    p.add_argument(
        "--day-stride", type=int, default=1,
        help="Evaluate every Nth calendar day instead of all 365. The year is ~366 "
             "sequential day-steps and ~100%% of an evaluation's cost, so this is the one "
             "lever that shortens a call (stride 5 ~= 5x cheaper). It is a DIFFERENT "
             "objective, not an approximation of the same number -- results are not "
             "comparable to stride-1 runs, and it gets its own cache key.",
    )
    p.add_argument("--case", choices=tuple(CASE_EPS_IR), default="case2")
    p.add_argument("--weather-cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--run-id", type=str, default="run")
    p.add_argument("--n-verify-neighbors", type=int, default=5)
    p.add_argument("--verify-perturbation-frac", type=float, default=0.10)
    return p.parse_args(argv)


def resolve_sites(args: argparse.Namespace) -> tuple:
    """The one site to optimize at, as a 1-tuple (BayesOptConfig.sites' shape)."""
    if args.lat_lon:
        lat, lon = args.lat_lon
        return (site_from_lat_lon(lat, lon, year=args.year),)
    return {"atacama": (ATACAMA,), "stanford": (STANFORD,)}[args.site]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sites = resolve_sites(args)

    cfg = BayesOptConfig(
        bounds=DesignBounds(complex_mode=args.complex),
        complex_mode=args.complex,
        backend=args.backend,
        sites=sites,
        n_init=args.n_init,
        n_total=args.n_total,
        batch_size=args.batch_size,
        seed=args.seed,
        ei_xi=args.ei_xi,
        stall_rel_tol=args.stall_rel_tol,
        stall_rounds=args.stall_rounds,
        weather_cache_dir=args.weather_cache_dir,
        case=args.case,
        day_stride=args.day_stride,
    )

    run_dir = _REPO / "outputs" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(cfg, run_dir / "config.json")

    print(
        f"Running BayesOpt: n_init={cfg.n_init} n_total={cfg.n_total} "
        f"sites={[s.name for s in cfg.sites]}",
        flush=True,
    )
    result = run_bayesopt(cfg, run_dir)
    print(f"Stopped: {result.stopped_reason} after {len(result.history)} design points.", flush=True)
    print(f"Best combined LCOW: {result.best.combined_lcow:.4f} USD/m3", flush=True)

    write_history_csv(result.history, run_dir / "history.csv", var_order=cfg.bounds.names())
    write_convergence_plot(result.history, run_dir / "convergence.png")
    save_state(result.surrogate, run_dir / "gp_state.joblib")
    write_de_diagnostics(result.de_diagnostics, run_dir / "diagnostics" / "de_diagnostics.json")
    n_de = len(result.de_diagnostics)
    n_hit_maxiter = sum(1 for d in result.de_diagnostics if d["hit_maxiter"])
    n_not_success = sum(1 for d in result.de_diagnostics if not d["success"])
    if n_de:
        print(
            f"EI proposal DE calls: {n_de} total, {n_hit_maxiter} hit maxiter, "
            f"{n_not_success} not marked success.",
            flush=True,
        )

    print("Verifying optimum against the true model...", flush=True)
    verification = verify_optimum(
        result,
        cfg,
        run_dir,
        n_neighbors=args.n_verify_neighbors,
        perturbation_frac=args.verify_perturbation_frac,
        seed=args.seed,
    )
    if verification.flagged_as_surrogate_artifact:
        print(
            f"WARNING: a perturbed neighbor beat the reported optimum by "
            f"{verification.max_neighbor_improvement_frac:.2%} -- possible surrogate artifact.",
            flush=True,
        )

    report = write_final_report(result, cfg, run_dir, verification, run_dir / "report.json")
    print(f"Report written to {run_dir / 'report.json'}", flush=True)
    if report["improvement_vs_baseline_frac"] is not None:
        print(
            f"Improvement vs Wilson Table S3 baseline: {report['improvement_vs_baseline_frac']:.2%}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
