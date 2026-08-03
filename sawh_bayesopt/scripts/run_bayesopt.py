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
from sawh_bayesopt.sites import (  # noqa: E402
    ATACAMA,
    CAMBRIDGE,
    DEFAULT_SITES,
    land_grid_sites,
    site_from_lat_lon,
)
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
    p.add_argument("--combine-rule", choices=("mean", "worst_case"), default="mean")

    # Site selection: the two validated field sites by name, arbitrary coordinates,
    # or indices into solar_lumped's land grid (the same grid the gpu_sweep global
    # sweep uses, so runs stay comparable).
    site = p.add_mutually_exclusive_group()
    site.add_argument("--sites", choices=("both", "cambridge", "atacama"), default="both")
    site.add_argument(
        "--lat-lon", type=float, nargs=2, action="append", metavar=("LAT", "LON"),
        help="Optimize at these coordinates. Repeatable for a multi-site design.",
    )
    site.add_argument("--num-sites", type=int, help="First N sites of the --step land grid.")
    site.add_argument("--site-indices", type=int, nargs="+", help="Indices into the --step land grid.")
    p.add_argument("--step", type=float, default=3.0, help="Land-grid spacing (deg).")
    p.add_argument("--year", type=int, default=2024)

    p.add_argument(
        "--complex", action="store_true",
        help="Run the complex-fidelity model (A1/B1/B2/B3/B4/B8): 13 design dims on "
             "solar_lumped's CPU path instead of 6 on the JAX fast path.",
    )
    p.add_argument(
        "--backend", choices=("jax", "cpu"), default="jax",
        help="jax: vmapped/jitted, batches designs x sites -- use for global sweeps. "
             "cpu: sequential solar_lumped ODE path, no GPU stack -- use for one site.",
    )
    p.add_argument("--case", choices=tuple(CASE_EPS_IR), default="case2")
    p.add_argument("--weather-cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--run-id", type=str, default="run")
    p.add_argument("--n-verify-neighbors", type=int, default=5)
    p.add_argument("--verify-perturbation-frac", type=float, default=0.10)
    return p.parse_args(argv)


def resolve_sites(args: argparse.Namespace) -> tuple:
    """Turn the mutually exclusive site flags into a concrete SiteSpec tuple."""
    if args.lat_lon:
        return tuple(
            site_from_lat_lon(lat, lon, year=args.year) for lat, lon in args.lat_lon
        )
    if args.num_sites is not None:
        return land_grid_sites(step_deg=args.step, indices=list(range(args.num_sites)), year=args.year)
    if args.site_indices is not None:
        return land_grid_sites(step_deg=args.step, indices=args.site_indices, year=args.year)
    return {"both": DEFAULT_SITES, "cambridge": (CAMBRIDGE,), "atacama": (ATACAMA,)}[args.sites]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sites = resolve_sites(args)

    cfg = BayesOptConfig(
        bounds=DesignBounds(complex_mode=args.complex),
        complex_mode=args.complex,
        backend=args.backend,
        sites=sites,
        combine_rule=args.combine_rule,
        n_init=args.n_init,
        n_total=args.n_total,
        batch_size=args.batch_size,
        seed=args.seed,
        ei_xi=args.ei_xi,
        stall_rel_tol=args.stall_rel_tol,
        stall_rounds=args.stall_rounds,
        weather_cache_dir=args.weather_cache_dir,
        case=args.case,
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
