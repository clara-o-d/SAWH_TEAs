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
from sawh_bayesopt.design_space import DesignBounds  # noqa: E402
from sawh_bayesopt.reporting import (  # noqa: E402
    write_convergence_plot,
    write_final_report,
    write_history_csv,
    write_run_config,
)
from sawh_bayesopt.sites import ATACAMA, CAMBRIDGE, DEFAULT_SITES  # noqa: E402
from sawh_bayesopt.surrogate import save_state  # noqa: E402
from sawh_bayesopt.verification import verify_optimum  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-init", type=int, default=24)
    p.add_argument("--n-total", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ei-xi", type=float, default=0.01)
    p.add_argument("--combine-rule", choices=("mean", "worst_case"), default="mean")
    p.add_argument("--sites", choices=("both", "cambridge", "atacama"), default="both")
    p.add_argument("--resolution", choices=("monthly", "single"), default="monthly")
    p.add_argument("--weather-cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--run-id", type=str, default="run")
    p.add_argument("--n-verify-neighbors", type=int, default=5)
    p.add_argument("--verify-perturbation-frac", type=float, default=0.10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sites = {
        "both": DEFAULT_SITES,
        "cambridge": (CAMBRIDGE,),
        "atacama": (ATACAMA,),
    }[args.sites]

    cfg = BayesOptConfig(
        bounds=DesignBounds(),
        sites=sites,
        combine_rule=args.combine_rule,
        n_init=args.n_init,
        n_total=args.n_total,
        batch_size=args.batch_size,
        seed=args.seed,
        ei_xi=args.ei_xi,
        resolution=args.resolution,
        weather_cache_dir=args.weather_cache_dir,
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

    write_history_csv(result.history, run_dir / "history.csv")
    write_convergence_plot(result.history, run_dir / "convergence.png")
    save_state(result.surrogate, run_dir / "gp_state.joblib")

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
