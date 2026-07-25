#!/usr/bin/env python3
"""Grid sweep over ei_xi x stall_rel_tol x n_init, running a full BayesOpt
loop + gp_diagnostics.py for every combination and aggregating the results.

Motivation: a single sawh_bayesopt run (outputs/runs/sherlock_gpu_run_1)
stalled after 33-39 evaluations with its EI-proposed points clustering
tightly around the incumbent -- exploitation converging quickly under
ei_xi=0.01 (small EI exploration bonus) and a stall rule (stall_rel_tol,
stall_rounds) that can stop the search early. This sweep asks: does raising
ei_xi (more exploration bonus), loosening/tightening stall_rel_tol (when the
loop gives up), or growing n_init (more guaranteed LHS coverage before EI
even starts) change search quality, and by how much?

Grid (3 x 3 x 3 = 27 combinations by default):
  - ei_xi: only values *greater* than the baseline run's 0.01 (per request) --
    default (0.02, 0.05, 0.1).
  - stall_rel_tol: both greater and lesser than the baseline's 0.005 --
    default (0.001, 0.005, 0.02).
  - n_init: only values *greater* than the baseline's 24 -- default
    (30, 36, 42). n_total is NOT held fixed across these -- it's set to
    n_init + --bo-budget (default 26, matching the baseline's implied
    50-24=26 post-init evaluations) so every combo gets the same *EI-based*
    evaluation budget regardless of n_init. Holding n_total fixed instead
    would confound "more LHS coverage" with "less EI budget", making higher
    n_init look worse for a reason that has nothing to do with LHS coverage.

Every combination reuses the *same* seed (--seed, default 0) -- the point is
isolating each hyperparameter's effect, not adding RNG-driven variance on
top of it.

For each combination, in its own outputs/runs/<sweep-id>/<combo-tag>/:
  - Runs the full BayesOpt loop (sawh_bayesopt.bayesopt.run_bayesopt) exactly
    as scripts/run_bayesopt.py's CLI does (same write_run_config/
    write_history_csv/write_convergence_plot/write_de_diagnostics/
    verify_optimum/write_final_report sequence).
  - Runs scripts/diagnostics/gp_diagnostics.py's main() against that run_dir
    (in-process, not a subprocess) -- this is what actually applies the
    updated result.success/result.nit-vs-maxiter and standardized-residual/
    MSLL checks to every combination, not just the run that originally
    prompted this sweep.

Then writes, in outputs/runs/<sweep-id>/:
  - sweep_results.csv / sweep_results.json: one row per combination with its
    hyperparameters, best LCOW found, stopped_reason, wall-clock, and the key
    gp_regression_report.json / de_diagnostics.json fields.
  - Nothing is silently dropped: a combination whose run_bayesopt/
    verify_optimum/gp_diagnostics step raises gets an "error" field instead
    of being skipped, and still appears as a row.

Runs combinations in parallel via a fresh subprocess per combination
(ProcessPoolExecutor, max_tasks_per_child=1 -- see the GPU-sharing note
below), across --n-workers workers spread over --gpu-ids.

GPU sharing: JAX initializes its GPU backend once per process and then
caches it -- if a worker process were reused for a second combination with a
different intended GPU/memory-fraction assignment, that second assignment
would silently never take effect. max_tasks_per_child=1 forces a brand new
process (and therefore a fresh, correctly-configured JAX backend) for every
single combination, at the cost of one process-spawn per combination (this
sweep's per-combination unit of work is minutes, not milliseconds, so that
overhead is negligible). Each combination's process sets
CUDA_VISIBLE_DEVICES to its assigned GPU and
XLA_PYTHON_CLIENT_MEM_FRACTION=1/n_workers *before* importing anything from
sawh_bayesopt/solar_lumped (which only imports jax lazily, inside
evaluate_batch's first call) -- so several combinations can safely share one
GPU's memory without one process preallocating 90% of it and starving the
rest (JAX's default behavior absent this env var).

Usage (full sweep, 4 workers on 1 GPU):
    python3 scripts/hp_sweep.py --sweep-id hp_sweep_1 --n-workers 4 --gpu-ids 0 \\
        --weather-cache-dir ../solar_lumped/.weather_cache

Usage (smoke test -- see docs/HP_SWEEP_RUNBOOK.md):
    python3 scripts/hp_sweep.py --sweep-id hp_sweep_smoke \\
        --ei-xi-values 0.02,0.1 --stall-rel-tol-values 0.005 --n-init-values 8 \\
        --bo-budget 4 --batch-size 2 --sites cambridge --resolution single \\
        --n-workers 2 --gpu-ids 0 --weather-cache-dir ../solar_lumped/.weather_cache
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
_DIAG_DIR = Path(__file__).resolve().parent / "diagnostics"


def _parse_float_list(s: str) -> list[float]:
    return [float(v) for v in s.split(",") if v.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-id", type=str, required=True, help="Written to outputs/runs/<sweep-id>/.")
    p.add_argument("--ei-xi-values", type=_parse_float_list, default=[0.02, 0.05, 0.1])
    p.add_argument("--stall-rel-tol-values", type=_parse_float_list, default=[0.001, 0.005, 0.02])
    p.add_argument("--n-init-values", type=_parse_int_list, default=[30, 36, 42])
    p.add_argument("--bo-budget", type=int, default=26, help="n_total = n_init + this, for every combination.")
    p.add_argument("--stall-rounds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="Shared across every combination on purpose -- see module docstring.")
    p.add_argument("--combine-rule", choices=("mean", "worst_case"), default="mean")
    p.add_argument("--sites", choices=("both", "cambridge", "atacama"), default="both")
    p.add_argument("--resolution", choices=("monthly", "single"), default="monthly")
    p.add_argument("--case", choices=("case1", "case2", "case3"), default="case1")
    p.add_argument("--weather-cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument(
        "--gpu-ids", type=str, default="",
        help="Comma-separated GPU indices to round-robin combinations across (e.g. '0,1'). "
        "Empty (default) means don't touch CUDA_VISIBLE_DEVICES/XLA_PYTHON_CLIENT_MEM_FRACTION "
        "at all -- appropriate for a CPU-only smoke test or a single pre-pinned GPU.",
    )
    return p.parse_args(argv)


def _combo_tag(ei_xi: float, stall_rel_tol: float, n_init: int) -> str:
    return f"eixi{ei_xi:g}_stall{stall_rel_tol:g}_ninit{n_init}"


def _run_one_combo(task: dict) -> dict:
    """Runs in its own (spawn, max_tasks_per_child=1) worker process -- see
    module docstring's GPU-sharing note for why that matters. Never raises:
    any failure at any stage is captured into the returned dict's "error"
    field instead of killing the whole sweep.
    """
    if task["gpu_id"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(task["gpu_id"])
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(task["mem_fraction"])

    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    if str(_DIAG_DIR) not in sys.path:
        sys.path.insert(0, str(_DIAG_DIR))

    import gp_diagnostics
    from sawh_bayesopt.bayesopt import BayesOptConfig, run_bayesopt
    from sawh_bayesopt.design_space import DesignBounds
    from sawh_bayesopt.reporting import (
        write_convergence_plot,
        write_de_diagnostics,
        write_final_report,
        write_history_csv,
        write_run_config,
    )
    from sawh_bayesopt.sites import ATACAMA, CAMBRIDGE, DEFAULT_SITES
    from sawh_bayesopt.surrogate import save_state
    from sawh_bayesopt.verification import verify_optimum

    sites = {"both": DEFAULT_SITES, "cambridge": (CAMBRIDGE,), "atacama": (ATACAMA,)}[task["sites"]]
    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    row: dict = {
        "combo_tag": task["combo_tag"],
        "ei_xi": task["ei_xi"],
        "stall_rel_tol": task["stall_rel_tol"],
        "n_init": task["n_init"],
        "n_total": task["n_total"],
        "run_dir": str(run_dir.relative_to(_REPO)),
    }

    cfg = BayesOptConfig(
        bounds=DesignBounds(),
        sites=sites,
        combine_rule=task["combine_rule"],
        n_init=task["n_init"],
        n_total=task["n_total"],
        batch_size=task["batch_size"],
        seed=task["seed"],
        ei_xi=task["ei_xi"],
        stall_rel_tol=task["stall_rel_tol"],
        stall_rounds=task["stall_rounds"],
        resolution=task["resolution"],
        weather_cache_dir=task["weather_cache_dir"],
        case=task["case"],
    )
    write_run_config(cfg, run_dir / "config.json")

    t0 = time.perf_counter()
    try:
        result = run_bayesopt(cfg, run_dir)
    except Exception as exc:  # noqa: BLE001 -- isolate one combo's failure from the rest of the sweep
        row["error"] = f"run_bayesopt: {exc!r}"
        return row
    row["wall_time_s"] = time.perf_counter() - t0
    row["stopped_reason"] = result.stopped_reason
    row["n_evaluations"] = len(result.history)
    row["best_combined_lcow_usd_per_m3"] = result.best.combined_lcow

    write_history_csv(result.history, run_dir / "history.csv")
    write_convergence_plot(result.history, run_dir / "convergence.png")
    save_state(result.surrogate, run_dir / "gp_state.joblib")
    write_de_diagnostics(result.de_diagnostics, run_dir / "diagnostics" / "de_diagnostics.json")

    try:
        verification = verify_optimum(result, cfg, run_dir, seed=task["seed"])
        report = write_final_report(result, cfg, run_dir, verification, run_dir / "report.json")
        row["improvement_vs_baseline_frac"] = report["improvement_vs_baseline_frac"]
        row["flagged_as_surrogate_artifact"] = verification.flagged_as_surrogate_artifact
    except Exception as exc:  # noqa: BLE001
        row["verify_error"] = f"verify_optimum/write_final_report: {exc!r}"

    try:
        gp_diagnostics.main(["--run-dir", str(run_dir)])
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

    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sweep_dir = _REPO / "outputs" / "runs" / args.sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    mem_fraction = 1.0 / max(1, args.n_workers)

    combos = list(itertools.product(args.ei_xi_values, args.stall_rel_tol_values, args.n_init_values))
    print(f"Sweeping {len(combos)} combinations into {sweep_dir}, {args.n_workers} worker(s)...", flush=True)

    tasks = []
    for i, (ei_xi, stall_rel_tol, n_init) in enumerate(combos):
        tag = _combo_tag(ei_xi, stall_rel_tol, n_init)
        tasks.append({
            "combo_tag": tag,
            "run_dir": str(sweep_dir / tag),
            "ei_xi": ei_xi,
            "stall_rel_tol": stall_rel_tol,
            "n_init": n_init,
            "n_total": n_init + args.bo_budget,
            "stall_rounds": args.stall_rounds,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "combine_rule": args.combine_rule,
            "sites": args.sites,
            "resolution": args.resolution,
            "case": args.case,
            "weather_cache_dir": args.weather_cache_dir,
            "gpu_id": gpu_ids[i % len(gpu_ids)] if gpu_ids else None,
            "mem_fraction": mem_fraction,
        })

    import multiprocessing

    rows: list[dict] = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {pool.submit(_run_one_combo, t): t["combo_tag"] for t in tasks}
        for fut in as_completed(futures):
            tag = futures[fut]
            row = fut.result()
            rows.append(row)
            status = "ERROR: " + row["error"] if "error" in row else (
                f"best={row.get('best_combined_lcow_usd_per_m3', float('nan')):.4g} "
                f"stopped={row.get('stopped_reason')} n_eval={row.get('n_evaluations')}"
            )
            print(f"  [{len(rows)}/{len(tasks)}] {tag}: {status}", flush=True)

    order = {t["combo_tag"]: i for i, t in enumerate(tasks)}
    rows.sort(key=lambda r: order[r["combo_tag"]])

    json_path = sweep_dir / "sweep_results.json"
    json_path.write_text(json.dumps(rows, indent=2))

    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    csv_path = sweep_dir / "sweep_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_errors = sum(1 for r in rows if "error" in r)
    print(f"\nWrote {csv_path} and {json_path} ({len(rows)} rows, {n_errors} error(s)).", flush=True)
    return 1 if n_errors == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
