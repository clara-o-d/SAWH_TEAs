#!/usr/bin/env python3
"""Grid sweep over ei_xi x stall_rel_tol x n_init (3x3x3 by default), running a full
BayesOpt loop + gp_diagnostics.py per combination and aggregating the results.

Motivation: a baseline run stalled after 33-39 evaluations with EI points clustered on the
incumbent. This asks whether more exploration bonus (ei_xi), a different stall rule
(stall_rel_tol), or more LHS coverage (n_init) changes search quality.

n_total is not held fixed: it is n_init + --bo-budget, so every combo gets the same EI
budget regardless of n_init -- otherwise "more LHS coverage" is confounded with "less EI
budget". All combos share one seed, isolating each hyperparameter's effect.

Each combo writes its own outputs/runs/<sweep-id>/<combo-tag>/ with the same artifacts
run_bayesopt.py's CLI produces, plus gp_diagnostics run in-process; the sweep root gets
sweep_results.csv/.json with one row per combo. A failing combo gets an "error" field
rather than being dropped.

GPU sharing: combos run in fresh subprocesses (max_tasks_per_child=1) because JAX caches
its GPU backend per process, so a reused worker would silently ignore the next combo's
CUDA_VISIBLE_DEVICES / XLA_PYTHON_CLIENT_MEM_FRACTION. Those are set before any
sawh_bayesopt/solar_lumped import (jax loads lazily), letting several combos share a GPU
without one preallocating 90% of its memory.

Usage (full sweep, 4 workers on 1 GPU):
    python3 scripts/hp_sweep.py --sweep-id hp_sweep_1 --n-workers 4 --gpu-ids 0 \\
        --weather-cache-dir ../solar_lumped/.weather_cache

Usage (smoke test -- see docs/HP_SWEEP_RUNBOOK.md):
    python3 scripts/hp_sweep.py --sweep-id hp_sweep_smoke \\
        --ei-xi-values 0.02,0.1 --stall-rel-tol-values 0.005 --n-init-values 8 \\
        --bo-budget 4 --batch-size 2 --sites cambridge \\
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
    p.add_argument("--case", choices=("case1", "case2", "case3"), default="case2")
    p.add_argument("--weather-cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument(
        "--resume", action="store_true",
        help="Skip any combination whose run_dir already has a report.json and "
        "gp_regression_report.json (reconstructing its row from those files instead of "
        "re-running it) -- lets a sweep that hit a SLURM time limit partway through be "
        "resubmitted with the same --sweep-id and pick up where it left off, instead of "
        "burning GPU time re-doing already-finished combinations.",
    )
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
    """Runs in its own spawn/max_tasks_per_child=1 worker process (see the module
    docstring's GPU-sharing note). Never raises -- failures land in the "error" field."""
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


def _load_completed_row(task: dict) -> dict | None:
    """Reconstruct a combo's row from a prior run's outputs if both report.json and
    gp_regression_report.json exist (the last two files written). None otherwise, so the
    caller re-runs the combination."""
    run_dir = Path(task["run_dir"])
    report_path = run_dir / "report.json"
    gp_report_path = run_dir / "diagnostics" / "gp_regression_report.json"
    if not (report_path.is_file() and gp_report_path.is_file()):
        return None
    try:
        report = json.loads(report_path.read_text())
        gp_report = json.loads(gp_report_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    row: dict = {
        "combo_tag": task["combo_tag"],
        "ei_xi": task["ei_xi"],
        "stall_rel_tol": task["stall_rel_tol"],
        "n_init": task["n_init"],
        "n_total": task["n_total"],
        "run_dir": str(run_dir.relative_to(_REPO)),
        "resumed_from_prior_run": True,
        "n_evaluations": report.get("n_evaluations"),
        "best_combined_lcow_usd_per_m3": report.get("recommended_combined_lcow_usd_per_m3"),
        "stopped_reason": report.get("stopped_reason"),
        "improvement_vs_baseline_frac": report.get("improvement_vs_baseline_frac"),
    }
    cv = gp_report.get("cross_validation", {})
    row["cv_rmse"] = cv.get("cv_rmse")
    row["standardized_residual_mean"] = cv.get("standardized_residual_mean")
    row["standardized_residual_std"] = cv.get("standardized_residual_std")
    row["msll_gp_minus_trivial"] = cv.get("msll_gp_minus_trivial")
    row["n_hyperparameter_warnings"] = len(gp_report.get("hyperparameter_convergence_warnings", []))
    de_summary = gp_report.get("de_diagnostics_summary")
    if de_summary and de_summary.get("n_de_calls"):
        row["n_de_calls"] = de_summary["n_de_calls"]
        row["frac_de_hit_maxiter"] = de_summary["frac_hit_maxiter"]
        row["frac_de_not_success"] = de_summary["frac_not_success"]
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
            "case": args.case,
            "weather_cache_dir": args.weather_cache_dir,
            "gpu_id": gpu_ids[i % len(gpu_ids)] if gpu_ids else None,
            "mem_fraction": mem_fraction,
        })

    import multiprocessing

    rows: list[dict] = []
    to_run = tasks
    if args.resume:
        to_run = []
        for t in tasks:
            resumed = _load_completed_row(t)
            if resumed is None:
                to_run.append(t)
                continue
            rows.append(resumed)
            print(
                f"  [resumed] {t['combo_tag']}: best={resumed.get('best_combined_lcow_usd_per_m3', float('nan')):.4g} "
                f"stopped={resumed.get('stopped_reason')} n_eval={resumed.get('n_evaluations')}",
                flush=True,
            )
        if rows:
            print(f"Resumed {len(rows)}/{len(tasks)} already-complete combination(s); running the remaining {len(to_run)}.", flush=True)

    order = {t["combo_tag"]: i for i, t in enumerate(tasks)}
    json_path = sweep_dir / "sweep_results.json"
    csv_path = sweep_dir / "sweep_results.csv"

    def _write_outputs() -> None:
        """Re-writes both output files from *rows* so far -- called after every
        single combination (resumed or freshly completed), not just once at
        the end, so a sweep killed partway through (SLURM time limit, node
        failure) still leaves an inspectable, up-to-date summary of whatever
        finished, instead of nothing at all until the very last combination lands.
        27 rows is small enough that a full rewrite each time is negligible cost.
        """
        ordered = sorted(rows, key=lambda r: order[r["combo_tag"]])
        json_path.write_text(json.dumps(ordered, indent=2))
        fieldnames: list[str] = []
        for r in ordered:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered)

    if rows:
        _write_outputs()

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {pool.submit(_run_one_combo, t): t["combo_tag"] for t in to_run}
        for fut in as_completed(futures):
            tag = futures[fut]
            row = fut.result()
            rows.append(row)
            _write_outputs()
            status = "ERROR: " + row["error"] if "error" in row else (
                f"best={row.get('best_combined_lcow_usd_per_m3', float('nan')):.4g} "
                f"stopped={row.get('stopped_reason')} n_eval={row.get('n_evaluations')}"
            )
            print(f"  [{len(rows)}/{len(tasks)}] {tag}: {status}", flush=True)

    n_errors = sum(1 for r in rows if "error" in r)
    print(f"\nWrote {csv_path} and {json_path} ({len(rows)} rows, {n_errors} error(s)).", flush=True)
    return 1 if n_errors == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
