"""History CSV, convergence plot, and a final JSON report comparing the
recommended design against Wilson's Table S3 baseline (through the same
two-site pipeline) and, where a genuinely comparable metric exists, the best
point already on disk in solar_lumped's own parameter-sweep outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from sawh_bayesopt.bayesopt import BayesOptConfig, BayesOptResult
from sawh_bayesopt.design_space import VAR_ORDER
from sawh_bayesopt.evaluator import DesignEvalResult, EvalCache, evaluate_batch
from sawh_bayesopt.sites import fetch_monthly_profiles
from sawh_bayesopt.verification import VerificationReport

# .../SAWH_TEAs/sawh_bayesopt/src/sawh_bayesopt/reporting.py -> .../SAWH_TEAs
_SAWH_TEAS_ROOT = Path(__file__).resolve().parents[3]
_SOLAR_LUMPED_SWEEP_DIR = _SAWH_TEAS_ROOT / "solar_lumped" / "outputs" / "parameter_sweeps"


def write_run_config(cfg: BayesOptConfig, path: str | Path) -> None:
    """Dump the BayesOptConfig actually used for a run to disk, so a later
    diagnostics pass (scripts/diagnostics/) doesn't have to have the caller
    re-type n_init/batch_size/seed/bounds from memory to replay a run's
    history in the same order it was originally evaluated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bounds": {name: list(getattr(cfg.bounds, name)) for name in VAR_ORDER},
        "sites": [s.name for s in cfg.sites],
        "combine_rule": cfg.combine_rule,
        "n_init": cfg.n_init,
        "n_total": cfg.n_total,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "ei_xi": cfg.ei_xi,
        "stall_rel_tol": cfg.stall_rel_tol,
        "stall_rounds": cfg.stall_rounds,
        "resolution": cfg.resolution,
    }
    path.write_text(json.dumps(payload, indent=2))


def write_history_csv(history: list[DesignEvalResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    site_names = sorted({r.site_name for res in history for r in res.site_results})
    fieldnames = ["index", *VAR_ORDER, "combined_lcow", "wall_time_s"]
    for name in site_names:
        fieldnames += [
            f"{name}_lcow",
            f"{name}_feasible",
            f"{name}_yield_kg_m2",
            f"{name}_eta_thermal",
            f"{name}_failure_reason",
        ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, res in enumerate(history):
            row: dict[str, object] = {
                "index": i,
                "combined_lcow": res.combined_lcow,
                "wall_time_s": res.wall_time_s,
            }
            row.update(dict(zip(VAR_ORDER, res.design_vector)))
            for r in res.site_results:
                row[f"{r.site_name}_lcow"] = r.lcow
                row[f"{r.site_name}_feasible"] = r.feasible
                row[f"{r.site_name}_yield_kg_m2"] = r.yield_kg_m2
                row[f"{r.site_name}_eta_thermal"] = r.eta_thermal
                row[f"{r.site_name}_failure_reason"] = r.failure_reason
            writer.writerow(row)


def write_convergence_plot(history: list[DesignEvalResult], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lcows = np.array([r.combined_lcow for r in history], dtype=float)
    best_so_far = np.minimum.accumulate(lcows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(lcows)), lcows, "o", alpha=0.3, label="evaluated")
    ax.plot(range(len(lcows)), best_so_far, "-", color="C1", label="best so far")
    ax.set_xlabel("design point index")
    ax.set_ylabel("combined LCOW (USD/m³)")
    ax.set_title("Bayesian optimization convergence")
    ax.legend()
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_baseline(cfg: BayesOptConfig, run_dir: str | Path) -> DesignEvalResult:
    """Wilson Table S3 baseline device (DeviceConfig.baseline()), run through
    the same two-site pipeline, for an apples-to-apples comparison."""
    from solar_lumped.economics.params import LCOEconomicParams
    from solar_lumped.simulation.device_config import DeviceConfig

    run_dir = Path(run_dir)
    econ = LCOEconomicParams()
    site_profiles = {
        s.name: fetch_monthly_profiles(s, cache_dir=cfg.weather_cache_dir) for s in cfg.sites
    }
    cache = EvalCache(run_dir / "cache.jsonl")
    baseline_cfg = DeviceConfig.baseline()
    x_baseline = np.array([getattr(baseline_cfg, name) for name in VAR_ORDER], dtype=float)
    [result] = evaluate_batch(
        [x_baseline],
        cache=cache,
        sites=cfg.sites,
        site_profiles=site_profiles,
        econ=econ,
        combine_rule=cfg.combine_rule,
        resolution=cfg.resolution,
    )
    return result


def _best_sweep_reference() -> dict | None:
    """Best-effort floor from solar_lumped's own sweep outputs, explicitly
    flagged as single-site (not the combined two-site metric this optimizer
    targets) rather than silently treated as comparable."""
    import pandas as pd

    for name in ("full_oat_sweep_cambridge.csv", "full_oat_sweep.csv", "parameter_sweep.csv"):
        path = _SOLAR_LUMPED_SWEEP_DIR / name
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "lcow_usd_per_m3" not in df.columns:
            continue
        df = df[np.isfinite(df["lcow_usd_per_m3"])]
        if df.empty:
            continue
        row = df.loc[df["lcow_usd_per_m3"].idxmin()]
        return {
            "source_file": str(path),
            "note": (
                "Single-site sweep result, not the two-site combined metric this "
                "optimizer targets -- a rough floor, not an apples-to-apples comparison."
            ),
            "lcow_usd_per_m3": float(row["lcow_usd_per_m3"]),
        }
    return None


def write_final_report(
    result: BayesOptResult,
    cfg: BayesOptConfig,
    run_dir: str | Path,
    verification: VerificationReport,
    path: str | Path,
) -> dict:
    baseline_result = evaluate_baseline(cfg, run_dir)
    sweep_ref = _best_sweep_reference()

    baseline_lcow = baseline_result.combined_lcow
    if math.isfinite(baseline_lcow) and baseline_lcow not in (0.0,):
        improvement = (baseline_lcow - result.best.combined_lcow) / baseline_lcow
    else:
        improvement = None

    report = {
        "recommended_design": dict(zip(VAR_ORDER, result.best.design_vector)),
        "recommended_combined_lcow_usd_per_m3": result.best.combined_lcow,
        "recommended_per_site": {
            r.site_name: {
                "lcow_usd_per_m3": r.lcow,
                "feasible": r.feasible,
                "yield_kg_m2": r.yield_kg_m2,
                "eta_thermal": r.eta_thermal,
                "failure_reason": r.failure_reason,
            }
            for r in result.best.site_results
        },
        "verification": {
            "true_combined_lcow_usd_per_m3": verification.best_true_combined_lcow,
            "surrogate_mu": verification.best_surrogate_mu,
            "surrogate_sigma": verification.best_surrogate_sigma,
            "max_neighbor_improvement_frac": verification.max_neighbor_improvement_frac,
            "flagged_as_surrogate_artifact": verification.flagged_as_surrogate_artifact,
        },
        "baseline_wilson_table_s3": {
            "design": dict(zip(VAR_ORDER, baseline_result.design_vector)),
            "combined_lcow_usd_per_m3": baseline_result.combined_lcow,
            "per_site": {
                r.site_name: {"lcow_usd_per_m3": r.lcow, "feasible": r.feasible}
                for r in baseline_result.site_results
            },
        },
        "improvement_vs_baseline_frac": improvement,
        "existing_sweep_reference": sweep_ref,
        "stopped_reason": result.stopped_reason,
        "n_evaluations": len(result.history),
        "caveats": [
            "combined_lcow comes from solar_lumped/gpu_sweep's JAX fast path "
            "(fixed-round-count Aitken, Tsit5), not solar_lumped's CPU ode_system.py "
            "directly -- gpu_sweep/FINDINGS.md documents <0.03% worst-case "
            "disagreement between the two, so this is not expected to be a "
            "meaningfully different physics model.",
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float))
    return report
