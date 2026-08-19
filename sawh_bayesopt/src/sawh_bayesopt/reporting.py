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
from sawh_bayesopt.evaluator import DesignEvalResult, EvalCache, evaluate_for_config
from sawh_bayesopt.verification import VerificationReport

# .../SAWH_TEAs/sawh_bayesopt/src/sawh_bayesopt/reporting.py -> .../SAWH_TEAs
_SAWH_TEAS_ROOT = Path(__file__).resolve().parents[3]
_SOLAR_LUMPED_SWEEP_DIR = _SAWH_TEAS_ROOT / "solar_lumped" / "outputs" / "parameter_sweeps"


def write_run_config(cfg: BayesOptConfig, path: str | Path) -> None:
    """Dump the BayesOptConfig actually used for a run to disk, so a later
    diagnostics pass (../analysis/performance/optimization/diagnostics_bo/) doesn't have to have the caller
    re-type n_init/batch_size/seed/bounds from memory to replay a run's
    history in the same order it was originally evaluated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bounds": {name: list(getattr(cfg.bounds, name)) for name in cfg.bounds.names()},
        "sites": [s.name for s in cfg.sites],
        "n_init": cfg.n_init,
        "n_total": cfg.n_total,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "ei_xi": cfg.ei_xi,
        "stall_rel_tol": cfg.stall_rel_tol,
        "stall_rounds": cfg.stall_rounds,
        "case": cfg.case,
        # Part of what the run optimized, not a tuning detail: a strided run's LCOW is a
        # different objective and its numbers are not comparable to a stride-1 run's.
        "day_stride": cfg.day_stride,
    }
    path.write_text(json.dumps(payload, indent=2))


def write_de_diagnostics(de_diagnostics: list[dict], path: str | Path) -> None:
    """Dump every EI-proposal round's differential_evolution result summary
    (success/nit/maxiter -- see acquisition.propose_next's `record` param) so
    ../analysis/performance/optimization/diagnostics_bo/gp_diagnostics.py can flag rounds where DE hit
    maxiter without its own convergence tolerance being satisfied, separately
    from the LCOW-GP calibration checks it already does."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(de_diagnostics, indent=2))


def write_history_csv(
    history: list[DesignEvalResult], path: str | Path, *, var_order=VAR_ORDER
) -> None:
    """``var_order`` is the run's design order -- 6 names simple, 13 complex."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    site_names = sorted({r.site_name for res in history for r in res.site_results})
    fieldnames = ["index", *var_order, "combined_lcow", "wall_time_s"]
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
            row.update(dict(zip(var_order, res.design_vector)))
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


def baseline_design_vector(cfg: BayesOptConfig) -> np.ndarray:
    """Wilson Table S3's system expressed in this run's design order.

    In complex mode the extra dimensions come from ComplexOptions' own defaults, which are
    exactly the simple model -- so the baseline stays Wilson's system, not a partly-complex
    hybrid. Site-independent: the same design is what every site is compared against.
    """
    from solar_lumped.complex_model import ComplexOptions
    from solar_lumped.simulation import SystemConfig

    from sawh_bayesopt.design_space import GLAZING_CONFIGS

    baseline_cfg = SystemConfig.baseline()
    defaults = ComplexOptions()
    baseline_extra = {
        "eps_abs_ir": defaults.eps_abs_ir,
        "glazing_config": float(GLAZING_CONFIGS.index((defaults.n_glazing_panes, defaults.evacuated_gap))),
        "condenser_air_speed_m_s": defaults.condenser_air_speed_m_s,
        "seal_offset_h": defaults.seal_offset_h,
        "open_offset_h": defaults.open_offset_h,
        "blend_u": 1.0,  # stick-breaking coords for pure LiCl
        "blend_v": 0.0,
    }
    return np.array(
        [
            baseline_extra[name] if name in baseline_extra else getattr(baseline_cfg, name)
            for name in cfg.bounds.names()
        ],
        dtype=float,
    )


def evaluate_baseline(cfg: BayesOptConfig, run_dir: str | Path) -> DesignEvalResult:
    """One site's Wilson Table S3 baseline. Wrapper over :func:`evaluate_baselines`."""
    if len(cfg.sites) != 1:
        raise ValueError(f"evaluate_baseline is single-site, got {len(cfg.sites)}; use evaluate_baselines")
    name = cfg.sites[0].name
    return evaluate_baselines(cfg, {name: Path(run_dir)})[name]


def evaluate_baselines(
    cfg: BayesOptConfig,
    run_dirs: dict[str, Path],
    *,
    site_inputs: tuple[dict, dict[str, float]] | None = None,
) -> dict[str, DesignEvalResult]:
    """The Table S3 baseline at every site, in ONE batched evaluation.

    Same design vector at every site, so this is one request per site. Batched for the
    same reason verification is: an evaluation call costs ~the same at any width, so a
    baseline-per-site loop would pay a full call per site.
    """
    from solar_lumped.economics import LCOEconomicParams

    econ = LCOEconomicParams()
    x_baseline = baseline_design_vector(cfg)
    specs = [spec for spec in cfg.sites if spec.name in run_dirs]
    caches = {spec.name: EvalCache(Path(run_dirs[spec.name]) / "cache.jsonl") for spec in specs}
    results = evaluate_for_config(
        [(spec, x_baseline) for spec in specs],
        cfg=cfg, caches=caches, econ=econ, site_inputs=site_inputs,
    )
    return {spec.name: result for spec, result in zip(specs, results)}


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
    # Pass the already-evaluated baseline when a many-site pass batched them all into one
    # call (evaluate_baselines); None evaluates this site's on its own.
    baseline_result: DesignEvalResult | None = None,
) -> dict:
    if baseline_result is None:
        baseline_result = evaluate_baseline(cfg, run_dir)
    sweep_ref = _best_sweep_reference()

    # Verification's perturbed neighbors are full true-model evaluations, not surrogate
    # predictions, so a neighbor that came back better is not an artifact to warn about --
    # it is the better design, already paid for. Recommending the loop's point anyway
    # discards it (a 13-dim run with a few hundred points is sparse enough that a +/-10%
    # perturbation beating the incumbent is routine, not pathological).
    recommended = min(
        [result.best, *verification.neighbor_results], key=lambda r: r.combined_lcow
    )
    from_verification = recommended is not result.best

    baseline_lcow = baseline_result.combined_lcow
    if math.isfinite(baseline_lcow) and baseline_lcow not in (0.0,):
        improvement = (baseline_lcow - recommended.combined_lcow) / baseline_lcow
    else:
        improvement = None

    report = {
        "case": cfg.case,
        "recommended_design": dict(zip(cfg.bounds.names(), recommended.design_vector)),
        "recommended_combined_lcow_usd_per_m3": recommended.combined_lcow,
        "recommended_from": "verification_neighbor" if from_verification else "bayesopt_loop",
        "recommended_per_site": {
            r.site_name: {
                "lcow_usd_per_m3": r.lcow,
                "feasible": r.feasible,
                "yield_kg_m2": r.yield_kg_m2,
                "eta_thermal": r.eta_thermal,
                "failure_reason": r.failure_reason,
            }
            for r in recommended.site_results
        },
        "verification": {
            "true_combined_lcow_usd_per_m3": verification.best_true_combined_lcow,
            "surrogate_mu": verification.best_surrogate_mu,
            "surrogate_sigma": verification.best_surrogate_sigma,
            "max_neighbor_improvement_frac": verification.max_neighbor_improvement_frac,
            "flagged_as_surrogate_artifact": verification.flagged_as_surrogate_artifact,
        },
        "baseline_wilson_table_s3": {
            "design": dict(zip(cfg.bounds.names(), baseline_result.design_vector)),
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
