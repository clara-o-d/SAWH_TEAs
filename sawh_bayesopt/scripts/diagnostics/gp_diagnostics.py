#!/usr/bin/env python3
"""GP surrogate regression diagnostics for a completed sawh_bayesopt run -- is the GP
modelling the LCOW surface, or just guessing? Works from cache.jsonl + config.json alone
(k-fold CV refits fresh GPs, so gp_state.joblib isn't needed).

Writes <run-dir>/diagnostics/gp_regression_report.json (CV MSE, standardized residuals,
MSLL vs. a mean/std baseline, final kernel hyperparameters) and gp_slices.png (1D posterior
mean ±95% CI through the incumbent, one subplot per variable, evaluated points overlaid).

Usage:
    python3 scripts/diagnostics/gp_diagnostics.py --run-dir outputs/runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from sawh_bayesopt.design_space import DesignBounds, VAR_ORDER  # noqa: E402
from sawh_bayesopt.evaluator import PENALTY_LCOW_USD_PER_M3, EvalCache  # noqa: E402
from sawh_bayesopt.surrogate import (  # noqa: E402
    SurrogateState,
    build_gp,
    check_hyperparameter_convergence,
    fit,
    predict_batch,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-slice-points", type=int, default=60)
    return p.parse_args(argv)


def _load_bounds(run_dir: Path) -> DesignBounds:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        print(f"WARNING: {config_path} not found (older run?) -- using DesignBounds() defaults.", file=sys.stderr)
        return DesignBounds()
    payload = json.loads(config_path.read_text())
    return DesignBounds(**{name: tuple(v) for name, v in payload["bounds"].items()})


def _load_xy(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = EvalCache(run_dir / "cache.jsonl")
    results = cache.all_results()
    if len(results) < 2:
        raise SystemExit(f"Only {len(results)} evaluated design(s) in {run_dir}/cache.jsonl -- need at least 2.")
    X = np.array([r.design_vector for r in results], dtype=float)
    y = np.array([r.combined_lcow for r in results], dtype=float)
    feasible = np.array([r.is_feasible for r in results], dtype=bool)
    return X, y, feasible


def _kfold_indices(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    out = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train_idx, test_idx))
    return out


def cross_validate(
    X: np.ndarray, y: np.ndarray, feasible: np.ndarray, bounds: DesignBounds, *, k: int, seed: int
) -> dict:
    n = len(y)
    n_penalized = int(np.sum(y >= 0.99 * PENALTY_LCOW_USD_PER_M3))
    k = min(k, n)  # can't have more folds than points
    mu_all = np.zeros(n)
    sigma_all = np.zeros(n)
    for train_idx, test_idx in _kfold_indices(n, k, seed):
        train_feasible = feasible[train_idx]
        if train_feasible.sum() < 2:
            continue
        state = SurrogateState(
            gp=build_gp(seed=seed), bounds=bounds, X_raw=X[train_idx], y=y[train_idx], feasible=train_feasible
        )
        state = fit(state)
        mu, sigma = predict_batch(state, X[test_idx])
        mu_all[test_idx] = mu
        sigma_all[test_idx] = np.where(sigma > 1e-12, sigma, 1e-12)

    residuals = y - mu_all
    mse = float(np.mean(residuals**2))
    z = residuals / sigma_all
    z_mean, z_std = float(np.mean(z)), float(np.std(z))

    msll_gp = float(np.mean(0.5 * np.log(2 * np.pi * sigma_all**2) + residuals**2 / (2 * sigma_all**2)))
    trivial_mu, trivial_sigma = float(np.mean(y)), max(float(np.std(y)), 1e-12)
    msll_trivial = float(
        np.mean(0.5 * np.log(2 * np.pi * trivial_sigma**2) + (y - trivial_mu) ** 2 / (2 * trivial_sigma**2))
    )

    return {
        "k_folds": k,
        "n_points": n,
        "n_penalized_points": n_penalized,
        "penalized_points_note": (
            f"{n_penalized}/{n} evaluated designs hit the infeasibility penalty "
            f"({PENALTY_LCOW_USD_PER_M3:.0f} USD/m3). With few total points, even "
            "one penalized outlier can dominate cv_mse/standardized residuals -- "
            "treat the metrics below with proportionally more suspicion the "
            "larger n_penalized_points is relative to n_points."
        ),
        "cv_mse": mse,
        "cv_rmse": float(np.sqrt(mse)),
        "standardized_residual_mean": z_mean,
        "standardized_residual_std": z_std,
        "interpretation": (
            "standardized_residual_mean should be near 0 and _std near 1 if the "
            "GP's uncertainty is well-calibrated; _std >> 1 means the GP is "
            "overconfident (real errors bigger than sigma predicts), _std << 1 "
            "means it's underconfident (sigma bigger than it needs to be)."
        ),
        "msll_gp": msll_gp,
        "msll_trivial_baseline": msll_trivial,
        "msll_gp_minus_trivial": msll_gp - msll_trivial,
        "msll_interpretation": (
            "msll_gp_minus_trivial should be clearly negative -- that means the "
            "fitted GP explains held-out points better than just predicting the "
            "training mean/std everywhere. Near 0 or positive means the GP isn't "
            "adding value over a constant baseline."
        ),
    }


def summarize_de_diagnostics(run_dir: Path) -> dict | None:
    """Summarize the per-round differential_evolution results, if de_diagnostics.json
    exists (None otherwise). A high hit_maxiter/not_success rate means EI proposals are
    approximate maximizers, so apparent explore/exploit behavior may be an under-budgeted
    inner optimizer rather than real acquisition preference -- rule that out before tuning
    ei_xi/stall_rel_tol/n_init."""
    path = run_dir / "diagnostics" / "de_diagnostics.json"
    if not path.is_file():
        return None
    records = json.loads(path.read_text())
    n = len(records)
    if n == 0:
        return {"n_de_calls": 0}
    n_hit_maxiter = sum(1 for d in records if d["hit_maxiter"])
    n_not_success = sum(1 for d in records if not d["success"])
    nit_frac = [d["nit"] / d["maxiter"] for d in records]
    return {
        "n_de_calls": n,
        "n_hit_maxiter": n_hit_maxiter,
        "frac_hit_maxiter": n_hit_maxiter / n,
        "n_not_success": n_not_success,
        "frac_not_success": n_not_success / n,
        "nit_over_maxiter_mean": float(np.mean(nit_frac)),
        "nit_over_maxiter_min": float(np.min(nit_frac)),
        "interpretation": (
            "frac_hit_maxiter/frac_not_success should be near 0 -- otherwise a "
            "meaningful fraction of EI-proposed points came from an "
            "unconverged inner differential_evolution search (raise maxiter "
            "and/or popsize in acquisition.propose_next before trusting "
            "apparent under-exploration as a property of EI itself)."
        ),
    }


def detect_outliers(y: np.ndarray, *, iqr_multiplier: float = 3.0) -> np.ndarray:
    """Tukey's "far out" fence (Q3 + iqr_multiplier*IQR) on percentiles of *y*. Being
    order-statistics-based, extreme points barely move Q1/Q3, so this also catches designs
    that are badly infeasible without landing exactly on the penalty sentinel."""
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    threshold = q3 + iqr_multiplier * iqr
    return y > threshold


def final_fit_hyperparameters(
    X: np.ndarray, y: np.ndarray, feasible: np.ndarray, bounds: DesignBounds, *, seed: int
) -> tuple[dict, SurrogateState]:
    state = SurrogateState(gp=build_gp(seed=seed), bounds=bounds, X_raw=X, y=y, feasible=feasible)
    state = fit(state)
    kernel = state.gp.kernel_
    # ConstantKernel * Matern(length_scale=[...]) + WhiteKernel -- see surrogate.py::build_gp.
    k1 = kernel.k1  # ConstantKernel * Matern
    white = kernel.k2
    return {
        "signal_variance": float(k1.k1.constant_value),
        "length_scales": {name: float(ls) for name, ls in zip(VAR_ORDER, np.atleast_1d(k1.k2.length_scale))},
        "noise_level": float(white.noise_level),
        "kernel_repr": str(kernel),
    }, state


def plot_slices(state: SurrogateState, X: np.ndarray, y: np.ndarray, bounds: DesignBounds, *, n_points: int, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_best = X[int(np.argmin(y))]
    bounds_arr = bounds.as_array()
    n_dims = len(VAR_ORDER)
    n_cols = 3
    n_rows = -(-n_dims // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for d, name in enumerate(VAR_ORDER):
        lo, hi = bounds_arr[d]
        grid = np.linspace(lo, hi, n_points)
        Xs = np.tile(x_best, (n_points, 1))
        Xs[:, d] = grid
        mu, sigma = predict_batch(state, Xs)

        ax = axes[d]
        ax.plot(grid, mu, "C0-", label="posterior mean")
        ax.fill_between(grid, mu - 1.96 * sigma, mu + 1.96 * sigma, color="C0", alpha=0.25, label="95% CI")
        ax.scatter(X[:, d], y, color="k", s=12, alpha=0.6, zorder=5, label="evaluated points")
        ax.axvline(x_best[d], color="C1", linestyle="--", linewidth=1, label="x_best")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in axes[n_dims:]:
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("GP posterior slices through the incumbent best design (1D, others held fixed)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir
    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = _load_bounds(run_dir)
    X, y, feasible = _load_xy(run_dir)
    print(
        f"Loaded {len(y)} evaluated designs from {run_dir}/cache.jsonl "
        f"({int(feasible.sum())} feasible, {int((~feasible).sum())} infeasible)",
        flush=True,
    )

    # CV and the final fit use feasible designs only (matching surrogate.py::fit):
    # penalized LCOWs aren't measurements, so calibrating against them is meaningless.
    X_feas, y_feas = X[feasible], y[feasible]

    print(f"Running {args.k_folds}-fold cross-validation on feasible points...", flush=True)
    cv = cross_validate(X_feas, y_feas, np.ones_like(y_feas, dtype=bool), bounds, k=args.k_folds, seed=args.seed)
    cv["n_infeasible_excluded"] = int((~feasible).sum())
    print(f"  n_infeasible_excluded: {cv['n_infeasible_excluded']}/{len(y)}", flush=True)
    for key in ("cv_mse", "cv_rmse", "standardized_residual_mean", "standardized_residual_std", "msll_gp_minus_trivial"):
        print(f"  {key}: {cv[key]:.4g}", flush=True)

    outlier_mask_feas = detect_outliers(y_feas)
    cv_no_outliers = None
    if outlier_mask_feas.sum() > 0 and (~outlier_mask_feas).sum() >= 2:
        print(
            f"\n{outlier_mask_feas.sum()} outlier(s) among feasible points beyond Tukey's far-out "
            f"fence (Q3 + 3*IQR): {sorted(y_feas[outlier_mask_feas].tolist())} -- re-running CV "
            "excluding them (a couple of extreme *genuine* LCOW measurements can still dominate "
            "cv_mse/standardized residuals when n is small, separately from anything to do with "
            "infeasibility, which is already excluded above)",
            flush=True,
        )
        cv_no_outliers = cross_validate(
            X_feas[~outlier_mask_feas], y_feas[~outlier_mask_feas],
            np.ones(int((~outlier_mask_feas).sum()), dtype=bool), bounds,
            k=min(args.k_folds, int((~outlier_mask_feas).sum())), seed=args.seed,
        )
        for key in ("cv_rmse", "standardized_residual_mean", "standardized_residual_std", "msll_gp_minus_trivial"):
            print(f"  (excl. outliers) {key}: {cv_no_outliers[key]:.4g}", flush=True)

    print("Fitting final GP on all feasible points...", flush=True)
    hyperparams, state = final_fit_hyperparameters(
        X_feas, y_feas, np.ones_like(y_feas, dtype=bool), bounds, seed=args.seed
    )
    print(f"  kernel: {hyperparams['kernel_repr']}", flush=True)

    hp_warnings = check_hyperparameter_convergence(state.gp)
    if hp_warnings:
        print(f"\n{len(hp_warnings)} hyperparameter(s) landed near an optimization bound:", flush=True)
        for w in hp_warnings:
            print(f"  WARNING: {w}", flush=True)
    else:
        print("  All fitted hyperparameters are comfortably within their bounds.", flush=True)

    de_summary = summarize_de_diagnostics(run_dir)
    if de_summary is None:
        print("\nNo diagnostics/de_diagnostics.json found (older run, or no EI-based proposals made).", flush=True)
    elif de_summary.get("n_de_calls", 0) == 0:
        print("\nde_diagnostics.json present but empty (run never got far enough to propose via EI).", flush=True)
    else:
        print(
            f"\nDE proposal diagnostics: {de_summary['n_de_calls']} calls, "
            f"{de_summary['n_hit_maxiter']} hit maxiter ({de_summary['frac_hit_maxiter']:.1%}), "
            f"{de_summary['n_not_success']} not marked success ({de_summary['frac_not_success']:.1%}), "
            f"mean nit/maxiter={de_summary['nit_over_maxiter_mean']:.2f}",
            flush=True,
        )
        if de_summary["frac_hit_maxiter"] > 0.1 or de_summary["frac_not_success"] > 0.1:
            print(
                "  WARNING: a meaningful fraction of EI proposals came from an unconverged "
                "differential_evolution search -- consider raising acquisition.propose_next's "
                "maxiter/popsize before drawing conclusions about explore/exploit behavior.",
                flush=True,
            )

    report = {
        "cross_validation": cv,
        "cross_validation_excluding_outliers": cv_no_outliers,
        "outlier_values_excluded": sorted(y_feas[outlier_mask_feas].tolist()) if outlier_mask_feas.sum() else [],
        "final_fit_hyperparameters": hyperparams,
        "hyperparameter_convergence_warnings": hp_warnings,
        "de_diagnostics_summary": de_summary,
    }
    report_path = out_dir / "gp_regression_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_path}", flush=True)

    slices_path = out_dir / "gp_slices.png"
    plot_slices(state, X_feas, y_feas, bounds, n_points=args.n_slice_points, out_path=slices_path)
    print(f"Posterior slice plot written to {slices_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
