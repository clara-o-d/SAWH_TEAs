#!/usr/bin/env python3
"""GP surrogate regression diagnostics for a completed sawh_bayesopt run.

Answers: "is the GP actually modeling the LCOW surface well, or are we
trusting a surrogate that's just guessing?" Everything here works from a
completed run directory's cache.jsonl (every evaluated design -- always
present) and config.json (bounds -- written by run_bayesopt.py); it doesn't
need gp_state.joblib, since k-fold CV refits fresh GPs on held-out folds by
construction.

Produces, in <run-dir>/diagnostics/:
  - gp_regression_report.json: k-fold cross-validated MSE, standardized
    residuals (mean/std), MSLL (vs. a trivial mean/std baseline), and the
    final full-data fit's kernel hyperparameters.
  - gp_slices.png: 1D posterior mean +/- 95% CI through the incumbent best
    design, one subplot per design variable, with the actually-evaluated
    points overlaid -- the variance band should pinch to ~0 right at
    observed points and widen away from them.

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
from sawh_bayesopt.surrogate import SurrogateState, build_gp, fit, predict_batch  # noqa: E402


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


def _load_xy(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cache = EvalCache(run_dir / "cache.jsonl")
    results = cache.all_results()
    if len(results) < 2:
        raise SystemExit(f"Only {len(results)} evaluated design(s) in {run_dir}/cache.jsonl -- need at least 2.")
    X = np.array([r.design_vector for r in results], dtype=float)
    y = np.array([r.combined_lcow for r in results], dtype=float)
    return X, y


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


def cross_validate(X: np.ndarray, y: np.ndarray, bounds: DesignBounds, *, k: int, seed: int) -> dict:
    n = len(y)
    n_penalized = int(np.sum(y >= 0.99 * PENALTY_LCOW_USD_PER_M3))
    k = min(k, n)  # can't have more folds than points
    mu_all = np.zeros(n)
    sigma_all = np.zeros(n)
    for train_idx, test_idx in _kfold_indices(n, k, seed):
        if len(train_idx) < 2:
            continue
        state = SurrogateState(gp=build_gp(seed=seed), bounds=bounds, X_raw=X[train_idx], y=y[train_idx])
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


def final_fit_hyperparameters(
    X: np.ndarray, y: np.ndarray, bounds: DesignBounds, *, seed: int
) -> tuple[dict, SurrogateState]:
    state = SurrogateState(gp=build_gp(seed=seed), bounds=bounds, X_raw=X, y=y)
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
    X, y = _load_xy(run_dir)
    print(f"Loaded {len(y)} evaluated designs from {run_dir}/cache.jsonl", flush=True)

    print(f"Running {args.k_folds}-fold cross-validation...", flush=True)
    cv = cross_validate(X, y, bounds, k=args.k_folds, seed=args.seed)
    print(f"  n_penalized_points: {cv['n_penalized_points']}/{cv['n_points']}", flush=True)
    for key in ("cv_mse", "cv_rmse", "standardized_residual_mean", "standardized_residual_std", "msll_gp_minus_trivial"):
        print(f"  {key}: {cv[key]:.4g}", flush=True)

    print("Fitting final GP on all evaluated points...", flush=True)
    hyperparams, state = final_fit_hyperparameters(X, y, bounds, seed=args.seed)
    print(f"  kernel: {hyperparams['kernel_repr']}", flush=True)

    report = {"cross_validation": cv, "final_fit_hyperparameters": hyperparams}
    report_path = out_dir / "gp_regression_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_path}", flush=True)

    slices_path = out_dir / "gp_slices.png"
    plot_slices(state, X, y, bounds, n_points=args.n_slice_points, out_path=slices_path)
    print(f"Posterior slice plot written to {slices_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
