#!/usr/bin/env python3
"""GP posterior slice plot, scattering only points near x_best in the other dims.

gp_slices_no_outliers.py's scatter overlay plots every fully-feasible cached
design against its raw combined_lcow on each 1D axis -- but each design
varies across *all* dimensions at once, so a point's position on (say) the
hydrogel_thickness_m axis says nothing about whether its other 5 parameters
were also near x_best. That makes the raw scatter look noisy relative to the
posterior mean line, which *does* hold the other dims fixed at x_best. This
script filters the scatter down to points whose other dims are all within
--tol (as a fraction of that dim's [lo, hi] bounds range) of x_best, so what's
shown is actually comparable to the conditional slice being plotted.

With few cached evaluations in a high-dimensional space, exact equality on
the other dims essentially never happens (continuous parameters, LHS/BO
sampling) -- hence the tolerance rather than a strict match. Tightening --tol
shows fewer, more faithful points; loosening it shows more, noisier ones.
This is purely a scatter-overlay filter: the GP fit and the posterior
mean/CI line are unchanged from gp_slices_no_outliers.py.

Usage:
    python3 scripts/diagnostics/gp_slices_near_xbest.py --run-dir outputs/runs/<run_id> [--tol 0.15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DIAG_DIR = Path(__file__).resolve().parent
if str(_DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(_DIAG_DIR))
_SRC = _DIAG_DIR.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from gp_diagnostics import _load_bounds  # noqa: E402
from gp_slices_no_outliers import _load_xy_feasible_only  # noqa: E402
from sawh_bayesopt.design_space import VAR_ORDER  # noqa: E402
from sawh_bayesopt.surrogate import SurrogateState, build_gp, fit, predict_batch  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-slice-points", type=int, default=60)
    p.add_argument(
        "--tol",
        type=float,
        default=0.15,
        help="Max normalized distance (fraction of bounds range) from x_best allowed on the OTHER dims for a "
        "cached point to be scattered on a given dim's slice. Smaller = stricter/fewer points.",
    )
    return p.parse_args(argv)


def plot_slices_near_xbest(state, X, y, bounds, *, n_points, tol, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_best = X[int(np.argmin(y))]
    bounds_arr = bounds.as_array()
    lo_all, hi_all = bounds_arr[:, 0], bounds_arr[:, 1]
    rng_all = hi_all - lo_all
    X_norm = (X - lo_all) / rng_all
    x_best_norm = (x_best - lo_all) / rng_all

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

        other_dims = [i for i in range(n_dims) if i != d]
        dist = np.max(np.abs(X_norm[:, other_dims] - x_best_norm[other_dims]), axis=1)
        mask = dist <= tol
        n_kept = int(mask.sum())
        print(f"  {name}: keeping {n_kept}/{len(y)} point(s) within tol={tol} of x_best on other dims", flush=True)

        ax = axes[d]
        ax.plot(grid, mu, "C0-", label="posterior mean")
        ax.fill_between(grid, mu - 1.96 * sigma, mu + 1.96 * sigma, color="C0", alpha=0.25, label="95% CI")
        ax.scatter(X[mask, d], y[mask], color="k", s=12, alpha=0.6, zorder=5, label="evaluated points (near x_best)")
        ax.axvline(x_best[d], color="C1", linestyle="--", linewidth=1, label="x_best")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in axes[n_dims:]:
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"GP posterior slices through the incumbent best design (points within tol={tol} of x_best on other dims)")
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
    print(f"Loading {run_dir}/cache.jsonl, dropping any design with an infeasible site...", flush=True)
    X, y, n_dropped = _load_xy_feasible_only(run_dir)
    print(f"Fitting GP on {len(y)} fully-feasible design(s) ({n_dropped} dropped).", flush=True)

    state = SurrogateState(gp=build_gp(seed=args.seed), bounds=bounds, X_raw=X, y=y, feasible=np.ones(len(y), dtype=bool))
    state = fit(state)
    print(f"  kernel: {state.gp.kernel_}", flush=True)

    out_path = out_dir / "gp_slices_near_xbest.png"
    plot_slices_near_xbest(state, X, y, bounds, n_points=args.n_slice_points, tol=args.tol, out_path=out_path)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
