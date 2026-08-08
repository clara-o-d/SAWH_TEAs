#!/usr/bin/env python3
"""GP posterior slice plot with infeasible designs dropped before fitting.

gp_diagnostics.py's gp_slices.png fits and plots against *every* cached
evaluation, including designs where a site hit the infeasibility penalty
(evaluator.PENALTY_LCOW_USD_PER_M3, or a partial-penalty blend when only one
of two sites failed). Even one or two such points (a) force the y-axis to
span into the thousands, squashing the real feasible-region posterior into
an unreadable flat line, and (b) drag the fitted kernel's own hyperparameters
toward whatever explains that outlier rather than the feasible-region
structure that actually matters for picking a next design. This refits a
second GP with those designs dropped first (using the same <site>.feasible
flags already recorded per evaluation -- not a statistical outlier rule,
since we know exactly why these points are extreme) and plots slices through
*that* GP instead.

This is a diagnostic view, not a replacement for gp_diagnostics.py's own
cross-validation pass -- that one deliberately keeps infeasible designs in,
since seeing how badly they distort CV/residuals is itself useful signal
(see its n_penalized_points field). This script is for actually looking at
the feasible-region posterior shape.

Usage:
    python3 scripts/diagnostics/gp_slices_no_outliers.py --run-dir outputs/runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DIAG_DIR = Path(__file__).resolve().parent
if str(_DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(_DIAG_DIR))
_SRC = _DIAG_DIR.parents[3] / "sawh_bayesopt" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from gp_diagnostics import _load_bounds, plot_slices  # noqa: E402
from sawh_bayesopt.evaluator import EvalCache  # noqa: E402
from sawh_bayesopt.surrogate import SurrogateState, build_gp, fit  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-slice-points", type=int, default=60)
    return p.parse_args(argv)


def _load_xy_feasible_only(run_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    cache = EvalCache(run_dir / "cache.jsonl")
    results = cache.all_results()

    feasible_results = []
    for i, r in enumerate(results):
        infeasible_sites = [sr.site_name for sr in r.site_results if not sr.feasible]
        if infeasible_sites:
            reasons = {sr.site_name: sr.failure_reason for sr in r.site_results if not sr.feasible}
            print(f"  dropping cached point {i}: combined_lcow={r.combined_lcow:.2f}, failing site(s): {reasons}", flush=True)
            continue
        feasible_results.append(r)

    n_dropped = len(results) - len(feasible_results)
    if len(feasible_results) < 2:
        raise SystemExit(f"Only {len(feasible_results)} fully-feasible design(s) in {run_dir}/cache.jsonl -- need at least 2.")
    X = np.array([r.design_vector for r in feasible_results], dtype=float)
    y = np.array([r.combined_lcow for r in feasible_results], dtype=float)
    return X, y, n_dropped


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

    out_path = out_dir / "gp_slices_no_outliers.png"
    plot_slices(state, X, y, bounds, n_points=args.n_slice_points, out_path=out_path)
    print(f"Wrote {out_path}", flush=True)

    (out_dir / "gp_slices_no_outliers_meta.json").write_text(json.dumps({
        "n_points_used": len(y),
        "n_points_dropped": n_dropped,
        "kernel_repr": str(state.gp.kernel_),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
