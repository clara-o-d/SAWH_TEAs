#!/usr/bin/env python3
"""Sanity baseline alongside baseline_random_search.py: does BayesOpt beat an exhaustive
grid sweep over the same 6-D space? Grid is 2 levels (bounds.lo/hi) x 6 variables = 64
hypercube corners -- close to a real BayesOpt evaluation budget, so neither side is starved
or given extra budget.

Usage:
    python3 scripts/diagnostics/grid_search_baseline.py \\
        --bayesopt-run-dir outputs/runs/<run_id> --run-id grid_baseline
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from sawh_bayesopt.design_space import DesignBounds, VAR_ORDER  # noqa: E402
from sawh_bayesopt.evaluator import EvalCache, evaluate_batch  # noqa: E402
from sawh_bayesopt.sites import ATACAMA, CAMBRIDGE, DEFAULT_SITES, fetch_monthly_profiles  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bayesopt-run-dir", type=Path, required=True, help="Completed run to compare against and copy config from.")
    p.add_argument("--run-id", type=str, default="grid_baseline")
    p.add_argument("--levels-per-dim", type=int, default=2, help="Grid levels per design variable (2 = corners only, the default).")
    p.add_argument("--eval-batch-size", type=int, default=64, help="How many grid points to evaluate per evaluate_batch() call.")
    p.add_argument("--seed", type=int, default=777, help="Only used to shuffle the reveal order for the best-so-far curve -- the grid itself is deterministic.")
    p.add_argument("--weather-cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    return p.parse_args(argv)


def full_factorial_design(bounds: DesignBounds, *, levels_per_dim: int) -> np.ndarray:
    """levels_per_dim evenly-spaced levels (inclusive of both bounds) per
    variable, every combination -- levels_per_dim ** len(VAR_ORDER) rows."""
    bounds_arr = bounds.as_array()
    axes = [np.linspace(lo, hi, levels_per_dim) for lo, hi in bounds_arr]
    return np.array(list(itertools.product(*axes)), dtype=float)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bo_config_path = args.bayesopt_run_dir / "config.json"
    if not bo_config_path.is_file():
        raise SystemExit(f"{bo_config_path} not found -- need a completed BayesOpt run's config.json for a fair sites/case match.")
    bo_config = json.loads(bo_config_path.read_text())

    bounds = DesignBounds(**{name: tuple(v) for name, v in bo_config["bounds"].items()})
    site_by_name = {s.name: s for s in DEFAULT_SITES + (CAMBRIDGE, ATACAMA)}
    sites = tuple(site_by_name[name] for name in bo_config["sites"])

    run_dir = _REPO / "outputs" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    X = full_factorial_design(bounds, levels_per_dim=args.levels_per_dim)
    n_total = len(X)
    # Exhaustive either way, so grid order can't change the final best -- but revealing in
    # grid order makes a steppy best-so-far curve, so shuffle with a fixed seed instead.
    rng = np.random.default_rng(args.seed)
    X = X[rng.permutation(n_total)]

    (run_dir / "config.json").write_text(json.dumps(
        {**bo_config, "seed": args.seed, "search": "full_factorial_grid", "levels_per_dim": args.levels_per_dim, "n_total": n_total},
        indent=2,
    ))

    site_profiles = {s.name: fetch_monthly_profiles(s, cache_dir=args.weather_cache_dir) for s in sites}
    cache = EvalCache(run_dir / "cache.jsonl")

    from solar_lumped.economics import LCOEconomicParams

    econ = LCOEconomicParams()

    print(
        f"Grid search: {args.levels_per_dim} levels/dim over {len(VAR_ORDER)} dims = {n_total} combinations, "
        f"sites={[s.name for s in sites]}",
        flush=True,
    )

    history = []
    for start in range(0, n_total, args.eval_batch_size):
        batch = list(X[start : start + args.eval_batch_size])
        results = evaluate_batch(
            batch, cache=cache, sites=sites, site_profiles=site_profiles, econ=econ,
            combine_rule=bo_config["combine_rule"], resolution=bo_config["resolution"],
            case=bo_config.get("case", "case2"),
        )
        history.extend(results)
        print(f"  {len(history)}/{n_total} evaluated, best so far: {min(r.combined_lcow for r in history):.4f}", flush=True)

    lcows = [r.combined_lcow for r in history]
    best_so_far_grid = list(np.minimum.accumulate(lcows))

    report = {
        "search": "full_factorial_grid",
        "levels_per_dim": args.levels_per_dim,
        "n_total": n_total,
        "best_so_far": best_so_far_grid,
        "final_best": best_so_far_grid[-1],
    }
    report_path = run_dir / "grid_search_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_path}", flush=True)
    print(f"Final best combined LCOW: {best_so_far_grid[-1]:.4f} USD/m3 over {n_total} grid points.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
