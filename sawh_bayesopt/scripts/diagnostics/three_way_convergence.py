#!/usr/bin/env python3
"""Combine a completed BayesOpt run, baseline_random_search.py's output, and
grid_search_baseline.py's output into one best-so-far convergence plot.

Usage (after running both baselines against the same --bayesopt-run-dir):
    python3 scripts/diagnostics/three_way_convergence.py \\
        --bayesopt-run-dir outputs/runs/<run_id> \\
        --random-run-dir outputs/runs/random_baseline \\
        --grid-run-dir outputs/runs/grid_baseline \\
        --out-path outputs/runs/<run_id>/diagnostics/three_way_convergence.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bayesopt-run-dir", type=Path, required=True)
    p.add_argument("--random-run-dir", type=Path, required=True)
    p.add_argument("--grid-run-dir", type=Path, required=True)
    p.add_argument("--out-path", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    bo_lcows = pd.read_csv(args.bayesopt_run_dir / "history.csv").sort_values("index")["combined_lcow"].to_numpy()
    best_so_far_bo = np.minimum.accumulate(bo_lcows)

    random_report = json.loads((args.random_run_dir / "random_search_report.json").read_text())
    best_so_far_random = np.array(random_report["random_search"]["best_so_far"])

    grid_report = json.loads((args.grid_run_dir / "grid_search_report.json").read_text())
    best_so_far_grid = np.array(grid_report["best_so_far"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(best_so_far_bo) + 1), best_so_far_bo, "-o", markersize=3, label=f"Bayesian optimization (n={len(best_so_far_bo)})")
    ax.plot(range(1, len(best_so_far_random) + 1), best_so_far_random, "-o", markersize=3, label=f"random search (n={len(best_so_far_random)})")
    ax.plot(range(1, len(best_so_far_grid) + 1), best_so_far_grid, "-o", markersize=3, label=f"full-factorial grid, {grid_report['levels_per_dim']} levels/dim (n={len(best_so_far_grid)})")
    ax.set_xlabel("evaluation count")
    ax.set_ylabel("incumbent best combined LCOW (USD/m³)")
    ax.set_title("BayesOpt vs. random search vs. brute-force grid, same design space")
    ax.legend()
    fig.tight_layout()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=150)
    plt.close(fig)

    summary = {
        "bayesopt": {"n": int(len(best_so_far_bo)), "final_best": float(best_so_far_bo[-1])},
        "random_search": {"n": int(len(best_so_far_random)), "final_best": float(best_so_far_random[-1])},
        "grid_search": {"n": int(len(best_so_far_grid)), "final_best": float(best_so_far_grid[-1])},
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
