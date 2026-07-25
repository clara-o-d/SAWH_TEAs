#!/usr/bin/env python3
"""Convergence plot restricted to fully-feasible designs.

write_convergence_plot (reporting.py) scatters every evaluated combined_lcow,
including designs where one or both sites hit the infeasibility penalty
(evaluator.PENALTY_LCOW_USD_PER_M3, 10000 USD/m^3, or a partial-penalty blend
when only one of two sites failed e.g. combine_rule="mean" averaging one
real ~20-30 USD/m^3 site with one 10000-penalty site into a ~5000ish point).
Even one or two such points force the y-axis to span into the thousands,
squashing the entire feasible-region detail into an unreadable line near 0
-- this drops any design where *any* site was infeasible (using history.csv's
own <site>_feasible columns, not a statistical outlier heuristic, since we
know exactly why these points are extreme) before plotting.

Usage:
    python3 scripts/diagnostics/convergence_no_outliers.py --run-dir outputs/runs/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir
    df = pd.read_csv(run_dir / "history.csv").sort_values("index")

    feasible_cols = [c for c in df.columns if c.endswith("_feasible")]
    if not feasible_cols:
        raise SystemExit(f"No <site>_feasible columns found in {run_dir}/history.csv")
    all_feasible = df[feasible_cols].all(axis=1)

    n_dropped = int((~all_feasible).sum())
    if n_dropped:
        dropped = df.loc[~all_feasible]
        print(f"Dropping {n_dropped}/{len(df)} design(s) with at least one infeasible site:", flush=True)
        for _, row in dropped.iterrows():
            reasons = {
                c.replace("_feasible", ""): df.loc[row.name, c.replace("_feasible", "_failure_reason")]
                for c in feasible_cols
                if not row[c]
            }
            print(f"  index {int(row['index'])}: combined_lcow={row['combined_lcow']:.2f}, failing site(s): {reasons}", flush=True)

    lcows_all = df["combined_lcow"].to_numpy(dtype=float)
    best_so_far_all = np.minimum.accumulate(lcows_all)  # unaffected by dropping infeasible points -- they're never the min

    feasible_df = df.loc[all_feasible]
    indices_feasible = feasible_df["index"].to_numpy()
    lcows_feasible = feasible_df["combined_lcow"].to_numpy(dtype=float)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(indices_feasible, lcows_feasible, "o", alpha=0.4, label="evaluated (feasible only)")
    ax.plot(df["index"], best_so_far_all, "-", color="C1", label="best so far (full history)")
    ax.set_xlabel("design point index")
    ax.set_ylabel("combined LCOW (USD/m³)")
    title = "Bayesian optimization convergence (infeasible designs excluded)"
    if n_dropped:
        title += f"\n({n_dropped} infeasible design(s) dropped from the scatter, not from best-so-far)"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = run_dir / "diagnostics" / "convergence_no_outliers.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
