#!/usr/bin/env python3
"""Plot scripts/hp_sweep.py's sweep_results.csv: performance metrics vs.
each swept hyperparameter (ei_xi, stall_rel_tol, n_init).

Produces, in <sweep-dir>/:
  - hp_sweep_marginals.png: one column per swept hyperparameter, one row per
    metric (best LCOW, cv_rmse, standardized_residual_std,
    msll_gp_minus_trivial, frac_de_hit_maxiter). Every combination is plotted
    as a point (jittered slightly on x so overlapping combinations are
    visible) against that column's hyperparameter, with the other two
    hyperparameters shown by marker color/shape; a connecting line through
    each color's mean marks the marginal trend.
  - hp_sweep_heatmap_<metric>.png: for best_combined_lcow_usd_per_m3 and
    standardized_residual_std, a row of heatmaps (one per n_init value) with
    ei_xi x stall_rel_tol on the axes -- the full 3-way interaction, not just
    marginals.

Rows with an "error" field (a combination whose run_bayesopt/verify_optimum/
gp_diagnostics step failed -- see hp_sweep.py) are excluded from the plots
but counted and reported on stderr, never silently dropped without a trace.

Usage:
    python3 scripts/plot_hp_sweep.py --sweep-dir outputs/runs/<sweep-id>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=Path, required=True)
    return p.parse_args(argv)


def _load_rows(sweep_dir: Path) -> tuple[list[dict], int]:
    csv_path = sweep_dir / "sweep_results.csv"
    if not csv_path.is_file():
        raise SystemExit(f"{csv_path} not found -- run scripts/hp_sweep.py first.")
    with csv_path.open() as f:
        all_rows = list(csv.DictReader(f))
    ok_rows = [r for r in all_rows if not r.get("error")]
    for r in ok_rows:
        for k, v in list(r.items()):
            if v == "":
                r[k] = None
            elif k in ("ei_xi", "stall_rel_tol", "best_combined_lcow_usd_per_m3", "cv_rmse",
                       "standardized_residual_mean", "standardized_residual_std", "msll_gp_minus_trivial",
                       "improvement_vs_baseline_frac", "frac_de_hit_maxiter", "frac_de_not_success", "wall_time_s"):
                r[k] = float(v) if v is not None else None
            elif k in ("n_init", "n_total", "n_evaluations", "n_hyperparameter_warnings", "n_de_calls"):
                r[k] = int(float(v)) if v is not None else None
    return ok_rows, len(all_rows) - len(ok_rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows, n_errors = _load_rows(args.sweep_dir)
    if n_errors:
        print(f"WARNING: {n_errors} combination(s) had errors and are excluded from these plots -- see sweep_results.csv's 'error' column.", file=sys.stderr)
    if not rows:
        raise SystemExit("No successful combinations to plot.")

    hp_names = ["ei_xi", "stall_rel_tol", "n_init"]
    metrics = [
        ("best_combined_lcow_usd_per_m3", "best combined LCOW (USD/m3)"),
        ("cv_rmse", "CV RMSE"),
        ("standardized_residual_std", "standardized residual std (want ~1)"),
        ("msll_gp_minus_trivial", "MSLL(GP) - MSLL(trivial) (want < 0)"),
        ("frac_de_hit_maxiter", "frac. of DE proposals hitting maxiter"),
    ]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(len(metrics), len(hp_names), figsize=(4.2 * len(hp_names), 3.0 * len(metrics)), squeeze=False)
    for col, hp in enumerate(hp_names):
        other = [h for h in hp_names if h != hp]
        color_key = [f"{r[other[0]]}/{r[other[1]]}" for r in rows]
        uniq_keys = sorted(set(color_key))
        cmap = plt.get_cmap("viridis", max(len(uniq_keys), 1))
        color_of = {k: cmap(i) for i, k in enumerate(uniq_keys)}

        x_vals = np.array([r[hp] for r in rows], dtype=float)
        x_uniq = np.array(sorted(set(x_vals)))
        jitter_scale = (x_uniq[1:] - x_uniq[:-1]).min() * 0.08 if len(x_uniq) > 1 else 0.0

        for row_i, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row_i][col]
            y_vals = [r.get(metric_key) for r in rows]
            has_data = any(v is not None for v in y_vals)
            if not has_data:
                ax.axis("off")
                continue
            for xi, yi, key in zip(x_vals, y_vals, color_key):
                if yi is None:
                    continue
                jitter = rng.uniform(-jitter_scale, jitter_scale)
                ax.scatter(xi + jitter, yi, color=color_of[key], s=28, alpha=0.85)
            for xi in x_uniq:
                ys = [y for x, y in zip(x_vals, y_vals) if x == xi and y is not None]
                if ys:
                    ax.plot(xi, float(np.mean(ys)), marker="_", markersize=22, color="black", markeredgewidth=2)
            if row_i == 0:
                ax.set_title(hp, fontsize=10)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("hp_sweep: metric vs. each hyperparameter (marginal; color = the other two hyperparameters' values, black bar = mean)", fontsize=10)
    fig.tight_layout()
    out1 = args.sweep_dir / "hp_sweep_marginals.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Wrote {out1}", flush=True)

    ei_vals = sorted({r["ei_xi"] for r in rows})
    stall_vals = sorted({r["stall_rel_tol"] for r in rows})
    n_init_vals = sorted({r["n_init"] for r in rows})

    for metric_key, metric_label in [
        ("best_combined_lcow_usd_per_m3", "best combined LCOW (USD/m3)"),
        ("standardized_residual_std", "standardized residual std"),
    ]:
        if not any(r.get(metric_key) is not None for r in rows):
            continue
        fig, axes = plt.subplots(1, len(n_init_vals), figsize=(4.5 * len(n_init_vals), 4), squeeze=False)
        axes = axes[0]
        grids = []
        for n_init in n_init_vals:
            grid = np.full((len(stall_vals), len(ei_vals)), np.nan)
            for r in rows:
                if r["n_init"] != n_init or r.get(metric_key) is None:
                    continue
                grid[stall_vals.index(r["stall_rel_tol"]), ei_vals.index(r["ei_xi"])] = r[metric_key]
            grids.append(grid)
        vmin = np.nanmin([np.nanmin(g) for g in grids if not np.all(np.isnan(g))])
        vmax = np.nanmax([np.nanmax(g) for g in grids if not np.all(np.isnan(g))])
        im = None
        for ax, n_init, grid in zip(axes, n_init_vals, grids):
            im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(ei_vals)))
            ax.set_xticklabels([f"{v:g}" for v in ei_vals])
            ax.set_yticks(range(len(stall_vals)))
            ax.set_yticklabels([f"{v:g}" for v in stall_vals])
            ax.set_xlabel("ei_xi")
            ax.set_title(f"n_init={n_init}")
            for si in range(len(stall_vals)):
                for ei in range(len(ei_vals)):
                    v = grid[si, ei]
                    if not np.isnan(v):
                        ax.text(ei, si, f"{v:.3g}", ha="center", va="center", fontsize=7, color="white")
        axes[0].set_ylabel("stall_rel_tol")
        fig.suptitle(f"{metric_label} across the full ei_xi x stall_rel_tol x n_init grid", fontsize=10)
        fig.colorbar(im, ax=list(axes), shrink=0.8, label=metric_label)
        out2 = args.sweep_dir / f"hp_sweep_heatmap_{metric_key}.png"
        fig.savefig(out2, dpi=150)
        plt.close(fig)
        print(f"Wrote {out2}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
