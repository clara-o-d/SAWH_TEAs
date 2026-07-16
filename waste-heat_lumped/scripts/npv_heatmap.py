#!/usr/bin/env python3
"""2D NPV / payback heatmaps over pairs of sweep parameters.

Reuses ``make_sweep_params``, ``_apply_combo``, and ``_metrics_from_result``
from ``parameter_sweep.py``: for each requested ``paramX:paramY`` pair, holds
every other parameter at its baseline value and evaluates a full
``grid_n x grid_n`` factorial grid over the pair, then renders an NPV
heatmap (diverging colormap, zero-NPV contour) and a payback-period heatmap
(infeasible cells hatched gray).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_SRC = _REPO / "src"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from parameter_sweep import (  # noqa: E402
    SweepParam,
    _apply_combo,
    _metrics_from_result,
    _sweep_grid,
    make_sweep_params,
)
from run_waste_heat_sim import (  # noqa: E402
    register_cyclic_warmup_arguments,
    register_waste_heat_sim_arguments,
)
from waste_heat_lumped.economics.params import LCOEconomicParams  # noqa: E402

_DEFAULT_PAIRS: tuple[str, ...] = (
    "water_price_usd_per_m3:discount_rate",
    "water_price_usd_per_m3:device_lifetime_years",
    "water_price_usd_per_m3:hydrogel_lifetime_years",
    "water_price_usd_per_m3:t_f_c",
    "hydrogel_thickness_mm:ua_gel_w_k",
    "discount_rate:device_lifetime_years",
)


def _compute_grid(
    x_param: SweepParam,
    y_param: SweepParam,
    *,
    grid_n: int,
    base_args: argparse.Namespace,
    base_econ: LCOEconomicParams,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict]]:
    """Full-factorial grid over exactly (x_param, y_param); all else at baseline."""
    x_vals = np.array(_sweep_grid(x_param, grid_n), dtype=float)
    y_vals = np.array(_sweep_grid(y_param, grid_n), dtype=float)

    npv = np.full((len(y_vals), len(x_vals)), np.nan)
    payback_simple = np.full_like(npv, np.nan)
    payback_discounted = np.full_like(npv, np.nan)
    lifetime_years = np.full_like(npv, float(base_econ.device_lifetime_years))

    rows: list[dict] = []
    for j, yv in enumerate(y_vals):
        for i, xv in enumerate(x_vals):
            combo = {x_param.key: float(xv), y_param.key: float(yv)}
            result, water_price = _apply_combo(combo, base_args, base_econ)
            metrics = _metrics_from_result(result, water_price_usd_per_m3=water_price)
            npv[j, i] = metrics["npv_usd_per_m2"]
            payback_simple[j, i] = metrics["payback_years_simple"]
            payback_discounted[j, i] = metrics["payback_years_discounted"]
            lifetime_years[j, i] = float(result.econ.device_lifetime_years)
            rows.append({x_param.key: float(xv), y_param.key: float(yv), **metrics})

    grids = {
        "npv_usd_per_m2": npv,
        "payback_years_simple": payback_simple,
        "payback_years_discounted": payback_discounted,
        "device_lifetime_years": lifetime_years,
    }
    return x_vals, y_vals, grids, rows


def _cell_edges(vals: np.ndarray) -> np.ndarray:
    """Cell boundaries for a 1D array of cell-center coordinates."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5])
    mid = (vals[:-1] + vals[1:]) / 2.0
    first = vals[0] - (mid[0] - vals[0])
    last = vals[-1] + (vals[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _plot_npv_heatmap(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    npv: np.ndarray,
    *,
    x_param: SweepParam,
    y_param: SweepParam,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    finite = npv[np.isfinite(npv)]
    vmin = float(finite.min()) if finite.size else -1.0
    vmax = float(finite.max()) if finite.size else 1.0
    # Center the diverging colormap on zero but let each side span only the
    # data actually present, so an all-negative (or all-positive) grid still
    # shows full contrast instead of collapsing into one hue.
    vmin = min(vmin, -1e-6)
    vmax = max(vmax, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    mesh = ax.pcolormesh(x_vals, y_vals, npv, cmap=cmap, norm=norm, shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("NPV (USD/m²)")

    if finite.size and finite.min() < 0.0 < finite.max():
        ax.contour(x_vals, y_vals, npv, levels=[0.0], colors="black", linewidths=1.5)

    ax.plot(
        [x_param.baseline],
        [y_param.baseline],
        marker="*",
        markersize=20,
        color="black",
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle="none",
        label="Baseline",
        zorder=5,
    )
    ax.set_xlabel(x_param.label.replace("\n", " "))
    ax.set_ylabel(y_param.label.replace("\n", " "))
    ax.set_title(f"NPV: {x_param.label.replace(chr(10), ' ')} vs {y_param.label.replace(chr(10), ' ')}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_payback_heatmap(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    payback: np.ndarray,
    lifetime_years: np.ndarray,
    *,
    x_param: SweepParam,
    y_param: SweepParam,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    vmax = float(np.nanmax(lifetime_years)) if np.isfinite(lifetime_years).any() else 20.0
    vmax = max(vmax, 1e-6)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax, clip=True)
    cmap = plt.get_cmap("viridis_r")

    infeasible = ~np.isfinite(payback) | (payback > lifetime_years)
    plot_vals = np.where(infeasible, np.nan, payback)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    mesh = ax.pcolormesh(x_vals, y_vals, plot_vals, cmap=cmap, norm=norm, shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Payback period (years)")

    x_edges = _cell_edges(x_vals)
    y_edges = _cell_edges(y_vals)
    for j in range(len(y_vals)):
        for i in range(len(x_vals)):
            if infeasible[j, i]:
                ax.add_patch(
                    Rectangle(
                        (x_edges[i], y_edges[j]),
                        x_edges[i + 1] - x_edges[i],
                        y_edges[j + 1] - y_edges[j],
                        facecolor="0.6",
                        edgecolor="0.2",
                        linewidth=0.25,
                        hatch="///",
                    )
                )

    ax.plot(
        [x_param.baseline],
        [y_param.baseline],
        marker="*",
        markersize=20,
        color="black",
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle="none",
        label="Baseline",
        zorder=5,
    )
    ax.set_xlabel(x_param.label.replace("\n", " "))
    ax.set_ylabel(y_param.label.replace("\n", " "))
    ax.set_title(
        f"Payback: {x_param.label.replace(chr(10), ' ')} vs {y_param.label.replace(chr(10), ' ')}"
        "\n(hatched = never pays back within device lifetime)"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="2D NPV/payback heatmaps (uses run_waste_heat_simulation)",
    )
    register_waste_heat_sim_arguments(ap)
    register_cyclic_warmup_arguments(ap)
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=list(_DEFAULT_PAIRS),
        help="paramX:paramY pairs (repeatable); keys from make_sweep_params in parameter_sweep.py",
    )
    ap.add_argument("--grid-n", type=int, default=25)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "outputs" / "npv_heatmaps",
    )
    args = ap.parse_args()

    base_args = copy.copy(args)
    base_econ = LCOEconomicParams()
    all_params = {p.key: p for p in make_sweep_params(base_args, base_econ)}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pair in args.pairs:
        if ":" not in pair:
            ap.error(f"--pairs entries must be 'paramX:paramY' (got {pair!r})")
        x_key, y_key = pair.split(":", 1)
        if x_key not in all_params or y_key not in all_params:
            unknown = [k for k in (x_key, y_key) if k not in all_params]
            ap.error(f"Unknown sweep parameter(s) in pair {pair!r}: {', '.join(unknown)}")
        x_param = all_params[x_key]
        y_param = all_params[y_key]

        print(f"Computing grid for {x_key} x {y_key} ({args.grid_n}x{args.grid_n})...", flush=True)
        x_vals, y_vals, grids, rows = _compute_grid(
            x_param,
            y_param,
            grid_n=args.grid_n,
            base_args=base_args,
            base_econ=base_econ,
        )

        csv_path = args.output_dir / f"npv_heatmap_{x_key}_x_{y_key}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

        npv_png = args.output_dir / f"npv_heatmap_{x_key}_x_{y_key}.png"
        _plot_npv_heatmap(
            x_vals,
            y_vals,
            grids["npv_usd_per_m2"],
            x_param=x_param,
            y_param=y_param,
            out_path=npv_png,
        )
        print(f"Wrote {npv_png}")

        payback_png = args.output_dir / f"payback_heatmap_{x_key}_x_{y_key}.png"
        _plot_payback_heatmap(
            x_vals,
            y_vals,
            grids["payback_years_simple"],
            grids["device_lifetime_years"],
            x_param=x_param,
            y_param=y_param,
            out_path=payback_png,
        )
        print(f"Wrote {payback_png}")


if __name__ == "__main__":
    main()
