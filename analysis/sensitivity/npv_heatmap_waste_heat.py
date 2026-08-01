#!/usr/bin/env python3
"""2D NPV / payback heatmaps over pairs of swept waste-heat parameters.

For each ``paramX:paramY`` pair, sweeps a ``grid_n x grid_n`` grid (every other
parameter at baseline) via ``parameter_sweep._apply_overrides`` and writes a CSV,
an NPV heatmap (zero-NPV contour) and a payback heatmap (infeasible cells hatched).

Usage: python npv_heatmap_waste_heat.py

``--grid-n`` defaults to 15 (vs 25 for the single-cycle packages): every point is a
full multi-cycle Radau solve, so 225 points per pair keeps a default run tractable.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent.parent / "waste-heat"

_DEFAULT_PAIRS: tuple[str, ...] = (
    "water_price_usd_per_m3:discount_rate",
    "water_price_usd_per_m3:device_lifetime_years",
    "water_price_usd_per_m3:hydrogel_lifetime_years",
    "water_price_usd_per_m3:t_wh_in_c",
    "hydrogel_thickness_mm:ua_wh_desorber_w_k",
    "discount_rate:device_lifetime_years",
)

_ROW_METRIC_KEYS: tuple[str, ...] = (
    "daily_yield_kg_m2",
    "thermal_efficiency",
    "n_cycles_per_day",
    "lcow_usd_per_m3",
    "npv_usd_per_m2",
    "payback_years_simple",
    "payback_years_discounted",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", action="append", default=None, help="paramX:paramY (repeatable)")
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    sys.path[:0] = [str(_REPO / "scripts"), str(_REPO / "src")]
    import parameter_sweep as ps  # noqa: E402  (needs _REPO on sys.path first)

    output_dir = args.output_dir or _REPO / "npv_heatmaps"
    params_by_key = {p.key: p for p in ps.make_sweep_params()}

    parsed_pairs = []
    for spec in args.pairs or _DEFAULT_PAIRS:
        keys = spec.split(":")
        if len(keys) != 2:
            raise SystemExit(f"--pairs entry must be 'paramX:paramY', got {spec!r}")
        unknown = [k for k in keys if k not in params_by_key]
        if unknown:
            raise SystemExit(
                f"Unknown sweep parameter(s) in --pairs: {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(params_by_key))}"
            )
        parsed_pairs.append(tuple(keys))

    device_lifetime_years = float(ps.BASELINE_ECON.device_lifetime_years)
    output_dir.mkdir(parents=True, exist_ok=True)

    for param_x, param_y in parsed_pairs:
        sp_x, sp_y = params_by_key[param_x], params_by_key[param_y]
        print(f"Sweeping {param_x} x {param_y} ({args.grid_n}x{args.grid_n})...", flush=True)
        grid = _compute_grid(ps, sp_x, sp_y, args.grid_n)

        slug = output_dir / f"{param_x}__{param_y}"
        out_csv = slug.with_name(slug.name + ".csv")
        _write_csv(out_csv, param_x, param_y, grid)
        _plot_npv_heatmap(slug.with_name(slug.name + "_npv.png"), sp_x, sp_y, grid)
        _plot_payback_heatmap(
            slug.with_name(slug.name + "_payback.png"), sp_x, sp_y, grid, device_lifetime_years
        )
        for suffix in (".csv", "_npv.png", "_payback.png"):
            print(f"  Wrote {slug.with_name(slug.name + suffix)}")


def _compute_grid(ps, sp_x, sp_y, grid_n: int) -> dict[str, np.ndarray]:
    xs = np.linspace(sp_x.lo, sp_x.hi, grid_n)
    ys = np.linspace(sp_y.lo, sp_y.hi, grid_n)
    if sp_x.is_int:
        xs = np.round(xs)
    if sp_y.is_int:
        ys = np.round(ys)

    grids = {key: np.full((grid_n, grid_n), np.nan) for key in _ROW_METRIC_KEYS}
    for iy, vy in enumerate(ys):
        for ix, vx in enumerate(xs):
            cfg, profile, econ = ps._apply_overrides({sp_x.key: float(vx), sp_y.key: float(vy)})
            water_price = ps._BASELINE_WATER_PRICE_USD_PER_M3
            for sp, v in ((sp_x, vx), (sp_y, vy)):
                if sp.key == "water_price_usd_per_m3":
                    water_price = float(v)
            row = ps._simulate_and_lcow(profile, cfg, econ, water_price)
            for key in _ROW_METRIC_KEYS:
                grids[key][iy, ix] = row[key]
    return {"xs": xs, "ys": ys, **grids}


def _write_csv(out_csv: Path, param_x: str, param_y: str, grid: dict[str, np.ndarray]) -> None:
    xs, ys = grid["xs"], grid["ys"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[param_x, param_y, *_ROW_METRIC_KEYS])
        w.writeheader()
        for iy in range(len(ys)):
            for ix in range(len(xs)):
                w.writerow({
                    param_x: float(xs[ix]),
                    param_y: float(ys[iy]),
                    **{k: float(grid[k][iy, ix]) for k in _ROW_METRIC_KEYS},
                })


def _plot_npv_heatmap(out_png: Path, sp_x, sp_y, grid: dict[str, np.ndarray]) -> None:
    xs, ys, npv = grid["xs"], grid["ys"], grid["npv_usd_per_m2"]
    finite = npv[np.isfinite(npv)]
    vmin = min(float(finite.min()), -1e-6) if finite.size else -1.0
    vmax = max(float(finite.max()), 1e-6) if finite.size else 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(
        xs, ys, npv, cmap=plt.get_cmap("RdYlGn"),
        norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), shading="nearest",
    )
    if finite.size and float(finite.min()) < 0.0 < float(finite.max()):
        try:
            ax.contour(xs, ys, npv, levels=[0.0], colors="black", linewidths=1.5)
        except ValueError:
            pass
    _finish_heatmap(fig, ax, mesh, out_png, sp_x, sp_y, "NPV (USD/m²)")


def _plot_payback_heatmap(
    out_png: Path, sp_x, sp_y, grid: dict[str, np.ndarray], device_lifetime_years: float
) -> None:
    xs, ys, payback = grid["xs"], grid["ys"], grid["payback_years_simple"]
    infeasible = ~np.isfinite(payback) | (payback < 0.0) | (payback > device_lifetime_years)

    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(
        xs, ys, np.where(infeasible, np.nan, payback), cmap=plt.get_cmap("viridis_r"),
        norm=mcolors.Normalize(vmin=0.0, vmax=max(device_lifetime_years, 1e-6)), shading="nearest",
    )

    # shading="nearest" centres cells on the sample points; hatch the infeasible ones.
    xe, ye = _cell_edges(xs), _cell_edges(ys)
    for iy, ix in np.argwhere(infeasible):
        ax.add_patch(Rectangle(
            (xe[ix], ye[iy]), xe[ix + 1] - xe[ix], ye[iy + 1] - ye[iy],
            facecolor="0.6", edgecolor="0.3", linewidth=0.2, hatch="///", zorder=3,
        ))

    _finish_heatmap(
        fig, ax, mesh, out_png, sp_x, sp_y, "Simple payback (years)",
        title=f"Simple payback (years)\nHatched = infeasible / never pays back "
              f"within {device_lifetime_years:g}-yr device lifetime",
    )


def _cell_edges(v: np.ndarray) -> np.ndarray:
    d = np.diff(v) if len(v) > 1 else np.array([1.0])
    return np.concatenate(([v[0] - d[0] / 2.0], v[:-1] + d / 2.0, [v[-1] + d[-1] / 2.0]))


def _finish_heatmap(fig, ax, mesh, out_png: Path, sp_x, sp_y, cbar_label: str, title: str | None = None) -> None:
    ax.plot(
        sp_x.baseline, sp_y.baseline, marker="*", markersize=16, markeredgecolor="black",
        markerfacecolor="white", linestyle="none", label="Baseline", zorder=5,
    )
    ax.legend(loc="best", frameon=True)
    ax.set_xlabel(sp_x.label)
    ax.set_ylabel(sp_y.label)
    ax.set_title(title or cbar_label)
    fig.colorbar(mesh, ax=ax).set_label(cbar_label)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
