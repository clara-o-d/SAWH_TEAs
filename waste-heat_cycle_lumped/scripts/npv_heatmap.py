#!/usr/bin/env python3
"""2D NPV / payback heatmaps over pairs of swept parameters.

Builds on ``_apply_overrides`` (see ``parameter_sweep.py``) so config/profile
/econ construction logic is not re-derived here. For each ``paramX:paramY``
pair, sweeps a ``grid_n x grid_n`` grid (holding every other parameter at
baseline) and renders:

  * an NPV heatmap (diverging colormap centered on zero, black zero-NPV
    contour, baseline point marked), and
  * a payback-period heatmap (viridis_r, capped at the device lifetime,
    gray-hatched cells where payback is infeasible / non-finite),

plus a CSV of every grid point's full metric set.

Note on ``--grid-n``: this package's baseline is default 15, lower than the
single-cycle sibling packages' default of 25. Each grid point here is a full
multi-cycle Radau ODE solve of ``run_daily_operation`` (which can run several
adsorption/desorption half-cycles per simulated day), so it is noticeably
stiffer/costlier per point than a single-cycle solve; 15x15=225 points per
pair keeps a default run tractable.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_SRC = _REPO / "src"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from parameter_sweep import (  # noqa: E402
    _BASELINE_WATER_PRICE_USD_PER_M3,
    BASELINE_ECON,
    _apply_overrides,
    _simulate_and_lcow,
    make_sweep_params,
)

_DEFAULT_OUTPUT_DIR = _REPO / "npv_heatmaps"
_DEFAULT_GRID_N = 15
_DEFAULT_PAIRS: tuple[str, ...] = (
    "water_price_usd_per_m3:discount_rate",
    "water_price_usd_per_m3:device_lifetime_years",
    "water_price_usd_per_m3:hydrogel_lifetime_years",
    "water_price_usd_per_m3:t_wh_in_c",
    "hydrogel_thickness_mm:wh_hx_ua_w_k",
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


def _parse_pair(spec: str) -> tuple[str, str]:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"--pairs entry must be 'paramX:paramY', got {spec!r}")
    return parts[0], parts[1]


def _grid_point(param_x: str, vx: float, param_y: str, vy: float) -> dict:
    cfg, profile, econ = _apply_overrides({param_x: vx, param_y: vy})
    water_price = _BASELINE_WATER_PRICE_USD_PER_M3
    if param_x == "water_price_usd_per_m3":
        water_price = vx
    elif param_y == "water_price_usd_per_m3":
        water_price = vy
    return _simulate_and_lcow(profile, cfg, econ, water_price)


def _compute_grid(
    param_x: str,
    param_y: str,
    grid_n: int,
    params_by_key: dict,
) -> dict[str, np.ndarray]:
    sp_x = params_by_key[param_x]
    sp_y = params_by_key[param_y]
    xs = np.linspace(sp_x.lo, sp_x.hi, grid_n)
    ys = np.linspace(sp_y.lo, sp_y.hi, grid_n)
    if sp_x.is_int:
        xs = np.round(xs)
    if sp_y.is_int:
        ys = np.round(ys)

    grids: dict[str, np.ndarray] = {
        key: np.full((grid_n, grid_n), np.nan) for key in _ROW_METRIC_KEYS
    }
    for iy, vy in enumerate(ys):
        for ix, vx in enumerate(xs):
            row = _grid_point(param_x, float(vx), param_y, float(vy))
            for key in _ROW_METRIC_KEYS:
                grids[key][iy, ix] = row[key]
    return {"xs": xs, "ys": ys, **grids}


def _write_csv(
    out_csv: Path,
    param_x: str,
    param_y: str,
    grid: dict[str, np.ndarray],
) -> None:
    xs = grid["xs"]
    ys = grid["ys"]
    fieldnames = [param_x, param_y, *_ROW_METRIC_KEYS]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for iy in range(len(ys)):
            for ix in range(len(xs)):
                row = {param_x: float(xs[ix]), param_y: float(ys[iy])}
                for key in _ROW_METRIC_KEYS:
                    row[key] = float(grid[key][iy, ix])
                w.writerow(row)


def _plot_npv_heatmap(
    out_png: Path,
    param_x_label: str,
    param_y_label: str,
    grid: dict[str, np.ndarray],
    baseline_x: float,
    baseline_y: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    xs = grid["xs"]
    ys = grid["ys"]
    npv = grid["npv_usd_per_m2"]

    finite = npv[np.isfinite(npv)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = min(float(finite.min()), -1e-6)
        vmax = max(float(finite.max()), 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(xs, ys, npv, cmap=cmap, norm=norm, shading="nearest")
    if finite.size > 0 and float(finite.min()) < 0.0 < float(finite.max()):
        try:
            ax.contour(xs, ys, npv, levels=[0.0], colors="black", linewidths=1.5)
        except ValueError:
            pass
    ax.plot(
        baseline_x,
        baseline_y,
        marker="*",
        markersize=16,
        markeredgecolor="black",
        markerfacecolor="white",
        linestyle="none",
        label="Baseline",
        zorder=5,
    )
    ax.legend(loc="best", frameon=True)
    ax.set_xlabel(param_x_label)
    ax.set_ylabel(param_y_label)
    ax.set_title("NPV (USD/m²)")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("NPV (USD/m²)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_payback_heatmap(
    out_png: Path,
    param_x_label: str,
    param_y_label: str,
    grid: dict[str, np.ndarray],
    baseline_x: float,
    baseline_y: float,
    device_lifetime_years: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    xs = grid["xs"]
    ys = grid["ys"]
    payback = grid["payback_years_simple"]

    infeasible = ~np.isfinite(payback) | (payback < 0.0) | (payback > device_lifetime_years)
    plot_vals = np.where(infeasible, np.nan, payback)

    norm = mcolors.Normalize(vmin=0.0, vmax=max(device_lifetime_years, 1e-6))
    cmap = plt.get_cmap("viridis_r")

    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(xs, ys, plot_vals, cmap=cmap, norm=norm, shading="nearest")

    # Cell edges for hatching rectangles (pcolormesh with shading="nearest"
    # centers cells on the xs/ys sample points).
    dx = np.diff(xs) if len(xs) > 1 else np.array([1.0])
    dy = np.diff(ys) if len(ys) > 1 else np.array([1.0])
    half_dx_lo = np.concatenate(([dx[0]], dx)) / 2.0
    half_dx_hi = np.concatenate((dx, [dx[-1]])) / 2.0
    half_dy_lo = np.concatenate(([dy[0]], dy)) / 2.0
    half_dy_hi = np.concatenate((dy, [dy[-1]])) / 2.0

    for iy in range(len(ys)):
        for ix in range(len(xs)):
            if not infeasible[iy, ix]:
                continue
            x0 = xs[ix] - half_dx_lo[ix]
            width = half_dx_lo[ix] + half_dx_hi[ix]
            y0 = ys[iy] - half_dy_lo[iy]
            height = half_dy_lo[iy] + half_dy_hi[iy]
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    width,
                    height,
                    facecolor="0.6",
                    edgecolor="0.3",
                    linewidth=0.2,
                    hatch="///",
                    zorder=3,
                )
            )

    ax.plot(
        baseline_x,
        baseline_y,
        marker="*",
        markersize=16,
        markeredgecolor="black",
        markerfacecolor="white",
        linestyle="none",
        label="Baseline",
        zorder=5,
    )
    ax.legend(loc="best", frameon=True)
    ax.set_xlabel(param_x_label)
    ax.set_ylabel(param_y_label)
    ax.set_title(
        f"Simple payback (years)\nHatched = infeasible / never pays back "
        f"within {device_lifetime_years:g}-yr device lifetime"
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Simple payback (years)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--pairs",
        action="append",
        default=None,
        help="paramX:paramY (repeatable). Default: a curated list, see _DEFAULT_PAIRS.",
    )
    ap.add_argument("--grid-n", type=int, default=_DEFAULT_GRID_N)
    ap.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    pairs = args.pairs if args.pairs else list(_DEFAULT_PAIRS)
    params_by_key = {p.key: p for p in make_sweep_params()}

    unknown: list[str] = []
    parsed_pairs: list[tuple[str, str]] = []
    for spec in pairs:
        px, py = _parse_pair(spec)
        parsed_pairs.append((px, py))
        for k in (px, py):
            if k not in params_by_key:
                unknown.append(k)
    if unknown:
        raise SystemExit(
            f"Unknown sweep parameter(s) in --pairs: {', '.join(sorted(set(unknown)))}. "
            f"Available: {', '.join(sorted(params_by_key))}"
        )

    device_lifetime_years = float(BASELINE_ECON.device_lifetime_years)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for param_x, param_y in parsed_pairs:
        sp_x = params_by_key[param_x]
        sp_y = params_by_key[param_y]
        print(f"Sweeping {param_x} x {param_y} ({args.grid_n}x{args.grid_n})...", flush=True)
        grid = _compute_grid(param_x, param_y, args.grid_n, params_by_key)

        pair_slug = f"{param_x}__{param_y}"
        out_csv = args.output_dir / f"{pair_slug}.csv"
        out_npv_png = args.output_dir / f"{pair_slug}_npv.png"
        out_payback_png = args.output_dir / f"{pair_slug}_payback.png"

        _write_csv(out_csv, param_x, param_y, grid)
        _plot_npv_heatmap(
            out_npv_png,
            sp_x.label,
            sp_y.label,
            grid,
            baseline_x=sp_x.baseline,
            baseline_y=sp_y.baseline,
        )
        _plot_payback_heatmap(
            out_payback_png,
            sp_x.label,
            sp_y.label,
            grid,
            baseline_x=sp_x.baseline,
            baseline_y=sp_y.baseline,
            device_lifetime_years=device_lifetime_years,
        )
        print(f"  Wrote {out_csv}")
        print(f"  Wrote {out_npv_png}")
        print(f"  Wrote {out_payback_png}")


if __name__ == "__main__":
    main()
