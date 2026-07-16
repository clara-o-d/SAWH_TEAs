#!/usr/bin/env python3
"""2D parameter-pair heatmaps of NPV, payback, and LCOW for solar lumped SAWH.

Extends the OAT tornado sensitivity (``parameter_sweep.py`` + ``tornado_plot.py``)
with two-parameter grids: for each ``paramX:paramY`` pair, a full-factorial grid
is evaluated at ``--grid-n`` levels per axis (all other sweep parameters held at
baseline) and rendered as:

  * an NPV heatmap (diverging colormap centered on zero, with the zero-NPV
    contour overlaid — the direct visual answer to "which parameter
    combinations are profitable?"), and
  * a payback-period heatmap (viridis, capped at the device lifetime, with
    infeasible/non-finite cells hatched gray).

A paired CSV (one row per grid point, all metrics) is written alongside each
PNG pair.

Examples::

  python scripts/npv_heatmap.py
  python scripts/npv_heatmap.py --pairs "water_price_usd_per_m3:discount_rate" --grid-n 8
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_SRC = _REPO / "src"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from parameter_sweep import (  # noqa: E402
    _BASELINE_WATER_PRICE_USD_PER_M3,
    _apply_combo,
    _factorial_combos,
    _metrics_from_result,
    _sweep_grid,
    make_sweep_params,
    SweepParam,
)
from run_solar_sim import (  # noqa: E402
    register_cyclic_warmup_arguments,
    register_solar_sim_arguments,
    resolve_solar_sim_arguments,
)
from solar_lumped.economics.params import LCOEconomicParams  # noqa: E402

_DEFAULT_OUTPUT_DIR = _REPO / "outputs" / "npv_heatmaps"

_DEFAULT_PAIRS: tuple[str, ...] = (
    "water_price_usd_per_m3:discount_rate",
    "water_price_usd_per_m3:device_lifetime_years",
    "water_price_usd_per_m3:hydrogel_lifetime_years",
    "water_price_usd_per_m3:solar_irradiance_w_per_m2",
    "hydrogel_thickness_mm:h_amb_w_m2_k",
    "discount_rate:device_lifetime_years",
)

_METRIC_FIELDS: tuple[str, ...] = (
    "daily_yield_kg_m2",
    "thermal_efficiency",
    "lcow_usd_per_m3",
    "capex_usd_per_m3",
    "opex_usd_per_m3",
    "npv_usd_per_m2",
    "payback_years_simple",
    "payback_years_discounted",
)


def _parse_pair(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Pair spec must be 'paramX:paramY', got {spec!r}")
    x_key, y_key = spec.split(":", 1)
    x_key, y_key = x_key.strip(), y_key.strip()
    if not x_key or not y_key:
        raise ValueError(f"Pair spec must be 'paramX:paramY', got {spec!r}")
    return x_key, y_key


def _lookup_params(
    all_params: list[SweepParam], x_key: str, y_key: str
) -> tuple[SweepParam, SweepParam]:
    by_key = {p.key: p for p in all_params}
    for key in (x_key, y_key):
        if key not in by_key:
            available = ", ".join(sorted(by_key))
            raise ValueError(f"Unknown sweep parameter {key!r}. Available: {available}")
    return by_key[x_key], by_key[y_key]


def compute_pair_grid(
    x_sp: SweepParam,
    y_sp: SweepParam,
    grid_n: int,
    base_args: argparse.Namespace,
    base_econ: LCOEconomicParams,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict]]:
    """Full-factorial 2-parameter grid; returns (x_vals, y_vals, metric grids, rows).

    Metric grids are shaped ``(len(y_vals), len(x_vals))`` (row=y, col=x), matching
    ``pcolormesh``'s ``(C.shape == (Y.shape[0]-1, X.shape[1]-1))`` convention.
    """
    x_vals = np.array(_sweep_grid(x_sp, grid_n), dtype=float)
    y_vals = np.array(_sweep_grid(y_sp, grid_n), dtype=float)
    nx, ny = len(x_vals), len(y_vals)

    combos = _factorial_combos([x_sp, y_sp], grid_n)
    grids = {field: np.full((ny, nx), np.nan) for field in _METRIC_FIELDS}
    rows: list[dict] = []

    for idx, combo in enumerate(combos):
        ix, iy = divmod(idx, ny)
        result = _apply_combo(combo, base_args, base_econ)
        water_price = combo.get(
            "water_price_usd_per_m3", _BASELINE_WATER_PRICE_USD_PER_M3
        )
        metrics = _metrics_from_result(result, water_price)
        for field in _METRIC_FIELDS:
            grids[field][iy, ix] = metrics[field]
        rows.append({x_sp.key: combo[x_sp.key], y_sp.key: combo[y_sp.key], **metrics})

    return x_vals, y_vals, grids, rows


def _cell_edges(vals: np.ndarray) -> np.ndarray:
    """Bin edges for ``pcolormesh`` given cell-center coordinates."""
    if len(vals) == 1:
        v = float(vals[0])
        half = 0.5 if v == 0.0 else abs(v) * 0.5
        return np.array([v - half, v + half])
    mid = (vals[:-1] + vals[1:]) / 2.0
    first = vals[0] - (mid[0] - vals[0])
    last = vals[-1] + (vals[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _lifetime_grid(
    x_sp: SweepParam,
    y_sp: SweepParam,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    base_econ: LCOEconomicParams,
) -> np.ndarray:
    """Per-cell device lifetime cap (years) for the payback heatmap.

    When ``device_lifetime_years`` is one of the two swept parameters, the cap
    varies point-to-point with that axis; otherwise it's the fixed baseline.
    """
    ny, nx = len(y_vals), len(x_vals)
    if x_sp.key == "device_lifetime_years":
        return np.tile(x_vals.reshape(1, nx), (ny, 1))
    if y_sp.key == "device_lifetime_years":
        return np.tile(y_vals.reshape(ny, 1), (1, nx))
    return np.full((ny, nx), float(base_econ.device_lifetime_years))


def plot_npv_heatmap(
    x_sp: SweepParam,
    y_sp: SweepParam,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    npv_grid: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    x_edges = _cell_edges(x_vals)
    y_edges = _cell_edges(y_vals)
    Xe, Ye = np.meshgrid(x_edges, y_edges)

    finite = npv_grid[np.isfinite(npv_grid)]
    eps = 1e-6
    if finite.size:
        vmin = min(0.0, float(finite.min())) - eps
        vmax = max(0.0, float(finite.max())) + eps
    else:
        vmin, vmax = -1.0, 1.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(Xe, Ye, npv_grid, cmap="RdYlGn", norm=norm, shading="flat")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("NPV (USD/m²)")

    if finite.size and finite.min() < 0.0 < finite.max():
        X, Y = np.meshgrid(x_vals, y_vals)
        ax.contour(X, Y, npv_grid, levels=[0.0], colors="black", linewidths=1.5)

    ax.plot(
        x_sp.baseline,
        y_sp.baseline,
        marker="*",
        markersize=18,
        markerfacecolor="black",
        markeredgecolor="white",
        markeredgewidth=1.0,
        linestyle="none",
        zorder=5,
        label="Baseline",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_xlabel(x_sp.label.replace("\n", " "))
    ax.set_ylabel(y_sp.label.replace("\n", " "))
    ax.set_title(
        f"NPV (USD/m²): {x_sp.label.replace(chr(10), ' ')} vs "
        f"{y_sp.label.replace(chr(10), ' ')}\n(black contour = break-even, star = baseline)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_payback_heatmap(
    x_sp: SweepParam,
    y_sp: SweepParam,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    payback_grid: np.ndarray,
    base_econ: LCOEconomicParams,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    x_edges = _cell_edges(x_vals)
    y_edges = _cell_edges(y_vals)
    Xe, Ye = np.meshgrid(x_edges, y_edges)

    lifetime_grid = _lifetime_grid(x_sp, y_sp, x_vals, y_vals, base_econ)
    infeasible = ~np.isfinite(payback_grid) | (payback_grid > lifetime_grid)

    vmax = max(float(np.nanmax(lifetime_grid)), 1e-6)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax, clip=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_grid = np.where(infeasible, np.nan, payback_grid)
    pcm = ax.pcolormesh(Xe, Ye, plot_grid, cmap="viridis_r", norm=norm, shading="flat")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Simple payback (years)")

    ny, nx = payback_grid.shape
    for iy in range(ny):
        for ix in range(nx):
            if infeasible[iy, ix]:
                ax.add_patch(
                    Rectangle(
                        (x_edges[ix], y_edges[iy]),
                        x_edges[ix + 1] - x_edges[ix],
                        y_edges[iy + 1] - y_edges[iy],
                        facecolor="0.75",
                        edgecolor="0.4",
                        linewidth=0.2,
                        hatch="///",
                        zorder=4,
                    )
                )

    ax.plot(
        x_sp.baseline,
        y_sp.baseline,
        marker="*",
        markersize=18,
        markerfacecolor="black",
        markeredgecolor="white",
        markeredgewidth=1.0,
        linestyle="none",
        zorder=5,
        label="Baseline",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_xlabel(x_sp.label.replace("\n", " "))
    ax.set_ylabel(y_sp.label.replace("\n", " "))
    ax.set_title(
        f"Simple payback (years): {x_sp.label.replace(chr(10), ' ')} vs "
        f"{y_sp.label.replace(chr(10), ' ')}\n(hatched = never pays back within device lifetime)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_pair_csv(x_key: str, y_key: str, rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [x_key, y_key, *_METRIC_FIELDS]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="2D parameter-pair NPV/payback heatmaps (uses parameter_sweep.py helpers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    register_solar_sim_arguments(ap)
    register_cyclic_warmup_arguments(ap)
    ap.set_defaults(weather_mode="baseline")
    ap.add_argument(
        "--pairs",
        action="append",
        default=None,
        metavar="paramX:paramY",
        help=(
            "Parameter pair to sweep, repeatable "
            f"(default: {', '.join(_DEFAULT_PAIRS)})"
        ),
    )
    ap.add_argument("--grid-n", type=int, default=25, help="Grid levels per axis (default: 25)")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Output directory for PNGs/CSVs (default: outputs/npv_heatmaps/)",
    )
    args = ap.parse_args()

    resolve_solar_sim_arguments(args, ap)
    if args.no_cyclic and args.cyclic:
        ap.error("Cannot use both --cyclic and --no-cyclic")
    if args.grid_n < 2:
        ap.error("--grid-n must be >= 2")

    base_args = copy.copy(args)
    base_econ = LCOEconomicParams()
    all_params = make_sweep_params(base_args, base_econ)

    pair_specs = args.pairs if args.pairs else list(_DEFAULT_PAIRS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for spec in pair_specs:
        x_key, y_key = _parse_pair(spec)
        x_sp, y_sp = _lookup_params(all_params, x_key, y_key)
        print(f"=== {x_key} x {y_key} ({args.grid_n}x{args.grid_n} grid) ===", flush=True)

        x_vals, y_vals, grids, rows = compute_pair_grid(
            x_sp, y_sp, args.grid_n, base_args, base_econ
        )

        csv_path = args.output_dir / f"npv_heatmap_{x_key}_x_{y_key}.csv"
        write_pair_csv(x_key, y_key, rows, csv_path)
        print(f"  Wrote {csv_path}", flush=True)

        npv_png = args.output_dir / f"npv_heatmap_{x_key}_x_{y_key}.png"
        plot_npv_heatmap(x_sp, y_sp, x_vals, y_vals, grids["npv_usd_per_m2"], npv_png)
        print(f"  Wrote {npv_png}", flush=True)

        payback_png = args.output_dir / f"payback_heatmap_{x_key}_x_{y_key}.png"
        plot_payback_heatmap(
            x_sp,
            y_sp,
            x_vals,
            y_vals,
            grids["payback_years_simple"],
            base_econ,
            payback_png,
        )
        print(f"  Wrote {payback_png}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
