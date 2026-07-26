#!/usr/bin/env python3
"""2D parameter-grid winner/margin maps across all four SAWH configs.

Answers "which parameter combinations make active vs. passive comparable,
and which make one dominate?" For each grid cell, every requested config is
simulated/evaluated and the winning config (best NPV, or lowest LCOW) and its
margin over the runner-up are recorded.

Optimization: ``heat_input_frac`` is the only parameter here that requires
re-solving the device ODEs; every other parameter (water price, financing
terms, ...) is purely economic and can be recomputed from a cached
``SimOutput`` at zero extra simulation cost. If ``heat_input_frac`` is one of
the two swept axes, each config is simulated once per unique value on that
axis and the other (economic) axis is swept for free. If neither axis is
physics-linked, each config is simulated exactly once total. If (in a future
extension) both axes were physics-linked, this script solves the full
``n_x * n_y`` grid and prints a warning to that effect.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from comparison.lib import plotting  # noqa: E402
from comparison.lib.adapters import ALL_CONFIG_IDS, _replace_econ, get_adapters  # noqa: E402
from comparison.lib.scenario import BASELINE_SCENARIO  # noqa: E402

_OUT_DIR = _REPO_ROOT / "comparison" / "outputs" / "heatmaps"

PHYSICS_PARAMS: frozenset[str] = frozenset({"heat_input_frac"})
ECON_PARAMS: frozenset[str] = frozenset(
    {
        "water_price_usd_per_m3",
        "total_investment_factor",
        "electricity_price_usd_per_kwh",
        "discount_rate",
        "device_lifetime_years",
        "maintenance_cost_fraction",
        "utilization_factor",
        "hydrogel_lifetime_years",
    }
)
ALL_PARAMS: frozenset[str] = PHYSICS_PARAMS | ECON_PARAMS

_PARAM_LABELS: dict[str, str] = {
    "water_price_usd_per_m3": "Water price (USD/m3)",
    "heat_input_frac": "Heat input fraction",
    "total_investment_factor": "Total investment factor",
    "electricity_price_usd_per_kwh": "Electricity price (USD/kWh)",
    "discount_rate": "Discount rate",
    "device_lifetime_years": "Device lifetime (yr)",
    "maintenance_cost_fraction": "Maintenance cost fraction",
    "utilization_factor": "Utilization factor",
    "hydrogel_lifetime_years": "Hydrogel lifetime (yr)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--param-x", default="water_price_usd_per_m3", choices=sorted(ALL_PARAMS))
    p.add_argument("--range-x", nargs=2, type=float, default=[0.5, 50.0])
    p.add_argument("--log-x", dest="log_x", action="store_true", default=True)
    p.add_argument("--no-log-x", dest="log_x", action="store_false")
    p.add_argument("--n-x", type=int, default=25)
    p.add_argument("--param-y", default="heat_input_frac", choices=sorted(ALL_PARAMS))
    p.add_argument("--range-y", nargs=2, type=float, default=[0.0, 1.0])
    p.add_argument("--n-y", type=int, default=21)
    p.add_argument(
        "--configs", nargs="+", default=list(ALL_CONFIG_IDS), choices=list(ALL_CONFIG_IDS)
    )
    p.add_argument("--metric", choices=("npv", "lcow"), default="npv")
    p.add_argument("--comparable-threshold", type=float, default=0.10)
    p.add_argument("--output-dir", type=Path, default=_OUT_DIR)
    p.add_argument("--tag", default=None, help="Output filename tag (default: paramx_x_paramy)")
    return p.parse_args()


def _axis_values(lo: float, hi: float, n: int, *, log: bool) -> np.ndarray:
    if log:
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError("--log-x requires a strictly positive range")
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def _metric_value(adapter, sim, econ, water_price: float, metric: str) -> tuple[float, dict]:
    """Compute the requested metric plus a small dict of extra reported fields."""
    npv = adapter.npv(
        sim.daily_yield_kg_per_m2,
        water_price,
        econ=econ,
        cycles_per_day=sim.cycles_per_day,
        **sim.material_kwargs,
    )
    lcow = adapter.lcow(
        sim.daily_yield_kg_per_m2,
        econ=econ,
        cycles_per_day=sim.cycles_per_day,
        **sim.material_kwargs,
    )
    extra = {
        "daily_yield_kg_per_m2": sim.daily_yield_kg_per_m2,
        "thermal_efficiency": sim.thermal_efficiency,
        "cycles_per_day": sim.cycles_per_day,
        "capex_usd_per_m2": npv.capex_usd_per_m2 if npv else float("nan"),
        "npv_usd_per_m2": npv.npv_usd_per_m2 if npv else float("nan"),
        "lcow_usd_per_m3": lcow,
        "payback_years_simple": npv.payback_years_simple if npv else float("inf"),
        "payback_years_discounted": npv.payback_years_discounted if npv else float("inf"),
    }
    value = extra["npv_usd_per_m2"] if metric == "npv" else extra["lcow_usd_per_m3"]
    return value, extra


def _apply_param(econ_base, water_price_base: float, param: str, value: float):
    """Return ``(econ, water_price)`` after applying one non-physics parameter override."""
    if param == "water_price_usd_per_m3":
        return econ_base, value
    if param in ECON_PARAMS:
        cast = int(round(value)) if param == "device_lifetime_years" else value
        return _replace_econ(econ_base, **{param: cast}), water_price_base
    raise ValueError(f"{param!r} is not an economic parameter")


def compute_grid(
    *,
    config_ids: list[str],
    param_x: str,
    x_vals: np.ndarray,
    param_y: str,
    y_vals: np.ndarray,
    metric: str,
    water_price_default: float,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    x_is_physics = param_x in PHYSICS_PARAMS
    y_is_physics = param_y in PHYSICS_PARAMS

    if x_is_physics and y_is_physics:
        print(
            f"WARNING: both axes ({param_x!r}, {param_y!r}) require re-solving the device "
            f"ODEs -- this will run the full {len(x_vals)}*{len(y_vals)}="
            f"{len(x_vals) * len(y_vals)}-point grid per config. Consider a coarser --n-x/--n-y.",
            file=sys.stderr,
        )

    adapters = get_adapters(config_ids)
    ny, nx = len(y_vals), len(x_vals)
    metric_grids: dict[str, np.ndarray] = {cid: np.full((ny, nx), np.nan) for cid in config_ids}
    rows: list[dict] = []

    for cid, adapter in adapters.items():
        econ_defaults = adapter.econ_defaults()

        # Cache SimOutput by physics-axis value (or a single baseline entry).
        sim_cache: dict[Any, Any] = {}

        def get_sim(x_val: float, y_val: float) -> Any:
            if x_is_physics and y_is_physics:
                key = (round(x_val, 12), round(y_val, 12))
                heat_input_frac = x_val  # degenerate: both claim heat_input_frac
            elif x_is_physics:
                key = round(x_val, 12)
                heat_input_frac = x_val
            elif y_is_physics:
                key = round(y_val, 12)
                heat_input_frac = y_val
            else:
                key = "baseline"
                heat_input_frac = 1.0
            if key not in sim_cache:
                sim_cache[key] = adapter.simulate(econ=econ_defaults, heat_input_frac=heat_input_frac)
            return sim_cache[key]

        for iy, y_val in enumerate(y_vals):
            for ix, x_val in enumerate(x_vals):
                sim = get_sim(x_val, y_val)
                econ = sim.econ
                water_price = water_price_default
                if not x_is_physics:
                    econ, water_price = _apply_param(econ, water_price, param_x, x_val)
                if not y_is_physics:
                    econ, water_price = _apply_param(econ, water_price, param_y, y_val)

                value, extra = _metric_value(adapter, sim, econ, water_price, metric)
                metric_grids[cid][iy, ix] = value
                rows.append(
                    {
                        "config_id": cid,
                        "param_x_value": x_val,
                        "param_y_value": y_val,
                        **extra,
                    }
                )

        n_sims = len(sim_cache)
        print(f"  {cid}: {n_sims} simulate() call(s), {nx * ny} grid cells", flush=True)

    return metric_grids, rows


def build_winner_table(
    config_ids: list[str],
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    metric_grids: dict[str, np.ndarray],
    metric: str,
    comparable_threshold: float,
) -> list[dict]:
    higher_is_better = metric == "npv"
    winner_rows: list[dict] = []
    for iy, y_val in enumerate(y_vals):
        for ix, x_val in enumerate(x_vals):
            values = [(cid, metric_grids[cid][iy, ix]) for cid in config_ids]
            values = [(cid, v) for cid, v in values if np.isfinite(v)]
            if not values:
                continue
            values.sort(key=lambda cv: cv[1], reverse=higher_is_better)
            best_cid, best_val = values[0]
            if len(values) > 1:
                second_cid, second_val = values[1]
            else:
                second_cid, second_val = None, float("nan")

            margin_abs = abs(best_val - second_val) if np.isfinite(second_val) else float("nan")
            denom = abs(best_val) if abs(best_val) > 1e-12 else np.nan
            margin_frac = margin_abs / denom if np.isfinite(margin_abs) and np.isfinite(denom) else float("nan")
            is_comparable = bool(np.isfinite(margin_frac) and margin_frac < comparable_threshold)

            winner_rows.append(
                {
                    "param_x_value": x_val,
                    "param_y_value": y_val,
                    "winner_config_id": best_cid,
                    "winner_metric_value": best_val,
                    "second_best_config_id": second_cid,
                    "second_best_metric_value": second_val,
                    "margin_abs": margin_abs,
                    "margin_frac": margin_frac,
                    "is_comparable": is_comparable,
                }
            )
    return winner_rows


def write_long_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id",
        "param_x_value",
        "param_y_value",
        "daily_yield_kg_per_m2",
        "thermal_efficiency",
        "cycles_per_day",
        "capex_usd_per_m2",
        "npv_usd_per_m2",
        "lcow_usd_per_m3",
        "payback_years_simple",
        "payback_years_discounted",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_winner_csv(winner_rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "param_x_value",
        "param_y_value",
        "winner_config_id",
        "winner_metric_value",
        "second_best_config_id",
        "second_best_metric_value",
        "margin_abs",
        "margin_frac",
        "is_comparable",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(winner_rows)


def _winner_grid(
    config_ids: list[str],
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    winner_rows: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (winner_code_grid, margin_abs_grid, margin_frac_grid), shape (ny, nx)."""
    codes = {cid: i for i, cid in enumerate(config_ids)}
    ny, nx = len(y_vals), len(x_vals)
    winner_code = np.full((ny, nx), -1, dtype=int)
    margin_abs = np.full((ny, nx), np.nan)
    margin_frac = np.full((ny, nx), np.nan)

    x_index = {round(float(v), 12): i for i, v in enumerate(x_vals)}
    y_index = {round(float(v), 12): i for i, v in enumerate(y_vals)}

    for r in winner_rows:
        ix = x_index[round(float(r["param_x_value"]), 12)]
        iy = y_index[round(float(r["param_y_value"]), 12)]
        winner_code[iy, ix] = codes.get(r["winner_config_id"], -1)
        margin_abs[iy, ix] = r["margin_abs"]
        margin_frac[iy, ix] = r["margin_frac"]
    return winner_code, margin_abs, margin_frac


def plot_winner_map(
    config_ids: list[str],
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    winner_code: np.ndarray,
    margin_frac: np.ndarray,
    comparable_threshold: float,
    param_x: str,
    param_y: str,
    metric: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from comparison.lib.adapters import get_adapter

    colors_by_id = {cid: get_adapter(cid).color for cid in config_ids}
    cmap, norm, _codes = plotting.categorical_colormap(config_ids, colors_by_id)

    x_edges = plotting.cell_edges(x_vals)
    y_edges = plotting.cell_edges(y_vals)
    Xe, Ye = np.meshgrid(x_edges, y_edges)

    plot_grid = np.where(winner_code < 0, np.nan, winner_code).astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(Xe, Ye, plot_grid, cmap=cmap, norm=norm, shading="flat")

    comparable = np.isfinite(margin_frac) & (margin_frac < comparable_threshold)
    plotting.add_comparable_hatch(ax, x_edges, y_edges, comparable)

    ax.set_xlabel(_PARAM_LABELS.get(param_x, param_x))
    ax.set_ylabel(_PARAM_LABELS.get(param_y, param_y))
    ax.set_title(
        f"Winning config by {metric.upper()}: {_PARAM_LABELS.get(param_x, param_x)} vs "
        f"{_PARAM_LABELS.get(param_y, param_y)}",
        fontsize=11,
    )

    handles = [Patch(facecolor=colors_by_id[cid], label=get_adapter(cid).display_name) for cid in config_ids]
    handles.append(
        Patch(
            facecolor="none",
            edgecolor=plotting.COMPARABLE_HATCH_COLOR,
            hatch=plotting.COMPARABLE_HATCH,
            label=f"Comparable (margin < {comparable_threshold:.0%})",
        )
    )
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_margin_map(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    margin_abs: np.ndarray,
    margin_frac: np.ndarray,
    comparable_threshold: float,
    param_x: str,
    param_y: str,
    metric: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    x_edges = plotting.cell_edges(x_vals)
    y_edges = plotting.cell_edges(y_vals)
    Xe, Ye = np.meshgrid(x_edges, y_edges)

    finite = margin_abs[np.isfinite(margin_abs)]
    vmax = float(finite.max()) if finite.size else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, 1e-9))

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(Xe, Ye, margin_abs, cmap=plotting.MARGIN_CMAP, norm=norm, shading="flat")
    unit = "USD/m^2" if metric == "npv" else "USD/m^3"
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(f"Margin: best - 2nd best {metric.upper()} ({unit})")

    X, Y = np.meshgrid(x_vals, y_vals)
    frac_finite = margin_frac[np.isfinite(margin_frac)]
    if frac_finite.size and frac_finite.min() < comparable_threshold < frac_finite.max():
        ax.contour(
            X,
            Y,
            np.where(np.isfinite(margin_frac), margin_frac, np.nan),
            levels=[comparable_threshold],
            colors="black",
            linestyles="dashed",
            linewidths=1.5,
        )

    ax.set_xlabel(_PARAM_LABELS.get(param_x, param_x))
    ax.set_ylabel(_PARAM_LABELS.get(param_y, param_y))
    ax.set_title(
        f"Margin (best - 2nd best {metric.upper()}): {_PARAM_LABELS.get(param_x, param_x)} vs "
        f"{_PARAM_LABELS.get(param_y, param_y)}\n(dashed contour = {comparable_threshold:.0%} margin boundary)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.param_x == args.param_y:
        print("--param-x and --param-y must differ", file=sys.stderr)
        return 2

    x_vals = _axis_values(args.range_x[0], args.range_x[1], args.n_x, log=args.log_x)
    y_vals = _axis_values(args.range_y[0], args.range_y[1], args.n_y, log=False)

    tag = args.tag or f"{args.param_x}_x_{args.param_y}"
    print(
        f"=== {args.param_x} ({args.n_x}{'  log' if args.log_x else ''}) x "
        f"{args.param_y} ({args.n_y}) -- metric={args.metric} ==="
    )

    metric_grids, rows = compute_grid(
        config_ids=args.configs,
        param_x=args.param_x,
        x_vals=x_vals,
        param_y=args.param_y,
        y_vals=y_vals,
        metric=args.metric,
        water_price_default=BASELINE_SCENARIO.water_price_usd_per_m3,
    )

    winner_rows = build_winner_table(
        args.configs, x_vals, y_vals, metric_grids, args.metric, args.comparable_threshold
    )
    winner_code, margin_abs, margin_frac = _winner_grid(args.configs, x_vals, y_vals, winner_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = args.output_dir / f"grid_{tag}_{args.metric}_long.csv"
    winner_csv = args.output_dir / f"grid_{tag}_{args.metric}_winner.csv"
    winner_png = args.output_dir / f"grid_{tag}_{args.metric}_winner_map.png"
    margin_png = args.output_dir / f"grid_{tag}_{args.metric}_margin_map.png"

    write_long_csv(rows, long_csv)
    print(f"Wrote {long_csv}")
    write_winner_csv(winner_rows, winner_csv)
    print(f"Wrote {winner_csv}")

    plot_winner_map(
        args.configs,
        x_vals,
        y_vals,
        winner_code,
        margin_frac,
        args.comparable_threshold,
        args.param_x,
        args.param_y,
        args.metric,
        winner_png,
    )
    print(f"Wrote {winner_png}")

    plot_margin_map(
        x_vals,
        y_vals,
        margin_abs,
        margin_frac,
        args.comparable_threshold,
        args.param_x,
        args.param_y,
        args.metric,
        margin_png,
    )
    print(f"Wrote {margin_png}")

    winners_seen = sorted({r["winner_config_id"] for r in winner_rows})
    n_comparable = sum(1 for r in winner_rows if r["is_comparable"])
    print(
        f"\nWinners present in grid: {winners_seen}; "
        f"{n_comparable}/{len(winner_rows)} cells within comparable threshold "
        f"({args.comparable_threshold:.0%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
