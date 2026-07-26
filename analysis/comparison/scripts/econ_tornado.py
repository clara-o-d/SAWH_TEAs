#!/usr/bin/env python3
"""Economic-parameter tornado sensitivity of NPV, shared across all four configs.

Since all four packages share the exact same ``LCOEconomicParams`` schema,
this needs no re-simulation: each config is simulated once at the baseline
heat input, then only the *economic* parameters (water price, financing,
electricity price, etc.) are perturbed and NPV is recomputed directly from
the cached daily yield / cycles-per-day -- cheap, and lets us compare
elasticities across configs on equal footing.

Elasticity metric and two-sided (increase/decrease) convention follow
``solar_lumped/scripts/tornado_plot.py``'s ``calculate_sensitivity``: percent
change in NPV per percent change in the parameter, computed separately for
the parameter-increase (baseline -> high) and parameter-decrease
(baseline -> low) directions.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from comparison.lib.adapters import ALL_CONFIG_IDS, _replace_econ, get_adapters  # noqa: E402
from comparison.lib.scenario import BASELINE_SCENARIO  # noqa: E402

_DEFAULT_OUT_NPV_CSV = _REPO_ROOT / "comparison" / "outputs" / "tornado" / "econ_tornado_npv.csv"
_DEFAULT_OUT_RANKING_CSV = (
    _REPO_ROOT / "comparison" / "outputs" / "tornado" / "econ_tornado_ranking.csv"
)
_DEFAULT_OUT_PNG = _REPO_ROOT / "comparison" / "outputs" / "tornado" / "econ_tornado_npv.png"

# (econ_field_or_special, low, high) -- "water_price_usd_per_m3" is special: it
# perturbs the price argument to npv(), not an LCOEconomicParams field.
_PERTURBATIONS: tuple[tuple[str, float, float], ...] = (
    ("water_price_usd_per_m3", 0.5, 50.0),
    ("total_investment_factor", 0.5, 2.0),
    ("electricity_price_usd_per_kwh", 0.05, 0.30),
    ("discount_rate", 0.04, 0.12),
    ("device_lifetime_years", 10, 30),
    ("maintenance_cost_fraction", 0.02, 0.10),
    ("utilization_factor", 0.7, 1.0),
    ("hydrogel_lifetime_years", 0.5, 2.0),
)

_PARAM_LABELS: dict[str, str] = {
    "water_price_usd_per_m3": "Water price\n(USD/m3)",
    "total_investment_factor": "Total investment\nfactor",
    "electricity_price_usd_per_kwh": "Electricity price\n(USD/kWh)",
    "discount_rate": "Discount rate",
    "device_lifetime_years": "Device lifetime\n(yr)",
    "maintenance_cost_fraction": "Maintenance cost\nfraction",
    "utilization_factor": "Utilization factor",
    "hydrogel_lifetime_years": "Hydrogel lifetime\n(yr)",
}


def calculate_sensitivity(x1: float, y1: float, x2: float, y2: float) -> tuple[float, bool]:
    """Percent sensitivity between two (x, y) points -- identical formula to
    ``solar_lumped/scripts/tornado_plot.py``'s ``calculate_sensitivity``."""
    if abs(x1) < 1e-10:
        pct_change_x = abs(x2 - x1)
    else:
        pct_change_x = (x2 - x1) / x1 * 100

    if abs(y1) < 1e-10:
        pct_change_y = y2 - y1
    else:
        pct_change_y = (y2 - y1) / y1 * 100

    if abs(pct_change_x) > 0.01:
        sensitivity = pct_change_y / pct_change_x
        if abs(sensitivity) < 1000:
            return sensitivity, True

    return 0.0, False


@dataclass
class TornadoRow:
    config_id: str
    parameter: str
    low_value: float
    high_value: float
    baseline_value: float
    npv_at_low: float
    npv_at_high: float
    npv_at_baseline: float
    increase_sensitivity: float
    decrease_sensitivity: float
    max_abs_sensitivity: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--configs", nargs="+", default=list(ALL_CONFIG_IDS), choices=list(ALL_CONFIG_IDS)
    )
    p.add_argument("--heat-input-frac", type=float, default=1.0)
    p.add_argument(
        "--water-price-usd-per-m3",
        type=float,
        default=BASELINE_SCENARIO.water_price_usd_per_m3,
    )
    p.add_argument("--out-csv", type=Path, default=_DEFAULT_OUT_NPV_CSV)
    p.add_argument("--out-ranking-csv", type=Path, default=_DEFAULT_OUT_RANKING_CSV)
    p.add_argument("--out-png", type=Path, default=_DEFAULT_OUT_PNG)
    return p.parse_args()


def _npv_value(adapter, sim, econ, water_price: float) -> float:
    result = adapter.npv(
        sim.daily_yield_kg_per_m2,
        water_price,
        econ=econ,
        cycles_per_day=sim.cycles_per_day,
        **sim.material_kwargs,
    )
    return result.npv_usd_per_m2 if result is not None else float("nan")


def compute_tornado(
    config_ids: list[str],
    *,
    heat_input_frac: float,
    baseline_water_price: float,
) -> list[TornadoRow]:
    adapters = get_adapters(config_ids)
    rows: list[TornadoRow] = []

    for config_id, adapter in adapters.items():
        econ_baseline = adapter.econ_defaults()
        sim = adapter.simulate(econ=econ_baseline, heat_input_frac=heat_input_frac)
        econ_baseline = sim.econ  # simulate() may not mutate econ, but stay consistent

        for param, low, high in _PERTURBATIONS:
            if param == "water_price_usd_per_m3":
                baseline_value = baseline_water_price
                npv_baseline = _npv_value(adapter, sim, econ_baseline, baseline_value)
                npv_low = _npv_value(adapter, sim, econ_baseline, low)
                npv_high = _npv_value(adapter, sim, econ_baseline, high)
            else:
                baseline_value = getattr(econ_baseline, param)
                econ_low = _replace_econ(econ_baseline, **{param: low})
                econ_high = _replace_econ(econ_baseline, **{param: high})
                npv_baseline = _npv_value(adapter, sim, econ_baseline, baseline_water_price)
                npv_low = _npv_value(adapter, sim, econ_low, baseline_water_price)
                npv_high = _npv_value(adapter, sim, econ_high, baseline_water_price)

            inc_sens, _ = calculate_sensitivity(baseline_value, npv_baseline, high, npv_high)
            dec_sens, _ = calculate_sensitivity(baseline_value, npv_baseline, low, npv_low)

            rows.append(
                TornadoRow(
                    config_id=config_id,
                    parameter=param,
                    low_value=low,
                    high_value=high,
                    baseline_value=baseline_value,
                    npv_at_low=npv_low,
                    npv_at_high=npv_high,
                    npv_at_baseline=npv_baseline,
                    increase_sensitivity=inc_sens,
                    decrease_sensitivity=dec_sens,
                    max_abs_sensitivity=max(abs(inc_sens), abs(dec_sens)),
                )
            )
    return rows


def write_npv_csv(rows: list[TornadoRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id",
        "parameter",
        "low_value",
        "high_value",
        "baseline_value",
        "npv_at_low",
        "npv_at_high",
        "npv_at_baseline",
        "increase_sensitivity",
        "decrease_sensitivity",
        "max_abs_sensitivity",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "config_id": r.config_id,
                    "parameter": r.parameter,
                    "low_value": r.low_value,
                    "high_value": r.high_value,
                    "baseline_value": r.baseline_value,
                    "npv_at_low": r.npv_at_low,
                    "npv_at_high": r.npv_at_high,
                    "npv_at_baseline": r.npv_at_baseline,
                    "increase_sensitivity": r.increase_sensitivity,
                    "decrease_sensitivity": r.decrease_sensitivity,
                    "max_abs_sensitivity": r.max_abs_sensitivity,
                }
            )


def build_ranking(rows: list[TornadoRow]) -> list[dict]:
    by_param: dict[str, list[TornadoRow]] = {}
    for r in rows:
        by_param.setdefault(r.parameter, []).append(r)

    ranking: list[dict] = []
    for param, param_rows in by_param.items():
        best = max(param_rows, key=lambda r: r.max_abs_sensitivity)
        mean_abs = sum(r.max_abs_sensitivity for r in param_rows) / len(param_rows)
        ranking.append(
            {
                "parameter": param,
                "max_abs_sensitivity_across_configs": best.max_abs_sensitivity,
                "config_with_max": best.config_id,
                "mean_abs_sensitivity_across_configs": mean_abs,
            }
        )
    ranking.sort(key=lambda r: r["max_abs_sensitivity_across_configs"], reverse=True)
    return ranking


def write_ranking_csv(ranking: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "parameter",
        "max_abs_sensitivity_across_configs",
        "config_with_max",
        "mean_abs_sensitivity_across_configs",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(ranking)


def plot_tornado(
    rows: list[TornadoRow],
    ranking: list[dict],
    config_ids: list[str],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    from comparison.lib.adapters import get_adapter

    params_sorted = [r["parameter"] for r in ranking]
    n_params = len(params_sorted)
    n_configs = len(config_ids)

    by_key: dict[tuple[str, str], TornadoRow] = {(r.config_id, r.parameter): r for r in rows}

    fig, ax = plt.subplots(figsize=(9, max(4, 1.0 * n_params)))
    group_height = 0.8
    bar_height = group_height / n_configs
    y_base = np.arange(n_params)[::-1]  # largest sensitivity at top

    for ci, cid in enumerate(config_ids):
        adapter = get_adapter(cid)
        offset = (ci - (n_configs - 1) / 2.0) * bar_height
        y = y_base + offset
        inc_vals = [by_key[(cid, p)].increase_sensitivity for p in params_sorted]
        dec_vals = [by_key[(cid, p)].decrease_sensitivity for p in params_sorted]
        ax.barh(
            y,
            inc_vals,
            height=bar_height * 0.9,
            color=adapter.color,
            label=adapter.display_name,
        )
        ax.barh(
            y,
            [-d for d in dec_vals],
            height=bar_height * 0.9,
            color=adapter.color,
            hatch="///",
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels([_PARAM_LABELS.get(p, p) for p in params_sorted], fontsize=10)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlabel("% change in NPV per % change in parameter\n(solid = parameter increase, hatched = parameter decrease)")
    ax.set_title("Economic-parameter NPV sensitivity by config", fontsize=13, fontweight="bold")

    config_handles = [
        Patch(facecolor=get_adapter(cid).color, label=get_adapter(cid).display_name)
        for cid in config_ids
    ]
    hatch_handle = Patch(facecolor="white", edgecolor="black", hatch="///", label="Parameter decrease")
    ax.legend(handles=[*config_handles, hatch_handle], loc="best", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = compute_tornado(
        args.configs,
        heat_input_frac=args.heat_input_frac,
        baseline_water_price=args.water_price_usd_per_m3,
    )
    write_npv_csv(rows, args.out_csv)
    print(f"Wrote {args.out_csv}")

    ranking = build_ranking(rows)
    write_ranking_csv(ranking, args.out_ranking_csv)
    print(f"Wrote {args.out_ranking_csv}")

    plot_tornado(rows, ranking, args.configs, args.out_png)
    print(f"Wrote {args.out_png}")

    print("\nParameter ranking (max |sensitivity| across configs, descending):")
    for r in ranking:
        print(
            f"  {r['parameter']:32s} max={r['max_abs_sensitivity_across_configs']:8.3f} "
            f"({r['config_with_max']:12s}) mean={r['mean_abs_sensitivity_across_configs']:8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
