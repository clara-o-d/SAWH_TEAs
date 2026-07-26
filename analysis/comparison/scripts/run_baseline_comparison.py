#!/usr/bin/env python3
"""Run all four SAWH configs at a shared baseline scenario and compare NPV/LCOW.

Answers the top-level question at a single operating point: "at a given
water price and heat-input level, how do passive and active SAWH devices
compare on cost?" For the parameter-space version of that question, see
``grid_heatmap.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from comparison.lib.adapters import ALL_CONFIG_IDS, get_adapters  # noqa: E402
from comparison.lib.scenario import BASELINE_SCENARIO  # noqa: E402

_DEFAULT_OUT_CSV = _REPO_ROOT / "comparison" / "outputs" / "baseline" / "baseline_comparison.csv"
_DEFAULT_OUT_SCENARIO_JSON = (
    _REPO_ROOT / "comparison" / "outputs" / "baseline" / "baseline_scenario.json"
)

_CSV_COLUMNS: tuple[str, ...] = (
    "config_id",
    "display_name",
    "cycles_per_day",
    "daily_yield_kg_per_m2",
    "thermal_efficiency",
    "heat_input_frac",
    "heat_input_physical_value",
    "heat_input_unit",
    "capex_usd_per_m2",
    "annual_revenue_usd_per_m2",
    "annual_opex_usd_per_m2",
    "annual_net_cash_flow_usd_per_m2",
    "npv_usd_per_m2",
    "payback_years_simple",
    "payback_years_discounted",
    "lcow_usd_per_m3",
    "water_price_usd_per_m3",
    "device_lifetime_years",
    "discount_rate",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--water-price-usd-per-m3",
        type=float,
        default=BASELINE_SCENARIO.water_price_usd_per_m3,
    )
    p.add_argument("--heat-input-frac", type=float, default=1.0)
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(ALL_CONFIG_IDS),
        choices=list(ALL_CONFIG_IDS),
    )
    p.add_argument("--out-csv", type=Path, default=_DEFAULT_OUT_CSV)
    p.add_argument("--out-scenario-json", type=Path, default=_DEFAULT_OUT_SCENARIO_JSON)
    return p.parse_args()


def run_comparison(
    *,
    config_ids: list[str],
    water_price_usd_per_m3: float,
    heat_input_frac: float,
) -> list[dict]:
    adapters = get_adapters(config_ids)
    rows: list[dict] = []
    for config_id, adapter in adapters.items():
        econ = adapter.econ_defaults()
        sim = adapter.simulate(econ=econ, heat_input_frac=heat_input_frac)
        npv = adapter.npv(
            sim.daily_yield_kg_per_m2,
            water_price_usd_per_m3,
            econ=sim.econ,
            cycles_per_day=sim.cycles_per_day,
            **sim.material_kwargs,
        )
        lcow = adapter.lcow(
            sim.daily_yield_kg_per_m2,
            econ=sim.econ,
            cycles_per_day=sim.cycles_per_day,
            **sim.material_kwargs,
        )
        row = {
            "config_id": config_id,
            "display_name": adapter.display_name,
            "cycles_per_day": sim.cycles_per_day,
            "daily_yield_kg_per_m2": sim.daily_yield_kg_per_m2,
            "thermal_efficiency": sim.thermal_efficiency,
            "heat_input_frac": sim.heat_input_frac,
            "heat_input_physical_value": sim.heat_input_physical_value,
            "heat_input_unit": sim.heat_input_unit,
            "capex_usd_per_m2": npv.capex_usd_per_m2 if npv else float("nan"),
            "annual_revenue_usd_per_m2": npv.annual_revenue_usd_per_m2 if npv else float("nan"),
            "annual_opex_usd_per_m2": npv.annual_opex_usd_per_m2 if npv else float("nan"),
            "annual_net_cash_flow_usd_per_m2": (
                npv.annual_net_cash_flow_usd_per_m2 if npv else float("nan")
            ),
            "npv_usd_per_m2": npv.npv_usd_per_m2 if npv else float("nan"),
            "payback_years_simple": npv.payback_years_simple if npv else float("inf"),
            "payback_years_discounted": npv.payback_years_discounted if npv else float("inf"),
            "lcow_usd_per_m3": lcow,
            "water_price_usd_per_m3": water_price_usd_per_m3,
            "device_lifetime_years": sim.econ.device_lifetime_years,
            "discount_rate": sim.econ.discount_rate,
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_CSV_COLUMNS))
        w.writeheader()
        w.writerows(rows)


def print_table(rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["npv_usd_per_m2"], reverse=True)
    headers = [
        "config_id",
        "cycles/day",
        "yield kg/m2/d",
        "eta",
        "capex $/m2",
        "NPV $/m2",
        "payback (yr)",
        "LCOW $/m3",
    ]
    widths = [14, 10, 13, 8, 11, 14, 12, 12]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows_sorted:
        payback = r["payback_years_simple"]
        payback_str = f"{payback:.2f}" if payback != float("inf") else "inf"
        vals = [
            r["config_id"],
            f"{r['cycles_per_day']:.1f}",
            f"{r['daily_yield_kg_per_m2']:.3f}",
            f"{r['thermal_efficiency']:.3f}",
            f"{r['capex_usd_per_m2']:.2f}",
            f"{r['npv_usd_per_m2']:.2f}",
            payback_str,
            f"{r['lcow_usd_per_m3']:.3f}",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(vals, widths)))


def main() -> int:
    args = parse_args()
    rows = run_comparison(
        config_ids=args.configs,
        water_price_usd_per_m3=args.water_price_usd_per_m3,
        heat_input_frac=args.heat_input_frac,
    )
    print(
        f"Baseline comparison: water_price=${args.water_price_usd_per_m3:.2f}/m3, "
        f"heat_input_frac={args.heat_input_frac:.2f}\n"
    )
    print_table(rows)

    write_csv(rows, args.out_csv)
    print(f"\nWrote {args.out_csv}")

    scenario_dict = {
        "t_amb_c": BASELINE_SCENARIO.t_amb_c,
        "rh_amb": BASELINE_SCENARIO.rh_amb,
        "h_amb_w_m2_k": BASELINE_SCENARIO.h_amb_w_m2_k,
        "salt_name": BASELINE_SCENARIO.salt_name,
        "salt_to_polymer_ratio": BASELINE_SCENARIO.salt_to_polymer_ratio,
        "hydrogel_thickness_m": BASELINE_SCENARIO.hydrogel_thickness_m,
        "device_lifetime_years": BASELINE_SCENARIO.device_lifetime_years,
        "discount_rate": BASELINE_SCENARIO.discount_rate,
        "electricity_price_usd_per_kwh": BASELINE_SCENARIO.electricity_price_usd_per_kwh,
        "water_price_usd_per_m3": args.water_price_usd_per_m3,
        "heat_input_frac": args.heat_input_frac,
        "configs": args.configs,
    }
    args.out_scenario_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_scenario_json.open("w") as f:
        json.dump(scenario_dict, f, indent=2)
    print(f"Wrote {args.out_scenario_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
