#!/usr/bin/env python3
"""Compare NPV/LCOW/payback across the four cataloged hygroscopic salts at a
fixed weather scenario. Salt choice is categorical (not a continuous sweep
axis), so it gets its own bar chart rather than a tornado-plot line item.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_SRC = _REPO / "src"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_solar_sim import (  # noqa: E402
    _lcow_kwargs,
    register_cyclic_warmup_arguments,
    register_solar_sim_arguments,
    resolve_solar_sim_arguments,
    run_solar_simulation,
)
from solar_lumped.economics.npv import npv_from_daily_yield  # noqa: E402
from solar_lumped.economics.params import LCOEconomicParams  # noqa: E402
from solar_lumped.physics.salt_properties import get_salt  # noqa: E402

_SALTS = ("LiCl", "NaCl", "CaCl2", "MgCl2")
_WATER_PRICE_USD_PER_M3 = 5.0
_FAIL_LCO_THRESHOLD = 1e20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    register_solar_sim_arguments(ap)
    register_cyclic_warmup_arguments(ap)
    ap.add_argument("--output-csv", type=Path, default=_REPO / "outputs" / "parameter_sweeps" / "salt_choice_comparison.csv")
    ap.add_argument("--output-png", type=Path, default=_REPO / "outputs" / "tornado_plot" / "salt_choice_comparison.png")
    args = ap.parse_args()
    resolve_solar_sim_arguments(args, ap)

    econ = LCOEconomicParams()
    rows: list[dict] = []
    for salt_name in _SALTS:
        salt_args = copy.copy(args)
        salt_args.salt = salt_name
        try:
            result = run_solar_simulation(salt_args, econ=econ)
        except Exception as exc:  # noqa: BLE001
            print(f"{salt_name}: simulation failed ({exc})")
            rows.append({
                "salt": salt_name, "price_usd_per_kg": get_salt(salt_name).price_usd_per_kg,
                "rh_min": get_salt(salt_name).rh_min, "rh_max": get_salt(salt_name).rh_max,
                "daily_yield_kg_m2": float("nan"), "lcow_usd_per_m3": float("nan"),
                "npv_usd_per_m2": float("nan"), "payback_years_simple": float("nan"),
            })
            continue

        feasible = result.lcow_usd_per_m3 < _FAIL_LCO_THRESHOLD and result.daily_yield_kg_per_m2 > 0
        lcow_kw = _lcow_kwargs(result.config)
        npv_result = None
        if feasible:
            npv_result = npv_from_daily_yield(
                result.daily_yield_kg_per_m2,
                _WATER_PRICE_USD_PER_M3,
                salt_name=result.config.salt_name,
                salt_to_polymer_ratio=result.config.salt_to_polymer_ratio,
                hydrogel_thickness_m=result.config.hydrogel_thickness_m,
                econ=result.econ,
                cycles_per_day=1.0,
                **lcow_kw,
            )
        salt = get_salt(salt_name)
        rows.append({
            "salt": salt_name,
            "price_usd_per_kg": salt.price_usd_per_kg,
            "rh_min": salt.rh_min,
            "rh_max": salt.rh_max,
            "daily_yield_kg_m2": result.daily_yield_kg_per_m2 if feasible else 0.0,
            "lcow_usd_per_m3": result.lcow_usd_per_m3 if feasible else float("nan"),
            "npv_usd_per_m2": npv_result.npv_usd_per_m2 if npv_result else float("nan"),
            "payback_years_simple": npv_result.payback_years_simple if npv_result else float("nan"),
            "feasible": feasible,
        })
        print(
            f"{salt_name}: feasible={feasible} yield={result.daily_yield_kg_per_m2:.3f} kg/m2 "
            f"lcow={result.lcow_usd_per_m3 if feasible else float('nan'):.2f} USD/m3 "
            f"npv={(npv_result.npv_usd_per_m2 if npv_result else float('nan')):.2f} USD/m2"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output_csv}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(_SALTS))
    colors = ["#1b9e77" if r["feasible"] else "#bbbbbb" for r in rows]

    yields_ = [r["daily_yield_kg_m2"] for r in rows]
    axes[0].bar(x, yields_, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(_SALTS)
    axes[0].set_ylabel("Daily yield (kg/m²)")
    axes[0].set_title("Yield")

    lcows = [r["lcow_usd_per_m3"] if np.isfinite(r["lcow_usd_per_m3"]) else 0.0 for r in rows]
    axes[1].bar(x, lcows, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(_SALTS)
    axes[1].set_ylabel("LCOW (USD/m³)")
    axes[1].set_title("Levelized cost of water")

    npvs = [r["npv_usd_per_m2"] if np.isfinite(r["npv_usd_per_m2"]) else 0.0 for r in rows]
    axes[2].bar(x, npvs, color=colors)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(_SALTS)
    axes[2].set_ylabel(f"NPV (USD/m²) @ ${_WATER_PRICE_USD_PER_M3}/m³ water")
    axes[2].set_title("Net present value")

    fig.suptitle(f"Salt choice comparison — {args.weather_mode}", fontsize=13, fontweight="bold")
    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
