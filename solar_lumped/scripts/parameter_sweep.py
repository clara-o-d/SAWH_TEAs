#!/usr/bin/env python3
"""One-at-a-time parameter sweep for solar lumped SAWH LCOW."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from solar_lumped.economics.lcow import lcow_from_daily_yield
from solar_lumped.economics.params import LCOEconomicParams
from solar_lumped.simulation.annual_yield import simulate_single_day
from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.weather.profiles import baseline_profile


@dataclass(frozen=True, slots=True)
class SweepParam:
    key: str
    label: str
    lo: float
    hi: float
    baseline: float
    is_int: bool = False


BASELINE_CONFIG = DeviceConfig.baseline()
BASELINE_ECON = LCOEconomicParams()


def _baseline_yield() -> float:
    y, _, _ = simulate_single_day(baseline_profile(), BASELINE_CONFIG)
    return y


def _sweep_grid(sp: SweepParam, n: int) -> list[float]:
    vals = list(
        sp.lo + (sp.hi - sp.lo) * i / (n - 1) for i in range(n)
    ) if n > 1 else [sp.baseline]
    if sp.baseline not in vals:
        vals.append(sp.baseline)
    vals = sorted(set(vals))
    if sp.is_int:
        return [float(int(round(v))) for v in vals]
    return vals


def _run_point(sp: SweepParam, value: float) -> dict:
    cfg = DeviceConfig(
        salt_name=BASELINE_CONFIG.salt_name,
        salt_to_polymer_ratio=BASELINE_CONFIG.salt_to_polymer_ratio,
        hydrogel_thickness_m=BASELINE_CONFIG.hydrogel_thickness_m,
        vapor_gap_m=BASELINE_CONFIG.vapor_gap_m,
    )
    econ = LCOEconomicParams()

    if sp.key == "hydrogel_thickness_mm":
        cfg = DeviceConfig(
            hydrogel_thickness_m=value * 1e-3,
            vapor_gap_m=cfg.vapor_gap_m,
            salt_name=cfg.salt_name,
            salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
        )
    elif sp.key == "vapor_gap_mm":
        cfg = DeviceConfig(
            hydrogel_thickness_m=cfg.hydrogel_thickness_m,
            vapor_gap_m=value * 1e-3,
            salt_name=cfg.salt_name,
            salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
        )
    elif sp.key == "humidity_high":
        prof = baseline_profile(relative_humidity=value)
        y, eta, _ = simulate_single_day(prof, cfg)
        lcow = lcow_from_daily_yield(
            y,
            salt_name=cfg.salt_name,
            salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
            hydrogel_thickness_m=cfg.hydrogel_thickness_m,
            econ=econ,
        )
        return {
            "daily_yield_kg_m2": y,
            "thermal_efficiency": eta,
            "lcow_usd_per_m3": lcow,
        }
    elif sp.key == "solar_irradiance_w_per_m2":
        prof = baseline_profile(solar_w_m2=value)
        y, eta, _ = simulate_single_day(prof, cfg)
        lcow = lcow_from_daily_yield(
            y,
            salt_name=cfg.salt_name,
            salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
            hydrogel_thickness_m=cfg.hydrogel_thickness_m,
            econ=econ,
        )
        return {
            "daily_yield_kg_m2": y,
            "thermal_efficiency": eta,
            "lcow_usd_per_m3": lcow,
        }
    elif sp.key == "h_amb_w_m2_k":
        prof = baseline_profile(h_amb_w_m2_k=value)
        y, eta, _ = simulate_single_day(prof, cfg)
        lcow = lcow_from_daily_yield(
            y,
            salt_name=cfg.salt_name,
            salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
            hydrogel_thickness_m=cfg.hydrogel_thickness_m,
            econ=econ,
        )
        return {
            "daily_yield_kg_m2": y,
            "thermal_efficiency": eta,
            "lcow_usd_per_m3": lcow,
        }
    elif sp.key == "discount_rate":
        econ = LCOEconomicParams(discount_rate=value)
    elif sp.key == "device_lifetime_years":
        econ = LCOEconomicParams(device_lifetime_years=int(value))
    elif sp.key == "hydrogel_lifetime_years":
        econ = LCOEconomicParams(hydrogel_lifetime_years=value)
    elif sp.key == "utilization_factor":
        econ = LCOEconomicParams(utilization_factor=value)

    y, eta, _ = simulate_single_day(baseline_profile(), cfg)
    lcow = lcow_from_daily_yield(
        y,
        salt_name=cfg.salt_name,
        salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
        hydrogel_thickness_m=cfg.hydrogel_thickness_m,
        econ=econ,
    )
    return {
        "daily_yield_kg_m2": y,
        "thermal_efficiency": eta,
        "lcow_usd_per_m3": lcow,
    }


def make_sweep_params() -> list[SweepParam]:
    return [
        SweepParam("hydrogel_thickness_mm", "Hydrogel thickness (mm)", 1.0, 10.0, 4.0),
        SweepParam("vapor_gap_mm", "Vapor gap (mm)", 7.0, 60.0, 40.0),
        SweepParam("humidity_high", "Uptake RH", 0.15, 0.80, 0.5),
        SweepParam("solar_irradiance_w_per_m2", "Solar GHI (W/m²)", 400.0, 800.0, 600.0),
        SweepParam("h_amb_w_m2_k", "h_amb (W/m²K)", 1.0, 12.5, 10.0),
        SweepParam("discount_rate", "Discount rate", 0.04, 0.12, 0.08),
        SweepParam("device_lifetime_years", "Device lifetime (yr)", 10, 30, 20, is_int=True),
        SweepParam("hydrogel_lifetime_years", "Hydrogel lifetime (yr)", 0.5, 2.0, 1.0),
        SweepParam("utilization_factor", "Utilization factor", 0.7, 1.0, 0.9),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=11)
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO / "parameter_sweeps" / "parameter_sweep.csv",
    )
    ap.add_argument("--params", nargs="*", default=None)
    args = ap.parse_args()

    params = make_sweep_params()
    if args.params:
        keys = set(args.params)
        params = [p for p in params if p.key in keys]

    rows: list[dict] = []
    bl_y = _baseline_yield()
    bl_lcow = lcow_from_daily_yield(
        bl_y,
        salt_name=BASELINE_CONFIG.salt_name,
        salt_to_polymer_ratio=BASELINE_CONFIG.salt_to_polymer_ratio,
        hydrogel_thickness_m=BASELINE_CONFIG.hydrogel_thickness_m,
        econ=BASELINE_ECON,
    )
    rows.append(
        {
            "sweep_param": "baseline",
            "param_value": "",
            "param_label": "baseline",
            "daily_yield_kg_m2": bl_y,
            "lcow_usd_per_m3": bl_lcow,
        }
    )

    for sp in params:
        for val in _sweep_grid(sp, args.n_points):
            res = _run_point(sp, val)
            rows.append(
                {
                    "sweep_param": sp.key,
                    "param_value": val,
                    "param_label": sp.label,
                    **res,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sweep_param",
        "param_value",
        "param_label",
        "daily_yield_kg_m2",
        "thermal_efficiency",
        "lcow_usd_per_m3",
    ]
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
