#!/usr/bin/env python3
"""One-at-a-time parameter sweep for waste-heat lumped SAWH LCOW."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from waste_heat_lumped.economics.lcow import lcow_from_daily_yield
from waste_heat_lumped.economics.params import LCOEconomicParams
from waste_heat_lumped.physics import device_defaults as dd
from waste_heat_lumped.physics.contactor_balances import ContactorThermalParams
from waste_heat_lumped.simulation.annual_yield import simulate_daily
from waste_heat_lumped.simulation.device_config import ControllerParams, DeviceConfig
from waste_heat_lumped.weather.profiles import datacenter_baseline_profile


@dataclass(frozen=True, slots=True)
class SweepParam:
    key: str
    label: str
    lo: float
    hi: float
    baseline: float
    is_int: bool = False


BASELINE_CONFIG = DeviceConfig.datacenter_baseline()
BASELINE_ECON = LCOEconomicParams()


def _baseline_profile():
    return datacenter_baseline_profile(
        tau_half_s=BASELINE_CONFIG.tau_half_s,
        t_amb_c=dd.T_AMB_C,
        rh=dd.RH_AMB,
        h_amb=dd.H_AMB_W_M2_K,
        t_wh_in_c=dd.T_WH_IN_C,
        m_dot_wh_kg_s_m2=dd.M_WH_KG_S_M2,
    )


def _baseline_yield() -> float:
    return simulate_daily(_baseline_profile(), BASELINE_CONFIG).mean_daily_yield_kg_m2


def _sweep_grid(sp: SweepParam, n: int) -> list[float]:
    vals = (
        list(sp.lo + (sp.hi - sp.lo) * i / (n - 1) for i in range(n))
        if n > 1
        else [sp.baseline]
    )
    if sp.baseline not in vals:
        vals.append(sp.baseline)
    vals = sorted(set(vals))
    if sp.is_int:
        return [float(int(round(v))) for v in vals]
    return vals


def _simulate_and_lcow(
    profile,
    cfg: DeviceConfig,
    econ: LCOEconomicParams,
) -> dict:
    result = simulate_daily(profile, cfg)
    lcow = lcow_from_daily_yield(
        result.mean_daily_yield_kg_m2,
        salt_name=cfg.salt_name,
        salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
        hydrogel_thickness_m=cfg.hydrogel_thickness_m,
        econ=econ,
    )
    return {
        "daily_yield_kg_m2": result.mean_daily_yield_kg_m2,
        "thermal_efficiency": result.mean_thermal_efficiency,
        "lcow_usd_per_m3": lcow,
    }


def _run_point(sp: SweepParam, value: float) -> dict:
    cfg = BASELINE_CONFIG
    profile = _baseline_profile()
    econ = LCOEconomicParams()

    if sp.key == "hydrogel_thickness_mm":
        cfg = DeviceConfig.datacenter_baseline(hydrogel_thickness_m=value * 1e-3)
    elif sp.key == "vapor_gap_mm":
        cfg = DeviceConfig.datacenter_baseline(vapor_gap_m=value * 1e-3)
    elif sp.key == "t_wh_in_c":
        profile = datacenter_baseline_profile(
            tau_half_s=cfg.tau_half_s,
            t_wh_in_c=value,
        )
    elif sp.key == "relative_humidity":
        profile = datacenter_baseline_profile(
            tau_half_s=cfg.tau_half_s,
            rh=value,
        )
    elif sp.key == "h_amb_w_m2_k":
        profile = datacenter_baseline_profile(
            tau_half_s=cfg.tau_half_s,
            h_amb=value,
        )
    elif sp.key == "m_dot_wh_kg_s_m2":
        profile = datacenter_baseline_profile(
            tau_half_s=cfg.tau_half_s,
            m_dot_wh_kg_s_m2=value,
        )
    elif sp.key == "rh_desorber_switch":
        cfg = DeviceConfig.datacenter_baseline(rh_desorber_switch=value)
        profile = datacenter_baseline_profile(tau_half_s=cfg.tau_half_s)
    elif sp.key == "tau_half_min":
        tau_s = value * 60.0
        cfg = DeviceConfig.datacenter_baseline(tau_half_s=tau_s)
        profile = datacenter_baseline_profile(tau_half_s=tau_s)
    elif sp.key == "c_vac_max":
        base = cfg.controller_params()
        controller = ControllerParams(
            m_f_base_kg_s_m2=base.m_f_base_kg_s_m2,
            m_f_min_kg_s_m2=base.m_f_min_kg_s_m2,
            m_f_max_kg_s_m2=base.m_f_max_kg_s_m2,
            c_vac_base_kg_s_pa_m2=base.c_vac_base_kg_s_pa_m2,
            c_vac_min_kg_s_pa_m2=base.c_vac_min_kg_s_pa_m2,
            c_vac_max_kg_s_pa_m2=value,
            k_t_per_k=base.k_t_per_k,
            k_m_per_kg_m2=base.k_m_per_kg_m2,
            k_p_per_kg_s_m2=base.k_p_per_kg_s_m2,
        )
        cfg = DeviceConfig.datacenter_baseline(controller=controller)
    elif sp.key == "wh_hx_ua_w_k":
        thermal = ContactorThermalParams(wh_hx_ua_w_k=value)
        cfg = DeviceConfig.datacenter_baseline(thermal=thermal)
    elif sp.key == "discount_rate":
        econ = LCOEconomicParams(discount_rate=value)
    elif sp.key == "device_lifetime_years":
        econ = LCOEconomicParams(device_lifetime_years=int(value))
    elif sp.key == "hydrogel_lifetime_years":
        econ = LCOEconomicParams(hydrogel_lifetime_years=value)
    elif sp.key == "utilization_factor":
        econ = LCOEconomicParams(utilization_factor=value)

    return _simulate_and_lcow(profile, cfg, econ)


def make_sweep_params() -> list[SweepParam]:
    return [
        SweepParam("hydrogel_thickness_mm", "Hydrogel thickness (mm)", 1.0, 10.0, 4.0),
        SweepParam("vapor_gap_mm", "Vapor gap (mm)", 7.0, 60.0, 40.0),
        SweepParam("t_wh_in_c", "Waste-heat inlet T (°C)", 50.0, 66.0, 58.0),
        SweepParam("relative_humidity", "Uptake RH", 0.15, 0.80, 0.45),
        SweepParam("m_dot_wh_kg_s_m2", "WH mass flow (kg/s/m²)", 0.05, 0.30, 0.15),
        SweepParam("rh_desorber_switch", "Desorber RH switch", 0.20, 0.50, 0.35),
        SweepParam("tau_half_min", "Max half-cycle (min)", 180.0, 540.0, 360.0),
        SweepParam("c_vac_max", "Vacuum pump c_max (kg/s/Pa/m²)", 1.0e-7, 1.0e-5, 5.0e-6),
        SweepParam("wh_hx_ua_w_k", "WH HX UA (W/K)", 400.0, 2000.0, 1200.0),
        SweepParam("h_amb_w_m2_k", "h_amb (W/m²K)", 5.0, 25.0, 15.0),
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
