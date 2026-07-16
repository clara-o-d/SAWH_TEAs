#!/usr/bin/env python3
"""One-at-a-time parameter sweep for waste-heat lumped SAWH LCOW and NPV."""

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

from waste_heat_cycle_lumped_no_loop.economics.lcow import lcow_from_daily_yield
from waste_heat_cycle_lumped_no_loop.economics.npv import npv_from_daily_yield
from waste_heat_cycle_lumped_no_loop.economics.params import LCOEconomicParams
from waste_heat_cycle_lumped_no_loop.physics import device_defaults as dd
from waste_heat_cycle_lumped_no_loop.physics.contactor_balances import ContactorThermalParams
from waste_heat_cycle_lumped_no_loop.simulation.annual_yield import SimulationResult, simulate_daily
from waste_heat_cycle_lumped_no_loop.simulation.device_config import ControllerParams, DeviceConfig
from waste_heat_cycle_lumped_no_loop.weather.profiles import HalfCycleProfile, datacenter_baseline_profile


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

# Baseline water price used for NPV/payback metrics when the sweep parameter
# in question isn't "water_price_usd_per_m3" itself; matches the baseline in
# the SweepParam registered by make_sweep_params().
_BASELINE_WATER_PRICE_USD_PER_M3: float = 5.0


def _baseline_profile() -> HalfCycleProfile:
    return datacenter_baseline_profile(
        tau_half_s=BASELINE_CONFIG.tau_half_s,
        t_amb_c=dd.T_AMB_C,
        rh=dd.RH_AMB,
        h_amb=dd.H_AMB_W_M2_K,
        t_wh_in_c=dd.T_WH_IN_C,
        m_dot_wh_kg_s_m2=dd.M_WH_KG_S_M2,
    )


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


def _apply_overrides(
    overrides: dict[str, float],
) -> tuple[DeviceConfig, HalfCycleProfile, LCOEconomicParams]:
    """Build (config, profile, econ) from a dict of {param_key: value}.

    Generalizes the old single-parameter ``_run_point`` if/elif dispatch so
    it can accept several simultaneous overrides (needed by the 2D heatmap
    script in npv_heatmap.py). Config/profile/econ kwargs are accumulated
    across all keys and applied once at the end, so combining e.g. a
    profile-level override with a config-level override in the same call
    behaves the same as applying either alone.
    """
    cfg_kwargs: dict[str, object] = {}
    profile_kwargs: dict[str, float] = {}
    econ_kwargs: dict[str, object] = {}

    base_controller = BASELINE_CONFIG.controller_params()
    controller_override: ControllerParams | None = None
    thermal_override: ContactorThermalParams | None = None

    for key, value in overrides.items():
        if key == "hydrogel_thickness_mm":
            cfg_kwargs["hydrogel_thickness_m"] = value * 1e-3
        elif key == "vapor_gap_mm":
            cfg_kwargs["vapor_gap_m"] = value * 1e-3
        elif key == "t_wh_in_c":
            profile_kwargs["t_wh_in_c"] = value
        elif key == "relative_humidity":
            profile_kwargs["rh"] = value
        elif key == "h_amb_w_m2_k":
            profile_kwargs["h_amb"] = value
        elif key == "m_dot_wh_kg_s_m2":
            profile_kwargs["m_dot_wh_kg_s_m2"] = value
        elif key == "rh_desorber_switch":
            cfg_kwargs["rh_desorber_switch"] = value
        elif key == "tau_half_min":
            cfg_kwargs["tau_half_s"] = value * 60.0
        elif key == "c_vac_max":
            controller_override = ControllerParams(
                c_vac_base_kg_s_pa_m2=base_controller.c_vac_base_kg_s_pa_m2,
                c_vac_min_kg_s_pa_m2=base_controller.c_vac_min_kg_s_pa_m2,
                c_vac_max_kg_s_pa_m2=value,
                k_m_per_kg_m2=base_controller.k_m_per_kg_m2,
                k_p_per_kg_s_m2=base_controller.k_p_per_kg_s_m2,
            )
        elif key == "ua_wh_desorber_w_k":
            thermal_override = ContactorThermalParams(ua_wh_desorber_w_k=value)
        elif key == "discount_rate":
            econ_kwargs["discount_rate"] = value
        elif key == "device_lifetime_years":
            econ_kwargs["device_lifetime_years"] = int(value)
        elif key == "hydrogel_lifetime_years":
            econ_kwargs["hydrogel_lifetime_years"] = value
        elif key == "utilization_factor":
            econ_kwargs["utilization_factor"] = value
        elif key == "water_price_usd_per_m3":
            # Pure economics input consumed directly by the caller (fed into
            # npv_from_daily_yield); no config/profile/econ dispatch here.
            pass
        else:
            raise ValueError(f"Unknown sweep parameter: {key}")

    if controller_override is not None:
        cfg_kwargs["controller"] = controller_override
    if thermal_override is not None:
        cfg_kwargs["thermal"] = thermal_override

    cfg = DeviceConfig.datacenter_baseline(**cfg_kwargs)
    profile = datacenter_baseline_profile(tau_half_s=cfg.tau_half_s, **profile_kwargs)
    econ = LCOEconomicParams(**econ_kwargs)
    return cfg, profile, econ


def _simulate_and_lcow(
    profile: HalfCycleProfile,
    cfg: DeviceConfig,
    econ: LCOEconomicParams,
    water_price_usd_per_m3: float = _BASELINE_WATER_PRICE_USD_PER_M3,
) -> dict:
    result: SimulationResult = simulate_daily(profile, cfg)
    cycles_per_day = float(result.n_cycles_per_day)
    lcow = lcow_from_daily_yield(
        result.mean_daily_yield_kg_m2,
        salt_name=cfg.salt_name,
        salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
        hydrogel_thickness_m=cfg.hydrogel_thickness_m,
        econ=econ,
        cycles_per_day=cycles_per_day,
    )
    npv_result = npv_from_daily_yield(
        result.mean_daily_yield_kg_m2,
        water_price_usd_per_m3,
        salt_name=cfg.salt_name,
        salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
        hydrogel_thickness_m=cfg.hydrogel_thickness_m,
        econ=econ,
        cycles_per_day=cycles_per_day,
    )
    if npv_result is None:
        npv_usd_per_m2 = float("nan")
        payback_years_simple = float("nan")
        payback_years_discounted = float("nan")
    else:
        npv_usd_per_m2 = npv_result.npv_usd_per_m2
        payback_years_simple = npv_result.payback_years_simple
        payback_years_discounted = npv_result.payback_years_discounted
    return {
        "daily_yield_kg_m2": result.mean_daily_yield_kg_m2,
        "thermal_efficiency": result.mean_thermal_efficiency,
        "specific_energy_wh_kwh_per_l": result.specific_energy_wh_kwh_per_l,
        "specific_energy_parasitic_kwh_per_l": result.specific_energy_parasitic_kwh_per_l,
        "specific_energy_total_kwh_per_l": result.specific_energy_total_kwh_per_l,
        "n_cycles_per_day": result.n_cycles_per_day,
        "lcow_usd_per_m3": lcow,
        "npv_usd_per_m2": npv_usd_per_m2,
        "payback_years_simple": payback_years_simple,
        "payback_years_discounted": payback_years_discounted,
    }


def _run_point(sp: SweepParam, value: float) -> dict:
    cfg, profile, econ = _apply_overrides({sp.key: value})
    water_price = (
        value if sp.key == "water_price_usd_per_m3" else _BASELINE_WATER_PRICE_USD_PER_M3
    )
    return _simulate_and_lcow(profile, cfg, econ, water_price)


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
        SweepParam(
            "ua_wh_desorber_w_k",
            "WH-to-desorber UA_eq (W/K)",
            160.0,
            800.0,
            dd.UA_WH_DESORBER_W_K,
        ),
        SweepParam("h_amb_w_m2_k", "h_amb (W/m²K)", 5.0, 25.0, 15.0),
        SweepParam("discount_rate", "Discount rate", 0.04, 0.12, 0.08),
        SweepParam("device_lifetime_years", "Device lifetime (yr)", 10, 30, 20, is_int=True),
        SweepParam("hydrogel_lifetime_years", "Hydrogel lifetime (yr)", 0.5, 2.0, 1.0),
        SweepParam("utilization_factor", "Utilization factor", 0.7, 1.0, 0.9),
        SweepParam(
            "water_price_usd_per_m3",
            "Water price (USD/m³)",
            1.0,
            25.0,
            _BASELINE_WATER_PRICE_USD_PER_M3,
        ),
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
    bl_res = _simulate_and_lcow(
        _baseline_profile(), BASELINE_CONFIG, BASELINE_ECON, _BASELINE_WATER_PRICE_USD_PER_M3
    )
    rows.append(
        {
            "sweep_param": "baseline",
            "param_value": "",
            "param_label": "baseline",
            **bl_res,
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
        "specific_energy_wh_kwh_per_l",
        "specific_energy_parasitic_kwh_per_l",
        "specific_energy_total_kwh_per_l",
        "n_cycles_per_day",
        "lcow_usd_per_m3",
        "npv_usd_per_m2",
        "payback_years_simple",
        "payback_years_discounted",
    ]
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
