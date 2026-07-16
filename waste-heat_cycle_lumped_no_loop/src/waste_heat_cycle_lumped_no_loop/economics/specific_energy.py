"""Specific energy per liter water — paired with LCOW economics."""

from __future__ import annotations

import math

from waste_heat_cycle_lumped_no_loop.economics.parasitic import (
    ElectricalLoadSpec,
    default_electrical_loads,
)
from waste_heat_cycle_lumped_no_loop.economics.params import LCOEconomicParams
from waste_heat_cycle_lumped_no_loop.physics import device_defaults as dd

_JOULES_PER_KWH = 3.6e6
_FAIL_SPECIFIC_ENERGY = float("inf")


def waste_heat_specific_energy_kwh_per_l(
    *,
    thermal_efficiency: float,
    h_fg_j_per_kg: float = dd.H_FG_J_PER_KG,
) -> float:
    """Waste-heat input per liter water produced (kWh/L).

    Derived from η = m_water h_fg / Q_wh ⇒ Q_wh / m_water = h_fg / η.
    """
    eta = float(thermal_efficiency)
    if eta <= 0.0 or not math.isfinite(eta):
        return _FAIL_SPECIFIC_ENERGY
    j_per_l = float(h_fg_j_per_kg) / eta
    if not math.isfinite(j_per_l):
        return _FAIL_SPECIFIC_ENERGY
    return j_per_l / _JOULES_PER_KWH


def parasitic_specific_energy_kwh_per_l(
    daily_yield_kg_per_m2: float,
    *,
    cycles_per_day: float = 1.0,
    loads: tuple[ElectricalLoadSpec, ...] | None = None,
) -> float:
    """Grid electricity for pumps and controls, amortized per liter water (kWh/L)."""
    yield_kg = float(daily_yield_kg_per_m2)
    if yield_kg <= 0.0 or not math.isfinite(yield_kg):
        return _FAIL_SPECIFIC_ENERGY
    load_specs = loads if loads is not None else default_electrical_loads()
    kwh_per_m2_yr = sum(load.annual_kwh_per_m2() for load in load_specs)
    water_l_per_m2_yr = float(cycles_per_day) * 365.0 * yield_kg
    return kwh_per_m2_yr / water_l_per_m2_yr


def supplemental_heat_specific_energy_kwh_per_l(
    *,
    electric_heat_w_per_m2: float,
    econ: LCOEconomicParams,
    daily_yield_kg_per_m2: float,
    cycles_per_day: float = 1.0,
) -> float:
    """Optional supplemental electric desorption heat per liter water (kWh/L)."""
    yield_kg = float(daily_yield_kg_per_m2)
    if yield_kg <= 0.0 or not math.isfinite(yield_kg):
        return _FAIL_SPECIFIC_ENERGY
    kwh_per_m2_yr = (
        float(electric_heat_w_per_m2)
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
    )
    water_l_per_m2_yr = float(cycles_per_day) * 365.0 * yield_kg
    return kwh_per_m2_yr / water_l_per_m2_yr


def total_specific_energy_kwh_per_l(
    daily_yield_kg_per_m2: float,
    *,
    thermal_efficiency: float,
    h_fg_j_per_kg: float = dd.H_FG_J_PER_KG,
    econ: LCOEconomicParams | None = None,
    cycles_per_day: float = 1.0,
    electric_heat_w_per_m2: float = 0.0,
    loads: tuple[ElectricalLoadSpec, ...] | None = None,
) -> float:
    """Waste heat + parasitic grid + supplemental electric heat per liter (kWh/L)."""
    wh = waste_heat_specific_energy_kwh_per_l(
        thermal_efficiency=thermal_efficiency,
        h_fg_j_per_kg=h_fg_j_per_kg,
    )
    parasitic = parasitic_specific_energy_kwh_per_l(
        daily_yield_kg_per_m2,
        cycles_per_day=cycles_per_day,
        loads=loads,
    )
    supplemental = 0.0
    if electric_heat_w_per_m2 > 0.0 and econ is not None:
        supplemental = supplemental_heat_specific_energy_kwh_per_l(
            electric_heat_w_per_m2=electric_heat_w_per_m2,
            econ=econ,
            daily_yield_kg_per_m2=daily_yield_kg_per_m2,
            cycles_per_day=cycles_per_day,
        )
    total = wh + parasitic + supplemental
    if not math.isfinite(total):
        return _FAIL_SPECIFIC_ENERGY
    return total
