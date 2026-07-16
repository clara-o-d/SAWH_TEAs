"""Specific energy per liter water — paired with LCOW economics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from waste_heat_cycle_lumped.economics.parasitic import (
    ElectricalLoadSpec,
    LoadCategory,
    ParasiticLoadOptions,
    electrical_loads_for_operation,
)
from waste_heat_cycle_lumped.economics.params import LCOEconomicParams
from waste_heat_cycle_lumped.physics import device_defaults as dd
from waste_heat_cycle_lumped.simulation.ode_system import CycleResult
from waste_heat_cycle_lumped.simulation.operation_hours import (
    DailyOperationHours,
    daily_operating_hours_from_results,
)

_JOULES_PER_KWH = 3.6e6
_FAIL_SPECIFIC_ENERGY = float("inf")

_CATEGORY_FIELDS: tuple[tuple[LoadCategory, str], ...] = (
    ("vacuum", "vacuum_kwh_per_l"),
    ("htf_pump", "htf_pump_kwh_per_l"),
    ("uptake_fan", "fans_kwh_per_l"),
    ("condenser_fan", "fans_kwh_per_l"),
    ("condenser_active", "condenser_active_kwh_per_l"),
    ("aux", "aux_kwh_per_l"),
)


@dataclass(frozen=True, slots=True)
class SpecificEnergyBreakdown:
    wh_kwh_per_l: float
    supplemental_kwh_per_l: float
    vacuum_kwh_per_l: float
    htf_pump_kwh_per_l: float
    fans_kwh_per_l: float
    condenser_active_kwh_per_l: float
    aux_kwh_per_l: float
    parasitic_kwh_per_l: float
    total_kwh_per_l: float
    min_kwh_per_l: float
    desorption_hours_per_day: float
    operating_hours_per_day: float
    n_cycles_per_day: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "specific_energy_wh_kwh_per_l": self.wh_kwh_per_l,
            "specific_energy_supplemental_kwh_per_l": self.supplemental_kwh_per_l,
            "specific_energy_vacuum_kwh_per_l": self.vacuum_kwh_per_l,
            "specific_energy_htf_pump_kwh_per_l": self.htf_pump_kwh_per_l,
            "specific_energy_fans_kwh_per_l": self.fans_kwh_per_l,
            "specific_energy_condenser_active_kwh_per_l": self.condenser_active_kwh_per_l,
            "specific_energy_aux_kwh_per_l": self.aux_kwh_per_l,
            "specific_energy_parasitic_kwh_per_l": self.parasitic_kwh_per_l,
            "specific_energy_total_kwh_per_l": self.total_kwh_per_l,
            "specific_energy_min_kwh_per_l": self.min_kwh_per_l,
            "desorption_hours_per_day": self.desorption_hours_per_day,
            "operating_hours_per_day": self.operating_hours_per_day,
            "n_cycles_per_day": self.n_cycles_per_day,
        }


def minimum_specific_energy_kwh_per_l(*, h_fg_j_per_kg: float = dd.H_FG_J_PER_KG) -> float:
    """Thermodynamic minimum to condense water (kWh/L)."""
    return float(h_fg_j_per_kg) / _JOULES_PER_KWH


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


def _annual_water_l_per_m2(*, daily_yield_kg_per_m2: float) -> float:
    return 365.0 * float(daily_yield_kg_per_m2)


def _kwh_per_l_from_annual_kwh(annual_kwh_per_m2: float, *, daily_yield_kg_per_m2: float) -> float:
    water_l = _annual_water_l_per_m2(daily_yield_kg_per_m2=daily_yield_kg_per_m2)
    if water_l <= 0.0:
        return _FAIL_SPECIFIC_ENERGY
    return annual_kwh_per_m2 / water_l


def _loads_kwh_per_l_by_category(
    loads: tuple[ElectricalLoadSpec, ...],
    *,
    daily_yield_kg_per_m2: float,
) -> dict[str, float]:
    totals = {
        "vacuum_kwh_per_l": 0.0,
        "htf_pump_kwh_per_l": 0.0,
        "fans_kwh_per_l": 0.0,
        "condenser_active_kwh_per_l": 0.0,
        "aux_kwh_per_l": 0.0,
    }
    for load in loads:
        kwh_per_l = _kwh_per_l_from_annual_kwh(
            load.annual_kwh_per_m2(),
            daily_yield_kg_per_m2=daily_yield_kg_per_m2,
        )
        field = dict(_CATEGORY_FIELDS)[load.category]
        totals[field] += kwh_per_l
    return totals


def supplemental_heat_specific_energy_kwh_per_l(
    *,
    electric_heat_w_per_m2: float,
    desorption_hours_per_day: float,
    daily_yield_kg_per_m2: float,
) -> float:
    """Optional supplemental electric desorption heat per liter water (kWh/L)."""
    yield_kg = float(daily_yield_kg_per_m2)
    if yield_kg <= 0.0 or not math.isfinite(yield_kg):
        return _FAIL_SPECIFIC_ENERGY
    kwh_per_m2_yr = float(electric_heat_w_per_m2) * float(desorption_hours_per_day) * 365.0 / 1000.0
    return _kwh_per_l_from_annual_kwh(kwh_per_m2_yr, daily_yield_kg_per_m2=yield_kg)


def specific_energy_breakdown_from_daily_operation(
    daily_yield_kg_per_m2: float,
    *,
    thermal_efficiency: float,
    cycle_results: list[CycleResult],
    h_fg_j_per_kg: float = dd.H_FG_J_PER_KG,
    parasitic_options: ParasiticLoadOptions | None = None,
    electric_heat_w_per_m2: float = 0.0,
    desorption_hours_per_day: float | None = None,
) -> SpecificEnergyBreakdown:
    """Full specific-energy breakdown using simulation-coupled parasitic loads."""
    opts = parasitic_options or ParasiticLoadOptions()
    hours: DailyOperationHours = daily_operating_hours_from_results(cycle_results)
    loads = electrical_loads_for_operation(cycle_results, options=opts)

    wh = waste_heat_specific_energy_kwh_per_l(
        thermal_efficiency=thermal_efficiency,
        h_fg_j_per_kg=h_fg_j_per_kg,
    )
    desorp_h = (
        float(desorption_hours_per_day)
        if desorption_hours_per_day is not None
        else hours.desorption_hours_per_day
    )
    supplemental = 0.0
    if electric_heat_w_per_m2 > 0.0:
        supplemental = supplemental_heat_specific_energy_kwh_per_l(
            electric_heat_w_per_m2=electric_heat_w_per_m2,
            desorption_hours_per_day=desorp_h,
            daily_yield_kg_per_m2=daily_yield_kg_per_m2,
        )

    by_cat = _loads_kwh_per_l_by_category(loads, daily_yield_kg_per_m2=daily_yield_kg_per_m2)
    parasitic = sum(by_cat.values())
    total = wh + supplemental + parasitic
    if not math.isfinite(total):
        total = _FAIL_SPECIFIC_ENERGY

    return SpecificEnergyBreakdown(
        wh_kwh_per_l=wh,
        supplemental_kwh_per_l=supplemental,
        vacuum_kwh_per_l=by_cat["vacuum_kwh_per_l"],
        htf_pump_kwh_per_l=by_cat["htf_pump_kwh_per_l"],
        fans_kwh_per_l=by_cat["fans_kwh_per_l"],
        condenser_active_kwh_per_l=by_cat["condenser_active_kwh_per_l"],
        aux_kwh_per_l=by_cat["aux_kwh_per_l"],
        parasitic_kwh_per_l=parasitic,
        total_kwh_per_l=total,
        min_kwh_per_l=minimum_specific_energy_kwh_per_l(h_fg_j_per_kg=h_fg_j_per_kg),
        desorption_hours_per_day=desorp_h,
        operating_hours_per_day=hours.operating_hours_per_day,
        n_cycles_per_day=hours.n_cycles,
    )


def parasitic_specific_energy_kwh_per_l(
    daily_yield_kg_per_m2: float,
    *,
    cycle_results: list[CycleResult] | None = None,
    parasitic_options: ParasiticLoadOptions | None = None,
) -> float:
    """Grid electricity for pumps and controls, amortized per liter water (kWh/L)."""
    loads = electrical_loads_for_operation(cycle_results, options=parasitic_options)
    yield_kg = float(daily_yield_kg_per_m2)
    if yield_kg <= 0.0 or not math.isfinite(yield_kg):
        return _FAIL_SPECIFIC_ENERGY
    kwh_per_m2_yr = sum(load.annual_kwh_per_m2() for load in loads)
    return _kwh_per_l_from_annual_kwh(kwh_per_m2_yr, daily_yield_kg_per_m2=yield_kg)


def total_specific_energy_kwh_per_l(
    daily_yield_kg_per_m2: float,
    *,
    thermal_efficiency: float,
    h_fg_j_per_kg: float = dd.H_FG_J_PER_KG,
    cycle_results: list[CycleResult] | None = None,
    parasitic_options: ParasiticLoadOptions | None = None,
    electric_heat_w_per_m2: float = 0.0,
    desorption_hours_per_day: float | None = None,
) -> float:
    """Waste heat + parasitic grid + supplemental electric heat per liter (kWh/L)."""
    if cycle_results is not None:
        return specific_energy_breakdown_from_daily_operation(
            daily_yield_kg_per_m2,
            thermal_efficiency=thermal_efficiency,
            cycle_results=cycle_results,
            h_fg_j_per_kg=h_fg_j_per_kg,
            parasitic_options=parasitic_options,
            electric_heat_w_per_m2=electric_heat_w_per_m2,
            desorption_hours_per_day=desorption_hours_per_day,
        ).total_kwh_per_l

    wh = waste_heat_specific_energy_kwh_per_l(
        thermal_efficiency=thermal_efficiency,
        h_fg_j_per_kg=h_fg_j_per_kg,
    )
    parasitic = parasitic_specific_energy_kwh_per_l(
        daily_yield_kg_per_m2,
        cycle_results=None,
        parasitic_options=parasitic_options,
    )
    supplemental = 0.0
    if electric_heat_w_per_m2 > 0.0 and desorption_hours_per_day is not None:
        supplemental = supplemental_heat_specific_energy_kwh_per_l(
            electric_heat_w_per_m2=electric_heat_w_per_m2,
            desorption_hours_per_day=desorption_hours_per_day,
            daily_yield_kg_per_m2=daily_yield_kg_per_m2,
        )
    total = wh + parasitic + supplemental
    if not math.isfinite(total):
        return _FAIL_SPECIFIC_ENERGY
    return total
