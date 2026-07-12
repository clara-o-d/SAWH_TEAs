"""LCOW from Wilson simulation daily yield — same equation as electrolyte_optimization lcow_zsr_at_sl."""

from __future__ import annotations

import math
from dataclasses import dataclass

from waste_heat_lumped.economics.params import (
    C_DEVICE_USD,
    DEVICE_BOM_USD_PER_M2,
    KG_WATER_PER_M3,
    LCOEconomicParams,
    dry_composite_mass_kg,
)
from waste_heat_lumped.physics.salt_properties import get_salt_price_usd_per_kg

FAIL_LCO: float = 1e30


@dataclass(frozen=True, slots=True)
class LcowCostBreakdown:
    items: tuple[tuple[str, float], ...]

    @property
    def total_usd_per_m3(self) -> float:
        return float(sum(v for _, v in self.items))


def _hydrogel_cost_per_kg(
    salt_price_usd_per_kg: float,
    salt_to_polymer_ratio: float,
    econ: LCOEconomicParams,
) -> float:
    sl = salt_to_polymer_ratio
    return (
        (salt_price_usd_per_kg * sl + econ.c_acrylamide_usd_per_kg) / (1.0 + sl)
        + econ.c_additives_usd_per_kg_composite
    )


def _sorbent_replacement_annual_usd(
    *,
    sorbent: str,
    salt_name: str,
    salt_to_polymer_ratio: float,
    hydrogel_thickness_m: float,
    mof_mass_kg_m2: float,
    mof_price_usd_per_kg: float,
    econ: LCOEconomicParams,
    salt_price_usd_per_kg: float | None = None,
) -> float:
    gel_lifetime = econ.hydrogel_lifetime_years
    if sorbent == "mof":
        return mof_mass_kg_m2 * mof_price_usd_per_kg / gel_lifetime
    sl = salt_to_polymer_ratio
    dry_mass = dry_composite_mass_kg(hydrogel_thickness_m)
    salt_price = (
        salt_price_usd_per_kg
        if salt_price_usd_per_kg is not None
        else get_salt_price_usd_per_kg(salt_name)
    )
    hydrogel_cost_per_kg = _hydrogel_cost_per_kg(salt_price, sl, econ)
    return hydrogel_cost_per_kg * dry_mass / gel_lifetime


def lcow_from_daily_yield(
    daily_yield_kg_per_m2: float,
    *,
    salt_name: str,
    salt_to_polymer_ratio: float,
    hydrogel_thickness_m: float,
    econ: LCOEconomicParams,
    cycles_per_day: float = 1.0,
    electric_heat_w_per_m2: float = 0.0,
    salt_price_usd_per_kg: float | None = None,
    sorbent: str = "hydrogel",
    mof_mass_kg_m2: float = 0.0,
    mof_price_usd_per_kg: float = 0.0,
) -> float:
    """Scalar LCOW (USD/m³) — identical structure to lcow_zsr_at_sl."""
    if daily_yield_kg_per_m2 <= 0.0 or not math.isfinite(daily_yield_kg_per_m2):
        return FAIL_LCO
    if sorbent == "hydrogel" and (
        salt_to_polymer_ratio <= 0.0 or not math.isfinite(salt_to_polymer_ratio)
    ):
        return FAIL_LCO

    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(daily_yield_kg_per_m2)
    sorbent_replacement = _sorbent_replacement_annual_usd(
        sorbent=sorbent,
        salt_name=salt_name,
        salt_to_polymer_ratio=salt_to_polymer_ratio,
        hydrogel_thickness_m=hydrogel_thickness_m,
        mof_mass_kg_m2=mof_mass_kg_m2,
        mof_price_usd_per_kg=mof_price_usd_per_kg,
        econ=econ,
        salt_price_usd_per_kg=salt_price_usd_per_kg,
    )

    annual_electricity_cost = (
        econ.electricity_price_usd_per_kwh
        * float(electric_heat_w_per_m2)
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
    )
    annual_extra_cycle_energy = econ.annual_extra_cycle_energy_cost_usd(cycles_per_day)

    annual_cost_usd = (
        econ.capital_recovery_factor() * econ.total_investment_factor * C_DEVICE_USD
        + sorbent_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * C_DEVICE_USD
        + econ.energy_cost_usd_per_year
        + annual_electricity_cost
        + annual_extra_cycle_energy
    )
    if not math.isfinite(annual_cost_usd):
        return FAIL_LCO
    return float(
        annual_cost_usd
        / (econ.utilization_factor * (annual_water_yield_kg / KG_WATER_PER_M3 + 1e-9))
    )


def lcow_cost_breakdown_from_daily_yield(
    daily_yield_kg_per_m2: float,
    *,
    salt_name: str,
    salt_to_polymer_ratio: float,
    hydrogel_thickness_m: float,
    econ: LCOEconomicParams,
    cycles_per_day: float = 1.0,
    electric_heat_w_per_m2: float = 0.0,
    salt_price_usd_per_kg: float | None = None,
    sorbent: str = "hydrogel",
    mof_mass_kg_m2: float = 0.0,
    mof_price_usd_per_kg: float = 0.0,
) -> LcowCostBreakdown | None:
    """Per-term LCOW breakdown — same segments as lcow_cost_breakdown_usd_per_m3."""
    lcow = lcow_from_daily_yield(
        daily_yield_kg_per_m2,
        salt_name=salt_name,
        salt_to_polymer_ratio=salt_to_polymer_ratio,
        hydrogel_thickness_m=hydrogel_thickness_m,
        econ=econ,
        cycles_per_day=cycles_per_day,
        electric_heat_w_per_m2=electric_heat_w_per_m2,
        salt_price_usd_per_kg=salt_price_usd_per_kg,
        sorbent=sorbent,
        mof_mass_kg_m2=mof_mass_kg_m2,
        mof_price_usd_per_kg=mof_price_usd_per_kg,
    )
    if not math.isfinite(lcow) or lcow >= 0.99 * FAIL_LCO:
        return None

    sl = salt_to_polymer_ratio
    dry_mass = dry_composite_mass_kg(hydrogel_thickness_m)
    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(daily_yield_kg_per_m2)
    if annual_water_yield_kg <= 0.0:
        return None

    denom = econ.utilization_factor * (annual_water_yield_kg / KG_WATER_PER_M3 + 1e-9)
    inv = econ.total_investment_factor
    crf = econ.capital_recovery_factor()
    maint_frac = econ.maintenance_cost_fraction
    gel_lifetime = econ.hydrogel_lifetime_years

    def _lcow_seg(annual_usd: float) -> float:
        return float(annual_usd / denom)

    segments: list[tuple[str, float]] = []
    maintenance_annual = 0.0
    for name, line_cost in DEVICE_BOM_USD_PER_M2:
        scaled = inv * line_cost
        segments.append((f"CAPEX: {name}", _lcow_seg(crf * scaled)))
        maintenance_annual += maint_frac * scaled
    segments.append(("Maintenance", _lcow_seg(maintenance_annual)))

    if sorbent == "mof":
        mof_annual = mof_mass_kg_m2 * mof_price_usd_per_kg / gel_lifetime
        segments.append(("MOF sorbent", _lcow_seg(mof_annual)))
    else:
        salt_price = (
            salt_price_usd_per_kg
            if salt_price_usd_per_kg is not None
            else get_salt_price_usd_per_kg(salt_name)
        )
        salt_annual = salt_price * sl / (1.0 + sl) * dry_mass / gel_lifetime
        acrylamide_annual = econ.c_acrylamide_usd_per_kg / (1.0 + sl) * dry_mass / gel_lifetime
        additives_annual = econ.c_additives_usd_per_kg_composite * dry_mass / gel_lifetime
        segments.append(("Hydrogel: salt", _lcow_seg(salt_annual)))
        segments.append(("Hydrogel: acrylamide", _lcow_seg(acrylamide_annual)))
        segments.append(("Hydrogel: additives", _lcow_seg(additives_annual)))

    annual_electricity_cost = (
        econ.electricity_price_usd_per_kwh
        * float(electric_heat_w_per_m2)
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
    )
    annual_extra = econ.annual_extra_cycle_energy_cost_usd(cycles_per_day)
    segments.append(("Fixed energy", _lcow_seg(econ.energy_cost_usd_per_year)))
    segments.append(("Electricity (active heat)", _lcow_seg(annual_electricity_cost)))
    segments.append(("Extra cycling energy", _lcow_seg(annual_extra)))

    return LcowCostBreakdown(items=tuple(segments))
