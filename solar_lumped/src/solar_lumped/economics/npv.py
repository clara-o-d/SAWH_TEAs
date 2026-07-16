"""Net present value (NPV) and payback period for the Wilson lumped SAWH device.

Same cost model as ``lcow.py`` (BOM CAPEX, hydrogel/sorbent replacement, energy),
but CAPEX is paid up front at year 0 instead of amortized via a capital
recovery factor, so year-by-year cash flows can be discounted directly. Annual
OPEX and revenue are held constant across the device lifetime (no
escalation), so NPV reduces to a level-annuity calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from solar_lumped.economics.lcow import _sorbent_replacement_annual_usd
from solar_lumped.economics.params import C_DEVICE_USD, KG_WATER_PER_M3, LCOEconomicParams


@dataclass(frozen=True, slots=True)
class NpvResult:
    capex_usd_per_m2: float
    annual_revenue_usd_per_m2: float
    annual_opex_usd_per_m2: float
    annual_net_cash_flow_usd_per_m2: float
    npv_usd_per_m2: float
    payback_years_simple: float
    payback_years_discounted: float


def _present_value_annuity_factor(discount_rate: float, years: float) -> float:
    i = discount_rate
    if i <= 0.0:
        return years
    return (1.0 - (1.0 + i) ** (-years)) / i


def npv_from_daily_yield(
    daily_yield_kg_per_m2: float,
    water_price_usd_per_m3: float,
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
) -> NpvResult | None:
    """NPV and payback period (USD/m2 of device footprint) for one site."""
    if daily_yield_kg_per_m2 <= 0.0 or not math.isfinite(daily_yield_kg_per_m2):
        return None
    if sorbent == "hydrogel" and (
        salt_to_polymer_ratio <= 0.0 or not math.isfinite(salt_to_polymer_ratio)
    ):
        return None

    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(daily_yield_kg_per_m2)
    gross_annual_water_m3 = econ.utilization_factor * (annual_water_yield_kg / KG_WATER_PER_M3)

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

    capex = econ.total_investment_factor * C_DEVICE_USD
    annual_opex = (
        sorbent_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * C_DEVICE_USD
        + econ.energy_cost_usd_per_year
        + annual_electricity_cost
        + annual_extra_cycle_energy
    )
    annual_revenue = gross_annual_water_m3 * float(water_price_usd_per_m3)
    annual_net_cash_flow = annual_revenue - annual_opex

    if not math.isfinite(annual_net_cash_flow):
        return None

    lifetime_years = float(econ.device_lifetime_years)
    i = econ.discount_rate
    pvaf = _present_value_annuity_factor(i, lifetime_years)
    npv = -capex + annual_net_cash_flow * pvaf

    payback_simple = capex / annual_net_cash_flow if annual_net_cash_flow > 0.0 else float("inf")

    if annual_net_cash_flow <= 0.0:
        payback_discounted = float("inf")
    elif i <= 0.0:
        payback_discounted = payback_simple
    else:
        ratio = 1.0 - i * capex / annual_net_cash_flow
        payback_discounted = -math.log(ratio) / math.log(1.0 + i) if ratio > 0.0 else float("inf")

    return NpvResult(
        capex_usd_per_m2=capex,
        annual_revenue_usd_per_m2=annual_revenue,
        annual_opex_usd_per_m2=annual_opex,
        annual_net_cash_flow_usd_per_m2=annual_net_cash_flow,
        npv_usd_per_m2=npv,
        payback_years_simple=payback_simple,
        payback_years_discounted=payback_discounted,
    )
