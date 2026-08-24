import pytest

from solar_lumped.economics import (
    HYDROGEL_DENSITY_KG_M3,
    LCOEconomicParams,
    dry_composite_mass_kg,
    lcow_cost_breakdown_from_daily_yield,
    lcow_from_daily_yield,
    npv_from_daily_yield,
)
from solar_lumped.physics import DRY_COMPOSITE_DENSITY_KG_M3

_KW = dict(
    yield_per_cycle_kg_per_m2=1.0,
    salt_name="NaCl",
    salt_loading=1.0,
    hydrogel_thickness_m=0.002,
)


def _maintenance_segment(econ):
    breakdown = lcow_cost_breakdown_from_daily_yield(econ=econ, **_KW)
    return dict(breakdown.items)["Maintenance"]


def test_maintenance_cost_ignores_total_investment_factor():
    base = LCOEconomicParams(total_investment_factor=1.0)
    inflated = LCOEconomicParams(total_investment_factor=2.5)

    assert _maintenance_segment(base) == _maintenance_segment(inflated)

    lcow_base = lcow_from_daily_yield(econ=base, **_KW)
    lcow_inflated = lcow_from_daily_yield(econ=inflated, **_KW)
    assert lcow_inflated > lcow_base  # CAPEX term still scales with TIC

    npv_base = npv_from_daily_yield(econ=base, water_price_usd_per_m3=1.0, **_KW)
    npv_inflated = npv_from_daily_yield(econ=inflated, water_price_usd_per_m3=1.0, **_KW)
    assert npv_base.annual_opex_usd_per_m2 == npv_inflated.annual_opex_usd_per_m2


def test_dry_composite_mass_uses_dry_basis_not_wet_20rh_density():
    # rho_gel (Table S3) is measured at 20% RH and includes ~126% equilibrium
    # water by mass -- using it directly would price purchased dry solids at
    # ~2.26x their real mass. dry_composite_mass_kg must use the moisture-corrected
    # dry-basis density instead (physics.py::DRY_COMPOSITE_DENSITY_KG_M3).
    assert DRY_COMPOSITE_DENSITY_KG_M3 < HYDROGEL_DENSITY_KG_M3
    thickness_m = 0.004
    dry_mass = dry_composite_mass_kg(thickness_m)
    assert dry_mass == thickness_m * DRY_COMPOSITE_DENSITY_KG_M3
    assert dry_mass < thickness_m * HYDROGEL_DENSITY_KG_M3


def test_simplified_tea_drops_water_and_non_am_polymer_costs():
    econ = LCOEconomicParams()
    full = lcow_cost_breakdown_from_daily_yield(econ=econ, **_KW)
    simple = lcow_cost_breakdown_from_daily_yield(econ=econ, simplified=True, **_KW)

    full_labels = {label for label, _ in full.items}
    simple_labels = {label for label, _ in simple.items}
    assert full_labels >= {"Hydrogel: water", "Hydrogel: APS", "Hydrogel: MBA", "Hydrogel: TEMED"}
    assert simple_labels.isdisjoint({"Hydrogel: water", "Hydrogel: APS", "Hydrogel: MBA", "Hydrogel: TEMED"})
    assert "Hydrogel: acrylamide (AM)" in simple_labels

    assert simple.total_usd_per_m3 < full.total_usd_per_m3
    lcow_simple = lcow_from_daily_yield(econ=econ, simplified=True, **_KW)
    assert simple.total_usd_per_m3 == pytest.approx(lcow_simple)
