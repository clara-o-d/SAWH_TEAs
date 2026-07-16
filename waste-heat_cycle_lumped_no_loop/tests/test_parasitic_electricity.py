"""Tests for waste-heat parasitic electricity costing."""

from __future__ import annotations

from waste_heat_cycle_lumped_no_loop.economics.bom import C_DEVICE_USD, DEVICE_BOM_USD_PER_M2
from waste_heat_cycle_lumped_no_loop.economics.lcow import lcow_from_daily_yield
from waste_heat_cycle_lumped_no_loop.economics.params import HYDROGEL_THICKNESS_M, LCOEconomicParams
from waste_heat_cycle_lumped_no_loop.economics.parasitic import (
    default_electrical_loads,
    total_parasitic_electricity_annual_usd_per_m2,
)


def test_device_bom_matches_patent_midpoints():
    assert len(DEVICE_BOM_USD_PER_M2) == 9
    assert C_DEVICE_USD == 8715.0


def test_default_parasitic_loads_include_efficiency():
    loads = default_electrical_loads()
    names = {load.name for load in loads}
    assert "Vacuum pump (28)" in names
    vacuum = next(load for load in loads if load.name == "Vacuum pump (28)")
    assert vacuum.motor_efficiency == 0.35
    assert vacuum.grid_power_w_per_m2 > vacuum.shaft_power_w_per_m2


def test_lcow_includes_parasitic_electricity():
    econ = LCOEconomicParams()
    yield_kg = 250.0
    without = lcow_from_daily_yield(
        yield_kg,
        salt_name="LiCl",
        salt_to_polymer_ratio=4.0,
        hydrogel_thickness_m=HYDROGEL_THICKNESS_M,
        econ=econ,
        electric_heat_w_per_m2=0.0,
    )
    parasitic = total_parasitic_electricity_annual_usd_per_m2(
        default_electrical_loads(),
        econ.electricity_price_usd_per_kwh,
    )
    assert parasitic > 0.0
    assert without > 0.0
