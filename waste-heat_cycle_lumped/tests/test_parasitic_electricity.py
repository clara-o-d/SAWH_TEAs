"""Tests for waste-heat parasitic electricity costing."""

from __future__ import annotations

from waste_heat_cycle_lumped.economics.bom import C_DEVICE_USD, DEVICE_BOM_USD_PER_M2
from waste_heat_cycle_lumped.economics.lcow import lcow_from_daily_yield
from waste_heat_cycle_lumped.economics.params import HYDROGEL_THICKNESS_M, LCOEconomicParams
from waste_heat_cycle_lumped.economics.parasitic import (
    default_electrical_loads,
    htf_pump_shaft_power_w_per_m2,
    total_parasitic_electricity_annual_usd_per_m2,
)
from waste_heat_cycle_lumped.physics import device_defaults as dd


def test_device_bom_matches_patent_midpoints():
    assert len(DEVICE_BOM_USD_PER_M2) == 10
    assert C_DEVICE_USD == 9265.0


def test_htf_pump_shaft_power_from_loop_flow():
    power = htf_pump_shaft_power_w_per_m2(
        m_dot_kg_s_m2=dd.M_F_BASE_KG_S_M2,
        head_m=8.0,
    )
    assert 19.0 < power < 20.5


def test_default_parasitic_loads_include_efficiency():
    loads = default_electrical_loads()
    names = {load.name for load in loads}
    assert "Transfer pump (18)" in names
    assert "Vacuum pump (28)" in names
    transfer = next(load for load in loads if load.name == "Transfer pump (18)")
    assert transfer.motor_efficiency == 0.55
    assert transfer.grid_power_w_per_m2 > transfer.shaft_power_w_per_m2


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
