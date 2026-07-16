"""Tests for specific energy per liter water."""

from __future__ import annotations

import math

import pytest

from waste_heat_cycle_lumped_no_loop.economics.params import LCOEconomicParams
from waste_heat_cycle_lumped_no_loop.economics.specific_energy import (
    parasitic_specific_energy_kwh_per_l,
    total_specific_energy_kwh_per_l,
    waste_heat_specific_energy_kwh_per_l,
)
from waste_heat_cycle_lumped_no_loop.physics import device_defaults as dd
from waste_heat_cycle_lumped_no_loop.simulation.annual_yield import simulate_daily
from waste_heat_cycle_lumped_no_loop.simulation.device_config import DeviceConfig
from waste_heat_cycle_lumped_no_loop.weather.profiles import datacenter_baseline_profile


def test_waste_heat_specific_energy_from_efficiency():
    eta = 0.02
    e = waste_heat_specific_energy_kwh_per_l(thermal_efficiency=eta)
    expected = (dd.H_FG_J_PER_KG / eta) / 3.6e6
    assert e == expected
    assert e > 30.0


def test_parasitic_specific_energy_scales_with_yield():
    low = parasitic_specific_energy_kwh_per_l(100.0)
    high = parasitic_specific_energy_kwh_per_l(300.0)
    assert high == pytest.approx(low / 3.0)


def test_total_specific_energy_sums_components():
    yield_kg = 250.0
    eta = 0.02
    total = total_specific_energy_kwh_per_l(yield_kg, thermal_efficiency=eta)
    wh = waste_heat_specific_energy_kwh_per_l(thermal_efficiency=eta)
    parasitic = parasitic_specific_energy_kwh_per_l(yield_kg)
    assert total == wh + parasitic


def test_simulate_daily_reports_specific_energy():
    cfg = DeviceConfig.datacenter_baseline()
    profile = datacenter_baseline_profile(tau_half_s=cfg.tau_half_s)
    result = simulate_daily(profile, cfg)
    assert math.isfinite(result.specific_energy_wh_kwh_per_l)
    assert math.isfinite(result.specific_energy_total_kwh_per_l)
    assert result.specific_energy_total_kwh_per_l > result.specific_energy_wh_kwh_per_l
    assert result.specific_energy_parasitic_kwh_per_l > 0.0


def test_supplemental_heat_adds_to_total():
    yield_kg = 250.0
    eta = 0.02
    econ = LCOEconomicParams()
    without = total_specific_energy_kwh_per_l(yield_kg, thermal_efficiency=eta, econ=econ)
    with_heat = total_specific_energy_kwh_per_l(
        yield_kg,
        thermal_efficiency=eta,
        econ=econ,
        electric_heat_w_per_m2=100.0,
    )
    assert with_heat > without
