"""Tests for specific energy per liter water."""

from __future__ import annotations

import math

import numpy as np
import pytest

from waste_heat_cycle_lumped.economics.parasitic import (
    ParasiticLoadOptions,
    default_electrical_loads,
    electrical_loads_for_operation,
)
from waste_heat_cycle_lumped.economics.specific_energy import (
    minimum_specific_energy_kwh_per_l,
    parasitic_specific_energy_kwh_per_l,
    specific_energy_breakdown_from_daily_operation,
    total_specific_energy_kwh_per_l,
    waste_heat_specific_energy_kwh_per_l,
)
from waste_heat_cycle_lumped.physics import device_defaults as dd
from waste_heat_cycle_lumped.simulation.annual_yield import simulate_daily
from waste_heat_cycle_lumped.simulation.device_config import DeviceConfig
from waste_heat_cycle_lumped.simulation.ode_system import CycleResult, HalfCycleResult
from waste_heat_cycle_lumped.simulation.operation_hours import daily_operating_hours_from_results
from waste_heat_cycle_lumped.weather.profiles import datacenter_baseline_profile


def _mock_cycle(duration_s: float) -> CycleResult:
    t = np.array([0.0, duration_s])
    zeros = np.zeros(2)
    half = HalfCycleResult(
        time_s=t,
        q_a=zeros,
        q_d=zeros,
        t_a_c=zeros,
        t_d_c=zeros,
        t_f_c=zeros,
        t_cond_c=zeros,
        m_ads_kg_s_m2=zeros,
        m_des_kg_s_m2=zeros,
        water_collected_kg_m2=0.0,
        integral_ads_kg_m2=0.0,
        integral_des_kg_m2=0.0,
    )
    return CycleResult(half_a=half, half_b=half, water_collected_kg_m2=0.0)


def test_waste_heat_specific_energy_from_efficiency():
    eta = 0.02
    e = waste_heat_specific_energy_kwh_per_l(thermal_efficiency=eta)
    expected = (dd.H_FG_J_PER_KG / eta) / 3.6e6
    assert e == expected
    assert e > 30.0


def test_minimum_specific_energy_is_latent_heat():
    assert minimum_specific_energy_kwh_per_l() == pytest.approx(dd.H_FG_J_PER_KG / 3.6e6)


def test_parasitic_specific_energy_scales_with_yield():
    low = parasitic_specific_energy_kwh_per_l(100.0)
    high = parasitic_specific_energy_kwh_per_l(300.0)
    assert high == pytest.approx(low / 3.0)


def test_simulation_couples_vacuum_hours_to_cycle_duration():
    short_day = [_mock_cycle(600.0)]  # 2 × 600 s = 1200 s ≈ 0.33 h
    long_day = [_mock_cycle(3600.0)]  # 2 h
    short_loads = electrical_loads_for_operation(short_day)
    long_loads = electrical_loads_for_operation(long_day)
    short_vac = next(load for load in short_loads if load.category == "vacuum")
    long_vac = next(load for load in long_loads if load.category == "vacuum")
    assert long_vac.operating_hours_per_day == pytest.approx(2.0)
    assert short_vac.operating_hours_per_day == pytest.approx(1200.0 / 3600.0)


def test_breakdown_sums_to_total():
    results = [_mock_cycle(1800.0), _mock_cycle(1800.0)]
    breakdown = specific_energy_breakdown_from_daily_operation(
        250.0,
        thermal_efficiency=0.02,
        cycle_results=results,
    )
    parasitic_sum = (
        breakdown.vacuum_kwh_per_l
        + breakdown.htf_pump_kwh_per_l
        + breakdown.fans_kwh_per_l
        + breakdown.condenser_active_kwh_per_l
        + breakdown.aux_kwh_per_l
    )
    assert breakdown.parasitic_kwh_per_l == pytest.approx(parasitic_sum)
    assert breakdown.total_kwh_per_l == pytest.approx(
        breakdown.wh_kwh_per_l + breakdown.supplemental_kwh_per_l + breakdown.parasitic_kwh_per_l
    )


def test_optional_fans_increase_parasitic_energy():
    results = [_mock_cycle(3600.0)]
    without = specific_energy_breakdown_from_daily_operation(
        250.0,
        thermal_efficiency=0.02,
        cycle_results=results,
    )
    with_fans = specific_energy_breakdown_from_daily_operation(
        250.0,
        thermal_efficiency=0.02,
        cycle_results=results,
        parasitic_options=ParasiticLoadOptions(
            include_uptake_fans=True,
            include_condenser_fans=True,
        ),
    )
    assert with_fans.fans_kwh_per_l > 0.0
    assert with_fans.parasitic_kwh_per_l > without.parasitic_kwh_per_l


def test_simulate_daily_reports_specific_energy():
    cfg = DeviceConfig.datacenter_baseline()
    profile = datacenter_baseline_profile(tau_half_s=cfg.tau_half_s)
    result = simulate_daily(profile, cfg)
    assert math.isfinite(result.specific_energy_wh_kwh_per_l)
    assert math.isfinite(result.specific_energy_total_kwh_per_l)
    assert result.specific_energy_total_kwh_per_l >= result.specific_energy_wh_kwh_per_l
    assert result.specific_energy_parasitic_kwh_per_l > 0.0
    assert result.specific_energy.desorption_hours_per_day > 0.0
    assert result.n_cycles_per_day > 0


def test_supplemental_heat_adds_to_total():
    yield_kg = 250.0
    eta = 0.02
    results = [_mock_cycle(3600.0)]
    without = total_specific_energy_kwh_per_l(
        yield_kg,
        thermal_efficiency=eta,
        cycle_results=results,
    )
    with_heat = total_specific_energy_kwh_per_l(
        yield_kg,
        thermal_efficiency=eta,
        cycle_results=results,
        electric_heat_w_per_m2=100.0,
    )
    assert with_heat > without


def test_daily_operating_hours_from_results():
    results = [_mock_cycle(1000.0), _mock_cycle(2000.0)]
    hours = daily_operating_hours_from_results(results)
    assert hours.n_cycles == 2
    assert hours.operating_hours_per_day == pytest.approx((1000.0 + 2000.0) * 2.0 / 3600.0)


def test_default_electrical_loads_unchanged_for_lcow():
    loads = default_electrical_loads()
    vacuum = next(load for load in loads if load.category == "vacuum")
    htf = next(load for load in loads if load.category == "htf_pump")
    assert vacuum.operating_hours_per_day == 12.0
    assert htf.operating_hours_per_day == 24.0
