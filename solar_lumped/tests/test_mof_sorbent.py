"""Tests for MOF sorbent option in solar_lumped."""

import math

import pytest

from solar_lumped.physics.adsorbent import (
    DEFAULT_MOF_NAME,
    equilibrium_loading_at_rh,
    get_mof,
    loading_at_rh,
    water_activity_from_loading,
)
from solar_lumped.physics.salt_properties import WATER_MOLAR_MASS_KG_MOL
from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.simulation.ode_system import run_daily_cycle
from solar_lumped.weather.profiles import baseline_profile


def test_mil100_isotherm_units():
    props = get_mof(DEFAULT_MOF_NAME)
    # 98.58 % RH → 29.404 mol/kg ≈ 0.530 kg/kg
    q = loading_at_rh(0.9858, props=props)
    assert q == pytest.approx(29.40438871473355 * WATER_MOLAR_MASS_KG_MOL, rel=1e-4)


def test_mof_isotherm_monotone():
    props = get_mof(DEFAULT_MOF_NAME)
    q_lo = loading_at_rh(0.2, props=props)
    q_hi = loading_at_rh(0.6, props=props)
    assert q_hi > q_lo


def test_mof_aw_inverts_loading():
    props = get_mof(DEFAULT_MOF_NAME)
    q = equilibrium_loading_at_rh(0.5, temperature_c=30.0, props=props)
    aw = water_activity_from_loading(q, temperature_c=30.0, props=props)
    assert aw == pytest.approx(0.5, abs=0.02)


def test_mof_baseline_simulation_runs():
    config = DeviceConfig.baseline(sorbent="mof", mof_name=DEFAULT_MOF_NAME)
    y, eta, abs_res, des_res = run_daily_cycle(baseline_profile(), config)
    assert y >= 0.0
    assert math.isfinite(eta)
    assert float(abs_res.c_w[-1]) > float(abs_res.c_w[0])
    assert des_res.water_collected_kg_m2 >= 0.0
