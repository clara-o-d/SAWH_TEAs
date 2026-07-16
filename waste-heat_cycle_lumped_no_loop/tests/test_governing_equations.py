"""Structural checks that Wilson Eqs. 5–6 are implemented as documented."""

from __future__ import annotations

import pytest

from waste_heat_cycle_lumped_no_loop.physics.mass_transfer import (
    concentration_ratio_absorption,
    concentration_ratio_desorption,
    dH_dt,
    dc_w_dt,
    mass_transfer_g_m_s,
)
from waste_heat_cycle_lumped_no_loop.physics.salt_properties import (
    GAS_CONSTANT_J_MOL_K,
    WATER_MOLAR_MASS_KG_MOL,
    saturation_vapor_pressure_pa,
)
from waste_heat_cycle_lumped_no_loop.physics.sorbent import mass_transfer_params
from waste_heat_cycle_lumped_no_loop.simulation.device_config import DeviceConfig


@pytest.fixture
def config() -> DeviceConfig:
    return DeviceConfig.datacenter_baseline()


@pytest.fixture
def mass(config: DeviceConfig):
    return mass_transfer_params(config)


def test_eq5_mass_transfer_formula_absorption(config: DeviceConfig, mass):
    """dc_w/dt = (g/H₀) · P_sat/(RT) · (C_R − a_w) during absorption."""
    from waste_heat_cycle_lumped_no_loop.physics.mass_transfer import _absorption_effective_water_activity

    h0 = config.hydrogel_thickness_m
    t_gel = 32.0
    rh = 0.45
    c_w = 70000.0
    c_r = concentration_ratio_absorption(rh)
    g = mass_transfer_g_m_s(phase="absorption", params=mass, h_m=h0, t_gel_c=t_gel)
    aw = _absorption_effective_water_activity(
        c_w, t_gel_c=t_gel, params=mass, h_m=h0
    )
    t_k = t_gel + 273.15
    p_sat = saturation_vapor_pressure_pa(t_gel)
    expected = (g / h0) * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * (c_r - aw)

    dc = dc_w_dt(
        c_w,
        t_gel_c=t_gel,
        c_r=c_r,
        params=mass,
        h_m=h0,
        phase="absorption",
    )
    assert dc == pytest.approx(expected, rel=1e-9)


def test_eq6_thickness_rate_ratio_to_eq5(config: DeviceConfig, mass):
    """dH/dt and dc_w/dt share the same driving force; ratio is MW/ρ_sol · H₀."""
    h0 = config.hydrogel_thickness_m
    t_gel = 40.0
    t_cond = 30.0
    c_w = 12000.0
    c_r = concentration_ratio_desorption(t_gel, t_cond)
    dc = dc_w_dt(
        c_w,
        t_gel_c=t_gel,
        c_r=c_r,
        params=mass,
        h_m=h0,
        phase="desorption",
        t_cond_c=t_cond,
    )
    dh = dH_dt(
        c_w,
        t_gel_c=t_gel,
        c_r=c_r,
        params=mass,
        h_m=h0,
        phase="desorption",
        t_cond_c=t_cond,
    )
    if abs(dc) < 1e-20:
        pytest.skip("No mass transfer at this state")
    expected_ratio = (WATER_MOLAR_MASS_KG_MOL / mass.rho_solution_kg_m3) * h0
    assert dh / dc == pytest.approx(expected_ratio, rel=1e-9)


def test_eq6_thickness_rate_ratio_absorption(config: DeviceConfig, mass):
    """Same g-limited ratio holds during absorption (g = g_chamber)."""
    h0 = config.hydrogel_thickness_m
    t_gel = 32.0
    rh = 0.45
    c_w = 70000.0
    c_r = concentration_ratio_absorption(rh)
    dc = dc_w_dt(
        c_w,
        t_gel_c=t_gel,
        c_r=c_r,
        params=mass,
        h_m=h0,
        phase="absorption",
    )
    dh = dH_dt(
        c_w,
        t_gel_c=t_gel,
        c_r=c_r,
        params=mass,
        h_m=h0,
        phase="absorption",
    )
    if abs(dc) < 1e-20:
        pytest.skip("No mass transfer at this state")
    expected_ratio = (WATER_MOLAR_MASS_KG_MOL / mass.rho_solution_kg_m3) * h0
    assert dh / dc == pytest.approx(expected_ratio, rel=1e-9)
