"""Structural checks that Wilson Eqs. 5–6 are implemented as documented."""

from __future__ import annotations

import pytest

from waste_heat.physics import (
    concentration_ratio_absorption,
    concentration_ratio_desorption,
    dH_dt,
    dc_w_dt,
    mass_transfer_g_m_s,
)
from waste_heat.physics import (
    GAS_CONSTANT_J_MOL_K,
    WATER_MOLAR_MASS_KG_MOL,
    saturation_vapor_pressure_pa,
)
from waste_heat.physics import mass_transfer_params
from waste_heat.simulation import SystemConfig


@pytest.fixture
def config() -> SystemConfig:
    return SystemConfig.datacenter_baseline()


@pytest.fixture
def mass(config: SystemConfig):
    return mass_transfer_params(config)


def test_eq5_mass_transfer_formula_absorption(config: SystemConfig, mass):
    """dc_w/dt = (g/H₀) · P_sat/(RT) · (C_R − a_w) during absorption."""
    from waste_heat.physics import _absorption_effective_water_activity

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


def test_eq6_thickness_rate_ratio_to_eq5(config: SystemConfig, mass):
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


def test_eq6_thickness_rate_ratio_absorption(config: SystemConfig, mass):
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


def test_hollands_nu_is_real_hollands_and_vacuum_suppressed():
    """Guards the vapor-gap Nu against the invented `0.720*Ra^0.25*(1+0.1cos t)` fit
    that used to sit here mislabelled as Hollands et al. (1976)."""
    import math

    from waste_heat.physics import (
        ALPHA_AIR_M2_S,
        GRAVITY_M_S2,
        K_AIR_W_M_K,
        NU_AIR_M2_S,
        P_REF_PA,
        hollands_nu,
    )

    gap, t_hot, t_cold, tilt = 0.01, 80.0, 30.0, 30.0

    # alpha must be a diffusivity (k/(rho*cp)), not the 1.8e-5 dynamic viscosity it was.
    assert ALPHA_AIR_M2_S == pytest.approx(2.371e-5, rel=1e-3)

    # At 1 atm: matches Hollands evaluated independently, with the exact ideal-gas
    # density-difference buoyancy and no spurious extra Prandtl factor.
    t_h, t_c = t_hot + 273.15, t_cold + 273.15
    d_rho_over_rho = 0.5 * (t_h + t_c) * (t_h - t_c) / (t_h * t_c)
    ra = GRAVITY_M_S2 * d_rho_over_rho * gap**3 / (NU_AIR_M2_S * ALPHA_AIR_M2_S)
    rc = ra * math.cos(math.radians(tilt))
    expect = (
        1.0
        + 1.44
        * max(0.0, 1.0 - 1708.0 * math.sin(math.radians(1.8 * tilt)) ** 1.6 / rc)
        * max(0.0, 1.0 - 1708.0 / rc)
        + max(0.0, (rc / 5830.0) ** (1.0 / 3.0) - 1.0)
    )
    assert hollands_nu(gap, t_hot, t_cold, tilt_deg=tilt, p_gap_pa=P_REF_PA) == pytest.approx(
        expect, rel=1e-12
    )

    # Ra ~ p^2, so the ~30 mbar operating gap collapses to the conduction limit Nu = 1.
    assert hollands_nu(gap, t_hot, t_cold, tilt_deg=tilt, p_gap_pa=3000.0) == pytest.approx(1.0)

    # Nu >= 1 always, so h = Nu*k/gap never falls below pure conduction.
    for p in (1.0, 3000.0, P_REF_PA, 5 * P_REF_PA):
        for dt in (0.0, 1.0, 60.0):
            assert hollands_nu(gap, 30.0 + dt, 30.0, tilt_deg=tilt, p_gap_pa=p) >= 1.0
    assert K_AIR_W_M_K > 0.0
