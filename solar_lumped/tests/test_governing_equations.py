"""Structural checks that Wilson Eqs. 1–6 are implemented as documented."""

from __future__ import annotations

import numpy as np
import pytest

from solar_lumped.physics.correlations import (
    condenser_h_conv_w_m2_k,
    mass_transfer_g_from_h_conv_m_s,
    parallel_plate_emissivity,
    radiative_exchange_w_m2,
)
from solar_lumped.physics.device_balances import DeviceThermalParams, _residuals, solve_steady_thermal
from solar_lumped.physics.mass_transfer import (
    concentration_ratio_absorption,
    concentration_ratio_desorption,
    dH_dt,
    dc_w_dt,
    mass_transfer_g_m_s,
    m_des_kg_s_m2_from_state,
)
from solar_lumped.physics.salt_properties import (
    GAS_CONSTANT_J_MOL_K,
    WATER_MOLAR_MASS_KG_MOL,
    saturation_vapor_pressure_pa,
)
from solar_lumped.simulation.coupled_dynamics import evaluate_coupled_rates
from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.simulation.ode_system import run_daily_cycle
from solar_lumped.weather.profiles import baseline_profile


@pytest.fixture
def config() -> DeviceConfig:
    return DeviceConfig.baseline()


@pytest.fixture
def mass(config: DeviceConfig):
    return config.mass_params()


@pytest.fixture
def thermal(config: DeviceConfig):
    return config.thermal_params()


def test_eq5_mass_transfer_formula_absorption(config: DeviceConfig, mass):
    """dc_w/dt = (g/H₀) · P_sat/(RT) · (C_R − a_w) during absorption."""
    from solar_lumped.physics.mass_transfer import _absorption_effective_water_activity

    h0 = config.hydrogel_thickness_m
    t_gel = 25.0
    rh = 0.5
    c_w = 8000.0
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
    """dH/dt and dc_w/dt share the same driving force; ratio is MW/ρ_sol · H₀.

    dc_w/dt = g/H₀ · (p_sat/RT) · driving  [mol/m³/s]
    dH/dt   = g    · (MW/ρ)   · (p_sat/RT) · driving  [m/s]
    ratio   = (MW/ρ) · H₀  [m⁴/mol]
    """
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


def test_concentration_ratio_desorption_formula():
    t_gel, t_cond = 45.0, 28.0
    p_g = saturation_vapor_pressure_pa(t_gel)
    p_c = saturation_vapor_pressure_pa(t_cond)
    expected = (p_c / p_g) * ((t_gel + 273.15) / (t_cond + 273.15))
    assert concentration_ratio_desorption(t_gel, t_cond) == pytest.approx(expected)


def test_desorption_g_uses_lewis_analogy(config: DeviceConfig, mass):
    """Note S1 Eq. S5: g = h_conv · D_air / k_air in desorption."""
    h0 = config.hydrogel_thickness_m
    t_gel, t_cond = 50.0, 30.0
    from solar_lumped.physics.correlations import hollands_vapor_gap_h_conv_w_m2_k

    gap = max(config.vapor_gap_m - h0, 1e-4)
    h_conv = hollands_vapor_gap_h_conv_w_m2_k(
        gap, t_gel, t_cond, tilt_deg=config.tilt_deg
    )
    expected_g = mass_transfer_g_from_h_conv_m_s(h_conv)
    g = mass_transfer_g_m_s(
        phase="desorption",
        params=mass,
        h_m=h0,
        t_gel_c=t_gel,
        t_cond_c=t_cond,
    )
    assert g == pytest.approx(expected_g, rel=1e-12)


def test_m_des_from_gel_inventory(config: DeviceConfig):
    """Eq. mdot: ṁ_des = MW · (−dc_w/dt · H − c_w · dH/dt), ṁ ≥ 0."""
    c_w, h_m = 15000.0, 0.0045
    dc, dh = -0.5, -1e-5
    expected = max(
        0.0,
        -WATER_MOLAR_MASS_KG_MOL * (dc * h_m + c_w * dh),
    )
    assert m_des_kg_s_m2_from_state(c_w, h_m, dc, dh) == pytest.approx(expected)


def test_steady_thermal_residuals_near_zero(config: DeviceConfig, thermal):
    """Eqs. 1, 3, 4 residuals ≈ 0 at solve_steady_thermal solution (effective gap)."""
    h0 = config.hydrogel_thickness_m
    gap_eff = config.vapor_gap_m - h0
    state = solve_steady_thermal(
        t_cond_c=30.0,
        t_amb_c=25.0,
        q_solar_w_m2=600.0,
        m_des_kg_s_m2=2e-6,
        h_amb=10.0,
        params=thermal,
        h_m=h0,
        vapor_gap_m=gap_eff,
    )
    r = _residuals(
        np.array([state.t_gel_c, state.t_abs_c, state.t_glass_c]),
        30.0,
        25.0,
        600.0,
        2e-6,
        10.0,
        thermal,
        gap_eff,
        h0,
    )
    assert float(np.linalg.norm(r)) < 1e-4


def test_absorption_coupled_rates_match_doc(config: DeviceConfig, mass, thermal):
    """Absorption: Q_solar=0, ṁ_des=0, dT_cond/dt=0; Note S1 T_gel = T_amb."""
    h0 = config.hydrogel_thickness_m
    t_amb = 20.0
    rates = evaluate_coupled_rates(
        c_w=9000.0,
        h_m=h0,
        t_cond_c=t_amb,
        t_amb_c=t_amb,
        rh=0.6,
        q_solar_w_m2=0.0,
        h_amb=8.0,
        phase="absorption",
        mass=mass,
        thermal=thermal,
        vapor_gap_m=config.vapor_gap_m,
        condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
        fin_area_ratio=config.fin_area_ratio,
        h_fg_j_per_kg=config.h_fg_j_per_kg,
    )
    assert rates.m_des_kg_s_m2 == 0.0
    assert rates.dT_cond_dt == 0.0
    assert rates.t_gel_c == pytest.approx(t_amb)
    assert rates.dc_w_dt > 0.0


def test_desorption_m_des_self_consistent(config: DeviceConfig, mass, thermal):
    """Desorption root find: ṁ_des matches Note S1 flux (Eq. 5 with H₀)."""
    from solar_lumped.physics.mass_transfer import m_des_kg_s_m2_from_dc_w

    h0 = config.hydrogel_thickness_m
    rates = evaluate_coupled_rates(
        c_w=14000.0,
        h_m=h0 * 1.05,
        t_cond_c=32.0,
        t_amb_c=25.0,
        rh=0.4,
        q_solar_w_m2=700.0,
        h_amb=10.0,
        phase="desorption",
        mass=mass,
        thermal=thermal,
        vapor_gap_m=config.vapor_gap_m,
        condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
        fin_area_ratio=config.fin_area_ratio,
        h_fg_j_per_kg=config.h_fg_j_per_kg,
    )
    m_calc = m_des_kg_s_m2_from_dc_w(rates.dc_w_dt, h0_ref_m=h0)
    assert rates.dc_w_dt <= 0.0
    assert rates.dH_dt <= 0.0
    if rates.m_des_kg_s_m2 > 0.0:
        assert m_calc == pytest.approx(rates.m_des_kg_s_m2, rel=1e-8, abs=1e-14)


def test_eq2_condenser_rate_matches_formula(config: DeviceConfig, mass, thermal):
    """Wilson Eq. 2: dT_cond/dt from evaluate_coupled_rates matches explicit balance."""
    h0 = config.hydrogel_thickness_m
    t_cond = 35.0
    t_amb = 25.0
    h_amb = 10.0
    rates = evaluate_coupled_rates(
        c_w=13000.0,
        h_m=h0,
        t_cond_c=t_cond,
        t_amb_c=t_amb,
        rh=0.4,
        q_solar_w_m2=650.0,
        h_amb=h_amb,
        phase="desorption",
        mass=mass,
        thermal=thermal,
        vapor_gap_m=config.vapor_gap_m,
        condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
        fin_area_ratio=config.fin_area_ratio,
        h_fg_j_per_kg=config.h_fg_j_per_kg,
    )
    t_gel = rates.t_gel_c
    m_des = rates.m_des_kg_s_m2
    h_conv_g = rates.thermal.h_conv_g
    h_conv_cond = condenser_h_conv_w_m2_k(h_amb, fin_area_ratio=config.fin_area_ratio)
    eps_gc = parallel_plate_emissivity(thermal.eps_gel, thermal.eps_al)
    q_rad = radiative_exchange_w_m2(t_gel, t_cond, emissivity=eps_gc)
    tmass = config.condenser_thermal_mass_j_m2_k()
    expected = (
        h_conv_g * (t_gel - t_cond)
        - h_conv_cond * (t_cond - t_amb)
        + m_des * config.h_fg_j_per_kg
        + q_rad
    ) / tmass
    assert rates.dT_cond_dt == pytest.approx(expected, rel=1e-10)


def test_thickness_constraints_at_h0(config: DeviceConfig, mass, thermal):
    """H = H₀: absorption allows swelling only; desorption forbids shrinkage."""
    h0 = config.hydrogel_thickness_m
    abs_rates = evaluate_coupled_rates(
        c_w=8000.0,
        h_m=h0,
        t_cond_c=22.0,
        t_amb_c=22.0,
        rh=0.55,
        q_solar_w_m2=0.0,
        h_amb=8.0,
        phase="absorption",
        mass=mass,
        thermal=thermal,
        vapor_gap_m=config.vapor_gap_m,
        condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
        fin_area_ratio=config.fin_area_ratio,
        h_fg_j_per_kg=config.h_fg_j_per_kg,
    )
    assert abs_rates.dH_dt >= 0.0

    des_rates = evaluate_coupled_rates(
        c_w=14000.0,
        h_m=h0,
        t_cond_c=40.0,
        t_amb_c=25.0,
        rh=0.4,
        q_solar_w_m2=800.0,
        h_amb=10.0,
        phase="desorption",
        mass=mass,
        thermal=thermal,
        vapor_gap_m=config.vapor_gap_m,
        condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
        fin_area_ratio=config.fin_area_ratio,
        h_fg_j_per_kg=config.h_fg_j_per_kg,
    )
    assert des_rates.dH_dt == 0.0


def test_integrated_cycle_state_dimensions(config: DeviceConfig):
    """Absorption integrates [c_w, H]; desorption adds transient T_cond."""
    _, _, abs_res, des_res = run_daily_cycle(baseline_profile(), config)
    assert abs_res.t_cond_c is None
    assert des_res.t_cond_c is not None
    assert len(abs_res.c_w) == len(abs_res.H) == len(abs_res.t_gel_c)
    assert len(des_res.c_w) == len(des_res.H) == len(des_res.t_cond_c)


def test_integrated_h_never_below_h0(config: DeviceConfig):
    h0 = config.hydrogel_thickness_m
    _, _, abs_res, des_res = run_daily_cycle(baseline_profile(), config)
    assert np.min(abs_res.H) >= h0 - 1e-12
    assert np.min(des_res.H) >= h0 - 1e-12


def test_thermal_efficiency_definition(config: DeviceConfig):
    """η_th = m_water · h_fg / ∫ Q_solar dt over desorption."""
    profile = baseline_profile()
    y, eta, _, des_res = run_daily_cycle(profile, config)
    q_solar_int = sum(
        profile.desorption.solar_w_m2[i] * profile.desorption.dt_s
        for i in range(len(profile.desorption.solar_w_m2))
    )
    expected_eta = (y * config.h_fg_j_per_kg / q_solar_int) if q_solar_int > 0 else 0.0
    assert eta == pytest.approx(expected_eta, rel=1e-12)
    assert y == des_res.water_collected_kg_m2
