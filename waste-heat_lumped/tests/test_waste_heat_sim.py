"""Tests for fluid-heated daily-cycle SAWH simulation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from waste_heat_lumped.physics import device_defaults as dd
from waste_heat_lumped.physics.correlations import hx_effectiveness_q
from waste_heat_lumped.physics.device_balances import q_f_to_gel_w_m2
from waste_heat_lumped.physics.mass_transfer import dc_w_dt, concentration_ratio_absorption
from waste_heat_lumped.physics.salt_properties import equilibrium_c_w_at_rh
from waste_heat_lumped.simulation.detailed_plots import detailed_series, write_detailed_csv
from waste_heat_lumped.simulation.device_config import DeviceConfig
from waste_heat_lumped.simulation.ode_system import run_daily_cycle
from waste_heat_lumped.weather.profiles import datacenter_baseline_profile


def test_hx_effectiveness_ntu_epsilon():
    mdot_cp = 0.25 * 4180.0
    ua = 800.0
    delta_t = 20.0
    q = hx_effectiveness_q(mdot_cp, ua, delta_t)
    ntu = ua / mdot_cp
    expected = mdot_cp * delta_t * (1.0 - math.exp(-ntu))
    assert q == pytest.approx(expected, rel=1e-9)


def test_q_f_to_gel_zero_when_no_flow():
    assert q_f_to_gel_w_m2(
        t_gel_c=40.0,
        t_f_c=58.0,
        m_dot_f_kg_s_m2=0.0,
        ua_gel_w_k=800.0,
        fluid_cp_j_kg_k=4180.0,
    ) == 0.0


def test_q_f_to_gel_positive_during_desorption():
    q = q_f_to_gel_w_m2(
        t_gel_c=40.0,
        t_f_c=58.0,
        m_dot_f_kg_s_m2=dd.M_DOT_F_KG_S_M2,
        ua_gel_w_k=dd.UA_GEL_W_K,
        fluid_cp_j_kg_k=dd.FLUID_CP_J_KG_K,
    )
    assert q > 0.0


def test_absorption_mass_transfer_positive():
    config = DeviceConfig.datacenter_baseline()
    mass = config.mass_params()
    c_w = equilibrium_c_w_at_rh(0.2, c_s=mass.c_s_mol_m3, ions_per_formula=mass.ions_per_formula)
    rate = dc_w_dt(
        c_w,
        t_gel_c=dd.T_AMB_C,
        c_r=concentration_ratio_absorption(dd.RH_AMB),
        params=mass,
        h_m=config.hydrogel_thickness_m,
        phase="absorption",
    )
    assert rate > 0.0


def test_daily_cycle_datacenter_baseline():
    config = DeviceConfig.datacenter_baseline()
    profile = datacenter_baseline_profile()
    yield_kg, eta, abs_res, des_res = run_daily_cycle(profile, config)
    assert yield_kg > 0.0
    assert 0.0 < eta < 1.0
    assert abs_res.water_collected_kg_m2 == 0.0
    assert des_res.water_collected_kg_m2 > 0.0
    assert all(q == 0.0 for q in abs_res.q_f_to_gel_w_m2)
    assert any(q > 0.0 for q in des_res.q_f_to_gel_w_m2)


def test_gel_balance_structure():
    """Gel residual uses Q_f→gel − ṁ h_des − gap − rad (no vacuum terms)."""
    from waste_heat_lumped.physics.device_balances import solve_steady_gel_thermal

    config = DeviceConfig.datacenter_baseline()
    thermal = config.thermal_params()
    state = solve_steady_gel_thermal(
        t_cond_c=30.0,
        m_des_kg_s_m2=1e-6,
        params=thermal,
        h_m=config.hydrogel_thickness_m,
        m_dot_f_kg_s_m2=thermal.m_dot_f_kg_s_m2,
    )
    assert state.t_gel_c > 30.0
    assert state.q_f_to_gel_w_m2 > 0.0


def test_detailed_series_full_cycle(tmp_path: Path):
    config = DeviceConfig.datacenter_baseline()
    profile = datacenter_baseline_profile()
    _, _, abs_res, des_res = run_daily_cycle(profile, config)
    detailed = detailed_series(profile, abs_res, des_res, config)

    n = len(detailed.time_s)
    assert n == len(abs_res.time_s) + len(des_res.time_s) - 1
    assert len(detailed.phase) == n
    assert detailed.phase[0] == "absorption"
    assert detailed.phase[-1] == "desorption"
    assert detailed.absorption_end_s == pytest.approx(float(abs_res.time_s[-1]))
    assert len(detailed.t_gel_c) == n
    assert len(detailed.t_cond_c) == n
    assert all(detailed.t_f_c == config.t_f_c)
    assert all(detailed.q_f_to_gel_w_m2[: len(abs_res.time_s)] == 0.0)
    assert any(detailed.q_f_to_gel_w_m2[len(abs_res.time_s) :] > 0.0)

    csv_path = tmp_path / "diagnostics.csv"
    write_detailed_csv(csv_path, detailed)
    text = csv_path.read_text()
    assert "t_gel_c" in text
    assert "q_f_to_gel_w_m2" in text
    assert "solar_w_m2" not in text
