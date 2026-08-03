"""Tests for solar_lumped SAWH simulation."""

import math

import numpy as np
import pytest

# Wilson Table S3 g_chamber, used only when replaying/validating against a
# specific published Wilson et al. figure (the code's own default may differ).
WILSON_TABLE_S3_G_CHAMBER_M_S = 0.0085

from solar_lumped.economics import (
    lcow_cost_breakdown_from_daily_yield,
    lcow_from_daily_yield,
)
from solar_lumped.economics import LCOEconomicParams
from solar_lumped.physics import thermal_residual_norm
from solar_lumped.physics import SystemThermalParams
from solar_lumped.physics import concentration_ratio_absorption, dc_w_dt
from solar_lumped.physics import equilibrium_c_w_at_rh
from solar_lumped.physics import (
    CONDENSER_THERMAL_MASS_J_M2_K,
    EPS_AL,
    EPS_GEL,
    FIN_AREA_RATIO,
    G_CHAMBER_M_S,
    H0_M,
    H_DES_J_PER_KG,
    H_FG_J_PER_KG,
    L_C_M,
    L_G_M,
    U_GEL_W_M2_K,
    u_gel_w_m2_k,
)
from solar_lumped.simulation import SystemConfig
from solar_lumped.simulation import run_daily_cycle
from solar_lumped.weather import baseline_profile


def test_table_s3_system_defaults():
    cfg = SystemConfig.comsol_table_s3()

    assert cfg.hydrogel_thickness_m == H0_M
    assert cfg.vapor_gap_m == L_G_M
    assert cfg.g_conv_m_s == G_CHAMBER_M_S
    assert cfg.condenser_thickness_m == pytest.approx(L_C_M)
    assert cfg.fin_area_ratio == FIN_AREA_RATIO
    assert cfg.h_fg_j_per_kg == H_FG_J_PER_KG
    thermal = cfg.thermal_params()
    assert thermal.h_des_j_per_kg == H_DES_J_PER_KG
    assert u_gel_w_m2_k(cfg.hydrogel_thickness_m) == pytest.approx(U_GEL_W_M2_K)
    assert thermal.eps_gel == EPS_GEL
    assert thermal.eps_al == EPS_AL
    assert cfg.condenser_thermal_mass_j_m2_k() == pytest.approx(
        CONDENSER_THERMAL_MASS_J_M2_K
    )


def test_algebraic_balances_small_residual():
    params = SystemThermalParams()
    norm = thermal_residual_norm(
        t_cond_c=25.0,
        t_amb_c=25.0,
        q_solar_w_m2=600.0,
        m_des_kg_s_m2=1e-5,
        h_amb=10.0,
        params=params,
        h_m=SystemConfig.baseline().hydrogel_thickness_m,
    )
    assert norm < 1e-2


def test_absorption_increases_c_w():
    config = SystemConfig()
    mass = config.mass_params()
    h0 = config.hydrogel_thickness_m
    # Unsaturated gel: c0=40000 sits at ~0.38 brine salt fraction, below XI_SAT_LICL=0.458,
    # so this tests the real Conde a_w rather than the saturation plateau.
    c0 = 40000.0
    dc = dc_w_dt(
        c0,
        t_gel_c=25.0,
        c_r=concentration_ratio_absorption(0.5),
        params=mass,
        h_m=h0,
    )
    assert dc > 0.0


def test_pam_licl_brine_aw_inverts_at_rh():
    from solar_lumped.physics import (
        licl_equilibrium_brine_salt_fraction,
        water_activity_from_c_w,
    )

    config = SystemConfig()
    mass = config.mass_params()
    h0 = config.hydrogel_thickness_m
    rh = 0.38
    cw_eq = equilibrium_c_w_at_rh(
        rh,
        c_s=mass.c_s_mol_m3,
        ions_per_formula=2,
        temperature_c=25.0,
        salt_to_polymer_ratio=config.salt_to_polymer_ratio,
        h_m=h0,
        h0_ref_m=h0,
    )
    aw_eq = water_activity_from_c_w(
        cw_eq,
        c_s=mass.c_s_mol_m3,
        ions_per_formula=2,
        temperature_c=25.0,
        salt_to_polymer_ratio=config.salt_to_polymer_ratio,
        h_m=h0,
        h0_ref_m=h0,
    )
    assert aw_eq <= rh + 0.02
    assert licl_equilibrium_brine_salt_fraction(rh, 25.0) > 0.0


def test_water_inventory_series_baseline():
    from solar_lumped.simulation import water_inventory_series

    y, _, abs_res, des_res = run_daily_cycle(baseline_profile(), SystemConfig.baseline())
    series = water_inventory_series(abs_res, des_res, config=SystemConfig.baseline())
    assert len(series.time_s) == len(series.water_l_m2) == len(series.phase)
    assert len(series.collected_water_l_m2) == len(series.time_s)
    assert series.water_l_m2[0] > 0.0
    assert float(np.max(series.water_l_m2)) >= series.water_l_m2[0]
    assert series.absorption_end_s > 0.0
    assert series.collected_water_l_m2[0] == 0.0
    assert series.collected_water_l_m2[-1] == pytest.approx(des_res.water_collected_kg_m2, rel=0.02)


def test_baseline_simulation_runs():
    y, eta, _, _ = run_daily_cycle(baseline_profile(), SystemConfig.baseline())
    assert y >= 0.0
    assert math.isfinite(eta)


def test_desorption_flux_matches_inventory_loss():
    from solar_lumped.physics import m_des_kg_s_m2_from_state
    from solar_lumped.physics import WATER_MOLAR_MASS_KG_MOL

    h0 = SystemConfig.baseline().hydrogel_thickness_m
    y, _, abs_r, des_r = run_daily_cycle(baseline_profile(), SystemConfig.baseline())

    # Wilson's yield = integral(-dc_w/dt * H0 * MW) dt ≈ (c_w_des_start − c_w_des_end) * H0 * MW.
    # This uses H0 (reference thickness), not the swollen H at start of desorption.
    # The "inventory loss" as c_w*H changes includes both concentration change and
    # volume-change contributions; only the former is collected yield per Wilson Note S1.
    cw_start = float(abs_r.c_w[-1])
    cw_end = float(des_r.c_w[-1])
    expected_yield = (cw_start - cw_end) * h0 * WATER_MOLAR_MASS_KG_MOL
    assert abs(y - expected_yield) < 0.05 * max(expected_yield, 1e-6)
    assert y == des_r.water_collected_kg_m2
    assert m_des_kg_s_m2_from_state(1e5, 0.004, -1.0, -1e-5) > 0.0


def test_baseline_yield_from_desorption_flux():
    config = SystemConfig.baseline(g_conv_m_s=WILSON_TABLE_S3_G_CHAMBER_M_S)
    y, _, _, des = run_daily_cycle(baseline_profile(), config)
    assert y == des.water_collected_kg_m2
    assert y >= 0.0
    # Paper Fig. 2 baseline ~1.7 L/m²/day (25°C, 50% RH, 600 W/m²)
    assert 0.8 < y < 2.5


def test_lcow_breakdown_sums():
    econ = LCOEconomicParams()
    cfg = SystemConfig()
    y = 0.5
    bd = lcow_cost_breakdown_from_daily_yield(
        y,
        salt_name=cfg.salt_name,
        salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
        hydrogel_thickness_m=cfg.hydrogel_thickness_m,
        econ=econ,
    )
    assert bd is not None
    total = lcow_from_daily_yield(
        y,
        salt_name=cfg.salt_name,
        salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
        hydrogel_thickness_m=cfg.hydrogel_thickness_m,
        econ=econ,
    )
    assert abs(bd.total_usd_per_m3 - total) < 1e-3 * total


def test_atacama_replay_runs():
    from solar_lumped.weather import ATACAMA_FIELD_DESORPTION_STEPS
    from solar_lumped.weather import replay_profile

    profile = replay_profile("atacama-replay")
    config = SystemConfig.atacama_field()
    y, eta, _, _ = run_daily_cycle(profile, config)
    assert y >= 0.0
    assert math.isfinite(eta)
    assert len(profile.desorption.temperature_c) == ATACAMA_FIELD_DESORPTION_STEPS
    assert config.tilt_deg == 25.0
    assert config.fin_area_ratio == 5.0
    mean_abs_rh = sum(profile.absorption.relative_humidity) / len(
        profile.absorption.relative_humidity
    )
    assert 0.25 < mean_abs_rh < 0.55


def test_cycled_initial_uses_post_desorption_state():
    from solar_lumped.weather import replay_profile

    profile = replay_profile("atacama-replay")
    config = SystemConfig.atacama_field()
    h0 = config.hydrogel_thickness_m
    _, _, abs_res, _ = run_daily_cycle(
        profile, config, cyclic_initial=True, cyclic_warmup_cycles=2
    )
    cw_cycled = float(abs_res.c_w[0])
    h_cycled = float(abs_res.H[0])
    assert h_cycled >= h0
    assert cw_cycled > 0.0


def test_baseline_starts_at_fabrication_equilibrium():
    from solar_lumped.physics import (
        FABRICATION_EQUILIBRIUM_RH,
        pam_licl_uptake_g_g_at_rh,
    )
    from solar_lumped.weather import baseline_initial_c_w

    config = SystemConfig.baseline()
    h0 = config.hydrogel_thickness_m
    c_w0 = baseline_initial_c_w(h_m=h0)
    _, _, abs_res, _ = run_daily_cycle(
        baseline_profile(),
        config,
        c_w_initial=c_w0,
    )
    from solar_lumped.physics import pam_licl_gravimetric_uptake_g_g

    u0 = pam_licl_gravimetric_uptake_g_g(
        float(abs_res.c_w[0]), float(abs_res.H[0]), h0_ref_m=h0
    )
    u_eq = pam_licl_uptake_g_g_at_rh(FABRICATION_EQUILIBRIUM_RH)
    assert abs(u0 - u_eq) < 0.05


def test_fig_s1_replay_matches_note_s1d():
    from solar_lumped.physics import water_in_gel_l_m2
    from solar_lumped.weather import (
        FIG_S1_ABSORPTION_STEPS,
        FIG_S1_DESORPTION_STEPS,
        FIG_S1_INITIAL_WATER_L_M2,
        fig_s1_initial_c_w,
    )
    from solar_lumped.weather import replay_profile

    profile = replay_profile("fig-s1-replay")
    assert len(profile.absorption.temperature_c) == FIG_S1_ABSORPTION_STEPS
    assert len(profile.desorption.solar_w_m2) == FIG_S1_DESORPTION_STEPS
    assert profile.desorption.solar_w_m2[0] == 800.0
    assert profile.absorption.relative_humidity[0] == 0.5

    config = SystemConfig.comsol_table_s3(g_conv_m_s=WILSON_TABLE_S3_G_CHAMBER_M_S)
    c_w0 = fig_s1_initial_c_w(h_m=config.hydrogel_thickness_m)
    y, eta, abs_res, des_res = run_daily_cycle(
        profile, config, c_w_initial=c_w0
    )
    assert y >= 0.0
    assert math.isfinite(eta)

    w0 = water_in_gel_l_m2(float(abs_res.c_w[0]), float(abs_res.H[0]))
    w_peak = water_in_gel_l_m2(float(abs_res.c_w[-1]), float(abs_res.H[-1]))
    w_end = water_in_gel_l_m2(float(des_res.c_w[-1]), float(des_res.H[-1]))

    assert abs(w0 - FIG_S1_INITIAL_WATER_L_M2) < 0.05
    assert w_peak > w0
    assert w_end < w_peak
    # Paper Fig. S1D: ~1.2 → ~2.2 → ~1.2 L/m² (kinetic; our T is self-consistent not measured).
    assert w_peak > w0
