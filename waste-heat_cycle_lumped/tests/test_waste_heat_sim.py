"""Tests for waste_heat_cycle_lumped two-bed SAWH simulation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from waste_heat_cycle_lumped.physics import device_defaults as dd
from waste_heat_cycle_lumped.physics.adsorbent import (
    dq_dt_adsorption,
    equilibrium_loading_at_rh,
    get_mof,
    m_des_kg_s_m2,
    water_activity_from_loading,
)
from waste_heat_cycle_lumped.physics.correlations import (
    condenser_h_conv_w_m2_k,
    hx_effectiveness_q,
    waste_heat_to_loop_q_w,
)
from waste_heat_cycle_lumped.physics.mass_transfer import dc_w_dt, concentration_ratio_absorption, rh_outside_desorber
from waste_heat_cycle_lumped.physics.salt_properties import (
    C_W_MAX_MOL_M3,
    C_W_MIN_MOL_M3,
    equilibrium_c_w_at_rh,
    water_activity_from_c_w,
)
from waste_heat_cycle_lumped.physics.sorbent import (
    h_ads_j_per_kg,
    h_des_j_per_kg,
    mass_transfer_params,
    water_kg_m2_bed,
)
from waste_heat_cycle_lumped.simulation import ode_system
from waste_heat_cycle_lumped.simulation.control import compute_controls
from waste_heat_cycle_lumped.simulation.detailed_plots import (
    detailed_daily_series,
    detailed_series,
    write_detailed_csv,
)
from waste_heat_cycle_lumped.simulation.device_config import DeviceConfig
from waste_heat_cycle_lumped.simulation.ode_system import (
    HalfCycleResult,
    run_cycle,
    run_daily_operation,
    swap_roles,
)
from waste_heat_cycle_lumped.simulation.water_inventory import water_inventory_series
from waste_heat_cycle_lumped.weather.profiles import datacenter_baseline_profile


@pytest.fixture
def config_hydrogel() -> DeviceConfig:
    return DeviceConfig.datacenter_baseline()


@pytest.fixture
def config_mof() -> DeviceConfig:
    return DeviceConfig.mof_baseline()


@pytest.fixture
def profile(config_hydrogel: DeviceConfig):
    return datacenter_baseline_profile(tau_half_s=config_hydrogel.tau_half_s)


def test_default_sorbent_is_hydrogel():
    assert DeviceConfig().sorbent == "hydrogel"


def test_half_cycle_ends_at_rh_threshold(config_hydrogel: DeviceConfig, profile):
    cyc = run_cycle(profile, config_hydrogel)
    ha = cyc.half_a
    end_rh = rh_outside_desorber(float(ha.t_d_c[-1]), float(ha.t_cond_c[-1]))
    assert end_rh <= config_hydrogel.rh_desorber_switch + 1e-6
    assert float(ha.time_s[-1]) < config_hydrogel.tau_half_s - 60.0


def test_equal_mass_transfer_over_half_cycle(config_hydrogel: DeviceConfig, profile):
    """``sorbent._equalize_mass_rates`` scales m_ads/m_des to a common value at every
    instant, so their time integrals must match to numerical precision, not just
    approximately -- a wide tolerance here would silently hide a broken equalizer."""
    cyc = run_cycle(profile, config_hydrogel)
    ha = cyc.half_a
    imb = abs(ha.integral_ads_kg_m2 - ha.integral_des_kg_m2)
    mean_m = 0.5 * (ha.integral_ads_kg_m2 + ha.integral_des_kg_m2)
    assert mean_m > 1e-12
    assert imb / mean_m < 1e-6


def test_hydrogel_state_water_content_matches_flux_integral(
    config_hydrogel: DeviceConfig, profile
):
    """Real mass-balance check: the water content implied by the integrated bed
    state (loading × thickness) must match the reported flux integral. Unlike the
    ads/des equality above, this is not true by construction -- it would catch a
    bug in ``m_ads_kg_s_m2_from_state``'s product-rule derivation, or a mismatch
    between the ODE's dy_mass and the accounted flux."""
    cyc = run_cycle(profile, config_hydrogel)
    ha = cyc.half_a
    assert ha.h_a is not None and ha.h_d is not None
    delta_w_a = water_kg_m2_bed(
        float(ha.q_a[-1]), config=config_hydrogel, h_m=float(ha.h_a[-1])
    ) - water_kg_m2_bed(float(ha.q_a[0]), config=config_hydrogel, h_m=float(ha.h_a[0]))
    delta_w_d = water_kg_m2_bed(
        float(ha.q_d[0]), config=config_hydrogel, h_m=float(ha.h_d[0])
    ) - water_kg_m2_bed(float(ha.q_d[-1]), config=config_hydrogel, h_m=float(ha.h_d[-1]))
    assert delta_w_a == pytest.approx(ha.integral_ads_kg_m2, rel=0.01)
    assert delta_w_d == pytest.approx(ha.integral_des_kg_m2, rel=0.01)


def test_hydrogel_cycle_runs(config_hydrogel: DeviceConfig, profile):
    cyc = run_cycle(profile, config_hydrogel)
    assert cyc.water_collected_kg_m2 >= 0.0
    assert math.isfinite(cyc.water_collected_kg_m2)


def test_water_inventory_tracks_one_bed_absorb_then_desorb(
    config_hydrogel: DeviceConfig, profile
):
    cyc = run_cycle(profile, config_hydrogel)
    inv = water_inventory_series(cyc, config=config_hydrogel)
    swing = float(np.max(inv.water_l_m2) - np.min(inv.water_l_m2))
    assert swing > 0.5
    assert len(inv.time_s) > 100
    # Same physical bed: rises in half A, falls in half B (no swap discontinuity).
    n_abs = int(np.sum(inv.phase == "absorption"))
    assert inv.water_l_m2[n_abs - 1] >= inv.water_l_m2[0]
    assert inv.water_l_m2[-1] < inv.water_l_m2[n_abs - 1]
    # First plotted desorption point is ~6 s after swap; fast kinetics can drop slightly.
    assert abs(inv.water_l_m2[n_abs - 1] - inv.water_l_m2[n_abs]) < 0.15
    assert inv.collected_water_l_m2[0] == 0.0
    assert inv.collected_water_l_m2[-1] == pytest.approx(cyc.water_collected_kg_m2, rel=0.02)


def test_hydrogel_swelling(config_hydrogel: DeviceConfig, profile):
    cyc = run_cycle(profile, config_hydrogel)
    ha = cyc.half_a
    assert ha.h_a is not None
    assert float(ha.h_a[-1]) >= config_hydrogel.hydrogel_thickness_m - 1e-9


def test_hydrogel_adsorption_increases_c_w(config_hydrogel: DeviceConfig):
    params = mass_transfer_params(config_hydrogel)
    h0 = config_hydrogel.hydrogel_thickness_m
    c0 = 5000.0
    dc = dc_w_dt(
        c0,
        t_gel_c=dd.T_AMB_C,
        c_r=concentration_ratio_absorption(dd.RH_AMB),
        params=params,
        h_m=h0,
    )
    assert dc > 0.0


def test_hydrogel_isotherm_inverts(config_hydrogel: DeviceConfig):
    params = mass_transfer_params(config_hydrogel)
    rh = 0.45
    cw = equilibrium_c_w_at_rh(
        rh,
        c_s=params.c_s_mol_m3,
        ions_per_formula=params.ions_per_formula,
        temperature_c=30.0,
        salt_to_polymer_ratio=config_hydrogel.salt_to_polymer_ratio,
        h_m=config_hydrogel.hydrogel_thickness_m,
        h0_ref_m=config_hydrogel.hydrogel_thickness_m,
    )
    aw = water_activity_from_c_w(
        cw,
        c_s=params.c_s_mol_m3,
        ions_per_formula=params.ions_per_formula,
        temperature_c=30.0,
        salt_to_polymer_ratio=config_hydrogel.salt_to_polymer_ratio,
        h_m=config_hydrogel.hydrogel_thickness_m,
        h0_ref_m=config_hydrogel.hydrogel_thickness_m,
    )
    assert aw <= rh + 0.05


def test_mof_isotherm_monotone(config_mof: DeviceConfig):
    props = config_mof.mof()
    aw_lo = water_activity_from_loading(0.05, temperature_c=30.0, props=props)
    aw_hi = water_activity_from_loading(0.25, temperature_c=30.0, props=props)
    assert aw_hi > aw_lo


def test_mof_adsorption_increases_q(config_mof: DeviceConfig):
    props = config_mof.mof()
    dq = dq_dt_adsorption(0.05, temperature_c=dd.T_AMB_C, rh_amb=dd.RH_AMB, props=props)
    assert dq > 0.0


def test_vacuum_desorption_flux():
    m1 = m_des_kg_s_m2(temperature_c=50.0, t_cond_c=30.0, c_vac_kg_s_pa_m2=1e-8)
    m2 = m_des_kg_s_m2(temperature_c=50.0, t_cond_c=30.0, c_vac_kg_s_pa_m2=2e-8)
    assert m2 == pytest.approx(2.0 * m1)
    m0 = m_des_kg_s_m2(temperature_c=10.0, t_cond_c=50.0, c_vac_kg_s_pa_m2=1e-8)
    assert m0 == 0.0
    m_hot_cond = m_des_kg_s_m2(temperature_c=50.0, t_cond_c=50.0, c_vac_kg_s_pa_m2=1e-8)
    assert m_hot_cond == 0.0


def test_hx_effectiveness_limits():
    ua = 500.0
    dt = 10.0
    q_low = hx_effectiveness_q(1.0, ua, dt)
    assert abs(q_low - 1.0 * dt) < 0.01 * abs(1.0 * dt)
    q_high = hx_effectiveness_q(1e6, ua, dt)
    assert abs(q_high - ua * dt) / (ua * dt) < 1e-3


def test_mof_cycle_runs(config_mof: DeviceConfig, profile):
    cyc = run_cycle(profile, config_mof)
    assert cyc.water_collected_kg_m2 >= 0.0


def test_mof_mass_balance_half_cycle(config_mof: DeviceConfig, profile):
    """Same equalizer invariant as test_equal_mass_transfer_over_half_cycle, for MOF."""
    cyc = run_cycle(profile, config_mof)
    ha = cyc.half_a
    imb = abs(ha.integral_ads_kg_m2 - ha.integral_des_kg_m2)
    mean_m = 0.5 * (ha.integral_ads_kg_m2 + ha.integral_des_kg_m2)
    assert mean_m > 1e-8
    assert imb / mean_m < 1e-6


def test_mof_state_water_content_matches_flux_integral(config_mof: DeviceConfig, profile):
    """Real mass-balance check for MOF: bed loading trajectory vs. reported flux
    integral (see test_hydrogel_state_water_content_matches_flux_integral)."""
    cyc = run_cycle(profile, config_mof)
    ha = cyc.half_a
    delta_w_a = water_kg_m2_bed(float(ha.q_a[-1]), config=config_mof) - water_kg_m2_bed(
        float(ha.q_a[0]), config=config_mof
    )
    delta_w_d = water_kg_m2_bed(float(ha.q_d[0]), config=config_mof) - water_kg_m2_bed(
        float(ha.q_d[-1]), config=config_mof
    )
    assert delta_w_a == pytest.approx(ha.integral_ads_kg_m2, rel=0.01)
    assert delta_w_d == pytest.approx(ha.integral_des_kg_m2, rel=0.01)


def test_role_swap_conserves_state_mof(config_mof: DeviceConfig):
    res = HalfCycleResult(
        time_s=np.array([0.0, 1.0]),
        q_a=np.array([0.2, 0.25]),
        q_d=np.array([0.1, 0.05]),
        t_a_c=np.array([30.0, 31.0]),
        t_d_c=np.array([35.0, 36.0]),
        t_f_c=np.array([32.0, 33.0]),
        t_cond_c=np.array([28.0, 29.0]),
        m_ads_kg_s_m2=np.array([1e-5, 1e-5]),
        m_des_kg_s_m2=np.array([1e-5, 1e-5]),
        water_collected_kg_m2=0.0,
        integral_ads_kg_m2=0.0,
        integral_des_kg_m2=0.0,
    )
    la, ld, ha, hd, ta, td, tf, tc = swap_roles(res, config_mof)
    assert la == pytest.approx(0.05)
    assert ld == pytest.approx(0.25)
    assert ha is None
    assert hd is None


def test_equilibrium_loading_at_rh(config_mof: DeviceConfig):
    props = get_mof(config_mof.mof_name)
    q = equilibrium_loading_at_rh(0.5, temperature_c=30.0, props=props)
    assert 0.0 < q < props.q_max_kg_kg


def test_detailed_series_single_cycle(config_hydrogel: DeviceConfig, profile, tmp_path: Path):
    cyc = run_cycle(profile, config_hydrogel)
    detailed = detailed_series(cyc, config=config_hydrogel, profile=profile)

    n = len(detailed.time_s)
    ha = cyc.half_a
    hb = cyc.half_b
    assert n == len(ha.time_s) + len(hb.time_s) - 1
    assert len(detailed.t_a_c) == n
    assert len(detailed.t_d_c) == n
    assert len(detailed.t_f_c) == n
    assert len(detailed.t_cond_c) == n
    assert detailed.half_cycle_end_s == pytest.approx(float(ha.time_s[-1]))
    assert detailed.n_cycles == 1
    assert set(detailed.half_cycle) <= {"A", "B"}

    csv_path = tmp_path / "diagnostics.csv"
    write_detailed_csv(csv_path, detailed)
    text = csv_path.read_text()
    assert "t_a_c" in text
    assert "t_wh_in_c" in text
    assert "t_tracked_c" not in text


def test_detailed_daily_series(config_hydrogel: DeviceConfig, profile, tmp_path: Path):
    _, _, results = run_daily_operation(profile, config_hydrogel, n_cycles=2)
    detailed = detailed_daily_series(results, config=config_hydrogel, profile=profile)

    assert detailed.n_cycles == 2
    assert len(detailed.cycle_index) == len(detailed.time_s)
    assert set(detailed.cycle_index) == {0, 1}
    assert len(detailed.t_a_c) == len(detailed.time_s)

    csv_path = tmp_path / "diagnostics_daily.csv"
    write_detailed_csv(csv_path, detailed)
    assert csv_path.exists()


def test_solver_convergence_hydrogel(config_hydrogel: DeviceConfig, profile):
    """Water yield and half-cycle duration should be near-invariant to ODE solver
    tolerance -- otherwise the reported results are an artifact of loose rtol/atol
    rather than the physics."""
    baseline = run_cycle(profile, config_hydrogel)
    orig_rtol, orig_atol = ode_system._ODE_RTOL, ode_system._ODE_ATOL
    try:
        ode_system._ODE_RTOL = orig_rtol * 1e-2
        ode_system._ODE_ATOL = orig_atol * 1e-2
        tight = run_cycle(profile, config_hydrogel)
    finally:
        ode_system._ODE_RTOL, ode_system._ODE_ATOL = orig_rtol, orig_atol

    assert tight.water_collected_kg_m2 == pytest.approx(
        baseline.water_collected_kg_m2, rel=1e-3
    )
    assert float(tight.half_a.time_s[-1]) == pytest.approx(
        float(baseline.half_a.time_s[-1]), rel=1e-3
    )


def test_solver_convergence_mof(config_mof: DeviceConfig, profile):
    baseline = run_cycle(profile, config_mof)
    orig_rtol, orig_atol = ode_system._ODE_RTOL, ode_system._ODE_ATOL
    try:
        ode_system._ODE_RTOL = orig_rtol * 1e-2
        ode_system._ODE_ATOL = orig_atol * 1e-2
        tight = run_cycle(profile, config_mof)
    finally:
        ode_system._ODE_RTOL, ode_system._ODE_ATOL = orig_rtol, orig_atol

    assert tight.water_collected_kg_m2 == pytest.approx(
        baseline.water_collected_kg_m2, rel=1e-3
    )


def _env_index(t_s: float, profile, n: int) -> int:
    return min(max(int(t_s / profile.dt_s), 0), n - 1)


def _energy_balance_residual(half: HalfCycleResult, config: DeviceConfig, profile) -> float:
    """Sum the four contactor/loop/condenser energy equations (governing_eq.tex).

    The internal exchange terms (contactor A <-> loop, loop <-> contactor D,
    vacuum-gap + radiative exchange between D and the condenser) cancel exactly
    when the four equations are added, leaving only the externally-visible
    terms below. A nonzero residual here means the recorded state trajectory
    isn't actually a solution of the coupled energy balances -- e.g. a missing
    or mis-signed term in one of the dT_*_dt functions. Returns the relative
    residual (integrated RHS vs. actual sensible-energy change).
    """
    params = config.thermal_params()
    ctrl_p = config.controller_params()
    t = np.asarray(half.time_s, dtype=float)
    t_a = np.asarray(half.t_a_c, dtype=float)
    t_d = np.asarray(half.t_d_c, dtype=float)
    t_f = np.asarray(half.t_f_c, dtype=float)
    t_cond = np.asarray(half.t_cond_c, dtype=float)
    m_ads = np.asarray(half.m_ads_kg_s_m2, dtype=float)
    m_des = np.asarray(half.m_des_kg_s_m2, dtype=float)
    n_env = len(profile.temperature_c)
    h_ads = h_ads_j_per_kg(config)
    h_des = h_des_j_per_kg(config)

    rhs = np.zeros(len(t))
    for k in range(len(t)):
        idx = _env_index(float(t[k]), profile, n_env)
        m_f = compute_controls(
            t_a_c=float(t_a[k]),
            t_d_c=float(t_d[k]),
            m_ads_kg_s_m2=float(m_ads[k]),
            m_des_kg_s_m2=float(m_des[k]),
            params=ctrl_p,
            integral_ads_kg_m2=0.0,
            integral_des_kg_m2=0.0,
        ).m_dot_f_kg_s_m2
        mdot_cp = m_f * params.fluid_cp_j_kg_k
        q_a_to_f = hx_effectiveness_q(mdot_cp, params.ua_adsorber_w_k, t_a[k] - t_f[k])
        q_f_to_d = hx_effectiveness_q(mdot_cp, params.ua_desorber_w_k, t_f[k] - t_d[k])
        q_f_loss = params.loop_loss_fraction * (abs(q_a_to_f) + abs(q_f_to_d))
        q_wh_to_f, _ = waste_heat_to_loop_q_w(
            m_dot_wh_kg_s=profile.m_dot_wh_kg_s_m2[idx],
            cp_wh_j_kg_k=params.cp_wh_j_kg_k,
            t_wh_in_c=profile.t_wh_in_c[idx],
            t_f_c=float(t_f[k]),
            ua_wh_w_k=params.wh_hx_ua_w_k,
        )
        q_conv_amb_a = (
            profile.h_amb_w_m2_k[idx] * params.contactor_area_m2 * (t_a[k] - profile.temperature_c[idx])
        )
        h_conv_cond = condenser_h_conv_w_m2_k(profile.h_amb_w_m2_k[idx], fin_area_ratio=params.fin_area_ratio)
        q_conv_cond = h_conv_cond * (t_cond[k] - profile.temperature_c[idx])
        rhs[k] = (
            q_wh_to_f
            - q_conv_amb_a
            - q_f_loss
            - q_conv_cond
            + m_ads[k] * h_ads
            - m_des[k] * h_des
            + m_des[k] * params.h_fg_j_per_kg
        )

    integral = np.trapezoid(rhs, t)
    delta_u = (
        params.contactor_thermal_mass_j_m2_k * (t_a[-1] - t_a[0])
        + params.contactor_thermal_mass_j_m2_k * (t_d[-1] - t_d[0])
        + params.fluid_thermal_mass_j_m2_k * (t_f[-1] - t_f[0])
        + params.condenser_thermal_mass_j_m2_k * (t_cond[-1] - t_cond[0])
    )
    scale = max(abs(integral), abs(delta_u), 1.0)
    return (integral - delta_u) / scale


def test_energy_balance_closes_hydrogel(config_hydrogel: DeviceConfig, profile):
    cyc = run_cycle(profile, config_hydrogel)
    assert abs(_energy_balance_residual(cyc.half_a, config_hydrogel, profile)) < 0.05
    assert abs(_energy_balance_residual(cyc.half_b, config_hydrogel, profile)) < 0.05


def test_energy_balance_closes_mof(config_mof: DeviceConfig, profile):
    cyc = run_cycle(profile, config_mof)
    assert abs(_energy_balance_residual(cyc.half_a, config_mof, profile)) < 0.05
    assert abs(_energy_balance_residual(cyc.half_b, config_mof, profile)) < 0.05


def test_no_waste_heat_yields_negligible_water_hydrogel(config_hydrogel: DeviceConfig):
    """Without a waste-heat driving force, desorption has no energy source, so
    cyclic-steady-state water production should vanish (after transients from the
    arbitrary initial condition die out over a few warmup cycles)."""
    baseline_profile = datacenter_baseline_profile(tau_half_s=config_hydrogel.tau_half_s)
    baseline = run_cycle(baseline_profile, config_hydrogel)

    no_wh_profile = datacenter_baseline_profile(
        tau_half_s=config_hydrogel.tau_half_s,
        t_wh_in_c=dd.T_AMB_C,
        m_dot_wh_kg_s_m2=0.0,
    )
    starved = run_cycle(no_wh_profile, config_hydrogel, warmup_cycles=4)
    assert starved.water_collected_kg_m2 < 0.01 * baseline.water_collected_kg_m2


def test_no_waste_heat_yields_negligible_water_mof(config_mof: DeviceConfig):
    baseline_profile = datacenter_baseline_profile(tau_half_s=config_mof.tau_half_s)
    baseline = run_cycle(baseline_profile, config_mof)

    no_wh_profile = datacenter_baseline_profile(
        tau_half_s=config_mof.tau_half_s,
        t_wh_in_c=dd.T_AMB_C,
        m_dot_wh_kg_s_m2=0.0,
    )
    starved = run_cycle(no_wh_profile, config_mof, warmup_cycles=4)
    assert starved.water_collected_kg_m2 < 0.01 * baseline.water_collected_kg_m2


def test_hydrogel_absorption_has_stable_equilibrium(config_hydrogel: DeviceConfig):
    """At fixed ambient RH, a very dry gel must absorb (dc/dt > 0) and a very wet
    gel must desorb (dc/dt < 0) -- otherwise the isotherm has no stable fixed
    point and the bed would run away to fully dry or fully saturated."""
    params = mass_transfer_params(config_hydrogel)
    rh = 0.45
    t_c = 30.0
    h_m = config_hydrogel.hydrogel_thickness_m
    dry = dc_w_dt(
        C_W_MIN_MOL_M3 * 1.01,
        t_gel_c=t_c,
        c_r=concentration_ratio_absorption(rh),
        params=params,
        h_m=h_m,
        phase="absorption",
    )
    wet = dc_w_dt(
        C_W_MAX_MOL_M3 * 0.99,
        t_gel_c=t_c,
        c_r=concentration_ratio_absorption(rh),
        params=params,
        h_m=h_m,
        phase="absorption",
    )
    assert dry > 0.0
    assert wet < 0.0


def test_mof_zero_driving_force_adsorption(config_mof: DeviceConfig):
    """At the MOF's own equilibrium loading for a given RH, the adsorption rate
    must vanish -- there is no driving force left."""
    props = get_mof(config_mof.mof_name)
    rh = 0.45
    t_c = 30.0
    q_eq = equilibrium_loading_at_rh(rh, temperature_c=t_c, props=props)
    dq = dq_dt_adsorption(q_eq, temperature_c=t_c, rh_amb=rh, props=props)
    assert dq == pytest.approx(0.0, abs=1e-12)
