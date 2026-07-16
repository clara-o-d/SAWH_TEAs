"""Tests for waste_heat_cycle_lumped_no_loop two-bed SAWH simulation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from waste_heat_cycle_lumped_no_loop.physics import device_defaults as dd
from waste_heat_cycle_lumped_no_loop.physics.adsorbent import (
    dq_dt_adsorption,
    equilibrium_loading_at_rh,
    get_mof,
    m_des_kg_s_m2,
    water_activity_from_loading,
)
from waste_heat_cycle_lumped_no_loop.physics.correlations import hx_effectiveness_q
from waste_heat_cycle_lumped_no_loop.physics.mass_transfer import dc_w_dt, concentration_ratio_absorption, rh_outside_desorber
from waste_heat_cycle_lumped_no_loop.physics.salt_properties import equilibrium_c_w_at_rh, water_activity_from_c_w
from waste_heat_cycle_lumped_no_loop.physics.sorbent import mass_transfer_params
from waste_heat_cycle_lumped_no_loop.simulation.detailed_plots import (
    detailed_daily_series,
    detailed_series,
    write_detailed_csv,
)
from waste_heat_cycle_lumped_no_loop.simulation.device_config import DeviceConfig
from waste_heat_cycle_lumped_no_loop.simulation.ode_system import HalfCycleResult, run_cycle, run_daily_operation, swap_roles
from waste_heat_cycle_lumped_no_loop.simulation.water_inventory import water_inventory_series
from waste_heat_cycle_lumped_no_loop.weather.profiles import datacenter_baseline_profile


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
    cyc = run_cycle(profile, config_hydrogel)
    ha = cyc.half_a
    imb = abs(ha.integral_ads_kg_m2 - ha.integral_des_kg_m2)
    mean_m = 0.5 * (ha.integral_ads_kg_m2 + ha.integral_des_kg_m2)
    assert mean_m > 1e-12
    assert imb / mean_m < 0.05


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
    cyc = run_cycle(profile, config_mof)
    ha = cyc.half_a
    imb = abs(ha.integral_ads_kg_m2 - ha.integral_des_kg_m2)
    mean_m = 0.5 * (ha.integral_ads_kg_m2 + ha.integral_des_kg_m2)
    assert mean_m > 1e-8
    assert imb / mean_m < 0.25


def test_role_swap_conserves_state_mof(config_mof: DeviceConfig):
    res = HalfCycleResult(
        time_s=np.array([0.0, 1.0]),
        q_a=np.array([0.2, 0.25]),
        q_d=np.array([0.1, 0.05]),
        t_a_c=np.array([30.0, 31.0]),
        t_d_c=np.array([35.0, 36.0]),
        t_cond_c=np.array([28.0, 29.0]),
        m_ads_kg_s_m2=np.array([1e-5, 1e-5]),
        m_des_kg_s_m2=np.array([1e-5, 1e-5]),
        water_collected_kg_m2=0.0,
        integral_ads_kg_m2=0.0,
        integral_des_kg_m2=0.0,
    )
    la, ld, ha, hd, ta, td, tc = swap_roles(res, config_mof)
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
