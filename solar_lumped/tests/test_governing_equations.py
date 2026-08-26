"""Structural checks that Wilson Eqs. 1–6 are implemented as documented."""

from __future__ import annotations

import math

import numpy as np
import pytest

from solar_lumped._parameters_xlsx import physics_value as _pv
from solar_lumped.physics import (
    condenser_h_conv_w_m2_k,
    mass_transfer_g_from_h_conv_m_s,
    parallel_plate_emissivity,
    radiative_exchange_w_m2,
)
from solar_lumped.physics import _residuals, solve_steady_thermal
from solar_lumped.physics import (
    concentration_ratio_absorption,
    concentration_ratio_desorption,
    dH_dt,
    dc_w_dt,
    mass_transfer_g_m_s,
    VAPOR_GAP_TRANSPORT_MIN_M,
    m_des_kg_s_m2_from_state,
)
from solar_lumped.physics import (
    GAS_CONSTANT_J_MOL_K,
    WATER_MOLAR_MASS_KG_MOL,
    saturation_vapor_pressure_pa,
)
from solar_lumped import simulation
from solar_lumped.simulation import evaluate_coupled_rates
from solar_lumped.simulation import SystemConfig
from solar_lumped.simulation import run_daily_cycle
from solar_lumped.weather import baseline_profile


@pytest.fixture
def config() -> SystemConfig:
    return SystemConfig.baseline()


@pytest.fixture
def mass(config: SystemConfig):
    return config.mass_params()


@pytest.fixture
def thermal(config: SystemConfig):
    return config.thermal_params()


def test_eq5_mass_transfer_formula_absorption(config: SystemConfig, mass):
    """dc_w/dt = (g/H₀) · P_sat/(RT) · (C_R − a_w) during absorption."""
    from solar_lumped.physics import _absorption_effective_water_activity

    h0 = config.hydrogel_thickness_m
    t_gel = 25.0
    rh = 0.5
    # Unsaturated gel state, so this exercises the real Conde a_w rather than the
    # saturation plateau: c_w=40000 sits at ~0.38 brine salt fraction, below
    # XI_SAT_LICL=0.458 (the sim's own operating range is 0.22-0.39).
    c_w = 40000.0
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


def test_desorption_g_uses_lewis_analogy(config: SystemConfig, mass):
    """Note S1 Eq. S5: g = h_conv · D_air / k_air in desorption."""
    h0 = config.hydrogel_thickness_m
    t_gel, t_cond = 50.0, 30.0
    from solar_lumped.physics import vapor_gap_h_conv_w_m2_k

    gap = max(config.vapor_gap_m - h0, 1e-4)
    h_conv = vapor_gap_h_conv_w_m2_k(
        gap, t_gel, t_cond, tilt_deg=config.tilt_deg
    )
    expected_g = mass_transfer_g_from_h_conv_m_s(h_conv, t_gel_c=t_gel, t_cond_c=t_cond)
    g = mass_transfer_g_m_s(
        phase="desorption",
        params=mass,
        h_m=h0,
        t_gel_c=t_gel,
        t_cond_c=t_cond,
    )
    assert g == pytest.approx(expected_g, rel=1e-12)


def test_no_thermobuoyancy_cutoff_below_7mm(mass):
    """Sub-7 mm gaps diffuse, they do not stop. Wilson's g = 0 cliff came from Hollands'
    Ra_crit = 1708; the stratified correlation we use has no onset, so Nu -> 1 smoothly and
    Sh = Nu = 1 is diffusion over a short path -- *faster* than the baseline gap."""
    t_gel, t_cond = 75.0, 27.0

    def g_at(gap_m: float) -> float:
        return mass_transfer_g_m_s(
            phase="desorption",
            params=mass,
            h_m=mass.vapor_gap_m - gap_m,
            t_gel_c=t_gel,
            t_cond_c=t_cond,
        )

    # Continuous across the old 7 mm wall, and no zero anywhere below it.
    assert g_at(VAPOR_GAP_TRANSPORT_MIN_M - 1e-6) == pytest.approx(
        g_at(VAPOR_GAP_TRANSPORT_MIN_M + 1e-6), rel=1e-3
    )
    for gap in (0.001, 0.003, 0.005, 0.007):
        assert g_at(gap) > 0.0
    # Short-path diffusion beats weak convection over the baseline gap.
    assert g_at(0.005) > 2.0 * g_at(0.036)


def test_m_des_from_gel_inventory(config: SystemConfig):
    """Eq. mdot: ṁ_des = MW · (−dc_w/dt · H − c_w · dH/dt), ṁ ≥ 0."""
    c_w, h_m = 15000.0, 0.0045
    dc, dh = -0.5, -1e-5
    expected = max(
        0.0,
        -WATER_MOLAR_MASS_KG_MOL * (dc * h_m + c_w * dh),
    )
    assert m_des_kg_s_m2_from_state(c_w, h_m, dc, dh) == pytest.approx(expected)


def test_steady_thermal_residuals_near_zero(config: SystemConfig, thermal):
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


def test_absorption_coupled_rates_match_doc(config: SystemConfig, mass, thermal):
    """Absorption: Q_solar=0, ṁ_des=0, dT_cond/dt=0; Note S1 T_gel = T_amb."""
    h0 = config.hydrogel_thickness_m
    t_amb = 20.0
    rates = evaluate_coupled_rates(
        c_w=40000.0,  # unsaturated (see test_eq5_mass_transfer_formula_absorption)
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
        config=config,
    )
    assert rates.m_des_kg_s_m2 == 0.0
    assert rates.dT_cond_dt == 0.0
    assert rates.t_gel_c == pytest.approx(t_amb)
    assert rates.dc_w_dt > 0.0


def test_desorption_m_des_self_consistent(config: SystemConfig, mass, thermal):
    """Desorption root find: ṁ_des matches Note S1 flux (Eq. 5 with H₀)."""
    from solar_lumped.physics import m_des_kg_s_m2_from_dc_w

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
        config=config,
    )
    m_calc = m_des_kg_s_m2_from_dc_w(rates.dc_w_dt, h0_ref_m=h0)
    assert rates.dc_w_dt <= 0.0
    assert rates.dH_dt <= 0.0
    if rates.m_des_kg_s_m2 > 0.0:
        assert m_calc == pytest.approx(rates.m_des_kg_s_m2, rel=1e-8, abs=1e-14)


def test_eq2_condenser_rate_matches_formula(config: SystemConfig, mass, thermal):
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
        config=config,
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


def test_thickness_constraints_at_h0(config: SystemConfig, mass, thermal):
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
        config=config,
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
        config=config,
    )
    # At H₀ the gel is still well above its hydrate floor, so it is free to keep
    # shrinking -- the thickness bound lives at c_w_min, not at the as-cast thickness.
    assert des_rates.dH_dt < 0.0


def test_integrated_cycle_state_dimensions(config: SystemConfig):
    """Absorption integrates [c_w, H]; desorption adds transient T_cond."""
    _, _, abs_res, des_res = run_daily_cycle(baseline_profile(), config)
    assert abs_res.t_cond_c is None
    assert des_res.t_cond_c is not None
    assert len(abs_res.c_w) == len(abs_res.H) == len(abs_res.t_gel_c)
    assert len(des_res.c_w) == len(des_res.H) == len(des_res.t_cond_c)


def test_integrated_h_bounded_by_the_hydrate_floor_not_h0(config: SystemConfig):
    """The gel shrinks past its as-cast thickness as it dries, but never past the
    thickness implied by c_w_min -- Eq. 6 says volume tracks water content, so the two
    states have to bottom out together."""
    floor = config.hydrogel_floor_thickness_m()
    assert 0.0 < floor < config.hydrogel_thickness_m
    _, _, abs_res, des_res = run_daily_cycle(baseline_profile(), config)
    assert np.min(abs_res.H) >= floor - 1e-12
    assert np.min(des_res.H) >= floor - 1e-12


def test_thermal_efficiency_definition(config: SystemConfig):
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


def test_c_w_floor_modes_bracket_correctly(config: SystemConfig):
    """The DRH floor is the wetter bound, and both sit under the dilution ceiling."""
    import dataclasses

    from solar_lumped.physics import saturation_brine_salt_fraction

    hyd = config.mass_params()
    drh = dataclasses.replace(config, c_w_floor_mode="drh").mass_params()

    # Hydrate floor is n·c_s exactly (LiCl·H₂O ⇒ n = 1).
    assert hyd.c_w_min_mol_m3 == pytest.approx(hyd.c_s_mol_m3, rel=1e-12)
    # DRH floor is the saturated-brine composition on the same H₀ basis.
    f_b = saturation_brine_salt_fraction(config.salt_name)
    expected = (
        hyd.c_s_mol_m3 * hyd.formula_weight_g_mol / 1000.0
        * (1.0 - f_b) / f_b / WATER_MOLAR_MASS_KG_MOL
    )
    assert drh.c_w_min_mol_m3 == pytest.approx(expected, rel=1e-12)
    assert drh.c_w_min_mol_m3 > hyd.c_w_min_mol_m3 < hyd.c_w_max_mol_m3
    assert drh.c_w_min_mol_m3 < drh.c_w_max_mol_m3
    # Only the floor changed, so the wetter floor can only cost yield.
    y_hyd, _, _, _ = run_daily_cycle(baseline_profile(), config)
    y_drh, _, _, _ = run_daily_cycle(
        baseline_profile(), dataclasses.replace(config, c_w_floor_mode="drh")
    )
    assert y_drh < y_hyd


def test_instant_equilibrium_sits_on_the_isotherm(config: SystemConfig):
    """g → ∞: the driving force (C_R − a_w) is ~0 all cycle, and yield goes up."""
    import dataclasses

    from solar_lumped.physics import _mass_transfer_driving_force

    ideal = dataclasses.replace(config, instant_equilibrium=True)
    profile = baseline_profile()
    y_base, _, _, _ = run_daily_cycle(profile, config)
    y_ideal, _, abs_res, des_res = run_daily_cycle(profile, ideal)
    # Removing the only kinetic resistance in the model cannot cost water.
    assert y_ideal > y_base

    mass = ideal.mass_params()
    # Absorbing: a_w == RH. Skip points sitting on the dilution ceiling, where the
    # gel is saturated and the clamp -- not equilibrium -- sets the state.
    for k in range(1, len(abs_res.c_w)):
        c_w = float(abs_res.c_w[k])
        if c_w >= mass.c_w_max_mol_m3 * (1.0 - 1e-6):
            continue
        i = min(k, len(profile.absorption.relative_humidity) - 1)
        driving = _mass_transfer_driving_force(
            c_w,
            t_gel_c=float(abs_res.t_gel_c[k]),
            c_r=concentration_ratio_absorption(profile.absorption.relative_humidity[i]),
            params=mass,
            h_m=float(abs_res.H[k]),
            phase="absorption",
        )
        assert abs(driving) < 1e-3, f"absorption step {k}: driving={driving}"

    # Desorbing: a_w == C_R(T_gel, T_cond), wherever water is actually leaving.
    assert des_res.t_cond_c is not None
    for k in range(len(des_res.c_w)):
        if des_res.m_des_kg_s_m2[k] <= 0.0:
            continue
        driving = _mass_transfer_driving_force(
            float(des_res.c_w[k]),
            t_gel_c=float(des_res.t_gel_c[k]),
            c_r=concentration_ratio_desorption(
                float(des_res.t_gel_c[k]), float(des_res.t_cond_c[k])
            ),
            params=mass,
            h_m=float(des_res.H[k]),
            phase="desorption",
        )
        assert abs(driving) < 1e-3, f"desorption step {k}: driving={driving}"


def test_zero_solar_collects_no_water(config: SystemConfig):
    """Q_solar = 0 at constant T/RH: T_gel = T_amb = T_cond, so C_R -> 1 and the driving
    force turns positive -- the gel wants to *absorb*, and the desorption branch clips it.

    Yield must be exactly 0.0, not merely small. A dropped clip or a flipped sign
    anywhere in the desorption branch fabricates water out of a dark panel, and every
    other whole-model test runs at 600 W/m2 where that stays hidden inside a real yield.
    """
    import dataclasses

    y, eta, _, des_res = run_daily_cycle(baseline_profile(solar_w_m2=0.0), config)
    assert y == 0.0
    assert eta == 0.0
    assert float(np.max(des_res.m_des_kg_s_m2)) == 0.0
    # The ambient-condenser mode pins T_cond by a different route; same limit must hold.
    y_amb, _, _, _ = run_daily_cycle(
        baseline_profile(solar_w_m2=0.0),
        dataclasses.replace(config, condenser_tracks_ambient=True),
    )
    assert y_amb == 0.0


def test_yield_is_monotone_in_solar_and_closes_on_energy(config: SystemConfig):
    """More sun cannot collect less water, and the collected latent heat cannot exceed
    the solar energy that drove it. Nothing desorbs below ~100 W/m2, which is physical:
    T_gel has to clear T_cond before C_R drops under a_w."""
    runs = [
        run_daily_cycle(baseline_profile(solar_w_m2=q), config)[:2]
        for q in (0.0, 200.0, 400.0, 600.0, 800.0)
    ]
    ys = [y for y, _ in runs]
    assert all(b >= a for a, b in zip(ys, ys[1:])), ys
    assert ys[0] == 0.0 and ys[-1] > 0.0
    assert all(0.0 <= eta <= 1.0 for _, eta in runs), [eta for _, eta in runs]


def test_implicit_and_explicit_solvers_agree_off_the_hydrate_floor(
    config: SystemConfig, monkeypatch
):
    """The control behind _integrate_desorption's LSODA-not-Radau choice.

    That choice is justified by one specific defect: dc_w/dt steps discontinuously to
    exactly 0.0 at the hydrate floor, so a Jacobian across it is meaningless. The claim
    only holds up if Radau is fine on this RHS *otherwise* -- if implicit methods failed
    here generally, the floor would not be the explanation and the real cause would still
    be unfound.

    The baseline trajectory bottoms out around 24000 mol/m3 against a floor near 10400,
    so it never touches the discontinuity, and the two integrators must agree closely.
    Asserted at SIMPLE-path tightness rather than merely 'both finite': the point is that
    nothing about being implicit is disqualifying, so a loose bound would prove nothing.
    """
    baseline_floor = config.mass_params().c_w_min_mol_m3
    lsoda_yield, _e, _a, des = run_daily_cycle(baseline_profile(), config)
    assert float(np.min(des.c_w)) > baseline_floor * 1.5, (
        "this control is only meaningful while the baseline stays off the hydrate floor; "
        f"c_w reached {float(np.min(des.c_w)):.1f} vs floor {baseline_floor:.1f}"
    )

    real_solve_ivp = simulation.solve_ivp

    def as_radau(*args, **kwargs):
        if kwargs.get("method") == "LSODA":
            kwargs["method"] = "Radau"
        return real_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(simulation, "solve_ivp", as_radau)
    radau_yield, _e, _a, _d = run_daily_cycle(baseline_profile(), config)
    assert radau_yield == pytest.approx(lsoda_yield, rel=5e-4)


def test_absorption_water_balance_guard_catches_a_state_its_rhs_never_produced(
    config: SystemConfig, monkeypatch
):
    """The absorption half has no independent yield integral, so its guard compares the
    dc_w/dt trapezoid against the returned c_w. Pinned by corrupting the returned state:
    the failure mode it exists for (an implicit step throwing the state while reporting
    success=True) is exactly a solution the solver's own RHS never generated.

    Silent on every sound configuration -- that direction is covered by the rest of this
    module running absorption on each of them.
    """
    real_solve_ivp = simulation.solve_ivp

    def corrupt(*args, **kwargs):
        sol = real_solve_ivp(*args, **kwargs)
        sol.y[0, -1] += 5e4
        return sol

    monkeypatch.setattr(simulation, "solve_ivp", corrupt)
    with pytest.raises(RuntimeError, match="Absorption integration did not conserve water"):
        simulation._integrate_absorption(
            simulation.initial_loading(config),
            config.hydrogel_thickness_m,
            baseline_profile().absorption,
            config,
        )


def test_zero_humidity_integrates_once_thickness_tracks_water(config: SystemConfig):
    """RH = 0 used to trip the water-balance guard: with H pinned at H₀ the gel drove
    c_w into its hydrate floor discontinuously and the yield trapezoid closed to ~1.2%
    against the inventory drop. Now that thickness bottoms out with water content
    instead, the floor is reached smoothly and the balance closes.

    The guard itself is unchanged and still covered by the corrupted-solution test
    above; this only asserts that RH = 0 is no longer the case that trips it.
    """
    yields = [
        run_daily_cycle(baseline_profile(relative_humidity=rh), config)[0]
        for rh in (0.0, 0.02, 0.2)
    ]
    assert all(y >= 0.0 for y in yields)
    assert yields[0] < yields[1] < yields[2]  # drier air, less water, no inversion


def test_vapor_gap_uses_heated_from_above_iso15099():
    """The gel is the UPPER gap surface, so the cavity is stably stratified and must
    use ISO 15099 sec. 5.3.3.5, not Hollands (a heated-from-BELOW correlation)."""
    import math

    from solar_lumped.physics import (
        CAVITY_HEIGHT_M,
        hollands_nu,
        humid_air_props,
        iso15099_nu_vertical,
        rayleigh_vapor_gap,
        vapor_gap_h_conv_w_m2_k,
        vapor_gap_water_mole_fraction,
    )

    def _nu_from_h_conv(gap: float, t_gel: float, t_cond: float, tilt: float) -> float:
        """Back out Nu from h_conv = Nu·k_air/L. k_air is no longer a constant, so it has
        to be re-evaluated at the same film temperature and composition the gap used."""
        x_v = vapor_gap_water_mole_fraction(t_cond)
        k_air = humid_air_props(0.5 * (t_gel + t_cond), x_v=x_v)[0]
        return vapor_gap_h_conv_w_m2_k(gap, t_gel, t_cond, tilt_deg=tilt) * gap / k_air

    gap, t_gel, t_cond, tilt = 0.04, 76.0, 28.8, 30.0
    ra = rayleigh_vapor_gap(gap, t_gel, t_cond, x_v=vapor_gap_water_mole_fraction(t_cond))

    # Eq. 53: Nu = 1 + [Nu_v - 1] sin(theta), theta = 180 - tilt.
    nu_v = iso15099_nu_vertical(ra, CAVITY_HEIGHT_M / gap)
    expect = 1.0 + (nu_v - 1.0) * math.sin(math.radians(180.0 - tilt))
    got = _nu_from_h_conv(gap, t_gel, t_cond, tilt)
    assert got == pytest.approx(expect, rel=1e-12)

    # Strictly below Hollands, which would over-predict convection on a stable layer.
    assert got < hollands_nu(ra, tilt_deg=tilt)
    assert 1.0 <= got

    # Inverted case (condenser hotter) IS heated from below -> Hollands applies.
    inv = _nu_from_h_conv(gap, 20.0, 40.0, tilt)
    ra_inv = rayleigh_vapor_gap(gap, 20.0, 40.0, x_v=vapor_gap_water_mole_fraction(40.0))
    assert inv == pytest.approx(hollands_nu(ra_inv, tilt_deg=tilt), rel=1e-12)


def test_conde_isotherm_cap_not_exceeded_by_temperature_bucketing():
    """A hot gel must not silently stop desorbing: the blend path buckets temperature,
    and rounding past the isotherm cap used to empty the table -> NaN -> a 0.0 driving
    force that reads as 'no desorption' rather than an error."""
    from solar_lumped.complex_model import zsr_water_activity_at_brine_fraction
    from solar_lumped.physics import isotherm_t_max_c

    for weights in ((1.0, 0.0, 0.0), (0.5, 0.5, 0.0)):
        for t in (99.0, 100.5, 149.0, 155.0, 155.5, 156.0, 200.0):
            aw = zsr_water_activity_at_brine_fraction(weights, 0.284, t)
            assert math.isfinite(aw), f"NaN water activity at {t} C for {weights}"

    # Pure LiCl gets its own 155.5 C ceiling; adding CaCl2 pulls the blend back to 100 C,
    # because a mixture is evaluated at one temperature and CaCl2's isotherm stops there.
    hot = [zsr_water_activity_at_brine_fraction((1.0, 0.0, 0.0), 0.284, t) for t in (100.0, 155.0)]
    assert hot[1] > hot[0], f"pure-LiCl blend still clamped at 100 C: {hot}"
    mixed = [zsr_water_activity_at_brine_fraction((0.5, 0.5, 0.0), 0.284, t) for t in (100.0, 155.0)]
    assert mixed[1] == pytest.approx(mixed[0]), "CaCl2 blend must still clamp at 100 C"
    assert isotherm_t_max_c("LiCl") > isotherm_t_max_c("CaCl2")


def test_note_s1_eq_s3_s5_corrected_forms():
    """Ra uses the exact Δρ buoyancy (Eq. S3), not the Boussinesq β·ΔT it was
    mis-transcribed as; and Sh = g_conv·(L_g−H)/D_air equals Nu (Eq. S5)."""
    from solar_lumped.physics import (
        GRAVITY_M_S2,
        d_h2o_air_m2_s,
        humid_air_props,
        vapor_gap_h_conv_w_m2_k,
        mass_transfer_g_from_h_conv_m_s,
        rayleigh_vapor_gap,
        vapor_gap_water_mole_fraction,
    )

    gap, t_gel, t_cond = 0.04, 76.0, 28.8
    t_film = 0.5 * (t_gel + t_cond)
    x_v = vapor_gap_water_mole_fraction(t_cond)
    k_air, nu_air, alpha_air = humid_air_props(t_film, x_v=x_v)

    # Eq. S3: ideal-gas Δρ/ρ_air at film temperature, reference density cancels. ν and α
    # are the film-temperature values too, so the whole group is at one temperature.
    t_h, t_c = t_gel + 273.15, t_cond + 273.15
    d_rho_over_rho = 0.5 * (t_h + t_c) * (t_h - t_c) / (t_h * t_c)
    expected = GRAVITY_M_S2 * d_rho_over_rho * gap**3 / (nu_air * alpha_air)
    assert rayleigh_vapor_gap(gap, t_gel, t_cond, x_v=x_v) == pytest.approx(expected, rel=1e-12)

    # The retired β·ΔT form (β = 1/300 K) overstates this peak-ΔT Ra by ~7%.
    t_ref = _pv("Air reference temperature for thermal expansion")
    boussinesq = GRAVITY_M_S2 * (t_h - t_c) / t_ref * gap**3 / (nu_air * alpha_air)
    assert boussinesq / expected == pytest.approx(1.07, abs=0.01)

    # Eq. S5: Sh == Nu under the Le ≈ 1 analogy. k_air cancels between h_conv and g, so
    # this identity is what pins the two to the SAME film temperature -- evaluate D_air or
    # k_air anywhere else and it breaks.
    h_conv = vapor_gap_h_conv_w_m2_k(gap, t_gel, t_cond, tilt_deg=30.0)
    g = mass_transfer_g_from_h_conv_m_s(h_conv, t_gel_c=t_gel, t_cond_c=t_cond)
    sh = g * gap / d_h2o_air_m2_s(t_film)
    assert sh == pytest.approx(h_conv * gap / k_air, rel=1e-12)


def test_isosteric_h_des_reproduces_diaz_marin():
    """h_des from Clausius-Clapeyron on Conde's a_w(xi,T), checked against Díaz-Marín:
    2800-2900 kJ/kg for PAM-LiCl at saturation, converging to h_fg when dilute."""
    from solar_lumped.physics import (
        H_FG_J_PER_KG,
        ISOSTERIC_H_DES_SALTS,
        isosteric_h_des_j_per_kg,
        saturation_brine_salt_fraction,
    )

    xi_sat = saturation_brine_salt_fraction("LiCl", 25.0)
    h_sat = isosteric_h_des_j_per_kg("LiCl", xi_sat, 25.0)
    assert 2.80e6 < h_sat < 2.90e6, h_sat

    # Dilute brine has almost no binding term left, so h_sorp -> h_fg (their Eq. 7).
    h_dilute = isosteric_h_des_j_per_kg("LiCl", 0.02, 25.0)
    assert h_dilute == pytest.approx(H_FG_J_PER_KG, rel=0.10)
    assert h_dilute < h_sat

    # Monotone: drier gel binds water harder.
    hs = [isosteric_h_des_j_per_kg("LiCl", x, 60.0) for x in (0.10, 0.25, 0.40)]
    assert hs[0] < hs[1] < hs[2]

    # LiCl and LiBr carry the Zeng & Zhou BET past 100 C (see test_bet_isotherm);
    # CaCl2 has only Conde, and out-of-domain must be NaN (never a number) so callers
    # fall back instead of poisoning the energy balance.
    assert ISOSTERIC_H_DES_SALTS == frozenset({"LiCl", "LiBr", "CaCl2"})
    assert math.isfinite(isosteric_h_des_j_per_kg("CaCl2", 0.30, 80.0))
    # Unsupported salts opt out rather than returning a bogus h_fg.
    assert math.isnan(isosteric_h_des_j_per_kg("MgCl2", 0.30, 60.0))
    assert math.isnan(isosteric_h_des_j_per_kg("NaCl", 0.20, 60.0))


def test_isosteric_h_des_gated_and_falls_back():
    """The mode must be off for blends (ZSR gives a mixture a_w with no per-salt slope)
    and must fall back to the tabulated constant rather than propagate NaN."""
    from solar_lumped.complex_model import ComplexOptions
    from solar_lumped.physics import H_DES_J_PER_KG, effective_h_des_j_per_kg
    from solar_lumped.simulation import SystemConfig

    assert SystemConfig.baseline().thermal_params().h_des_salt_name == "LiCl"
    # Table S3 is a recreation target and keeps COMSOL's 2320 kJ/kg.
    s3 = SystemConfig.comsol_table_s3().thermal_params()
    assert s3.h_des_salt_name is None and s3.h_des_j_per_kg == H_DES_J_PER_KG
    # Complex mode routes a_w through ZSR, so the per-salt slope is not defined.
    assert (
        SystemConfig.baseline(complex=ComplexOptions()).thermal_params().h_des_salt_name
        is None
    )

    p = SystemConfig.baseline().thermal_params()
    assert effective_h_des_j_per_kg(p, 70.0, None) == p.h_des_j_per_kg
    # 900 C is far past Conde's cap; the resolver must return the constant, not NaN.
    assert math.isfinite(effective_h_des_j_per_kg(p, 900.0, 0.30))
