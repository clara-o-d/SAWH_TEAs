"""SciPy Radau integration for Wilson half-cycles (coupled Eqs. 1–6 + Eq. 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from solar_lumped.physics import table_s3
from solar_lumped.physics.correlations import (
    STEFAN_BOLTZMANN_W_M2_K4,
    condenser_h_conv_w_m2_k,
    hollands_vapor_gap_h_conv_w_m2_k,
    parallel_plate_emissivity,
    radiative_exchange_w_m2,
)
from solar_lumped.physics.device_balances import _residuals as _thermal_residuals
from solar_lumped.physics.mass_transfer import (
    C_W_MAX_MOL_M3,
    C_W_MIN_MOL_M3,
    concentration_ratio_desorption,
    dH_dt,
    dc_w_dt,
    m_des_kg_s_m2_from_dc_w,
)
from solar_lumped.physics.salt_properties import clamp_temperature_c
from solar_lumped.physics.sorbent import clip_loading, evaluate_mass_rates, initial_loading
from solar_lumped.simulation.coupled_dynamics import evaluate_coupled_rates
from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.weather.profiles import DailyWeatherProfile, PhaseProfile

# c_w is O(1e4) mol/m³; atol=1e-9 forces steps below float64 spacing in stiff desorption.
_ODE_RTOL = 1e-4
_ODE_ATOL = 1e-7


@dataclass
class PhaseResult:
    time_s: np.ndarray
    c_w: np.ndarray
    H: np.ndarray
    t_cond_c: np.ndarray | None
    t_gel_c: np.ndarray
    water_collected_kg_m2: float
    m_des_kg_s_m2: np.ndarray
    # Surface temperatures along the trajectory. Populated by transient solvers and
    # by quasi_steady (algebraic Eqs 1/3/4 each step, with k=0 pinned when ICs set).
    t_abs_c: np.ndarray | None = None
    t_glass_c: np.ndarray | None = None


def _profile_index(t: float, dt_s: float, n: int) -> int:
    return min(int(t / dt_s), n - 1)


def _integrate_absorption(
    c_w0: float,
    h0: float,
    profile: PhaseProfile,
    config: DeviceConfig,
) -> PhaseResult:
    mass = config.mass_params()
    thermal = config.thermal_params()
    n = len(profile.temperature_c)
    dt = profile.dt_s
    t_span = (0.0, dt * n)
    t_eval = np.linspace(0.0, t_span[1], n + 1)
    h_min = config.hydrogel_thickness_m
    # Gel cannot swell into the condenser; keep ≥7 mm effective gap (Wilson §2.2).
    h_max = max(
        config.vapor_gap_m - table_s3.VAPOR_GAP_TRANSPORT_MIN_M,
        h_min + 1e-6,
    )
    t_guess: tuple[float, float, float] | None = None

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal t_guess
        i = _profile_index(t, dt, n)
        h_m = max(float(y[1]), h_min)
        rates = evaluate_coupled_rates(
            c_w=float(y[0]),
            h_m=h_m,
            t_cond_c=profile.temperature_c[i],
            t_amb_c=profile.temperature_c[i],
            rh=profile.relative_humidity[i],
            q_solar_w_m2=0.0,
            h_amb=profile.h_amb_w_m2_k[i],
            phase="absorption",
            mass=mass,
            thermal=thermal,
            vapor_gap_m=config.vapor_gap_m,
            condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
            fin_area_ratio=config.fin_area_ratio,
            h_fg_j_per_kg=config.h_fg_j_per_kg,
            config=config,
            t_guess=t_guess,
        )
        dh = rates.dH_dt if h_m > h_min + 1e-12 else max(0.0, rates.dH_dt)
        if h_m >= h_max and dh > 0.0:
            dh = 0.0
        t_guess = (
            rates.thermal.t_gel_c,
            rates.thermal.t_abs_c,
            rates.thermal.t_glass_c,
        )
        return np.array([rates.dc_w_dt, dh])

    sol = solve_ivp(
        rhs,
        t_span,
        y0=np.array([c_w0, max(h0, h_min)]),
        method="Radau",
        t_eval=t_eval,
        max_step=dt,
        rtol=_ODE_RTOL,
        atol=_ODE_ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"Absorption integration failed: {sol.message}")

    t_gel_hist: list[float] = []
    guess: tuple[float, float, float] | None = None
    for k in range(len(sol.t)):
        i = _profile_index(float(sol.t[k]), dt, n)
        rates = evaluate_coupled_rates(
            c_w=float(sol.y[0, k]),
            h_m=max(float(sol.y[1, k]), h_min),
            t_cond_c=profile.temperature_c[i],
            t_amb_c=profile.temperature_c[i],
            rh=profile.relative_humidity[i],
            q_solar_w_m2=0.0,
            h_amb=profile.h_amb_w_m2_k[i],
            phase="absorption",
            mass=mass,
            thermal=thermal,
            vapor_gap_m=config.vapor_gap_m,
            condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
            fin_area_ratio=config.fin_area_ratio,
            h_fg_j_per_kg=config.h_fg_j_per_kg,
            config=config,
            t_guess=guess,
        )
        guess = (
            rates.thermal.t_gel_c,
            rates.thermal.t_abs_c,
            rates.thermal.t_glass_c,
        )
        t_gel_hist.append(rates.t_gel_c)

    c_w_out = np.array([clip_loading(float(v), config=config) for v in sol.y[0]])
    h_out = np.clip(sol.y[1], h_min, h_max)
    return PhaseResult(
        time_s=sol.t,
        c_w=c_w_out,
        H=h_out,
        t_cond_c=None,
        t_gel_c=np.array(t_gel_hist),
        water_collected_kg_m2=0.0,
        m_des_kg_s_m2=np.zeros(len(sol.t)),
    )


def _integrate_desorption(
    c_w0: float,
    h0: float,
    profile: PhaseProfile,
    config: DeviceConfig,
    *,
    t_guess0: tuple[float, float, float] | None = None,
) -> PhaseResult:
    mass = config.mass_params()
    thermal = config.thermal_params()
    tmass = config.condenser_thermal_mass_j_m2_k()
    n = len(profile.temperature_c)
    dt = profile.dt_s
    t_span = (0.0, dt * n)
    t_eval = np.linspace(0.0, t_span[1], n + 1)
    h_min = config.hydrogel_thickness_m
    surface_ic = config.desorption_surface_ic_c()
    if surface_ic is not None:
        t_gel_ic, t_abs_ic, t_glass_ic, t_cond_ic = (
            clamp_temperature_c(t) for t in surface_ic
        )
        t_cond0 = t_cond_ic
        t_guess0 = (t_gel_ic, t_abs_ic, t_glass_ic)
    else:
        t_cond0 = clamp_temperature_c(profile.temperature_c[0])
        if t_guess0 is None:
            t_amb = t_cond0
            t_guess0 = (t_amb, t_amb, t_amb)
    t_guess: tuple[float, float, float] | None = t_guess0

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal t_guess
        i = _profile_index(t, dt, n)
        h_m = max(float(y[1]), h_min)
        rates = evaluate_coupled_rates(
            c_w=float(y[0]),
            h_m=h_m,
            t_cond_c=float(y[2]),
            t_amb_c=profile.temperature_c[i],
            rh=profile.relative_humidity[i],
            q_solar_w_m2=profile.solar_w_m2[i],
            h_amb=profile.h_amb_w_m2_k[i],
            phase="desorption",
            mass=mass,
            thermal=thermal,
            vapor_gap_m=config.vapor_gap_m,
            condenser_thermal_mass_j_m2_k=tmass,
            fin_area_ratio=config.fin_area_ratio,
            h_fg_j_per_kg=config.h_fg_j_per_kg,
            config=config,
            t_guess=t_guess,
            h_amb_cond=(
                profile.h_amb_cond_w_m2_k[i]
                if profile.h_amb_cond_w_m2_k is not None
                else None
            ),
        )
        dh = rates.dH_dt if h_m > h_min + 1e-12 else 0.0
        dc = min(0.0, rates.dc_w_dt)
        dh = min(0.0, dh)
        t_guess = (
            rates.thermal.t_gel_c,
            rates.thermal.t_abs_c,
            rates.thermal.t_glass_c,
        )
        return np.array([dc, dh, rates.dT_cond_dt])

    sol = solve_ivp(
        rhs,
        t_span,
        y0=np.array([c_w0, max(h0, h_min), t_cond0]),
        method="Radau",
        t_eval=t_eval,
        max_step=dt,
        rtol=_ODE_RTOL,
        atol=_ODE_ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"Desorption integration failed: {sol.message}")

    t_gel_hist: list[float] = []
    t_abs_hist: list[float] = []
    t_glass_hist: list[float] = []
    t_cond_hist: list[float] = []
    m_des_hist: list[float] = []
    guess: tuple[float, float, float] | None = t_guess0
    for k in range(len(sol.t)):
        i = _profile_index(float(sol.t[k]), dt, n)
        rates = evaluate_coupled_rates(
            c_w=float(sol.y[0, k]),
            h_m=max(float(sol.y[1, k]), h_min),
            t_cond_c=float(sol.y[2, k]),
            t_amb_c=profile.temperature_c[i],
            rh=profile.relative_humidity[i],
            q_solar_w_m2=profile.solar_w_m2[i],
            h_amb=profile.h_amb_w_m2_k[i],
            phase="desorption",
            mass=mass,
            thermal=thermal,
            vapor_gap_m=config.vapor_gap_m,
            condenser_thermal_mass_j_m2_k=tmass,
            fin_area_ratio=config.fin_area_ratio,
            h_fg_j_per_kg=config.h_fg_j_per_kg,
            config=config,
            t_guess=guess,
            h_amb_cond=(
                profile.h_amb_cond_w_m2_k[i]
                if profile.h_amb_cond_w_m2_k is not None
                else None
            ),
        )
        guess = (
            rates.thermal.t_gel_c,
            rates.thermal.t_abs_c,
            rates.thermal.t_glass_c,
        )
        if k == 0 and surface_ic is not None:
            t_gel_hist.append(t_gel_ic)
            t_abs_hist.append(t_abs_ic)
            t_glass_hist.append(t_glass_ic)
        else:
            t_gel_hist.append(rates.t_gel_c)
            t_abs_hist.append(rates.thermal.t_abs_c)
            t_glass_hist.append(rates.thermal.t_glass_c)
        t_cond_hist.append(float(sol.y[2, k]))
        m_des_hist.append(rates.m_des_kg_s_m2)

    water = 0.0
    for k in range(len(sol.t) - 1):
        dt_step = float(sol.t[k + 1] - sol.t[k])
        water += 0.5 * (m_des_hist[k] + m_des_hist[k + 1]) * dt_step

    c_w_out = np.array([clip_loading(float(v), config=config) for v in sol.y[0]])
    h_out = np.maximum(sol.y[1], h_min)
    return PhaseResult(
        time_s=sol.t,
        c_w=c_w_out,
        H=h_out,
        t_cond_c=np.array(t_cond_hist),
        t_gel_c=np.array(t_gel_hist),
        water_collected_kg_m2=max(0.0, water),
        m_des_kg_s_m2=np.array(m_des_hist),
        t_abs_c=np.array(t_abs_hist),
        t_glass_c=np.array(t_glass_hist),
    )


def _h_rad_w_m2_k(t_hot_c: float, t_cold_c: float, emissivity: float) -> float:
    """Linearized radiative exchange coefficient σε(Tₕ²+T_c²)(Tₕ+T_c) [W/m²K]."""
    if emissivity <= 0.0:
        return 0.0
    th = t_hot_c + 273.15
    tc = t_cold_c + 273.15
    return emissivity * STEFAN_BOLTZMANN_W_M2_K4 * (th * th + tc * tc) * (th + tc)


def _integrate_desorption_segregated(
    c_w0: float,
    h0: float,
    profile: PhaseProfile,
    config: DeviceConfig,
) -> PhaseResult:
    """COMSOL-style segregated desorption integrator (Note S1).

    Absorber, glass, and gel carry finite lumped thermal capacitance and are
    updated sequentially (Gauss-Seidel) once per fixed time step using each
    other's latest values. Each surface update is backward-Euler in its own
    temperature (radiation linearized at lagged temps), so the scheme is
    unconditionally stable and relaxes to the same quasi-steady Eqs. 1/3/4
    solution at long times while warming gradually from ambient ICs.
    """
    mass = config.mass_params()
    thermal = config.thermal_params()
    tmass_cond = max(config.condenser_thermal_mass_j_m2_k(), 1.0)
    c_glass = table_s3.GLASS_THERMAL_MASS_J_M2_K
    c_abs = table_s3.ABSORBER_THERMAL_MASS_J_M2_K

    n = len(profile.temperature_c)
    dt = profile.dt_s
    h_min = config.hydrogel_thickness_m
    k_air = table_s3.K_AIR_W_M_K
    l_c = thermal.insulation_gap_m
    eps_gc = parallel_plate_emissivity(thermal.eps_gel, thermal.eps_al)
    h_des = thermal.h_des_j_per_kg
    h_fg = config.h_fg_j_per_kg

    # Cold-start ICs: gel/surfaces begin at the desorption-start ambient temp,
    # unless per-component temperatures (e.g. the first digitized data points) or a
    # single explicit initial temperature is configured.
    if config.coupled_initial_temps_c is not None:
        t_gel, t_abs, t_glass, t_cond = (
            clamp_temperature_c(t) for t in config.coupled_initial_temps_c
        )
    else:
        if config.segregated_initial_temp_c is not None:
            t_init = clamp_temperature_c(config.segregated_initial_temp_c)
        else:
            t_init = clamp_temperature_c(profile.temperature_c[0])
        t_gel = t_abs = t_glass = t_cond = t_init
    c_w = c_w0
    h_m = max(h0, h_min)
    m_des = 0.0

    time_s = np.arange(n + 1, dtype=float) * dt
    c_w_hist = np.empty(n + 1)
    h_hist = np.empty(n + 1)
    t_gel_hist = np.empty(n + 1)
    t_abs_hist = np.empty(n + 1)
    t_glass_hist = np.empty(n + 1)
    t_cond_hist = np.empty(n + 1)
    m_des_hist = np.empty(n + 1)

    def record(k: int) -> None:
        c_w_hist[k] = c_w
        h_hist[k] = h_m
        t_gel_hist[k] = t_gel
        t_abs_hist[k] = t_abs
        t_glass_hist[k] = t_glass
        t_cond_hist[k] = t_cond
        m_des_hist[k] = m_des

    record(0)

    for k in range(n):
        i = min(k, n - 1)
        t_amb = profile.temperature_c[i]
        q_solar = max(0.0, profile.solar_w_m2[i])
        h_amb = profile.h_amb_w_m2_k[i]
        h_amb_cond = (
            profile.h_amb_cond_w_m2_k[i]
            if profile.h_amb_cond_w_m2_k is not None
            else h_amb
        )
        u_gel = table_s3.u_gel_w_m2_k(h_m)
        gap_eff = max(config.vapor_gap_m - h_m, 0.0)
        c_gel = table_s3.gel_thermal_mass_j_m2_k(h_m)

        cond_ag = k_air / l_c
        a_a = c_abs / dt
        if thermal.has_glass:
            # 1) Glass (Eq. 3) — backward-Euler in T_glass, uses lagged T_abs.
            h_r_ag = _h_rad_w_m2_k(t_abs, t_glass, 1.0)
            h_r_ga = _h_rad_w_m2_k(t_glass, t_amb, 1.0)
            a_g = c_glass / dt
            t_glass = (
                a_g * t_glass
                + (cond_ag + h_r_ag) * t_abs
                + (h_amb + h_r_ga) * t_amb
            ) / (a_g + cond_ag + h_r_ag + h_amb + h_r_ga)

            # 2) Absorber (Eq. 4) — backward-Euler in T_abs, uses updated T_glass.
            h_r_ag = _h_rad_w_m2_k(t_abs, t_glass, 1.0)
            q_abs_in = thermal.eps_abs * thermal.tau_glass * q_solar
            t_abs = (
                a_a * t_abs
                + q_abs_in
                + (cond_ag + h_r_ag) * t_glass
                + u_gel * t_gel
            ) / (a_a + cond_ag + h_r_ag + u_gel)
        else:
            # No cover: absorber exchanges directly with ambient (Eq. 4 no-glass
            # form in _residuals); solar not attenuated by τ_glass, glass ≡ ambient.
            t_glass = t_amb
            h_r_aa = _h_rad_w_m2_k(t_abs, t_amb, thermal.eps_abs)
            q_abs_in = thermal.eps_abs * q_solar
            t_abs = (
                a_a * t_abs
                + q_abs_in
                + (h_amb + h_r_aa) * t_amb
                + u_gel * t_gel
            ) / (a_a + h_amb + h_r_aa + u_gel)

        # 3) Gel (Eq. 1) — backward-Euler in T_gel, uses updated T_abs; desorption
        #    heat sink held at the previous step's ṁ_des (segregated lag).
        h_conv_g = (
            hollands_vapor_gap_h_conv_w_m2_k(
                gap_eff, t_gel, t_cond, tilt_deg=thermal.tilt_deg
            )
            if gap_eff > 0.0
            else 0.0
        )
        h_r_gc = _h_rad_w_m2_k(t_gel, t_cond, eps_gc)
        a_gel = c_gel / dt
        q_des = m_des * h_des
        t_gel = (
            a_gel * t_gel
            + u_gel * t_abs
            + (h_conv_g + h_r_gc) * t_cond
            - q_des
        ) / (a_gel + u_gel + h_conv_g + h_r_gc)

        # 4) Mass transfer (Eqs. 5/6) at the updated gel temperature.
        dc, dh, m_des = evaluate_mass_rates(
            loading=c_w,
            h_m=h_m,
            t_gel_c=t_gel,
            t_cond_c=t_cond,
            rh=profile.relative_humidity[i],
            phase="desorption",
            mass=mass,
            config=config,
            vapor_gap_m=config.vapor_gap_m,
        )
        dc = min(0.0, dc)
        dh = min(0.0, dh)
        if h_m <= h_min + 1e-12:
            dh = 0.0
        c_w = clip_loading(c_w + dt * dc, config=config)
        h_m = max(h_m + dt * dh, h_min)

        # 5) Condenser (Eq. 2) — backward-Euler in T_cond.
        h_conv_cond = condenser_h_conv_w_m2_k(
            h_amb_cond, fin_area_ratio=config.fin_area_ratio
        )
        a_c = tmass_cond / dt
        q_rad_gc = h_r_gc * (t_gel - t_cond)
        t_cond = (
            a_c * t_cond
            + h_conv_g * t_gel
            + h_conv_cond * t_amb
            + m_des * h_fg
            + q_rad_gc
        ) / (a_c + h_conv_g + h_conv_cond)

        record(k + 1)

    water = 0.0
    for k in range(n):
        water += 0.5 * (m_des_hist[k] + m_des_hist[k + 1]) * dt

    return PhaseResult(
        time_s=time_s,
        c_w=c_w_hist,
        H=h_hist,
        t_cond_c=t_cond_hist,
        t_gel_c=t_gel_hist,
        water_collected_kg_m2=max(0.0, water),
        m_des_kg_s_m2=m_des_hist,
        t_abs_c=t_abs_hist,
        t_glass_c=t_glass_hist,
    )


def _integrate_desorption_coupled_ode(
    c_w0: float,
    h0: float,
    profile: PhaseProfile,
    config: DeviceConfig,
) -> PhaseResult:
    """Coupled BDF ODE desorption integrator (COMSOL v6.2 lumped ODE model).

    Wilson Methods: "our heat and mass transport equations were iteratively solved
    using a time-dependent segregated solver in COMSOL v6.2, with time steps of
    100 s." The model is a *lumped system of ODEs* (Bi ≪ 1), not spatial FEM.
    COMSOL's time integrator is implicit BDF (variable order 1–5); "segregated" is
    its per-step Newton solver, iterated to convergence, so the converged per-step
    solution equals a fully coupled solve. We integrate the coupled system with
    SciPy's variable-order BDF at a 100 s max step. Here the six lumped states

        y = [T_gel, T_abs, T_glass, T_cond, c_w, H]

    are advanced together as one stiff system with SciPy's variable-order BDF
    integrator (the closest analogue to COMSOL's scheme). The surface heat balances
    reuse Wilson's exact Eq. 1/3/4 residuals (net flux = C·dT/dt), each node carries
    finite *physical* thermal capacitance, and at long times dT/dt → 0 recovers the
    quasi-steady algebraic solution. The transient warm-up therefore follows the
    true relaxation time constants rather than a solver-stepping artifact.
    """
    mass = config.mass_params()
    thermal = config.thermal_params()
    n = len(profile.temperature_c)
    dt = profile.dt_s
    t_span = (0.0, dt * n)
    t_eval = np.linspace(0.0, t_span[1], n + 1)
    h_min = config.hydrogel_thickness_m

    c_glass = max(table_s3.GLASS_THERMAL_MASS_J_M2_K, 1.0)
    c_abs = max(table_s3.ABSORBER_THERMAL_MASS_J_M2_K, 1.0)
    c_cond = max(config.condenser_thermal_mass_j_m2_k(), 1.0)
    h_fg = config.h_fg_j_per_kg
    eps_gc = parallel_plate_emissivity(thermal.eps_gel, thermal.eps_al)

    if config.coupled_initial_temps_c is not None:
        t_gel0, t_abs0, t_glass0, t_cond0 = (
            clamp_temperature_c(t) for t in config.coupled_initial_temps_c
        )
    else:
        if config.segregated_initial_temp_c is not None:
            t_init = clamp_temperature_c(config.segregated_initial_temp_c)
        else:
            t_init = clamp_temperature_c(profile.temperature_c[0])
        t_gel0 = t_abs0 = t_glass0 = t_cond0 = t_init

    def _mass_rates(
        c_w: float, h_m: float, t_gel: float, t_cond: float
    ) -> tuple[float, float, float]:
        c_w = clip_loading(c_w, config=config)
        dc, dh, m_des = evaluate_mass_rates(
            loading=c_w,
            h_m=h_m,
            t_gel_c=t_gel,
            t_cond_c=t_cond,
            rh=0.0,
            phase="desorption",
            mass=mass,
            config=config,
            vapor_gap_m=config.vapor_gap_m,
        )
        dc = min(0.0, dc)
        dh = min(0.0, dh)
        if h_m <= h_min + 1e-12:
            dh = 0.0
        return dc, dh, m_des

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        t_gel, t_abs, t_glass, t_cond = (
            float(y[0]), float(y[1]), float(y[2]), float(y[3])
        )
        c_w = float(y[4])
        h_m = max(float(y[5]), h_min)
        i = _profile_index(t, dt, n)
        t_amb = profile.temperature_c[i]
        q_solar = max(0.0, profile.solar_w_m2[i])
        h_amb = profile.h_amb_w_m2_k[i]
        h_amb_cond = (
            profile.h_amb_cond_w_m2_k[i]
            if profile.h_amb_cond_w_m2_k is not None
            else h_amb
        )
        gap_eff = max(config.vapor_gap_m - h_m, 0.0)

        dc, dh, m_des = _mass_rates(c_w, h_m, t_gel, t_cond)

        # Net surface fluxes (W/m²) — identical to Wilson Eq. 1 (gel), Eq. 3 (glass),
        # Eq. 4 (absorber). C·dT/dt = net flux, so steady state == quasi-steady solve.
        r = _thermal_residuals(
            np.array([t_gel, t_abs, t_glass]),
            t_cond, t_amb, q_solar, m_des, h_amb, thermal, gap_eff, h_m,
        )
        c_gel = max(table_s3.gel_thermal_mass_j_m2_k(h_m), 1.0)
        dT_gel = float(r[0]) / c_gel
        dT_abs = float(r[2]) / c_abs
        if thermal.has_glass:
            dT_glass = float(r[1]) / c_glass
        else:
            # No cover: glass is not a real surface; relax it to ambient for plotting.
            dT_glass = h_amb * (t_amb - t_glass) / c_glass

        # Condenser transient (Eq. 2).
        h_conv_g = (
            hollands_vapor_gap_h_conv_w_m2_k(
                gap_eff, t_gel, t_cond, tilt_deg=thermal.tilt_deg
            )
            if gap_eff > 0.0
            else 0.0
        )
        h_conv_cond = condenser_h_conv_w_m2_k(
            h_amb_cond, fin_area_ratio=config.fin_area_ratio
        )
        q_rad_gc = radiative_exchange_w_m2(t_gel, t_cond, emissivity=eps_gc)
        dT_cond = (
            h_conv_g * (t_gel - t_cond)
            - h_conv_cond * (t_cond - t_amb)
            + m_des * h_fg
            + q_rad_gc
        ) / c_cond

        return np.array([dT_gel, dT_abs, dT_glass, dT_cond, dc, dh])

    sol = solve_ivp(
        rhs,
        t_span,
        y0=np.array([t_gel0, t_abs0, t_glass0, t_cond0, c_w0, max(h0, h_min)]),
        method="BDF",
        t_eval=t_eval,
        max_step=dt,
        rtol=_ODE_RTOL,
        atol=_ODE_ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"Coupled desorption integration failed: {sol.message}")

    t_gel_hist = sol.y[0]
    t_abs_hist = sol.y[1]
    t_glass_hist = sol.y[2]
    t_cond_hist = sol.y[3]
    c_w_hist = np.array([clip_loading(float(v), config=config) for v in sol.y[4]])
    h_hist = np.maximum(sol.y[5], h_min)

    m_des_hist = np.empty(len(sol.t))
    for k in range(len(sol.t)):
        _, _, m_des_hist[k] = _mass_rates(
            float(c_w_hist[k]),
            float(h_hist[k]),
            float(t_gel_hist[k]),
            float(t_cond_hist[k]),
        )

    water = 0.0
    for k in range(len(sol.t) - 1):
        dt_step = float(sol.t[k + 1] - sol.t[k])
        water += 0.5 * (m_des_hist[k] + m_des_hist[k + 1]) * dt_step

    return PhaseResult(
        time_s=sol.t,
        c_w=c_w_hist,
        H=h_hist,
        t_cond_c=t_cond_hist,
        t_gel_c=t_gel_hist,
        water_collected_kg_m2=max(0.0, water),
        m_des_kg_s_m2=m_des_hist,
        t_abs_c=t_abs_hist,
        t_glass_c=t_glass_hist,
    )


def cycle_end_state(des_res: PhaseResult) -> tuple[float, float]:
    """Gel state (c_w, H) at end of a full absorption–desorption cycle."""
    return float(des_res.c_w[-1]), float(des_res.H[-1])


def warmup_cycle_state(
    profile: DailyWeatherProfile,
    config: DeviceConfig,
    *,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
) -> tuple[float, float]:
    """Run one full cycle; return gel state after desorption (for next-day IC)."""
    _, _, _, des_res = run_daily_cycle(
        profile,
        config,
        c_w_initial=c_w_initial,
        h_initial=h_initial,
    )
    return cycle_end_state(des_res)


def warmup_to_cyclic_state(
    profile: DailyWeatherProfile,
    config: DeviceConfig,
    *,
    n_cycles: int = 2,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
) -> tuple[float, float]:
    """Run repeated daily cycles until a periodic post-desorption (c_w, H) is reached."""
    cw, h = c_w_initial, h_initial
    for _ in range(max(1, n_cycles)):
        _, _, _, des_res = run_daily_cycle(
            profile,
            config,
            c_w_initial=cw,
            h_initial=h,
        )
        cw, h = cycle_end_state(des_res)
    return cw, h


def run_daily_cycle(
    profile: DailyWeatherProfile,
    config: DeviceConfig,
    *,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
    cyclic_initial: bool = False,
    cyclic_warmup_cycles: int = 2,
) -> tuple[float, float, PhaseResult, PhaseResult]:
    """Run absorption then desorption; return (yield kg/m2, eta_thermal, abs_res, des_res).

    If ``cyclic_initial`` is True, run ``cyclic_warmup_cycles`` full days from the
    fabrication default first, then simulate one reporting day from that end state.
    """
    if cyclic_initial:
        cw, h = warmup_to_cyclic_state(
            profile,
            config,
            n_cycles=cyclic_warmup_cycles,
            c_w_initial=c_w_initial,
            h_initial=h_initial,
        )
        c_w_initial, h_initial = cw, h

    h0 = config.hydrogel_thickness_m
    if h_initial is None:
        h_initial = h0
    if c_w_initial is None:
        c_w_initial = initial_loading(config)

    abs_res = _integrate_absorption(c_w_initial, h_initial, profile.absorption, config)
    if config.desorption_solver == "coupled_bdf":
        des_res = _integrate_desorption_coupled_ode(
            float(abs_res.c_w[-1]),
            float(abs_res.H[-1]),
            profile.desorption,
            config,
        )
    elif config.desorption_solver == "segregated":
        des_res = _integrate_desorption_segregated(
            float(abs_res.c_w[-1]),
            float(abs_res.H[-1]),
            profile.desorption,
            config,
        )
    else:
        des_res = _integrate_desorption(
            float(abs_res.c_w[-1]),
            float(abs_res.H[-1]),
            profile.desorption,
            config,
        )
    yield_kg = max(0.0, des_res.water_collected_kg_m2)

    q_solar_int = sum(
        profile.desorption.solar_w_m2[i] * profile.desorption.dt_s
        for i in range(len(profile.desorption.solar_w_m2))
    )
    eta = (yield_kg * config.h_fg_j_per_kg / q_solar_int) if q_solar_int > 0 else 0.0
    return yield_kg, eta, abs_res, des_res
