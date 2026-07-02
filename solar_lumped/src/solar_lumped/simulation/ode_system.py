"""SciPy Radau integration for Wilson half-cycles (coupled Eqs. 1–6 + Eq. 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from solar_lumped.physics.mass_transfer import C_W_MAX_MOL_M3, C_W_MIN_MOL_M3
from solar_lumped.physics.salt_properties import (
    clamp_temperature_c,
    fabrication_c_w_initial,
)
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
            t_guess=t_guess,
        )
        dh = rates.dH_dt if h_m > h_min + 1e-12 else max(0.0, rates.dH_dt)
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
            t_guess=guess,
        )
        guess = (
            rates.thermal.t_gel_c,
            rates.thermal.t_abs_c,
            rates.thermal.t_glass_c,
        )
        t_gel_hist.append(rates.t_gel_c)

    c_w_out = np.clip(sol.y[0], C_W_MIN_MOL_M3, C_W_MAX_MOL_M3)
    h_out = np.maximum(sol.y[1], h_min)
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
) -> PhaseResult:
    mass = config.mass_params()
    thermal = config.thermal_params()
    tmass = config.condenser_thermal_mass_j_m2_k()
    n = len(profile.temperature_c)
    dt = profile.dt_s
    t_span = (0.0, dt * n)
    t_eval = np.linspace(0.0, t_span[1], n + 1)
    h_min = config.hydrogel_thickness_m
    t_cond0 = clamp_temperature_c(profile.temperature_c[0])
    t_guess: tuple[float, float, float] | None = None

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
            t_guess=t_guess,
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
    t_cond_hist: list[float] = []
    m_des_hist: list[float] = []
    guess: tuple[float, float, float] | None = None
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
            t_guess=guess,
        )
        guess = (
            rates.thermal.t_gel_c,
            rates.thermal.t_abs_c,
            rates.thermal.t_glass_c,
        )
        t_gel_hist.append(rates.t_gel_c)
        t_cond_hist.append(float(sol.y[2, k]))
        m_des_hist.append(rates.m_des_kg_s_m2)

    water = 0.0
    for k in range(len(sol.t) - 1):
        dt_step = float(sol.t[k + 1] - sol.t[k])
        water += 0.5 * (m_des_hist[k] + m_des_hist[k + 1]) * dt_step

    c_w_out = np.clip(sol.y[0], C_W_MIN_MOL_M3, C_W_MAX_MOL_M3)
    h_out = np.maximum(sol.y[1], h_min)
    return PhaseResult(
        time_s=sol.t,
        c_w=c_w_out,
        H=h_out,
        t_cond_c=np.array(t_cond_hist),
        t_gel_c=np.array(t_gel_hist),
        water_collected_kg_m2=max(0.0, water),
        m_des_kg_s_m2=np.array(m_des_hist),
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
        c_w_initial = fabrication_c_w_initial(
            salt_name=config.salt_name,
            salt_to_polymer_ratio=config.salt_to_polymer_ratio,
            hydrogel_thickness_m=h0,
            hydrogel_density_kg_m3=config.hydrogel_density_kg_m3,
        )

    abs_res = _integrate_absorption(c_w_initial, h_initial, profile.absorption, config)
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
