"""COMSOL lumped 0D cycle integrator (6-state model, BDF)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from waste_heat_lumped.physics import comsol_lumped as cl
from waste_heat_lumped.physics.comsol_balances import (
    cond_heating_w_m2,
    gel_heating_w_m2,
    solve_glass_absorber_temps,
)
from waste_heat_lumped.physics.device_balances import comsol_optics
from waste_heat_lumped.physics.salt_properties import clamp_temperature_c
from waste_heat_lumped.physics.sorbent import clip_loading
from waste_heat_lumped.simulation.device_config import DeviceConfig
from waste_heat_lumped.simulation.ode_system import PhaseResult, _profile_index
from waste_heat_lumped.weather.profiles import DailyWeatherProfile, PhaseProfile

_ODE_RTOL = 1e-4
_ODE_ATOL = 1e-7


def _comsol_params(config: DeviceConfig) -> dict:
    h0 = config.hydrogel_thickness_m
    return {
        "h0_m": h0,
        "vapor_gap_m": config.vapor_gap_m,
        "h_cond": config.comsol_h_cond_w_m2_k(),
        "tint_c": config.comsol_tint_c(),
        "rh_high": config.comsol_rh_high(),
        "gel_mass": cl.gel_thermal_mass_j_m2_k(h0),
        "cond_mass": config.condenser_thermal_mass_j_m2_k(),
        "h_fg": config.h_fg_j_per_kg,
    }


def _integrate_comsol_phase(
    c_w0: float,
    h0: float,
    t_gel0: float,
    t_cond0: float,
    profile: PhaseProfile,
    config: DeviceConfig,
    *,
    phase: str,
    collect_water: bool,
) -> PhaseResult:
    p = _comsol_params(config)
    optics = comsol_optics(config.thermal_params())
    h_min = config.hydrogel_thickness_m
    h_max = max(config.vapor_gap_m - 1e-4, h_min + 1e-6)
    n = len(profile.temperature_c)
    dt = profile.dt_s
    t_span = (0.0, dt * n)
    t_eval = np.linspace(0.0, t_span[1], n + 1)

    glass_guess = (t_gel0, p["tint_c"])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal glass_guess
        i = _profile_index(t, dt, n)
        c_w = float(y[0])
        h_m = max(float(y[1]), h_min)
        t_gel = clamp_temperature_c(float(y[2]))
        t_cond = clamp_temperature_c(float(y[3]))

        q_solar = max(0.0, profile.solar_w_m2[i])
        h_front = profile.h_amb_w_m2_k[i]
        solar_on = phase == "desorption" and q_solar > 0.0

        t_ads, t_glass = solve_glass_absorber_temps(
            t_gel_c=t_gel,
            q_solar_w_m2=q_solar,
            solar_on=solar_on,
            h_front=h_front,
            tint_c=p["tint_c"],
            optics=optics,
            t_guess=glass_guess,
        )
        glass_guess = (t_ads, t_glass)

        dc, dh, _m = cl.mass_rates(
            c_w,
            h_m,
            t_gel,
            t_cond,
            phase=phase if phase != "cooling" else "absorption",
            h0_m=p["h0_m"],
            vapor_gap_m=p["vapor_gap_m"],
            rh_high=p["rh_high"],
        )
        if phase == "desorption":
            dc = min(0.0, dc)
            dh = min(0.0, dh)
        elif phase == "absorption":
            dh = dh if h_m > h_min + 1e-12 else max(0.0, dh)
            if h_m >= h_max and dh > 0.0:
                dh = 0.0
        else:
            dc = 0.0
            dh = 0.0

        qdes = dc * p["h0_m"] * (cl.MW_W_G_MOL / 1000.0) * p["h_fg"]
        gap = max(p["vapor_gap_m"] - h_m, 0.0)
        if phase == "desorption" and gap > 0.0:
            gm1 = cl.comsol_g_conv_m_s(gap, t_gel, t_cond)
            cdif = cl.desorption_flux_mol_m3_s(
                c_w, h_m, t_gel, t_cond, h0_m=p["h0_m"]
            )
            qcond = cdif * gm1 * (cl.MW_W_G_MOL / 1000.0) * p["h_fg"]
        else:
            qcond = 0.0

        gel_heat = gel_heating_w_m2(
            t_gel_c=t_gel,
            t_cond_c=t_cond,
            t_ads_c=t_ads,
            h_m=h_m,
            vapor_gap_m=p["vapor_gap_m"],
            qdes_w_m2=qdes,
            phase=phase,
            h_cond=p["h_cond"],
            tint_c=p["tint_c"],
        )
        cond_heat = cond_heating_w_m2(
            t_gel_c=t_gel,
            t_cond_c=t_cond,
            h_m=h_m,
            vapor_gap_m=p["vapor_gap_m"],
            qcond_w_m2=qcond,
            phase=phase,
            h_cond=p["h_cond"],
            tint_c=p["tint_c"],
        )

        dT_gel = -gel_heat / max(p["gel_mass"], 1.0)
        dT_cond = -cond_heat / max(p["cond_mass"], 1.0)
        return np.array([dc, dh, dT_gel, dT_cond])

    sol = solve_ivp(
        rhs,
        t_span,
        y0=np.array([c_w0, max(h0, h_min), t_gel0, t_cond0]),
        method="BDF",
        t_eval=t_eval,
        max_step=dt,
        rtol=_ODE_RTOL,
        atol=_ODE_ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"COMSOL {phase} integration failed: {sol.message}")

    c_w_hist = np.array([clip_loading(float(v), config=config) for v in sol.y[0]])
    h_hist = np.clip(sol.y[1], h_min, h_max)
    t_gel_hist = sol.y[2]
    t_cond_hist = sol.y[3]
    t_abs_hist: list[float] = []
    t_glass_hist: list[float] = []
    m_des_hist: list[float] = []
    guess = glass_guess
    for k in range(len(sol.t)):
        i = _profile_index(float(sol.t[k]), dt, n)
        t_gel = float(t_gel_hist[k])
        t_cond = float(t_cond_hist[k])
        q_solar = max(0.0, profile.solar_w_m2[i])
        solar_on = phase == "desorption" and q_solar > 0.0
        t_ads, t_glass = solve_glass_absorber_temps(
            t_gel_c=t_gel,
            q_solar_w_m2=q_solar,
            solar_on=solar_on,
            h_front=profile.h_amb_w_m2_k[i],
            tint_c=p["tint_c"],
            optics=optics,
            t_guess=guess,
        )
        guess = (t_ads, t_glass)
        t_abs_hist.append(t_ads)
        t_glass_hist.append(t_glass)
        _, _, m_des = cl.mass_rates(
            float(c_w_hist[k]),
            float(h_hist[k]),
            t_gel,
            t_cond,
            phase=phase if phase != "cooling" else "absorption",
            h0_m=p["h0_m"],
            vapor_gap_m=p["vapor_gap_m"],
            rh_high=p["rh_high"],
        )
        m_des_hist.append(m_des if phase == "desorption" else 0.0)

    water = 0.0
    if collect_water:
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
        m_des_kg_s_m2=np.array(m_des_hist),
        t_abs_c=np.array(t_abs_hist),
        t_glass_c=np.array(t_glass_hist),
    )


def run_comsol_daily_cycle(
    profile: DailyWeatherProfile,
    config: DeviceConfig,
    *,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
) -> tuple[float, float, PhaseResult, PhaseResult]:
    """COMSOL lumped model: 12 h absorption + desorption (+ optional cooling)."""
    h0 = config.hydrogel_thickness_m
    tint = config.comsol_tint_c()
    if c_w_initial is None:
        cw0 = cl.CW0_MOL_M3
    else:
        cw0 = c_w_initial
    h_init = h0 if h_initial is None else h_initial

    abs_res = _integrate_comsol_phase(
        cw0,
        h_init,
        tint,
        tint - 0.1,
        profile.absorption,
        config,
        phase="absorption",
        collect_water=False,
    )

    des_res = _integrate_comsol_phase(
        float(abs_res.c_w[-1]),
        float(abs_res.H[-1]),
        float(abs_res.t_gel_c[-1]),
        float(abs_res.t_cond_c[-1]),
        profile.desorption,
        config,
        phase="desorption",
        collect_water=True,
    )

    if profile.cooling is not None:
        _integrate_comsol_phase(
            float(des_res.c_w[-1]),
            float(des_res.H[-1]),
            float(des_res.t_gel_c[-1]),
            float(des_res.t_cond_c[-1]),
            profile.cooling,
            config,
            phase="cooling",
            collect_water=False,
        )

    yield_kg = des_res.water_collected_kg_m2
    q_solar_int = sum(
        profile.desorption.solar_w_m2[i] * profile.desorption.dt_s
        for i in range(len(profile.desorption.solar_w_m2))
    )
    eta = (yield_kg * config.h_fg_j_per_kg / q_solar_int) if q_solar_int > 0 else 0.0
    return yield_kg, eta, abs_res, des_res
