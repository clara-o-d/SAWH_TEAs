"""Coupled Wilson Eqs. 1–6 + condenser transient (Eq. 2) rate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy.optimize import brentq

from solar_lumped.physics.correlations import (
    condenser_h_conv_w_m2_k,
    parallel_plate_emissivity,
    radiative_exchange_w_m2,
)
from solar_lumped.physics.device_balances import (
    DeviceThermalParams,
    ThermalState,
    solve_steady_thermal,
)
from solar_lumped.physics.mass_transfer import (
    MassTransferParams,
    concentration_ratio_absorption,
    concentration_ratio_desorption,
    dH_dt,
    dc_w_dt,
    m_des_kg_s_m2_from_state,
)
from solar_lumped.physics.salt_properties import clamp_temperature_c

CyclePhase = Literal["absorption", "desorption"]

_M_DES_BRACKET_MAX = 0.01  # kg/m²/s upper search bound for brentq bracket


@dataclass(frozen=True, slots=True)
class CoupledRates:
    dc_w_dt: float
    dH_dt: float
    dT_cond_dt: float
    t_gel_c: float
    m_des_kg_s_m2: float
    thermal: ThermalState


def _m_des_calc(
    m_des: float,
    *,
    c_w: float,
    h_m: float,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    h_amb: float,
    mass: MassTransferParams,
    thermal: DeviceThermalParams,
    vapor_gap_m: float,
    t_guess: tuple[float, float, float] | None,
) -> tuple[float, float, float, ThermalState]:
    state = solve_steady_thermal(
        t_cond_c=t_cond_c,
        t_amb_c=t_amb_c,
        q_solar_w_m2=q_solar_w_m2,
        m_des_kg_s_m2=max(0.0, m_des),
        h_amb=h_amb,
        params=thermal,
        t_guess=t_guess,
        vapor_gap_m=vapor_gap_m,
    )
    c_r = concentration_ratio_desorption(state.t_gel_c, t_cond_c)
    dc = dc_w_dt(
        c_w,
        t_gel_c=state.t_gel_c,
        c_r=c_r,
        params=mass,
        h_m=h_m,
        phase="desorption",
        t_cond_c=t_cond_c,
    )
    dh = dH_dt(
        c_w,
        t_gel_c=state.t_gel_c,
        c_r=c_r,
        params=mass,
        h_m=h_m,
        phase="desorption",
        t_cond_c=t_cond_c,
    )
    if h_m <= mass.h0_ref_m + 1e-12:
        dh = 0.0
    if dc > 0.0:
        dc = 0.0
    if dh > 0.0:
        dh = 0.0
    m_calc = m_des_kg_s_m2_from_state(c_w, h_m, dc, dh)
    return m_calc, state.t_gel_c, dc, state


def _solve_m_des_coupled(
    *,
    c_w: float,
    h_m: float,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    h_amb: float,
    mass: MassTransferParams,
    thermal: DeviceThermalParams,
    vapor_gap_m: float,
    t_guess: tuple[float, float, float] | None,
) -> tuple[float, float, float, ThermalState]:
    """Root of m_calc(m) - m = 0 so Eqs. 1–4 and Eq. 5 agree (avoids fixed-point cycling)."""

    def residual(m_des: float) -> float:
        m_calc, _, _, _ = _m_des_calc(
            m_des,
            c_w=c_w,
            h_m=h_m,
            t_cond_c=t_cond_c,
            t_amb_c=t_amb_c,
            q_solar_w_m2=q_solar_w_m2,
            h_amb=h_amb,
            mass=mass,
            thermal=thermal,
            vapor_gap_m=vapor_gap_m,
            t_guess=t_guess,
        )
        return m_calc - m_des

    m_at_zero, t_gel0, dc0, state0 = _m_des_calc(
        0.0,
        c_w=c_w,
        h_m=h_m,
        t_cond_c=t_cond_c,
        t_amb_c=t_amb_c,
        q_solar_w_m2=q_solar_w_m2,
        h_amb=h_amb,
        mass=mass,
        thermal=thermal,
        vapor_gap_m=vapor_gap_m,
        t_guess=t_guess,
    )
    if m_at_zero <= 0.0:
        return 0.0, t_gel0, dc0, state0

    hi = max(m_at_zero * 2.0, 1e-8)
    while hi < _M_DES_BRACKET_MAX and residual(hi) > 0.0:
        hi *= 2.0
    if residual(hi) >= 0.0:
        return 0.0, t_gel0, dc0, state0

    m_star = float(brentq(residual, 0.0, hi, xtol=1e-14))
    m_calc, t_gel, dc, state = _m_des_calc(
        m_star,
        c_w=c_w,
        h_m=h_m,
        t_cond_c=t_cond_c,
        t_amb_c=t_amb_c,
        q_solar_w_m2=q_solar_w_m2,
        h_amb=h_amb,
        mass=mass,
        thermal=thermal,
        vapor_gap_m=vapor_gap_m,
        t_guess=(state0.t_gel_c, state0.t_abs_c, state0.t_glass_c),
    )
    return m_calc, t_gel, dc, state


def evaluate_coupled_rates(
    *,
    c_w: float,
    h_m: float,
    t_cond_c: float,
    t_amb_c: float,
    rh: float,
    q_solar_w_m2: float,
    h_amb: float,
    phase: CyclePhase,
    mass: MassTransferParams,
    thermal: DeviceThermalParams,
    vapor_gap_m: float,
    condenser_thermal_mass_j_m2_k: float,
    fin_area_ratio: float,
    h_fg_j_per_kg: float,
    t_guess: tuple[float, float, float] | None = None,
) -> CoupledRates:
    """Return (dc_w/dt, dH/dt, dT_cond/dt) with self-consistent T_gel and m_des."""
    gap_eff = max(vapor_gap_m - h_m, 1e-6)
    q_sol = max(0.0, q_solar_w_m2)

    if phase == "absorption":
        state = solve_steady_thermal(
            t_cond_c=t_cond_c,
            t_amb_c=t_amb_c,
            q_solar_w_m2=0.0,
            m_des_kg_s_m2=0.0,
            h_amb=h_amb,
            params=thermal,
            t_guess=t_guess,
            vapor_gap_m=gap_eff,
        )
        t_gel = state.t_gel_c
        c_r = concentration_ratio_absorption(rh)
        dc = dc_w_dt(
            c_w,
            t_gel_c=t_gel,
            c_r=c_r,
            params=mass,
            h_m=h_m,
            phase="absorption",
        )
        dh = dH_dt(
            c_w,
            t_gel_c=t_gel,
            c_r=c_r,
            params=mass,
            h_m=h_m,
            phase="absorption",
        )
        if h_m <= mass.h0_ref_m + 1e-12:
            dh = max(0.0, dh)
        return CoupledRates(
            dc_w_dt=dc,
            dH_dt=dh,
            dT_cond_dt=0.0,
            t_gel_c=t_gel,
            m_des_kg_s_m2=0.0,
            thermal=state,
        )

    t_cond = clamp_temperature_c(t_cond_c)
    m_des, t_gel, dc, state = _solve_m_des_coupled(
        c_w=c_w,
        h_m=h_m,
        t_cond_c=t_cond,
        t_amb_c=t_amb_c,
        q_solar_w_m2=q_sol,
        h_amb=h_amb,
        mass=mass,
        thermal=thermal,
        vapor_gap_m=gap_eff,
        t_guess=t_guess,
    )
    c_r = concentration_ratio_desorption(t_gel, t_cond)
    dh = dH_dt(
        c_w,
        t_gel_c=t_gel,
        c_r=c_r,
        params=mass,
        h_m=h_m,
        phase="desorption",
        t_cond_c=t_cond,
    )
    h0 = mass.h0_ref_m
    if h_m <= h0 + 1e-12:
        dh = 0.0
    if dc > 0.0:
        dc = 0.0
    if dh > 0.0:
        dh = 0.0

    h_conv_g = state.h_conv_g
    h_conv_cond = condenser_h_conv_w_m2_k(h_amb, fin_area_ratio=fin_area_ratio)
    eps_gc = parallel_plate_emissivity(thermal.eps_gel, thermal.eps_al)
    q_rad = radiative_exchange_w_m2(t_gel, t_cond, emissivity=eps_gc)
    tmass = max(condenser_thermal_mass_j_m2_k, 1.0)
    dT_cond = (
        h_conv_g * (t_gel - t_cond)
        - h_conv_cond * (t_cond - t_amb_c)
        + m_des * h_fg_j_per_kg
        + q_rad
    ) / tmass

    return CoupledRates(
        dc_w_dt=dc,
        dH_dt=dh,
        dT_cond_dt=dT_cond,
        t_gel_c=t_gel,
        m_des_kg_s_m2=m_des,
        thermal=state,
    )
