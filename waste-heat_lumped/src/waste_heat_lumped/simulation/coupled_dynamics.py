"""Coupled gel + condenser rates with fixed-T_f loop-fluid HX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.optimize import brentq

from waste_heat_lumped.physics.correlations import (
    condenser_h_conv_w_m2_k,
    hollands_vapor_gap_h_conv_w_m2_k,
    parallel_plate_emissivity,
    radiative_exchange_w_m2,
)
from waste_heat_lumped.physics.device_balances import (
    DeviceThermalParams,
    ThermalState,
    solve_steady_gel_thermal,
)
from waste_heat_lumped.physics.mass_transfer import (
    MassTransferParams,
    m_des_kg_s_m2_from_dc_w,
)
from waste_heat_lumped.physics.salt_properties import clamp_temperature_c
from waste_heat_lumped.physics.sorbent import evaluate_mass_rates
from waste_heat_lumped.simulation.device_config import DeviceConfig

CyclePhase = Literal["absorption", "desorption"]

_M_DES_BRACKET_MAX = 0.01


@dataclass(frozen=True, slots=True)
class CoupledRates:
    dc_w_dt: float
    dH_dt: float
    dT_cond_dt: float
    t_gel_c: float
    m_des_kg_s_m2: float
    q_f_to_gel_w_m2: float
    thermal: ThermalState


def _m_des_calc(
    m_des: float,
    *,
    loading: float,
    h_m: float,
    t_cond_c: float,
    mass: MassTransferParams,
    thermal: DeviceThermalParams,
    vapor_gap_m: float,
    config: DeviceConfig,
    t_guess: float | None,
    fluid_active: bool,
) -> tuple[float, float, float, ThermalState]:
    mdot_f = thermal.m_dot_f_kg_s_m2 if fluid_active else 0.0
    state = solve_steady_gel_thermal(
        t_cond_c=t_cond_c,
        m_des_kg_s_m2=max(0.0, m_des),
        params=thermal,
        h_m=h_m,
        t_guess=t_guess,
        vapor_gap_m=vapor_gap_m,
        m_dot_f_kg_s_m2=mdot_f,
    )
    dc, dh, m_calc = evaluate_mass_rates(
        loading=loading,
        h_m=h_m,
        t_gel_c=state.t_gel_c,
        t_cond_c=t_cond_c,
        rh=0.0,
        phase="desorption",
        mass=mass,
        config=config,
        vapor_gap_m=vapor_gap_m,
    )
    if config.sorbent == "hydrogel":
        m_calc = m_des_kg_s_m2_from_dc_w(dc, h0_ref_m=mass.h0_ref_m)
    return m_calc, state.t_gel_c, dc, state


def _solve_m_des_coupled(
    *,
    loading: float,
    h_m: float,
    t_cond_c: float,
    mass: MassTransferParams,
    thermal: DeviceThermalParams,
    vapor_gap_m: float,
    config: DeviceConfig,
    t_guess: float | None,
    fluid_active: bool,
) -> tuple[float, float, float, ThermalState]:
    def residual(m_des: float) -> float:
        m_calc, _, _, _ = _m_des_calc(
            m_des,
            loading=loading,
            h_m=h_m,
            t_cond_c=t_cond_c,
            mass=mass,
            thermal=thermal,
            vapor_gap_m=vapor_gap_m,
            config=config,
            t_guess=t_guess,
            fluid_active=fluid_active,
        )
        if not math.isfinite(m_calc):
            return -m_des
        return m_calc - m_des

    m_at_zero, t_gel0, dc0, state0 = _m_des_calc(
        0.0,
        loading=loading,
        h_m=h_m,
        t_cond_c=t_cond_c,
        mass=mass,
        thermal=thermal,
        vapor_gap_m=vapor_gap_m,
        config=config,
        t_guess=t_guess,
        fluid_active=fluid_active,
    )
    if not math.isfinite(m_at_zero) or not math.isfinite(t_gel0):
        state0 = solve_steady_gel_thermal(
            t_cond_c=t_cond_c,
            m_des_kg_s_m2=0.0,
            params=thermal,
            h_m=h_m,
            t_guess=t_guess,
            vapor_gap_m=vapor_gap_m,
            m_dot_f_kg_s_m2=thermal.m_dot_f_kg_s_m2 if fluid_active else 0.0,
        )
        return 0.0, state0.t_gel_c, 0.0, state0
    if m_at_zero <= 0.0:
        return 0.0, t_gel0, dc0, state0

    hi = max(m_at_zero * 2.0, 1e-8)
    while hi < _M_DES_BRACKET_MAX and residual(hi) > 0.0:
        hi *= 2.0
    if residual(hi) >= 0.0:
        return 0.0, t_gel0, dc0, state0

    try:
        m_star = float(brentq(residual, 0.0, hi, xtol=1e-14))
    except ValueError:
        return 0.0, t_gel0, dc0, state0
    m_calc, t_gel, dc, state = _m_des_calc(
        m_star,
        loading=loading,
        h_m=h_m,
        t_cond_c=t_cond_c,
        mass=mass,
        thermal=thermal,
        vapor_gap_m=vapor_gap_m,
        config=config,
        t_guess=state0.t_gel_c,
        fluid_active=fluid_active,
    )
    return m_calc, t_gel, dc, state


def evaluate_coupled_rates(
    *,
    c_w: float,
    h_m: float,
    t_cond_c: float,
    t_amb_c: float,
    rh: float,
    h_amb: float,
    phase: CyclePhase,
    mass: MassTransferParams,
    thermal: DeviceThermalParams,
    vapor_gap_m: float,
    condenser_thermal_mass_j_m2_k: float,
    fin_area_ratio: float,
    h_fg_j_per_kg: float,
    config: DeviceConfig,
    fluid_active: bool = False,
    t_guess: float | None = None,
) -> CoupledRates:
    """Return (dc_w/dt, dH/dt, dT_cond/dt) with self-consistent T_gel and m_des."""
    gap_eff = max(vapor_gap_m - h_m, 0.0)

    if phase == "absorption":
        t_gel = t_amb_c
        h_conv_g = (
            hollands_vapor_gap_h_conv_w_m2_k(
                gap_eff, t_gel, t_cond_c, tilt_deg=thermal.tilt_deg
            )
            if gap_eff > 0.0
            else 0.0
        )
        state = ThermalState(
            t_gel_c=t_gel,
            h_conv_g=h_conv_g,
            m_des_kg_s_m2=0.0,
            q_f_to_gel_w_m2=0.0,
        )
        dc, dh, _ = evaluate_mass_rates(
            loading=c_w,
            h_m=h_m,
            t_gel_c=t_gel,
            t_cond_c=None,
            rh=rh,
            phase="absorption",
            mass=mass,
            config=config,
            vapor_gap_m=vapor_gap_m,
        )
        return CoupledRates(
            dc_w_dt=dc,
            dH_dt=dh,
            dT_cond_dt=0.0,
            t_gel_c=t_gel,
            m_des_kg_s_m2=0.0,
            q_f_to_gel_w_m2=0.0,
            thermal=state,
        )

    t_cond = clamp_temperature_c(t_cond_c)
    m_des, t_gel, dc, state = _solve_m_des_coupled(
        loading=c_w,
        h_m=h_m,
        t_cond_c=t_cond,
        mass=mass,
        thermal=thermal,
        vapor_gap_m=gap_eff,
        config=config,
        t_guess=t_guess,
        fluid_active=fluid_active,
    )
    _, dh, _ = evaluate_mass_rates(
        loading=c_w,
        h_m=h_m,
        t_gel_c=t_gel,
        t_cond_c=t_cond,
        rh=rh,
        phase="desorption",
        mass=mass,
        config=config,
        vapor_gap_m=vapor_gap_m,
    )

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
        q_f_to_gel_w_m2=state.q_f_to_gel_w_m2,
        thermal=state,
    )
