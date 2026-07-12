"""Gel energy balance with fixed-T_f loop-fluid HX (replaces Wilson solar stack)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from waste_heat_lumped.physics import device_defaults as dd
from waste_heat_lumped.physics.correlations import (
    hx_effectiveness_q,
    hollands_vapor_gap_h_conv_w_m2_k,
    parallel_plate_emissivity,
    radiative_exchange_w_m2,
)
from waste_heat_lumped.physics.salt_properties import clamp_temperature_c
from waste_heat_lumped.physics import table_s3


@dataclass(frozen=True, slots=True)
class ThermalState:
    t_gel_c: float
    h_conv_g: float
    m_des_kg_s_m2: float
    q_f_to_gel_w_m2: float


@dataclass(frozen=True, slots=True)
class DeviceThermalParams:
    vapor_gap_m: float = dd.VAPOR_GAP_M
    eps_gel: float = dd.GEL_EMISSIVITY
    eps_al: float = dd.CONDENSER_EMISSIVITY
    tilt_deg: float = dd.TILT_DEG
    h_des_j_per_kg: float = table_s3.H_DES_J_PER_KG
    gel_thermal_mass_j_m2_k: float = dd.GEL_THERMAL_MASS_J_M2_K
    t_f_c: float = dd.T_F_C
    m_dot_f_kg_s_m2: float = dd.M_DOT_F_KG_S_M2
    ua_gel_w_k: float = dd.UA_GEL_W_K
    fluid_cp_j_kg_k: float = dd.FLUID_CP_J_KG_K


def q_f_to_gel_w_m2(
    *,
    t_gel_c: float,
    t_f_c: float,
    m_dot_f_kg_s_m2: float,
    ua_gel_w_k: float,
    fluid_cp_j_kg_k: float,
) -> float:
    """NTU–ε heat flux from loop fluid (fixed T_f) to gel."""
    if m_dot_f_kg_s_m2 <= 0.0:
        return 0.0
    mdot_cp = m_dot_f_kg_s_m2 * fluid_cp_j_kg_k
    return hx_effectiveness_q(mdot_cp, ua_gel_w_k, t_f_c - t_gel_c)


def _gel_residual(
    t_gel: float,
    *,
    t_cond_c: float,
    t_f_c: float,
    m_dot_f_kg_s_m2: float,
    m_des_kg_s_m2: float,
    params: DeviceThermalParams,
    vapor_gap_effective_m: float,
) -> float:
    """Steady gel balance: Q_f→gel − ṁ_des h_des − q_gap − q_rad = 0."""
    t_gel = clamp_temperature_c(t_gel)
    q_in = q_f_to_gel_w_m2(
        t_gel_c=t_gel,
        t_f_c=t_f_c,
        m_dot_f_kg_s_m2=m_dot_f_kg_s_m2,
        ua_gel_w_k=params.ua_gel_w_k,
        fluid_cp_j_kg_k=params.fluid_cp_j_kg_k,
    )
    h_conv_g = hollands_vapor_gap_h_conv_w_m2_k(
        vapor_gap_effective_m, t_gel, t_cond_c, tilt_deg=params.tilt_deg
    )
    eps_gc = parallel_plate_emissivity(params.eps_gel, params.eps_al)
    q_rad = radiative_exchange_w_m2(t_gel, t_cond_c, emissivity=eps_gc)
    q_des = m_des_kg_s_m2 * params.h_des_j_per_kg
    return q_in - q_des - h_conv_g * (t_gel - t_cond_c) - q_rad


def solve_steady_gel_thermal(
    *,
    t_cond_c: float,
    m_des_kg_s_m2: float,
    params: DeviceThermalParams,
    h_m: float,
    t_guess: float | None = None,
    vapor_gap_m: float | None = None,
    m_dot_f_kg_s_m2: float | None = None,
    t_f_c: float | None = None,
) -> ThermalState:
    """Solve quasi-steady gel temperature for given ṁ_des and T_cond."""
    if vapor_gap_m is None:
        gap_m = max(params.vapor_gap_m - h_m, 0.0)
    else:
        gap_m = vapor_gap_m
    mdot_f = params.m_dot_f_kg_s_m2 if m_dot_f_kg_s_m2 is None else m_dot_f_kg_s_m2
    t_f = params.t_f_c if t_f_c is None else t_f_c

    if mdot_f <= 0.0:
        t_gel = clamp_temperature_c(t_cond_c)
        h_conv_g = (
            hollands_vapor_gap_h_conv_w_m2_k(
                gap_m, t_gel, t_cond_c, tilt_deg=params.tilt_deg
            )
            if gap_m > 0.0
            else 0.0
        )
        return ThermalState(
            t_gel_c=t_gel,
            h_conv_g=h_conv_g,
            m_des_kg_s_m2=m_des_kg_s_m2,
            q_f_to_gel_w_m2=0.0,
        )

    t_gel0 = clamp_temperature_c(t_guess if t_guess is not None else t_f)
    sol = root(
        lambda x: _gel_residual(
            float(x[0]),
            t_cond_c=t_cond_c,
            t_f_c=t_f,
            m_dot_f_kg_s_m2=mdot_f,
            m_des_kg_s_m2=m_des_kg_s_m2,
            params=params,
            vapor_gap_effective_m=gap_m,
        ),
        x0=np.array([t_gel0]),
        method="hybr",
        tol=1e-8,
    )
    t_gel = clamp_temperature_c(float(sol.x[0]) if sol.success else t_gel0)
    h_conv_g = hollands_vapor_gap_h_conv_w_m2_k(
        gap_m, t_gel, t_cond_c, tilt_deg=params.tilt_deg
    )
    q_flux = q_f_to_gel_w_m2(
        t_gel_c=t_gel,
        t_f_c=t_f,
        m_dot_f_kg_s_m2=mdot_f,
        ua_gel_w_k=params.ua_gel_w_k,
        fluid_cp_j_kg_k=params.fluid_cp_j_kg_k,
    )
    return ThermalState(
        t_gel_c=t_gel,
        h_conv_g=h_conv_g,
        m_des_kg_s_m2=m_des_kg_s_m2,
        q_f_to_gel_w_m2=q_flux,
    )
