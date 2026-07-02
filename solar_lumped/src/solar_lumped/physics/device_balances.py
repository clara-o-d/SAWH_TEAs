"""Wilson et al. 2025 Eqs. 1, 3, 4 — steady absorber, glass, and gel temperatures."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from solar_lumped.physics.correlations import (
    STEFAN_BOLTZMANN_W_M2_K4,
    conduction_air_gap_w_m2,
    hollands_vapor_gap_h_conv_w_m2_k,
    parallel_plate_emissivity,
    radiative_exchange_w_m2,
)
from solar_lumped.physics.salt_properties import clamp_temperature_c
from solar_lumped.physics import table_s3


@dataclass(frozen=True, slots=True)
class ThermalState:
    t_gel_c: float
    t_abs_c: float
    t_glass_c: float
    h_conv_g: float
    m_des_kg_s_m2: float


@dataclass(frozen=True, slots=True)
class DeviceThermalParams:
    insulation_gap_m: float = table_s3.L_INS_M
    vapor_gap_m: float = table_s3.L_G_M
    u_gel_w_m2_k: float = table_s3.U_GEL_W_M2_K
    eps_abs: float = table_s3.EPS_ABS
    tau_glass: float = table_s3.TAU_GLASS
    eps_gel: float = table_s3.EPS_GEL
    eps_al: float = table_s3.EPS_AL
    tilt_deg: float = table_s3.TILT_DEG
    h_des_j_per_kg: float = table_s3.H_DES_J_PER_KG


def _residuals(
    x: np.ndarray,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    m_des_kg_s_m2: float,
    h_amb: float,
    params: DeviceThermalParams,
    vapor_gap_effective_m: float,
) -> np.ndarray:
    t_gel, t_abs, t_glass = float(x[0]), float(x[1]), float(x[2])
    h_conv_g = hollands_vapor_gap_h_conv_w_m2_k(
        vapor_gap_effective_m, t_gel, t_cond_c, tilt_deg=params.tilt_deg
    )
    q_cond_ag = conduction_air_gap_w_m2(t_abs, t_glass, params.insulation_gap_m)
    q_rad_ag = radiative_exchange_w_m2(t_abs, t_glass, emissivity=1.0)
    q_rad_ga = radiative_exchange_w_m2(t_glass, t_amb_c, emissivity=1.0)
    eps_gc = parallel_plate_emissivity(params.eps_gel, params.eps_al)
    q_rad_gc = radiative_exchange_w_m2(t_gel, t_cond_c, emissivity=eps_gc)

    # Eq 3 glass
    r3 = q_cond_ag + q_rad_ag - h_amb * (t_glass - t_amb_c) - q_rad_ga
    # Eq 4 absorber
    r4 = (
        params.eps_abs * params.tau_glass * q_solar_w_m2
        - q_cond_ag
        - q_rad_ag
        - params.u_gel_w_m2_k * (t_abs - t_gel)
    )
    # Eq 1 gel
    q_des = m_des_kg_s_m2 * params.h_des_j_per_kg
    r1 = (
        params.u_gel_w_m2_k * (t_abs - t_gel)
        - h_conv_g * (t_gel - t_cond_c)
        - q_des
        - q_rad_gc
    )
    return np.array([r1, r3, r4], dtype=float)


def solve_steady_thermal(
    *,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    m_des_kg_s_m2: float,
    h_amb: float,
    params: DeviceThermalParams,
    t_guess: tuple[float, float, float] | None = None,
    vapor_gap_m: float | None = None,
) -> ThermalState:
    """Solve Eqs. 1, 3, 4 for (T_gel, T_abs, T_glass)."""
    gap_m = vapor_gap_m if vapor_gap_m is not None else params.vapor_gap_m
    if t_guess is None:
        t_gel0 = clamp_temperature_c(max(t_amb_c + 5.0, t_cond_c + 5.0))
        t_abs0 = clamp_temperature_c(t_gel0 + min(30.0, max(5.0, q_solar_w_m2 / 40.0)))
        t_glass0 = clamp_temperature_c(t_amb_c + 2.0)
    else:
        t_gel0, t_abs0, t_glass0 = (
            clamp_temperature_c(t_guess[0]),
            clamp_temperature_c(t_guess[1]),
            clamp_temperature_c(t_guess[2]),
        )

    sol = root(
        _residuals,
        x0=np.array([t_gel0, t_abs0, t_glass0]),
        args=(t_cond_c, t_amb_c, q_solar_w_m2, m_des_kg_s_m2, h_amb, params, gap_m),
        method="hybr",
        tol=1e-8,
    )
    if not sol.success:
        t_gel, t_abs, t_glass = t_gel0, t_abs0, t_glass0
    else:
        t_gel = clamp_temperature_c(float(sol.x[0]))
        t_abs = clamp_temperature_c(float(sol.x[1]))
        t_glass = clamp_temperature_c(float(sol.x[2]))

    h_conv_g = hollands_vapor_gap_h_conv_w_m2_k(
        gap_m, t_gel, t_cond_c, tilt_deg=params.tilt_deg
    )
    return ThermalState(
        t_gel_c=t_gel,
        t_abs_c=t_abs,
        t_glass_c=t_glass,
        h_conv_g=h_conv_g,
        m_des_kg_s_m2=m_des_kg_s_m2,
    )


def thermal_residual_norm(
    *,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    m_des_kg_s_m2: float,
    h_amb: float,
    params: DeviceThermalParams,
) -> float:
    state = solve_steady_thermal(
        t_cond_c=t_cond_c,
        t_amb_c=t_amb_c,
        q_solar_w_m2=q_solar_w_m2,
        m_des_kg_s_m2=m_des_kg_s_m2,
        h_amb=h_amb,
        params=params,
    )
    r = _residuals(
        np.array([state.t_gel_c, state.t_abs_c, state.t_glass_c]),
        t_cond_c,
        t_amb_c,
        q_solar_w_m2,
        m_des_kg_s_m2,
        h_amb,
        params,
        params.vapor_gap_m,
    )
    return float(np.linalg.norm(r))
