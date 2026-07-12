"""Wilson et al. 2025 Eqs. 1, 3, 4 — steady absorber, glass, and gel temperatures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import root

from solar_lumped.physics.correlations import (
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
    eps_abs: float = table_s3.EPS_ABS
    tau_glass: float = table_s3.TAU_GLASS
    eps_gel: float = table_s3.EPS_GEL
    eps_al: float = table_s3.EPS_AL
    eps_glass: float = table_s3.EPS_GLASS
    tilt_deg: float = table_s3.TILT_DEG
    h_des_j_per_kg: float = table_s3.H_DES_J_PER_KG
    has_glass: bool = True
    physics_model: Literal["note_s1", "comsol_lumped"] = "note_s1"
    # COMSOL lumped optics (None → file defaults in comsol_lumped.py).
    comsol_solar_ads_frac: float | None = None
    comsol_solar_glass_frac: float | None = None
    comsol_eps_ads: float | None = None
    comsol_eps_glass_ir: float | None = None
    comsol_reflect_glass: float | None = None
    # Depression (K) of the effective radiant sink temperature below air temperature
    # for the outermost surface. Wilson's typeset Eq. 3 radiates to T_amb (0 K here,
    # used for Fig. 2 / Cambridge). Their COMSOL Atacama field model radiates the glass
    # to a surroundings/sky temperature below air temperature (Surface-to-Ambient
    # Radiation); reproducing the digitized field-test curves requires this term.
    sky_temp_depression_c: float = 0.0


def comsol_optics(params: DeviceThermalParams) -> dict[str, float | bool]:
    from solar_lumped.physics import comsol_lumped as cl

    return {
        "solar_ads_frac": (
            params.comsol_solar_ads_frac
            if params.comsol_solar_ads_frac is not None
            else cl.SOLAR_ADS_FRAC
        ),
        "solar_glass_frac": (
            params.comsol_solar_glass_frac
            if params.comsol_solar_glass_frac is not None
            else cl.SOLAR_GLASS_FRAC
        ),
        "eps_ads": (
            params.comsol_eps_ads if params.comsol_eps_ads is not None else cl.EPS_ADS
        ),
        "eps_glass_ir": (
            params.comsol_eps_glass_ir
            if params.comsol_eps_glass_ir is not None
            else cl.EPS_GLASS_IR
        ),
        "reflect_glass": (
            params.comsol_reflect_glass
            if params.comsol_reflect_glass is not None
            else cl.REFLECT_GLASS
        ),
        "has_glass": params.has_glass,
    }


def _residuals(
    x: np.ndarray,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    m_des_kg_s_m2: float,
    h_amb: float,
    params: DeviceThermalParams,
    vapor_gap_effective_m: float,
    h_m: float,
) -> np.ndarray:
    t_gel, t_abs, t_glass = float(x[0]), float(x[1]), float(x[2])
    u_gel = table_s3.u_gel_w_m2_k(h_m)
    h_conv_g = hollands_vapor_gap_h_conv_w_m2_k(
        vapor_gap_effective_m, t_gel, t_cond_c, tilt_deg=params.tilt_deg
    )
    eps_gc = parallel_plate_emissivity(params.eps_gel, params.eps_al)
    q_rad_gc = radiative_exchange_w_m2(t_gel, t_cond_c, emissivity=eps_gc)

    # Eq 1 gel
    q_des = m_des_kg_s_m2 * params.h_des_j_per_kg
    r1 = (
        u_gel * (t_abs - t_gel)
        - h_conv_g * (t_gel - t_cond_c)
        - q_des
        - q_rad_gc
    )

    # Absorber→glass: Wilson Eq. 4 writes σ(T_abs⁴ − T_glass⁴) without an
    # explicit emissivity factor (cavity / blackbody approximation).
    eps_ag = 1.0
    # Glass→surroundings: Wilson Eq. 3 writes σ(T_glass⁴ − T_amb⁴) with no emissivity
    # factor (blackbody in the IR). The sink temperature is ambient by default; the
    # COMSOL Atacama field model uses a surroundings/sky temperature below air temp.
    eps_ga = 1.0
    t_sky_c = t_amb_c - params.sky_temp_depression_c

    if not params.has_glass:
        q_rad_abs_amb = radiative_exchange_w_m2(t_abs, t_sky_c, emissivity=params.eps_abs)
        r4 = (
            params.eps_abs * q_solar_w_m2
            - h_amb * (t_abs - t_amb_c)
            - q_rad_abs_amb
            - u_gel * (t_abs - t_gel)
        )
        r3 = t_glass - t_amb_c
    else:
        q_cond_ag = conduction_air_gap_w_m2(t_abs, t_glass, params.insulation_gap_m)
        q_rad_ag = radiative_exchange_w_m2(t_abs, t_glass, emissivity=eps_ag)
        q_rad_ga = radiative_exchange_w_m2(t_glass, t_sky_c, emissivity=eps_ga)
        r3 = q_cond_ag + q_rad_ag - h_amb * (t_glass - t_amb_c) - q_rad_ga
        r4 = (
            params.eps_abs * params.tau_glass * q_solar_w_m2
            - q_cond_ag
            - q_rad_ag
            - u_gel * (t_abs - t_gel)
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
    h_m: float,
    t_guess: tuple[float, float, float] | None = None,
    vapor_gap_m: float | None = None,
) -> ThermalState:
    """Solve Eqs. 1, 3, 4 for (T_gel, T_abs, T_glass)."""
    if vapor_gap_m is None:
        gap_m = max(params.vapor_gap_m - h_m, 0.0)
    else:
        gap_m = vapor_gap_m
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
        args=(t_cond_c, t_amb_c, q_solar_w_m2, m_des_kg_s_m2, h_amb, params, gap_m, h_m),
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
    h_m: float = table_s3.H0_M,
) -> float:
    gap_m = max(params.vapor_gap_m - h_m, 0.0)
    state = solve_steady_thermal(
        t_cond_c=t_cond_c,
        t_amb_c=t_amb_c,
        q_solar_w_m2=q_solar_w_m2,
        m_des_kg_s_m2=m_des_kg_s_m2,
        h_amb=h_amb,
        params=params,
        h_m=h_m,
        vapor_gap_m=gap_m,
    )
    r = _residuals(
        np.array([state.t_gel_c, state.t_abs_c, state.t_glass_c]),
        t_cond_c,
        t_amb_c,
        q_solar_w_m2,
        m_des_kg_s_m2,
        h_amb,
        params,
        gap_m,
        h_m,
    )
    return float(np.linalg.norm(r))
