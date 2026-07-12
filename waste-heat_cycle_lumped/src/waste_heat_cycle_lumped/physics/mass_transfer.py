"""Wilson et al. 2025 Eqs. 5–6 — PAM-LiCl hydrogel mass transfer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from waste_heat_cycle_lumped.physics.salt_properties import (
    C_W_MAX_MOL_M3,
    C_W_MIN_MOL_M3,
    GAS_CONSTANT_J_MOL_K,
    WATER_MOLAR_MASS_KG_MOL,
    pam_licl_gravimetric_uptake_g_g,
    pam_licl_water_activity_from_uptake_g_g,
    saturation_vapor_pressure_pa,
    water_activity_from_c_w,
)

MassTransferPhase = Literal["absorption", "desorption"]

K_AIR_W_M_K: float = 0.0286
D_AIR_M2_S: float = 2.62e-5
GRAVITY_M_S2: float = 9.81
BETA_AIR_K: float = 1.0 / 300.0
NU_AIR_M2_S: float = 1.5e-5
PR_AIR: float = 0.71


def hollands_vapor_gap_h_conv_w_m2_k(
    gap_m: float,
    t_hot_c: float,
    t_cold_c: float,
    *,
    tilt_deg: float = 35.0,
) -> float:
    """Natural convection between parallel plates (for Wilson desorption g ratio)."""
    if gap_m <= 0.0:
        return 50.0
    delta_t = max(abs(t_hot_c - t_cold_c), 0.1)
    ra = GRAVITY_M_S2 * BETA_AIR_K * delta_t * gap_m**3 / (NU_AIR_M2_S * 1.8e-5) * PR_AIR
    if ra < 1708.0:
        return max(K_AIR_W_M_K / gap_m, 0.5)
    nu = 0.720 * ra**0.25 * (1.0 + math.cos(math.radians(tilt_deg)) * 0.1)
    return max(nu * K_AIR_W_M_K / gap_m, K_AIR_W_M_K / gap_m)


def mass_transfer_g_from_h_conv_m_s(h_conv_w_m2_k: float) -> float:
    return h_conv_w_m2_k * D_AIR_M2_S / K_AIR_W_M_K


def _absorption_effective_water_activity(
    c_w: float,
    *,
    t_gel_c: float,
    params: MassTransferParams,
    h_m: float,
) -> float:
    aw_brine = water_activity_from_c_w(
        c_w,
        c_s=params.c_s_mol_m3,
        ions_per_formula=params.ions_per_formula,
        temperature_c=t_gel_c,
        salt_name=params.salt_name,
        formula_weight_g_mol=params.formula_weight_g_mol,
        salt_to_polymer_ratio=params.salt_to_polymer_ratio,
        h_m=h_m,
        h0_ref_m=params.h0_ref_m,
    )
    u = pam_licl_gravimetric_uptake_g_g(c_w, h_m, h0_ref_m=params.h0_ref_m)
    aw_dvs = pam_licl_water_activity_from_uptake_g_g(u)
    return max(aw_brine, aw_dvs)


def _mass_transfer_driving_force(
    c_w: float,
    *,
    t_gel_c: float,
    c_r: float,
    params: MassTransferParams,
    h_m: float,
    phase: MassTransferPhase,
) -> float:
    if phase == "absorption":
        aw = _absorption_effective_water_activity(
            c_w, t_gel_c=t_gel_c, params=params, h_m=h_m
        )
        return c_r - aw
    aw = water_activity_from_c_w(
        c_w,
        c_s=params.c_s_mol_m3,
        ions_per_formula=params.ions_per_formula,
        temperature_c=t_gel_c,
        salt_name=params.salt_name,
        formula_weight_g_mol=params.formula_weight_g_mol,
        salt_to_polymer_ratio=params.salt_to_polymer_ratio,
        h_m=h_m,
        h0_ref_m=params.h0_ref_m,
    )
    return c_r - aw


@dataclass(frozen=True, slots=True)
class MassTransferParams:
    g_conv_m_s: float
    h0_ref_m: float
    vapor_gap_m: float
    tilt_deg: float
    c_s_mol_m3: float
    ions_per_formula: int
    rho_solution_kg_m3: float
    salt_name: str = "LiCl"
    formula_weight_g_mol: float = 42.394
    salt_to_polymer_ratio: float = 4.0


def mass_transfer_g_m_s(
    *,
    phase: MassTransferPhase,
    params: MassTransferParams,
    h_m: float,
    t_gel_c: float,
    t_cond_c: float | None = None,
) -> float:
    if phase == "absorption":
        return params.g_conv_m_s
    if t_cond_c is None:
        raise ValueError("t_cond_c required for desorption mass transfer")
    gap_m = max(params.vapor_gap_m - h_m, 1e-4)
    h_conv = hollands_vapor_gap_h_conv_w_m2_k(
        gap_m, t_gel_c, t_cond_c, tilt_deg=params.tilt_deg
    )
    return mass_transfer_g_from_h_conv_m_s(h_conv)


def _mass_transfer_prefactor(
    *,
    phase: MassTransferPhase,
    params: MassTransferParams,
    h_m: float,
    t_gel_c: float,
    t_cond_c: float | None = None,
) -> float:
    g = mass_transfer_g_m_s(
        phase=phase,
        params=params,
        h_m=h_m,
        t_gel_c=t_gel_c,
        t_cond_c=t_cond_c,
    )
    return g / params.h0_ref_m


def concentration_ratio_absorption(rh: float) -> float:
    return float(rh)


def concentration_ratio_desorption(t_gel_c: float, t_cond_c: float) -> float:
    p_g = saturation_vapor_pressure_pa(t_gel_c)
    p_c = saturation_vapor_pressure_pa(t_cond_c)
    t_g_k = t_gel_c + 273.15
    t_c_k = t_cond_c + 273.15
    if p_g <= 0.0 or t_g_k <= 0.0 or t_c_k <= 0.0:
        return 0.0
    return (p_c / p_g) * (t_g_k / t_c_k)


def rh_outside_desorber(t_d_c: float, t_cond_c: float) -> float:
    """Relative humidity in the vapor gap outside the desorbing gel (0–1)."""
    return concentration_ratio_desorption(t_d_c, t_cond_c)


def dc_w_dt(
    c_w: float,
    *,
    t_gel_c: float,
    c_r: float,
    params: MassTransferParams,
    h_m: float,
    phase: MassTransferPhase = "absorption",
    t_cond_c: float | None = None,
) -> float:
    t_k = max(t_gel_c + 273.15, 200.0)
    p_sat = saturation_vapor_pressure_pa(t_gel_c)
    pref = _mass_transfer_prefactor(
        phase=phase,
        params=params,
        h_m=h_m,
        t_gel_c=t_gel_c,
        t_cond_c=t_cond_c,
    )
    driving = _mass_transfer_driving_force(
        c_w,
        t_gel_c=t_gel_c,
        c_r=c_r,
        params=params,
        h_m=h_m,
        phase=phase,
    )
    rate = pref * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving
    if c_w >= C_W_MAX_MOL_M3 and rate > 0.0:
        return 0.0
    if c_w <= C_W_MIN_MOL_M3 and rate < 0.0:
        return 0.0
    return rate


def dH_dt(
    c_w: float,
    *,
    t_gel_c: float,
    c_r: float,
    params: MassTransferParams,
    h_m: float,
    phase: MassTransferPhase = "absorption",
    t_cond_c: float | None = None,
) -> float:
    """Eq. 6: dH/dt (m/s) — hydrogel thickness rate.

    Consistent with Eq. 5 dc_w/dt:
        dH/dt = g · (MW / ρ_sol) · (p_sat / RT) · driving
    """
    t_k = max(t_gel_c + 273.15, 200.0)
    p_sat = saturation_vapor_pressure_pa(t_gel_c)
    driving = _mass_transfer_driving_force(
        c_w,
        t_gel_c=t_gel_c,
        c_r=c_r,
        params=params,
        h_m=h_m,
        phase=phase,
    )
    g = mass_transfer_g_m_s(
        phase=phase,
        params=params,
        h_m=h_m,
        t_gel_c=t_gel_c,
        t_cond_c=t_cond_c,
    )
    return (
        g
        * WATER_MOLAR_MASS_KG_MOL
        / params.rho_solution_kg_m3
        * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k))
        * driving
    )


def m_ads_kg_s_m2_from_state(c_w: float, h_m: float, dc_w_dt_val: float, dH_dt_val: float) -> float:
    return max(0.0, WATER_MOLAR_MASS_KG_MOL * (dc_w_dt_val * h_m + c_w * dH_dt_val))


def m_des_kg_s_m2_from_state(
    c_w: float,
    h_m: float,
    dc_w_dt_val: float,
    dH_dt_val: float,
) -> float:
    flux = -WATER_MOLAR_MASS_KG_MOL * (dc_w_dt_val * h_m + c_w * dH_dt_val)
    return max(0.0, flux)
