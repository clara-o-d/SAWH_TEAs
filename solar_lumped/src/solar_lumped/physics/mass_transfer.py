"""Wilson et al. 2025 Eqs. 5–6 — convection-limited mass transfer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from solar_lumped.physics.correlations import (
    hollands_vapor_gap_h_conv_w_m2_k,
    mass_transfer_g_from_h_conv_m_s,
    vapor_gap_mass_transfer_inhibited,
)
from solar_lumped.physics.salt_properties import (
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


def _absorption_effective_water_activity(
    c_w: float,
    *,
    t_gel_c: float,
    params: MassTransferParams,
    h_m: float,
) -> float:
    """Composite gel a_w for Eq. 5 during open absorption.

    LiCl uses brine activity plus PAM-LiCl DVS cap (Note S2). Other salts use brine only.
    """
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
        salt_weight_factor=params.salt_weight_factor,
    )
    if params.salt_name != "LiCl":
        return aw_brine
    u = pam_licl_gravimetric_uptake_g_g(
        c_w,
        h_m,
        h0_ref_m=params.h0_ref_m,
        c_s_mol_m3=params.c_s_mol_m3,
        formula_weight_g_mol=params.formula_weight_g_mol,
        salt_to_polymer_ratio=params.salt_to_polymer_ratio,
        salt_weight_factor=params.salt_weight_factor,
    )
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
            c_w,
            t_gel_c=t_gel_c,
            params=params,
            h_m=h_m,
        )
        if not math.isfinite(aw):
            return 0.0
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
        salt_weight_factor=params.salt_weight_factor,
    )
    if not math.isfinite(aw):
        return 0.0
    return c_r - aw


@dataclass(frozen=True, slots=True)
class MassTransferParams:
    g_conv_m_s: float  # g_chamber (Table S3) for open absorption
    h0_ref_m: float
    vapor_gap_m: float
    tilt_deg: float
    c_s_mol_m3: float
    ions_per_formula: int
    rho_solution_kg_m3: float
    salt_name: str = "LiCl"
    formula_weight_g_mol: float = 42.394
    salt_to_polymer_ratio: float = 4.0
    salt_weight_factor: float = 1.0


def mass_transfer_g_m_s(
    *,
    phase: MassTransferPhase,
    params: MassTransferParams,
    h_m: float,
    t_gel_c: float,
    t_cond_c: float | None = None,
) -> float:
    """Note S1: g_chamber in absorption; heat–mass analogy in desorption (Eq. S5)."""
    if phase == "absorption":
        return params.g_conv_m_s
    if t_cond_c is None:
        raise ValueError("t_cond_c required for desorption mass transfer")
    gap_m = max(params.vapor_gap_m - h_m, 0.0)
    if vapor_gap_mass_transfer_inhibited(gap_m):
        return 0.0
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
    """Eq. 5: dc_w/dt (mol/m³/s); g_chamber/H₀ (abs) or heat–mass analogy (des)."""
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
    if not math.isfinite(driving):
        return 0.0
    rate = pref * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving
    if not math.isfinite(rate):
        return 0.0
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

    Consistent with Note S1 dc_w/dt:
        dH/dt = g · (MW / ρ_sol) · (p_sat / RT) · driving
    This equals dc_w/dt · (MW · H₀ / ρ_sol), ensuring H and c_w evolve at
    the same timescale (both driven by the mass-transfer velocity g).
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


def m_des_kg_s_m2_from_state(
    c_w: float,
    h_m: float,
    dc_w_dt_val: float,
    dH_dt_val: float,
) -> float:
    """Desorption flux (kg/m²/s) from Eqs. 5–6 gel water inventory rate.

    Paper Eq. 1 ṁ_des removes water from the gel. With c_w (mol/m³) and H (m),
    inventory N = c_w H (mol/m²) gives ṁ = -MW · dN/dt = -MW · (dc_w/dt·H + c_w·dH/dt).

    Note S1 ṁ = -dc_w/dt · MW · H₀ is the H ≈ H₀, dH/dt ≈ 0 limit of Eq. 5 alone.
    """
    flux = -WATER_MOLAR_MASS_KG_MOL * (dc_w_dt_val * h_m + c_w * dH_dt_val)
    return max(0.0, flux)


def m_des_kg_s_m2_from_dc_w(
    dc_w_dt_val: float,
    *,
    h0_ref_m: float,
) -> float:
    """Note S1 limit: ṁ_des = -dc_w/dt · MW · H₀ (reference thickness, negligible dH/dt)."""
    if dc_w_dt_val >= 0.0:
        return 0.0
    return -dc_w_dt_val * WATER_MOLAR_MASS_KG_MOL * h0_ref_m
