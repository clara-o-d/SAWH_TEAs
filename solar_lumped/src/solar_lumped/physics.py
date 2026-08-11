"""System physics: geometry/material constants, brine/salt thermodynamics, heat-transfer
correlations, thermal balances, mass transfer, and the PAM-salt hydrogel sorbent model."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import root

from solar_lumped._parameters_xlsx import SALTS, physics_value as _pv
from solar_lumped.utils import find_root_bracketed, load_two_column_csv

if TYPE_CHECKING:
    from solar_lumped.simulation import SystemConfig


# --- Table S3 / Note S1 system parameters (Wilson & Díaz-Marín *Device* 2025) ---
# All values load from docs/parameters.xlsx (Physics sheet).

# Geometry
H0_M: float = _pv("Hydrogel reference thickness (H0)", mm_to_m=True)  # hydrogel reference thickness H₀ (m)
L_G_M: float = _pv("Vapor gap (L_g)", mm_to_m=True)  # vapor gap L_g (m)
L_INS_M: float = _pv("Insulation gap (L_ins)", mm_to_m=True)  # insulation gap L_ins (m)
L_C_M: float = _pv("Condenser aluminum plate thickness (L_c)", mm_to_m=True)  # condenser aluminum plate thickness L_c (m)

# Wilson §2.2 / Note S1: thermobuoyancy and mass transport inhibited below ~7 mm gap
VAPOR_GAP_TRANSPORT_MIN_M: float = _pv("Vapor-gap transport floor", mm_to_m=True)

# Baseline ambient convection coefficient, Wilson et al. (2025) Methods (Cambridge test).
H_AMB_W_M2_K: float = _pv("Ambient convection coefficient (h_amb)")

# Salt:polymer ratio default (g salt / g polymer); dual role, also economics.py sorbent cost.
SALT_TO_POLYMER_RATIO_DEFAULT: float = _pv("Salt:polymer ratio (S/L)")

# Materials / transport
G_CHAMBER_M_S: float = _pv("Chamber convection coefficient, absorption (g_chamber)")
RHO_GEL_KG_M3: float = _pv("Composite (hydrogel) density at 20% RH (rho_gel)")  # ρ_gel composite/hydrogel density (kg/m³)
RHO_COMPOSITE_KG_M3: float = RHO_GEL_KG_M3  # alias -- composite density at fabrication (25 °C, 20% RH)
H_DES_J_PER_KG: float = _pv("Desorption enthalpy, LiCl (h_des)")  # h_des (J/kg)
H_FG_J_PER_KG: float = _pv("Condensation enthalpy (h_fg)")  # h_fg condensation (J/kg)
K_AIR_W_M_K: float = _pv("Air thermal conductivity (k_air)")  # k_air (W/m·K)
K_GEL_W_M_K: float = _pv("Hydrogel thermal conductivity (k_gel)")  # k_w hydrogel (W/m·K) — Table S3
RABS_M2_K_W: float = _pv("Absorber-to-gel constant resistance (Rabs)")  # R_aluminum + R_silicone, lumped and rounded
RHO_AL_KG_M3: float = _pv("Aluminum density (rho_Al)")
CP_AL_J_KG_K: float = _pv("Aluminum specific heat (cp_Al)")

# Optical / radiative
EPS_GEL: float = _pv("Gel emissivity (eps_gel)")
EPS_AL: float = _pv("Condenser (Al) emissivity (eps_Al)")
EPS_ABS: float = _pv("Absorber emissivity (eps_abs)")
TAU_GLASS: float = _pv("Glass transmittance (tau_glass)")
# This package's Case 2 ("selective surface") base-case IR emissivities -- see
# SystemConfig.thermal_params() in simulation.py, which is where these are applied.
EPS_ABS_IR_CASE2: float = _pv("Absorber IR emissivity (eps_abs_ir)")
EPS_GLASS_IR_CASE2: float = _pv("Glass IR emissivity (eps_glass_ir)")

# System orientation / condenser fins
TILT_DEG: float = _pv("Tilt angle (theta)")
FIN_AREA_RATIO: float = _pv("Condenser fin area ratio (A_r)")  # A_r


def u_gel_w_m2_k(h_m: float) -> float:
    """Note S1 gel–absorber conductance in series:
    1/U_gel = Rabs + H(t)/k_hydrogel, where Rabs lumps the (fixed) aluminum
    and silicone resistances into one rounded constant."""
    h = max(float(h_m), H0_M * 0.25)
    resistance = RABS_M2_K_W + h / K_GEL_W_M_K
    return 1.0 / resistance


# Reference value at fabrication thickness H₀ (for tests / docs)
U_GEL_W_M2_K: float = u_gel_w_m2_k(H0_M)

# Condenser thermal mass per footprint area (ρ_al c_p L_c)
CONDENSER_THERMAL_MASS_J_M2_K: float = RHO_AL_KG_M3 * CP_AL_J_KG_K * L_C_M


# --- Conde (2004) aqueous LiCl and CaCl2 solution properties ---
# Table 3: π ≡ p_sol(ξ,T)/p_H2O(T) = π_25(ξ)·f(ξ,θ), ξ = brine salt mass fraction,
# θ = T/T_c,H2O. π is the interface water activity a_w; p_H2O from Saul–Wagner.

# Conde (2004): θ ≡ T / T_c,H2O
T_CRIT_H2O_K: float = _pv("Water critical temperature (T_crit,H2O)")
P_CRIT_H2O_PA: float = _pv("Water critical pressure (P_crit,H2O)")

# Saturation (solubility) brine salt mass fraction: the most concentrated the liquid can
# get. Past it the excess salt is precipitated solid, so a_w is pinned here rather than
# following the correlation into its supersaturated tail. Derived from solubility in grams
# per litre of water, so xi = s / (s + 1000): LiCl 845 g/L -> 45.8 wt%, CaCl2 745 g/L ->
# 42.7 wt%. Both sit inside Conde's stated validity range (§ Density: LiCl 0.56, CaCl2
# 0.60), so the correlation is never evaluated outside its own domain.
SOLUBILITY_G_PER_L: dict[str, float] = {
    name: float(row["solubility_g_per_l"]) for name, row in SALTS.items()
}


def saturation_brine_salt_fraction(salt_name: str) -> float:
    """Saturation brine salt mass fraction from solubility in g per litre of water."""
    try:
        s = SOLUBILITY_G_PER_L[salt_name]
    except KeyError:
        raise KeyError(
            f"No solubility for {salt_name!r}; add it to SOLUBILITY_G_PER_L "
            "(its deliquescence RH is derived from it)."
        ) from None
    return s / (s + 1000.0)


XI_SAT_LICL: float = saturation_brine_salt_fraction("LiCl")
XI_SAT_CACL2: float = saturation_brine_salt_fraction("CaCl2")

_BRACKET_LO: float = _pv("Brine mass-fraction bracket lower")
_BRACKET_HI: float = _pv("Brine mass-fraction bracket upper")


@dataclass(frozen=True, slots=True)
class VaporPressureParams:
    """Table 3 parameters for the Conde vapour-pressure equation."""

    pi0: float
    pi1: float
    pi2: float
    pi3: float
    pi4: float
    pi5: float
    pi6: float
    pi7: float
    pi8: float
    pi9: float
    xi_sat: float


LICL_VAPOR_PRESSURE = VaporPressureParams(
    pi0=0.28,
    pi1=4.30,
    pi2=0.60,
    pi3=0.21,
    pi4=5.10,
    pi5=0.49,
    pi6=0.362,
    pi7=-4.75,
    pi8=-0.40,
    pi9=0.03,
    xi_sat=XI_SAT_LICL,
)

CACL2_VAPOR_PRESSURE = VaporPressureParams(
    pi0=0.31,
    pi1=3.698,
    pi2=0.60,
    pi3=0.231,
    pi4=4.584,
    pi5=0.49,
    pi6=0.478,
    pi7=-5.20,
    pi8=-0.40,
    pi9=0.018,
    xi_sat=XI_SAT_CACL2,
)

# Saul–Wagner (Appendix A, Table 12)
_SAUL_WAGNER_A: tuple[float, ...] = (
    -7.858230,
    1.839910,
    -11.781100,
    22.670500,
    -15.939300,
    1.775160,
)
_SAUL_WAGNER_EXP: tuple[float, ...] = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)


def vapor_pressure_ratio(
    salt_mass_fraction: float,
    temperature_c: float,
    params: VaporPressureParams,
) -> float:
    """π = p_sol / p_H2O; equals brine water activity a_w."""
    xi = float(salt_mass_fraction)
    if not math.isfinite(xi):
        return float("nan")
    if xi <= 0.0:
        return 1.0
    if xi >= 1.0:
        return float("nan")
    # Past saturation the excess salt is precipitated, not dissolved: the liquid stays at
    # its saturated composition, so a_w is pinned there rather than undefined. Returning
    # nan instead made dc_w_dt clamp to 0, freezing a dry gel exactly when deliquescence
    # should drive the strongest uptake. Mirrored by jax_physics.vapor_pressure_ratio_licl
    # (_XI_SAT_LICL) -- keep the two in step or CPU and GPU diverge above saturation.
    xi = min(xi, params.xi_sat)
    theta = (float(temperature_c) + 273.15) / T_CRIT_H2O_K  # θ = T / T_c,H2O
    pi_25 = (
        1.0
        - (1.0 + (xi / params.pi6) ** params.pi7) ** params.pi8
        - params.pi9 * math.exp(-((xi - 0.1) ** 2) / 0.005)
    )
    a_term = 2.0 - (1.0 + (xi / params.pi0) ** params.pi1) ** params.pi2
    b_term = (1.0 + (xi / params.pi3) ** params.pi4) ** params.pi5 - 1.0
    pi = pi_25 * (a_term + b_term * theta)
    return max(0.0, min(1.0, float(pi)))


def water_activity_licl(
    salt_mass_fraction: float,
    temperature_c: float = 25.0,
) -> float:
    """LiCl–H2O brine water activity (Conde 2004 Table 3)."""
    return vapor_pressure_ratio(salt_mass_fraction, temperature_c, LICL_VAPOR_PRESSURE)


def equilibrium_salt_mass_fraction(
    relative_humidity: float,
    params: VaporPressureParams,
    *,
    temperature_c: float = 25.0,
    temperature_max_c: float = 150.0,
) -> float:
    """Invert a_w(ξ) = RH for brine salt mass fraction ξ."""
    rh = float(relative_humidity)
    if rh <= 0.0:
        return 1.0
    if temperature_c > temperature_max_c:
        return float("nan")
    # Bracket exhaustion, not a dilution cap: above the activity of the most dilute
    # bracketed brine the root lies below _BRACKET_LO and find_root_bracketed has
    # nothing to bisect, so _BRACKET_LO is the closest representable answer. Derived
    # per salt and temperature rather than approximated by a flat 0.99 (the true
    # threshold is 0.9930 for LiCl, 0.9961 for CaCl2).
    if rh >= vapor_pressure_ratio(_BRACKET_LO, temperature_c, params):
        return _BRACKET_LO

    def residual(xi: float) -> float:
        return rh - vapor_pressure_ratio(xi, temperature_c, params)

    hi = min(_BRACKET_HI, params.xi_sat)
    return find_root_bracketed(residual, _BRACKET_LO, hi)


def equilibrium_salt_mass_fraction_licl(
    relative_humidity: float,
    temperature_c: float = 25.0,
) -> float:
    """Equilibrium LiCl brine salt mass fraction at RH and T."""
    return equilibrium_salt_mass_fraction(
        relative_humidity,
        LICL_VAPOR_PRESSURE,
        temperature_c=temperature_c,
    )


def equilibrium_salt_mass_fraction_cacl2(
    relative_humidity: float,
    temperature_c: float = 25.0,
) -> float:
    """Equilibrium CaCl2 brine salt mass fraction at RH and T."""
    return equilibrium_salt_mass_fraction(
        relative_humidity,
        CACL2_VAPOR_PRESSURE,
        temperature_c=temperature_c,
    )


def water_vapor_pressure_pa(temperature_c: float) -> float:
    """Saul–Wagner vapour pressure of pure liquid water (Conde 2004 Appendix A)."""
    t_k = float(temperature_c) + 273.15
    if t_k <= 273.15 or t_k >= T_CRIT_H2O_K:
        return float("nan")
    tau = 1.0 - t_k / T_CRIT_H2O_K
    numer = sum(a * tau**exp for a, exp in zip(_SAUL_WAGNER_A, _SAUL_WAGNER_EXP, strict=True))
    ln_p_pc = numer / (1.0 - tau)
    return float(P_CRIT_H2O_PA * math.exp(ln_p_pc))


# --- Heat-transfer correlations (Hollands et al. 1976; Wilson Note S1) ---

STEFAN_BOLTZMANN_W_M2_K4: float = _pv("Stefan-Boltzmann constant (sigma)")
D_AIR_M2_S: float = _pv("Water-vapor-in-air diffusivity (D_air)")  # H2O in air ~25 °C (Note S1 Sh = Nu analogy)
GRAVITY_M_S2: float = _pv("Gravitational acceleration (g)")
BETA_AIR_K: float = 1.0 / _pv("Air reference temperature for thermal expansion")
NU_AIR_M2_S: float = _pv("Air kinematic viscosity (nu_air)")
PR_AIR: float = _pv("Air Prandtl number (Pr_air)")
RHO_AIR_KG_M3: float = _pv("Air density (rho_air)")
CP_AIR_J_KG_K: float = _pv("Air specific heat (cp_air)")
ALPHA_AIR_M2_S: float = K_AIR_W_M_K / (RHO_AIR_KG_M3 * CP_AIR_J_KG_K)


def parallel_plate_emissivity(eps_a: float, eps_b: float) -> float:
    """Note S1 Eq. S2 — infinite parallel plates."""
    if eps_a <= 0.0 or eps_b <= 0.0:
        return 0.0
    return 1.0 / (1.0 / eps_a + 1.0 / eps_b - 1.0)


def mass_transfer_g_from_h_conv_m_s(h_conv_w_m2_k: float) -> float:
    """Note S1 Eq. S5 (Le ≈ 1): g = h_conv · D_air / k_air."""
    if h_conv_w_m2_k <= 0.0:
        return 0.0
    return h_conv_w_m2_k * D_AIR_M2_S / K_AIR_W_M_K


def radiative_exchange_w_m2(t_hot_c: float, t_cold_c: float, *, emissivity: float = 0.9) -> float:
    t_hot_k = t_hot_c + 273.15
    t_cold_k = t_cold_c + 273.15
    return emissivity * STEFAN_BOLTZMANN_W_M2_K4 * (t_hot_k**4 - t_cold_k**4)


def hollands_vapor_gap_h_conv_w_m2_k(
    gap_m: float,
    t_hot_c: float,
    t_cold_c: float,
    *,
    tilt_deg: float = TILT_DEG,
) -> float:
    """Note S1 Eqs. S3–S4: h_conv,g = Nu·k_air/(L_g − H), with Nu from the Hollands et al.
    1976 tilted-parallel-plates correlation at the vapor-gap cavity Rayleigh number."""
    if gap_m <= 0.0:
        return 0.0
    delta_t = max(abs(t_hot_c - t_cold_c), 1e-6)
    ra = GRAVITY_M_S2 * BETA_AIR_K * delta_t * gap_m**3 / (NU_AIR_M2_S * ALPHA_AIR_M2_S)
    cos_t = max(math.cos(math.radians(tilt_deg)), 1e-6)
    ra_cos = ra * cos_t
    if ra_cos <= 0.0:
        nu = 1.0
    else:
        sin_18t_16 = math.sin(math.radians(1.8 * tilt_deg)) ** 1.6
        f1 = max(0.0, 1.0 - 1708.0 * sin_18t_16 / ra_cos)
        f2 = max(0.0, 1.0 - 1708.0 / ra_cos)
        f3 = max(0.0, (ra_cos / 5830.0) ** (1.0 / 3.0) - 1.0)
        nu = 1.0 + 1.44 * f1 * f2 + f3
    return nu * K_AIR_W_M_K / gap_m


def wind_to_h_amb_w_m2_k(wind_speed_m_s: float, *, base: float = H_AMB_W_M2_K) -> float:
    """Map 10 m wind speed to external convection coefficient (paper: ~10 at 0.5 m/s)."""
    w = max(0.0, float(wind_speed_m_s))
    return base * (0.5 + w) / 1.0


def condenser_h_conv_w_m2_k(
    h_amb: float,
    *,
    fin_area_ratio: float = FIN_AREA_RATIO,
    fin_thickness_m: float | None = None,
    fin_height_m: float | None = None,
) -> float:
    """Condenser-side convection coefficient referenced to the base plate area.

    Wilson assumes an ideally efficient fin, i.e. ``A_r * h_amb``, which is what
    both fin geometry arguments being ``None`` (the default) reproduces. Complex
    mode (B3) supplies the geometry and derates the added area by the straight-fin
    efficiency, so fin area stops paying off linearly forever.
    """
    if fin_thickness_m is None or fin_height_m is None:
        return fin_area_ratio * h_amb
    from solar_lumped.complex_model import fin_efficiency

    eta_f = fin_efficiency(
        h_amb, fin_thickness_m=fin_thickness_m, fin_height_m=fin_height_m
    )
    # Only the finned area (A_r - 1) is derated; the exposed base plate is not.
    return h_amb * (1.0 + eta_f * max(fin_area_ratio - 1.0, 0.0))


# --- Salt catalog and PAM-LiCl water-activity models for Wilson Eq. 5 ---

WATER_MOLAR_MASS_KG_MOL: float = _pv("Water molar mass (MW_w)")
GAS_CONSTANT_J_MOL_K: float = _pv("Universal gas constant (R)")
C_W_MAX_MOL_M3: float = _pv("Gel water concentration upper bound (c_w,max)")
C_W_MIN_MOL_M3: float = _pv("Gel water concentration lower bound (c_w,min)")

# The one dilution cap. a_w -> 1 is infinite dilution: c_w diverges and the a_w -> f_b
# inversion goes ill-conditioned (da_w/df_b -> 0), so an upper bound is structural, not
# cosmetic. It used to be expressed three inconsistent ways -- per-salt rh_max, an
# rh>=0.99 short-circuit, and C_W_MAX -- of which the tightest (C_W_MAX, a_w~0.948) was
# the one that actually bound. Now one number gates the isotherm, clamps the equilibrium
# inversion, and sets each salt's c_w ceiling through dilution_ceiling_c_w.
DILUTION_CAP_RH: float = _pv("Dilution cap (maximum water activity)")


@dataclass(frozen=True, slots=True)
class SaltProperties:
    name: str
    formula_weight_g_mol: float
    ions_per_formula: int
    price_usd_per_kg: float
    h_des_j_per_kg: float
    rho_solution_kg_m3: float
    default_sl: float
    rh_min: float
    rh_max: float
    # Moles of water per mole of salt in the lowest hydrate that is stable at this
    # model's desorption temperatures (gel runs ~60-120 C). This is the physical floor
    # on gel water: below it the remaining water is crystal-bound, and removing it
    # costs far more than h_fg, which Eq. 5 does not model. See hydrate_floor_c_w.
    hydrate_h2o_per_formula: float


@lru_cache(maxsize=1)
def _load_salt_catalog() -> dict[str, SaltProperties]:
    """Per-salt properties from the Salts sheet of docs/parameters.xlsx.

    Formerly two CSVs under data/materials; salt_heat_of_desorption.csv duplicated the
    catalog's h_des exactly (verified equal before the collapse) and contributed only
    its provenance notes, which now live in the sheet's Source column."""
    out: dict[str, SaltProperties] = {}
    for name, row in SALTS.items():
        out[name] = SaltProperties(
            name=name,
            formula_weight_g_mol=float(row["formula_weight_g_mol"]),
            ions_per_formula=int(row["ions_per_formula"]),
            price_usd_per_kg=float(row["price_usd_per_kg"]),
            h_des_j_per_kg=float(row["h_des_j_per_kg"]),
            rho_solution_kg_m3=float(row["rho_solution_kg_m3"]),
            default_sl=float(row["default_sl"]),
            # Both bounds are derived, not tabulated per salt: rh_min from solubility
            # (deliquescence_rh), rh_max from the one dilution cap shared by every salt.
            rh_min=deliquescence_rh(name),
            rh_max=DILUTION_CAP_RH,
            hydrate_h2o_per_formula=float(row["hydrate_h2o_per_formula"]),
        )
    return out


def get_salt(name: str) -> SaltProperties:
    catalog = _load_salt_catalog()
    if name not in catalog:
        raise KeyError(f"Unknown salt {name!r}; available: {sorted(catalog)}")
    return catalog[name]


def get_salt_price_usd_per_kg(name: str) -> float:
    return get_salt(name).price_usd_per_kg


# Shared with the JAX backend, which reads the same two rows -- these used to be a
# hardcoded tuple here and separate constants there, i.e. one clamp with two sources.
TEMPERATURE_CLAMP_C: tuple[float, float] = (
    _pv("Temperature clamp lower bound"),
    _pv("Temperature clamp upper bound"),
)


def clamp_temperature_c(temperature_c: float) -> float:
    lo, hi = TEMPERATURE_CLAMP_C
    if not math.isfinite(temperature_c):
        return 25.0
    return max(lo, min(hi, float(temperature_c)))


def saturation_vapor_pressure_pa(temperature_c: float) -> float:
    """Pure-water vapour pressure (Pa): Conde (2004) Saul–Wagner, Tetens fallback."""
    t = clamp_temperature_c(temperature_c)
    p = water_vapor_pressure_pa(t)
    if math.isfinite(p):
        return p
    return 1000.0 * 0.61078 * math.exp(17.27 * t / (t + 237.3))


def salt_molarity_from_composite(
    salt_to_polymer_ratio: float,
    hydrogel_density_kg_m3: float,
    formula_weight_g_mol: float,
) -> float:
    """Fixed salt molar concentration c_s (mol/m³ gel) in desorbed composite."""
    f_salt = salt_to_polymer_ratio / (1.0 + salt_to_polymer_ratio)
    mass_salt_kg_m3 = hydrogel_density_kg_m3 * f_salt
    return mass_salt_kg_m3 / (formula_weight_g_mol / 1000.0)


# Díaz-Marín Methods — one-pot batch in 50 mL, poured into 60 mm petri dishes.
_CHAMBER_DISH_DIAMETER_M: float = _pv("Chamber dish diameter")
_SYNTHESIS_BATCH_ML: float = _pv("Synthesis batch volume")
_PAM_LICL_STANDARD_POUR_ML: float = _pv("PAM-LiCl standard pour volume")
_PAM_LICL_2GG_CHAMBER_POUR_ML: float = _pv("PAM-LiCl 2 g/g chamber pour volume")
_LICL_BATCH_G_BY_SL: dict[int, float] = {
    1: 4.18,
    2: 8.36,
    4: 16.72,
    8: 33.44,
}
# Table S3 reference for anchoring synthesis c_s to the 4 g/g DVS dry-basis density.
_CHAMBER_CS_CALIB_SL: float = _pv("Chamber c_s calibration salt:polymer ratio")
_CHAMBER_CS_CALIB_H0_MM: float = _pv("Chamber c_s calibration hydrogel thickness")
_CHAMBER_CS_CALIB_POUR_ML: float = _PAM_LICL_STANDARD_POUR_ML


def chamber_pour_volume_ml(
    salt_to_polymer_ratio: float,
    *,
    pam_licl_chamber: bool = True,
) -> float:
    """Solution pour volume (mL) for environmental-chamber kinetics samples."""
    if pam_licl_chamber and int(round(salt_to_polymer_ratio)) == 2:
        return _PAM_LICL_2GG_CHAMBER_POUR_ML
    return _PAM_LICL_STANDARD_POUR_ML


def _chamber_c_s_from_pour_inventory(
    salt_to_polymer_ratio: float,
    h0_mm: float,
    *,
    pour_ml: float,
    formula_weight_g_mol: float,
) -> float:
    """c_s [mol/m³ gel] from LiCl mass in pour / gel volume at measured H₀."""
    sl_key = int(round(salt_to_polymer_ratio))
    if sl_key not in _LICL_BATCH_G_BY_SL:
        raise ValueError(f"unsupported PAM-LiCl salt loading for synthesis c_s: {sl_key}")
    salt_in_pour_kg = _LICL_BATCH_G_BY_SL[sl_key] * (pour_ml / _SYNTHESIS_BATCH_ML) / 1000.0
    moles = salt_in_pour_kg / (formula_weight_g_mol / 1000.0)
    area_m2 = math.pi * (_CHAMBER_DISH_DIAMETER_M / 2.0) ** 2
    vol_m3 = area_m2 * max(h0_mm * 1e-3, 1e-6)
    return moles / vol_m3


def chamber_c_s_from_synthesis(
    salt_to_polymer_ratio: float,
    h0_mm: float,
    *,
    formula_weight_g_mol: float = 42.394,
    pour_ml: float | None = None,
    calibrate_to_dvs: bool = True,
) -> float:
    """Fixed c_s for Eq. 8: poured LiCl moles spread over footprint × H₀ (SI Note S9),
    with the 4 g/g reference scaled to DRY_COMPOSITE_DENSITY for DVS consistency."""
    pour = pour_ml if pour_ml is not None else chamber_pour_volume_ml(salt_to_polymer_ratio)
    cs_synth = _chamber_c_s_from_pour_inventory(
        salt_to_polymer_ratio,
        h0_mm,
        pour_ml=pour,
        formula_weight_g_mol=formula_weight_g_mol,
    )
    if not calibrate_to_dvs:
        return cs_synth

    cs_dvs_ref = salt_molarity_from_composite(
        _CHAMBER_CS_CALIB_SL,
        DRY_COMPOSITE_DENSITY_KG_M3,
        formula_weight_g_mol,
    )
    cs_synth_ref = _chamber_c_s_from_pour_inventory(
        _CHAMBER_CS_CALIB_SL,
        _CHAMBER_CS_CALIB_H0_MM,
        pour_ml=_CHAMBER_CS_CALIB_POUR_ML,
        formula_weight_g_mol=formula_weight_g_mol,
    )
    return cs_dvs_ref * (cs_synth / cs_synth_ref)


def chamber_c_s_with_constant_density(
    salt_to_polymer_ratio: float,
    h0_mm: float,
    *,
    formula_weight_g_mol: float = 42.394,
    pour_ml: float | None = None,
) -> float:
    """``c_s`` for Eq. 8 at SI Note S7 constant 20 % RH solution density. c_s stays on the
    4 g/g calibration thickness while g/H₀ uses measured H₀ -- matches panel 5d."""
    cs = chamber_c_s_from_synthesis(
        salt_to_polymer_ratio,
        h0_mm,
        formula_weight_g_mol=formula_weight_g_mol,
        pour_ml=pour_ml,
    )
    if int(round(salt_to_polymer_ratio)) != 2:
        return cs
    if abs(h0_mm - _CHAMBER_CS_CALIB_H0_MM) < 1e-6:
        return cs
    cs_ref = chamber_c_s_from_synthesis(
        _CHAMBER_CS_CALIB_SL,
        _CHAMBER_CS_CALIB_H0_MM,
        formula_weight_g_mol=formula_weight_g_mol,
    )
    return cs_ref * (_CHAMBER_CS_CALIB_H0_MM / h0_mm)


@lru_cache(maxsize=1)
def _load_pam_licl_dvs_isotherm() -> tuple[np.ndarray, np.ndarray]:
    """Note S2 DVS isotherm: RH (%), gravimetric uptake (g water / g dry composite)."""
    path = Path(__file__).resolve().parent / "data" / "materials" / "PAM-LiCL_isotherm.csv"
    return load_two_column_csv(path)


def pam_licl_uptake_g_g_at_rh(rh_fraction: float) -> float:
    """Forward DVS isotherm: equilibrium uptake (g/g) at relative humidity."""
    rh_pct, uptake = _load_pam_licl_dvs_isotherm()
    r = max(0.0, min(100.0, float(rh_fraction) * 100.0))
    return float(np.interp(r, rh_pct, uptake))


# Dry-basis composite density = rho_composite(20% RH) / (1 + uptake(20% RH)); the wet
# density would over-count dry sorbent mass by (1 + u20) ≈ 2.26x.
DRY_COMPOSITE_DENSITY_KG_M3: float = RHO_COMPOSITE_KG_M3 / (
    1.0 + pam_licl_uptake_g_g_at_rh(0.20)
)


def pam_licl_dry_mass_kg_m2(
    h0_ref_m: float,
    *,
    dry_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3,
) -> float:
    """Dry PAM-LiCl composite mass per m² at reference thickness H₀ (DVS basis)."""
    return dry_density_kg_m3 * h0_ref_m


def pam_licl_gravimetric_uptake_g_g(
    c_w: float,
    h_m: float,
    *,
    h0_ref_m: float,
    dry_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3,
    c_s_mol_m3: float | None = None,
    formula_weight_g_mol: float | None = None,
    salt_to_polymer_ratio: float | None = None,
    salt_weight_factor: float = 1.0,
) -> float:
    """Gravimetric moisture content m_w/m_dry (g/g) per footprint, referenced to the fixed
    fabrication thickness H₀ (Note S1), not swollen H(t) -- ``h_m`` is unused. With composite
    state, dry mass uses ``formula_weight_g_mol * salt_weight_factor``; else the DVS density."""
    del h_m
    if (
        c_s_mol_m3 is not None
        and formula_weight_g_mol is not None
        and salt_to_polymer_ratio is not None
    ):
        mw_eff = formula_weight_g_mol * salt_weight_factor
        mass_salt = max(0.0, c_s_mol_m3) * h0_ref_m * mw_eff / 1000.0
        mass_polymer = mass_salt / max(salt_to_polymer_ratio, 1e-9)
        m_dry = mass_salt + mass_polymer
    else:
        m_dry = pam_licl_dry_mass_kg_m2(h0_ref_m, dry_density_kg_m3=dry_density_kg_m3)
    if m_dry <= 0.0:
        return 0.0
    mass_water = max(0.0, c_w) * h0_ref_m * WATER_MOLAR_MASS_KG_MOL
    return mass_water / m_dry


def licl_water_activity_at_brine_fraction(
    brine_salt_fraction: float,
    temperature_c: float,
) -> float:
    """LiCl brine water activity — Conde (2004) Table 3 vapour-pressure correlation."""
    t_corr = min(float(temperature_c), 150.0)
    return water_activity_licl(brine_salt_fraction, t_corr)


def licl_equilibrium_brine_salt_fraction(
    relative_humidity: float,
    temperature_c: float = 25.0,
) -> float:
    """Invert Conde (2004) LiCl isotherm: brine salt fraction at equilibrium with RH."""
    t_corr = min(float(temperature_c), 150.0)
    return equilibrium_salt_mass_fraction_licl(relative_humidity, t_corr)


def water_activity_from_c_w(
    c_w: float,
    *,
    c_s: float,
    ions_per_formula: int,
    temperature_c: float = 25.0,
    salt_name: str = "LiCl",
    formula_weight_g_mol: float = 42.394,
    salt_to_polymer_ratio: float = SALT_TO_POLYMER_RATIO_DEFAULT,
    h_m: float | None = None,
    h0_ref_m: float | None = None,
    salt_weight_factor: float = 1.0,
    blend_weights: tuple[float, ...] | None = None,
) -> float:
    """Brine a_w,s in Eq. 5 (Wilson Device); activity of water in the salt solution.

    With ``blend_weights`` (complex mode, B8) the brine is a ZSR mixture and a_w
    comes from inverting the mixing rule at the current salt mass fraction,
    instead of a single salt's closed-form isotherm.
    """
    del ions_per_formula, salt_to_polymer_ratio, h_m  # h_m unused; inventory is on H₀ basis
    if c_w <= 0.0 or c_s <= 0.0:
        return 1.0
    h_ref = h0_ref_m if h0_ref_m is not None else H0_M
    mw_eff = formula_weight_g_mol * salt_weight_factor

    if blend_weights is not None:
        from solar_lumped.complex_model import zsr_water_activity_at_brine_fraction

        mass_water = max(0.0, c_w) * h_ref * WATER_MOLAR_MASS_KG_MOL
        mass_salt = c_s * h_ref * mw_eff / 1000.0
        total = mass_salt + mass_water
        if total <= 0.0:
            return float("nan")
        aw = zsr_water_activity_at_brine_fraction(blend_weights, mass_salt / total, temperature_c)
        return aw if math.isfinite(aw) else float("nan")

    if salt_name == "LiCl":
        # Brine salt mass fraction m_s / (m_s + m_w) -- LiCl solution a_w,s (Eq. 5).
        salt_mol_m2 = c_s * h_ref
        mass_salt = salt_mol_m2 * mw_eff / 1000.0
        mass_water = max(0.0, c_w) * h_ref * WATER_MOLAR_MASS_KG_MOL
        total = mass_salt + mass_water
        f_b = 1.0 if total <= 0.0 else mass_salt / total
        aw = licl_water_activity_at_brine_fraction(f_b, temperature_c)
        return aw if math.isfinite(aw) else float("nan")

    # Brine salt mass fraction from gel water/salt molarities (mol/m³ gel).
    if not all(map(math.isfinite, (c_w, c_s, mw_eff))) or c_w < 0.0 or c_s < 0.0:
        return float("nan")
    mass_water = c_w * 18.015 / 1000.0
    mass_salt = c_s * mw_eff / 1000.0
    total = mass_water + mass_salt
    if total <= 0.0:
        return float("nan")
    f_b = float(mass_salt / total)
    aw = water_activity_at_brine_fraction(salt_name, f_b, temperature_c)
    return aw if math.isfinite(aw) else float("nan")


def equilibrium_c_w_at_rh(
    rh: float,
    *,
    c_s: float,
    ions_per_formula: int,
    temperature_c: float = 25.0,
    salt_name: str = "LiCl",
    formula_weight_g_mol: float = 42.394,
    salt_to_polymer_ratio: float = SALT_TO_POLYMER_RATIO_DEFAULT,
    h_m: float | None = None,
    h0_ref_m: float | None = None,
    blend_weights: tuple[float, ...] | None = None,
) -> float:
    """Invert a_w(RH) to c_w at reference hydrogel thickness H₀."""
    del ions_per_formula
    if rh <= 0.0:
        return 0.0
    # Clamp to the dilution cap rather than short-circuiting to a constant: above the
    # cap the isotherm has no finite answer, and the capped equilibrium *is* the ceiling
    # (that is exactly what dilution_ceiling_c_w asks for), so returning it keeps the
    # bound self-consistent instead of pinning to an unrelated C_W_MAX.
    rh = min(float(rh), DILUTION_CAP_RH)

    h_ref = h0_ref_m if h0_ref_m is not None else H0_M
    del h_m  # inventory referenced to H₀ (see pam_licl_gravimetric_uptake_g_g)

    if blend_weights is not None:
        # ZSR is posed a_w -> composition, so the forward direction is the direct
        # evaluation here -- no root solve needed on this path.
        from solar_lumped.complex_model import zsr_brine_state

        f_b, _ions, _mw = zsr_brine_state(blend_weights, rh, temperature_c)
        if not math.isfinite(f_b):
            return C_W_MIN_MOL_M3
    elif salt_name == "LiCl":
        f_b = licl_equilibrium_brine_salt_fraction(rh, temperature_c)
    else:
        f_b = equilibrate_salt_mf(salt_name, rh, temperature_c)
        if not math.isfinite(f_b):
            return C_W_MIN_MOL_M3

    salt_mol_m2 = c_s * h_ref
    mass_salt = salt_mol_m2 * formula_weight_g_mol / 1000.0
    if f_b <= 0.0:
        return C_W_MAX_MOL_M3
    mass_water = mass_salt * (1.0 - f_b) / f_b
    if mass_water <= 0.0:
        return C_W_MIN_MOL_M3
    # No C_W_MAX clamp: rh is capped above, so c_w cannot exceed the dilution ceiling.
    return mass_water / (h_ref * WATER_MOLAR_MASS_KG_MOL)


# Methods: hydrogel cast at equilibrium with ~20% RH ambient.
FABRICATION_EQUILIBRIUM_RH: float = _pv("Fabrication equilibrium RH")


def equilibrium_c_w_from_dvs_at_rh(
    rh: float,
    *,
    h_m: float,
    h0_ref_m: float,
    dry_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3,
) -> float:
    """Paper Note S2: DVS isotherm sets sorbent equilibrium uptake at ambient RH."""
    del h_m  # inventory referenced to H₀ (see pam_licl_gravimetric_uptake_g_g)
    if rh <= 0.0:
        return C_W_MIN_MOL_M3
    u = pam_licl_uptake_g_g_at_rh(rh)
    m_dry = dry_density_kg_m3 * h0_ref_m
    mass_water_kg_m2 = u * m_dry
    c_w = mass_water_kg_m2 / (h0_ref_m * WATER_MOLAR_MASS_KG_MOL)
    return max(C_W_MIN_MOL_M3, min(C_W_MAX_MOL_M3, c_w))


def desorption_water_activity(
    condenser_temperature_c: float,
    gel_temperature_c: float,
) -> float:
    """Effective desorption water activity at sealed condenser / sun-heated gel equilibrium."""
    p_sat_cond = saturation_vapor_pressure_pa(condenser_temperature_c)
    p_sat_gel = saturation_vapor_pressure_pa(gel_temperature_c)
    if p_sat_gel <= 0.0 or not math.isfinite(p_sat_gel) or not math.isfinite(p_sat_cond):
        return float("nan")
    t_cond_k = condenser_temperature_c + 273.15
    t_gel_k = gel_temperature_c + 273.15
    if t_cond_k <= 0.0 or t_gel_k <= 0.0:
        return float("nan")
    return p_sat_cond * t_gel_k / (p_sat_gel * t_cond_k)


def fabrication_c_w_initial(
    *,
    salt_name: str,
    salt_to_polymer_ratio: float,
    hydrogel_thickness_m: float,
    hydrogel_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3,
    formula_weight_g_mol: float | None = None,
    blend_weights: tuple[float, ...] | None = None,
    use_dvs_cap: bool = True,
) -> float:
    """Initial gel water state after fabrication at ~20% RH ambient.

    A ZSR blend (complex mode, B8) never takes the PAM-LiCl DVS branch: that
    isotherm was measured on the LiCl composite and says nothing about a mixed
    brine. It also casts at the blend's own clamped fabrication RH, since CaCl2 and
    MgCl2 have no brine at 20% RH and would otherwise start the gel bone dry --
    which silently zeroed a blend's whole year.
    """
    h0 = hydrogel_thickness_m
    if blend_weights is not None:
        from solar_lumped.complex_model import (
            clamp_reference_rh,
            zsr_effective_formula_weight_g_mol,
        )

        fw = (
            formula_weight_g_mol
            if formula_weight_g_mol is not None
            else zsr_effective_formula_weight_g_mol(
                blend_weights, reference_rh=FABRICATION_EQUILIBRIUM_RH
            )
        )
        return equilibrium_c_w_at_rh(
            clamp_reference_rh(blend_weights, FABRICATION_EQUILIBRIUM_RH),
            c_s=salt_molarity_from_composite(salt_to_polymer_ratio, hydrogel_density_kg_m3, fw),
            ions_per_formula=2,  # unused on this path; ZSR carries effective ions
            salt_name=salt_name,
            formula_weight_g_mol=fw,
            salt_to_polymer_ratio=salt_to_polymer_ratio,
            h_m=h0,
            h0_ref_m=h0,
            blend_weights=blend_weights,
        )
    if salt_name == "LiCl" and use_dvs_cap:
        return equilibrium_c_w_from_dvs_at_rh(
            FABRICATION_EQUILIBRIUM_RH,
            h_m=h0,
            h0_ref_m=h0,
            dry_density_kg_m3=hydrogel_density_kg_m3,
        )
    s = get_salt(salt_name)
    fw = formula_weight_g_mol if formula_weight_g_mol is not None else s.formula_weight_g_mol
    c_s = salt_molarity_from_composite(
        salt_to_polymer_ratio,
        hydrogel_density_kg_m3,
        fw,
    )
    # Same clamp as the blend branch, for the same reason: CaCl2 (DRH 0.35) and
    # MgCl2 (0.33) have no brine at Wilson's 20% RH casting condition, and leaving
    # it unclamped starts the gel dry and silently zeroes the year.
    return equilibrium_c_w_at_rh(
        min(max(FABRICATION_EQUILIBRIUM_RH, s.rh_min), s.rh_max),
        c_s=c_s,
        ions_per_formula=s.ions_per_formula,
        salt_name=salt_name,
        formula_weight_g_mol=fw,
        salt_to_polymer_ratio=salt_to_polymer_ratio,
        h_m=h0,
        h0_ref_m=h0,
    )


# --- Equilibrium brine isotherms for NaCl, LiCl, CaCl2, and MgCl2 ---


def _aw_polynomial(salt_fraction: float, coeffs: tuple[float, ...]) -> float:
    """a_w(ξ) = Σ coeffs[k]·ξ^k (coeffs[0] first, i.e. increasing powers)."""
    if not (0.0 <= salt_fraction < 1.0) or not math.isfinite(salt_fraction):
        return float("nan")
    a_w = 0.0
    for k, coeff in enumerate(coeffs):
        a_w += coeff * (salt_fraction**k)
    return float(a_w)


# a_w(ξ) polynomial coefficients (increasing powers of ξ): forward isotherm, and inverted
# via find_root_bracketed for the equilibrium brine fraction.
_NACL_AW_COEFFS: tuple[float, ...] = (0.9998, -0.5597, -0.332, -5.545, 5.863)
_MGCL2_AW_COEFFS: tuple[float, ...] = (1.16231287, -4.86704441, 38.21982328, -153.67496570, 186.32487108)


def mf_NaCl(relative_humidity: float) -> float:
    """Equilibrium brine salt fraction for NaCl at 25°C."""
    if not (0.0 < relative_humidity < 1.0):
        return float("nan")
    return find_root_bracketed(
        lambda xi: relative_humidity - _aw_polynomial(xi, _NACL_AW_COEFFS),
        0.0116,
        0.264,
    )


def mf_LiCl(relative_humidity: float, temperature_c: float = 25.0) -> float:
    """Equilibrium brine salt fraction for LiCl (Conde 2004)."""
    if not (0.0 < relative_humidity < 1.0) or temperature_c > 150.0:
        return float("nan")
    return equilibrium_salt_mass_fraction_licl(relative_humidity, temperature_c)


def mf_CaCl2(relative_humidity: float, temperature_c: float = 25.0) -> float:
    """Equilibrium brine salt fraction for CaCl2 (Conde 2004)."""
    if not (0.0 < relative_humidity < 1.0) or temperature_c > 100.0:
        return float("nan")
    return equilibrium_salt_mass_fraction_cacl2(relative_humidity, temperature_c)


def mf_MgCl2(relative_humidity: float) -> float:
    """Equilibrium brine salt fraction for MgCl2 (polynomial fit)."""
    if not (0.0 < relative_humidity < 1.0):
        return float("nan")
    return find_root_bracketed(
        lambda xi: relative_humidity - _aw_polynomial(xi, _MGCL2_AW_COEFFS),
        0.01,
        0.75,
        scan=True,
        n_intervals=19,
    )


_isotherm_by_salt: dict[str, Callable[[float, float], float]] = {
    "NaCl": lambda rh, t: mf_NaCl(rh),
    "LiCl": lambda rh, t: mf_LiCl(rh, t),
    "CaCl2": lambda rh, t: mf_CaCl2(rh, t),
    "MgCl2": lambda rh, t: mf_MgCl2(rh),
}


def equilibrate_salt_mf(
    salt_name: str,
    relative_humidity: float,
    temperature_c: float = 25.0,
) -> float:
    """Return equilibrium brine salt mass fraction, or nan if outside the salt's RH range."""
    rec = get_salt(salt_name)
    if rec.name not in _isotherm_by_salt:
        return float("nan")
    if not (rec.rh_min <= relative_humidity <= rec.rh_max):
        return float("nan")
    return float(_isotherm_by_salt[rec.name](relative_humidity, temperature_c))


def _water_activity_at_brine_fraction_by_name(
    salt_name: str,
    brine_salt_fraction: float,
    temperature_c: float,
) -> float:
    """Forward isotherm dispatch keyed on the salt name only.

    Deliberately does not call ``get_salt``: the catalog loader derives each salt's
    deliquescence RH through here, and looking the record up would re-enter the loader.
    """
    f = float(brine_salt_fraction)
    if not (0.0 <= f < 1.0) or not math.isfinite(f):
        return float("nan")
    if salt_name == "NaCl":
        return _aw_polynomial(f, _NACL_AW_COEFFS)
    if salt_name == "MgCl2":
        return _aw_polynomial(f, _MGCL2_AW_COEFFS)
    if salt_name == "LiCl":
        if temperature_c > 150.0:
            return float("nan")
        return water_activity_licl(f, min(temperature_c, 150.0))
    if salt_name == "CaCl2":
        if temperature_c > 100.0:
            return float("nan")
        return vapor_pressure_ratio(f, temperature_c, CACL2_VAPOR_PRESSURE)
    return float("nan")


@lru_cache(maxsize=1024)
def deliquescence_rh(salt_name: str, temperature_c: float = 25.0) -> float:
    """Deliquescence RH: the brine water activity at saturation, at this temperature.

    Derived, not tabulated. Below it no brine exists at equilibrium, so the salt stays
    solid and cannot take up water -- which also makes it the floor a_w pins to once
    the excess salt has precipitated.

    Temperature matters: the saturation *mass fraction* is held at its solubility value
    but the *activity* at that composition rises with temperature (LiCl 0.081 at 0 C ->
    0.182 at 100 C), so a gel desorbing at 80 C pins ~50% higher than the 25 C value.
    ``SaltProperties.rh_min`` keeps the 25 C default as the catalog reference.

    NaCl and MgCl2 use temperature-independent polynomial isotherms, so their
    deliquescence point does not move -- a limit of those correlations, not of this.
    """
    return _water_activity_at_brine_fraction_by_name(
        salt_name, saturation_brine_salt_fraction(salt_name), float(temperature_c)
    )


def water_activity_at_brine_fraction(
    salt_name: str,
    brine_salt_fraction: float,
    temperature_c: float = 25.0,
) -> float:
    """Forward isotherm: brine water activity at salt mass fraction and temperature."""
    return _water_activity_at_brine_fraction_by_name(
        get_salt(salt_name).name, brine_salt_fraction, temperature_c
    )


# --- Wilson et al. 2025 Eqs. 1, 3, 4 — steady absorber, glass, and gel temperatures ---


@dataclass(frozen=True, slots=True)
class ThermalState:
    t_gel_c: float
    t_abs_c: float
    t_glass_c: float
    h_conv_g: float
    m_des_kg_s_m2: float
    # Complex mode (B2) only: outer pane temperature of a 2-pane stack. None for
    # the single-pane / uncovered system, where t_glass_c is the whole story.
    t_glass_outer_c: float | None = None


@dataclass(frozen=True, slots=True)
class SystemThermalParams:
    insulation_gap_m: float = L_INS_M
    vapor_gap_m: float = L_G_M
    eps_abs: float = EPS_ABS
    tau_glass: float = TAU_GLASS
    eps_gel: float = EPS_GEL
    eps_al: float = EPS_AL
    # Real absorber/glass IR emissivities for the modified Eqs. 3/4 radiative terms.
    # Case 2 (selective surface) is the default; pass both as 1.0 (or None) for
    # Case 1's original Wilson blackbody/cavity approximation -- see _residuals.
    eps_abs_ir: float | None = EPS_ABS_IR_CASE2
    eps_glass_ir: float | None = EPS_GLASS_IR_CASE2
    tilt_deg: float = TILT_DEG
    h_des_j_per_kg: float = H_DES_J_PER_KG
    has_glass: bool = True
    # Complex mode (B2). n_glazing_panes=1 (default) is Wilson's single cover and
    # leaves _residuals a 3-unknown solve; 2 adds an outer pane as a real 4th
    # unknown; 0 is the uncovered system (equivalent to has_glass=False).
    n_glazing_panes: int = 1
    # An evacuated inter-pane gap suppresses conduction but not radiation.
    evacuated_gap: bool = False


def _residuals(
    x: np.ndarray,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    m_des_kg_s_m2: float,
    h_amb: float,
    params: SystemThermalParams,
    vapor_gap_effective_m: float,
    h_m: float,
) -> np.ndarray:
    t_gel, t_abs, t_glass = float(x[0]), float(x[1]), float(x[2])
    # Complex mode (B2): a second pane carries its own temperature as a 4th unknown.
    two_pane = params.has_glass and params.n_glazing_panes >= 2
    t_glass_outer = float(x[3]) if two_pane else t_glass
    u_gel = u_gel_w_m2_k(h_m)
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

    if params.eps_abs_ir is not None and params.eps_glass_ir is not None:
        # Modified Eqs. 3/4: eps_ag from the parallel-plate formula (as eps_gel/eps_al),
        # eps_ga is the glass IR emissivity directly (it radiates to ambient, not a cavity).
        eps_ag = parallel_plate_emissivity(params.eps_abs_ir, params.eps_glass_ir)
        eps_ga = params.eps_glass_ir
    else:
        # Absorber→glass: Wilson Eq. 4 writes σ(T_abs⁴ − T_glass⁴) without an
        # explicit emissivity factor (cavity / blackbody approximation).
        eps_ag = 1.0
        # Glass→surroundings: Wilson Eq. 3 writes σ(T_glass⁴ − T_amb⁴) with no emissivity
        # factor (blackbody in the IR).
        eps_ga = 1.0

    if not params.has_glass:
        q_rad_abs_amb = radiative_exchange_w_m2(t_abs, t_amb_c, emissivity=params.eps_abs)
        r4 = (
            params.eps_abs * q_solar_w_m2
            - h_amb * (t_abs - t_amb_c)
            - q_rad_abs_amb
            - u_gel * (t_abs - t_gel)
        )
        r3 = t_glass - t_amb_c
    else:
        gap_m = params.insulation_gap_m
        # An evacuated cover assembly (B2) kills gas conduction across every
        # glazing gap but leaves radiation untouched -- that asymmetry is exactly
        # why it buys anything, and why it only pays off with a low-eps absorber.
        cond_coeff = 0.0 if (gap_m <= 0.0 or params.evacuated_gap) else K_AIR_W_M_K / gap_m
        q_cond_ag = cond_coeff * (t_abs - t_glass)
        q_rad_ag = radiative_exchange_w_m2(t_abs, t_glass, emissivity=eps_ag)
        r4 = (
            params.eps_abs * params.tau_glass * q_solar_w_m2
            - q_cond_ag
            - q_rad_ag
            - u_gel * (t_abs - t_gel)
        )
        if two_pane:
            # Inner pane exchanges with the outer pane, which alone sees ambient.
            eps_pane_pane = parallel_plate_emissivity(eps_ga, eps_ga)
            q_cond_io = cond_coeff * (t_glass - t_glass_outer)
            q_rad_io = radiative_exchange_w_m2(t_glass, t_glass_outer, emissivity=eps_pane_pane)
            q_rad_oa = radiative_exchange_w_m2(t_glass_outer, t_amb_c, emissivity=eps_ga)
            r3 = q_cond_ag + q_rad_ag - q_cond_io - q_rad_io
            r5 = q_cond_io + q_rad_io - h_amb * (t_glass_outer - t_amb_c) - q_rad_oa
            return np.array([r1, r3, r4, r5], dtype=float)
        q_rad_ga = radiative_exchange_w_m2(t_glass, t_amb_c, emissivity=eps_ga)
        r3 = q_cond_ag + q_rad_ag - h_amb * (t_glass - t_amb_c) - q_rad_ga

    return np.array([r1, r3, r4], dtype=float)


def solve_steady_thermal(
    *,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    m_des_kg_s_m2: float,
    h_amb: float,
    params: SystemThermalParams,
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

    # A 2-pane stack (B2) adds the outer pane as a 4th unknown; it starts a little
    # cooler than the inner pane, which is the physically correct ordering.
    two_pane = params.has_glass and params.n_glazing_panes >= 2
    x0 = [t_gel0, t_abs0, t_glass0]
    if two_pane:
        x0.append(clamp_temperature_c(0.5 * (t_glass0 + t_amb_c)))

    sol = root(
        _residuals,
        x0=np.array(x0),
        args=(t_cond_c, t_amb_c, q_solar_w_m2, m_des_kg_s_m2, h_amb, params, gap_m, h_m),
        method="hybr",
        tol=1e-8,
    )
    if not sol.success:
        t_gel, t_abs, t_glass = t_gel0, t_abs0, t_glass0
        t_glass_outer = x0[3] if two_pane else None
    else:
        t_gel = clamp_temperature_c(float(sol.x[0]))
        t_abs = clamp_temperature_c(float(sol.x[1]))
        t_glass = clamp_temperature_c(float(sol.x[2]))
        t_glass_outer = clamp_temperature_c(float(sol.x[3])) if two_pane else None

    h_conv_g = hollands_vapor_gap_h_conv_w_m2_k(
        gap_m, t_gel, t_cond_c, tilt_deg=params.tilt_deg
    )
    return ThermalState(
        t_gel_c=t_gel,
        t_abs_c=t_abs,
        t_glass_c=t_glass,
        h_conv_g=h_conv_g,
        m_des_kg_s_m2=m_des_kg_s_m2,
        t_glass_outer_c=t_glass_outer,
    )


def thermal_residual_norm(
    *,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    m_des_kg_s_m2: float,
    h_amb: float,
    params: SystemThermalParams,
    h_m: float = H0_M,
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
    x = [state.t_gel_c, state.t_abs_c, state.t_glass_c]
    if state.t_glass_outer_c is not None:
        x.append(state.t_glass_outer_c)
    r = _residuals(
        np.array(x),
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


# --- Wilson et al. 2025 Eqs. 5–6 — convection-limited mass transfer ---

MassTransferPhase = Literal["absorption", "desorption"]


def _absorption_effective_water_activity(
    c_w: float,
    *,
    t_gel_c: float,
    params: MassTransferParams,
    h_m: float,
) -> float:
    """Composite gel a_w for Eq. 5 during open absorption: LiCl uses brine activity plus
    the PAM-LiCl DVS cap (Note S2); other salts use brine only."""
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
        blend_weights=params.blend_weights,
    )
    # The DVS cap is a measurement of *PAM-LiCl* specifically (Note S2): it applies
    # only to the pure-LiCl composite, never to a ZSR blend (whose polymer-bound
    # uptake was never characterized), and complex mode disables it outright via
    # use_dvs_cap so the whole simplex rests on one brine model.
    if params.salt_name != "LiCl" or params.blend_weights is not None or not params.use_dvs_cap:
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
    rh_pct, uptake = _load_pam_licl_dvs_isotherm()
    if u <= float(uptake[0]):
        aw_dvs = max(0.0, float(rh_pct[0]) / 100.0)
    elif u >= float(uptake[-1]):
        aw_dvs = min(1.0, float(rh_pct[-1]) / 100.0)
    else:
        aw_dvs = max(0.0, min(1.0, float(np.interp(u, uptake, rh_pct)) / 100.0))
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
        blend_weights=params.blend_weights,
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
    # Gel-water bounds (mol/m³), both salt- and blend-dependent so neither can be a
    # module constant: hydrate_floor_c_w below, dilution_ceiling_c_w above.
    c_w_min_mol_m3: float
    c_w_max_mol_m3: float
    salt_name: str = "LiCl"
    formula_weight_g_mol: float = 42.394
    salt_to_polymer_ratio: float = SALT_TO_POLYMER_RATIO_DEFAULT
    salt_weight_factor: float = 1.0
    # Complex mode (B8): ZSR molality weights over complex_model.ZSR_SALTS. None
    # (default) keeps the single-salt path, so gpu_sweep and every existing caller
    # are untouched.
    blend_weights: tuple[float, ...] | None = None
    # The PAM-LiCl DVS cap (Note S2) is a measurement of the LiCl composite only.
    # Complex mode switches it off so the brine model is self-consistent across the
    # whole blend simplex -- otherwise the pure-LiCl corner sits on a cliff that a
    # GP would chase for a modeling reason rather than a physical one.
    use_dvs_cap: bool = True


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
    if gap_m < VAPOR_GAP_TRANSPORT_MIN_M:  # Wilson's ~7 mm thermobuoyancy / transport limit
        return 0.0
    h_conv = hollands_vapor_gap_h_conv_w_m2_k(
        gap_m, t_gel_c, t_cond_c, tilt_deg=params.tilt_deg
    )
    return mass_transfer_g_from_h_conv_m_s(h_conv)


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


def _mass_transfer_rate_terms(
    c_w: float,
    *,
    t_gel_c: float,
    c_r: float,
    params: MassTransferParams,
    h_m: float,
    phase: MassTransferPhase,
    t_cond_c: float | None,
) -> tuple[float, float, float, float]:
    """Shared (T_k, p_sat, g, driving) terms behind Eqs. 5-6 (dc_w/dt, dH/dt)."""
    t_k = max(t_gel_c + 273.15, 200.0)
    p_sat = saturation_vapor_pressure_pa(t_gel_c)
    g = mass_transfer_g_m_s(phase=phase, params=params, h_m=h_m, t_gel_c=t_gel_c, t_cond_c=t_cond_c)
    driving = _mass_transfer_driving_force(
        c_w,
        t_gel_c=t_gel_c,
        c_r=c_r,
        params=params,
        h_m=h_m,
        phase=phase,
    )
    return t_k, p_sat, g, driving


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
    t_k, p_sat, g, driving = _mass_transfer_rate_terms(
        c_w, t_gel_c=t_gel_c, c_r=c_r, params=params, h_m=h_m, phase=phase, t_cond_c=t_cond_c
    )
    if not math.isfinite(driving):
        return 0.0
    rate = (g / params.h0_ref_m) * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving
    if not math.isfinite(rate):
        return 0.0
    if c_w >= params.c_w_max_mol_m3 and rate > 0.0:
        return 0.0
    if c_w <= params.c_w_min_mol_m3 and rate < 0.0:
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
    """Eq. 6 hydrogel thickness rate dH/dt = g·(MW/ρ_sol)·(p_sat/RT)·driving (m/s), i.e.
    dc_w/dt·(MW·H₀/ρ_sol), so H and c_w evolve on the same timescale."""
    t_k, p_sat, g, driving = _mass_transfer_rate_terms(
        c_w, t_gel_c=t_gel_c, c_r=c_r, params=params, h_m=h_m, phase=phase, t_cond_c=t_cond_c
    )
    return g * WATER_MOLAR_MASS_KG_MOL / params.rho_solution_kg_m3 * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving


def hydrate_floor_c_w(
    *,
    c_s_mol_m3: float,
    salt_name: str,
    blend_weights: tuple[float, ...] | None = None,
) -> float:
    """Lowest physically reachable gel water concentration (mol/m³), on the H₀ basis.

    Eq. 5's rate depends on c_w only through a_w, and a_w is floored at the salt's
    deliquescence RH -- correct for the brine (below DRH the salt precipitates and a
    saturated solution's activity stops falling), but it leaves the rate with no
    knowledge of how much water is actually left. Desorption therefore never
    self-terminates for a salt whose DRH sits above the condenser's concentration
    ratio, and the ODE will drive c_w through zero. The bound that is missing is
    chemical, not numerical: once every remaining water molecule is crystal-bound as
    ``salt·nH₂O``, removing more is a dehydration reaction costing far more than
    h_fg, which this model does not represent.

    So the floor is n·c_s, with n blend-weighted over ZSR_SALTS. That replaces the
    single global C_W_MIN_MOL_M3, which corresponds to 0.01-0.03 water per salt --
    roughly 100x drier than any hydrate, i.e. no constraint at all.
    """
    if blend_weights is None:
        n_eff = get_salt(salt_name).hydrate_h2o_per_formula
    else:
        from solar_lumped.complex_model import ZSR_SALTS, normalized_blend_weights

        w = normalized_blend_weights(blend_weights)
        n_eff = float(
            sum(wi * get_salt(name).hydrate_h2o_per_formula for name, wi in zip(ZSR_SALTS, w))
        )
    return max(0.0, n_eff * float(c_s_mol_m3))


def drh_floor_c_w(
    *,
    c_s_mol_m3: float,
    salt_name: str,
    formula_weight_g_mol: float,
    blend_weights: tuple[float, ...] | None = None,
    temperature_c: float = 25.0,
) -> float:
    """Alternative lower bound to :func:`hydrate_floor_c_w`: stop at brine saturation.

    The equilibrium c_w at the deliquescence RH -- the water still held when the brine
    reaches its solubility limit and any further drying precipitates solid salt. That is
    where the Conde/ZSR activity model stops being a description of a real solution, so
    it is the conservative place to stop: strictly wetter than the hydrate floor (a
    saturated LiCl brine holds ~2.8 H2O per LiCl vs. 1 in the monohydrate), ending
    desorption earlier and yielding less. Selected by ``SystemConfig(c_w_floor_mode="drh")``.

    Taken as the saturation *composition* rather than by inverting a_w at the DRH: the
    DRH is by construction the activity at saturation, so the inversion is being asked
    for a root sitting exactly on its own bracket edge and returns NaN. The saturated
    brine fraction is the same answer, directly.

    Composition-only, so temperature enters solely through the blend window; a single
    salt's solubility is not temperature-resolved in this model.
    """
    if blend_weights is None:
        f_b = saturation_brine_salt_fraction(salt_name)
    else:
        from solar_lumped.complex_model import blend_water_activity_window, zsr_brine_state

        aw_lo, _hi = blend_water_activity_window(blend_weights, temperature_c)
        f_b, _ions, _mw = zsr_brine_state(blend_weights, aw_lo, temperature_c)
    if not (0.0 < f_b <= 1.0):
        return float("nan")
    # Same H₀-basis algebra as equilibrium_c_w_at_rh, with H₀ cancelling out.
    mass_salt_per_h = c_s_mol_m3 * formula_weight_g_mol / 1000.0
    return mass_salt_per_h * (1.0 - f_b) / f_b / WATER_MOLAR_MASS_KG_MOL


def dilution_ceiling_c_w(
    *,
    c_s_mol_m3: float,
    salt_name: str,
    formula_weight_g_mol: float,
    blend_weights: tuple[float, ...] | None = None,
    temperature_c: float = 25.0,
) -> float:
    """Highest physically modelled gel water concentration (mol/m³), on the H₀ basis.

    The upper mirror of :func:`hydrate_floor_c_w`: the equilibrium c_w at
    ``DILUTION_CAP_RH``. Beyond the cap the brine is so dilute that a_w -> 1, c_w
    diverges, and the a_w -> f_b inversion loses conditioning, so the model has
    nothing meaningful to say. Per salt and blend, because the same a_w corresponds
    to a different water inventory for each.
    """
    return equilibrium_c_w_at_rh(
        DILUTION_CAP_RH,
        c_s=c_s_mol_m3,
        ions_per_formula=0,
        temperature_c=temperature_c,
        salt_name=salt_name,
        formula_weight_g_mol=formula_weight_g_mol,
        blend_weights=blend_weights,
    )


def m_des_kg_s_m2_from_state(
    c_w: float,
    h_m: float,
    dc_w_dt_val: float,
    dH_dt_val: float,
) -> float:
    """Desorption flux (kg/m²/s) from the Eqs. 5–6 inventory N = c_w·H:
    ṁ = -MW·(dc_w/dt·H + c_w·dH/dt); Note S1's -dc_w/dt·MW·H₀ is its dH/dt ≈ 0 limit."""
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


# --- Sorbent interface: PAM-salt hydrogel; PhaseResult.c_w stores mol/m³. ---


def evaluate_mass_rates(
    *,
    loading: float,
    h_m: float,
    t_gel_c: float,
    t_cond_c: float | None,
    rh: float,
    phase: str,
    mass: MassTransferParams,
    config: SystemConfig,
    vapor_gap_m: float,
) -> tuple[float, float, float]:
    """Return (dloading/dt, dH/dt, m_des_kg_s_m2)."""
    if phase == "absorption":
        c_r = concentration_ratio_absorption(rh)
        dc = dc_w_dt(
            loading,
            t_gel_c=t_gel_c,
            c_r=c_r,
            params=mass,
            h_m=h_m,
            phase="absorption",
        )
        dh = dH_dt(
            loading,
            t_gel_c=t_gel_c,
            c_r=c_r,
            params=mass,
            h_m=h_m,
            phase="absorption",
        )
        if h_m <= mass.h0_ref_m + 1e-12:
            dh = max(0.0, dh)
        return dc, dh, 0.0

    assert t_cond_c is not None
    c_r = concentration_ratio_desorption(t_gel_c, t_cond_c)
    dc = dc_w_dt(
        loading,
        t_gel_c=t_gel_c,
        c_r=c_r,
        params=mass,
        h_m=h_m,
        phase="desorption",
        t_cond_c=t_cond_c,
    )
    dh = dH_dt(
        loading,
        t_gel_c=t_gel_c,
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
    m_des = m_des_kg_s_m2_from_dc_w(dc, h0_ref_m=mass.h0_ref_m)
    return dc, dh, m_des


def inventory_label(config: SystemConfig) -> str:
    return "gel"


def inventory_ylabel(config: SystemConfig) -> str:
    return "Water in gel (L/m²)"


def inventory_prefix(config: SystemConfig) -> str:
    return "water_in_gel"


def initial_loading(config: SystemConfig) -> float:
    # Route the resolved blend (None for single-salt, including complex mode's
    # single-salt corners) so fabrication state and transport agree on the brine.
    mass = config.mass_params()
    return fabrication_c_w_initial(
        salt_name=mass.salt_name,
        salt_to_polymer_ratio=config.salt_to_polymer_ratio,
        hydrogel_thickness_m=config.hydrogel_thickness_m,
        hydrogel_density_kg_m3=config.hydrogel_density_kg_m3,
        blend_weights=mass.blend_weights,
        use_dvs_cap=mass.use_dvs_cap,
    )


def water_in_gel_l_m2(
    c_w: float,
    h_m: float,
    *,
    h0_ref_m: float = H0_M,
    dvs_basis: bool = True,
) -> float:
    """Water in gel (L/m²). Paper Fig. S1D uses DVS gravimetric basis (g/g × m_dry)."""
    if dvs_basis:
        u = pam_licl_gravimetric_uptake_g_g(c_w, h_m, h0_ref_m=h0_ref_m)
        return u * pam_licl_dry_mass_kg_m2(h0_ref_m)
    return max(0.0, c_w) * h_m * WATER_MOLAR_MASS_KG_MOL


def c_w_from_water_in_gel_l_m2(water_l_m2: float, h_m: float) -> float:
    """Invert water-in-gel inventory to uniform c_w (mol/m³) at thickness h_m."""
    if h_m <= 0.0:
        return 0.0
    return max(0.0, water_l_m2) / (h_m * WATER_MOLAR_MASS_KG_MOL)


def water_in_sorbent_l_m2(loading: float, h_m: float, *, config: SystemConfig) -> float:
    return water_in_gel_l_m2(loading, h_m, h0_ref_m=config.hydrogel_thickness_m)

