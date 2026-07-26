"""Device physics: geometry/material constants, brine/salt thermodynamics, heat-transfer
correlations, gel-fluid HX thermal balance, mass transfer, and sorbent (hydrogel/MOF) models.

Consolidated from the former physics/{table_s3, device_defaults, conde2004, correlations,
salt_properties, brine_equilibrium, device_balances, mass_transfer, adsorbent, sorbent}.py.
Section headers below mark each former module's boundary for traceability.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.optimize import root

from waste_heat_lumped.utils import find_root_bracketed

if TYPE_CHECKING:
    from waste_heat_lumped.simulation import DeviceConfig


# =============================================================================
# Table S3 / Note S1 device parameters (Wilson & Diaz-Marin Device 2025)
# =============================================================================

H0_M: float = 0.004  # hydrogel reference thickness H₀ (m)
L_G_M: float = 0.04  # vapor gap L_g (m)
L_INS_M: float = 0.005  # insulation gap L_ins (m)
L_C_M: float = 0.005  # condenser aluminum plate thickness L_c (m)
L_GLASS_IN: float = 0.125  # cover glass thickness (in)
L_GLASS_M: float = L_GLASS_IN * 0.0254
L_AL_STACK_M: float = L_C_M  # aluminum in gel–absorber stack (Table S3 L_c)
L_SILICONE_M: float = 0.001  # silicone coating (m)

# Wilson §2.2 / Note S1: thermobuoyancy and mass transport inhibited below ~7 mm gap
VAPOR_GAP_TRANSPORT_MIN_M: float = 0.007

# Materials / transport
G_CHAMBER_M_S: float = 0.015  # g_chamber (m/s)
RHO_SOL_KG_M3: float = 1250.2  # ρ_sol brine solution density (kg/m³)
RHO_COMPOSITE_KG_M3: float = 1250.2  # composite density at fabrication (25 °C, 20% RH), Table S3
H_DES_J_PER_KG: float = 2.32e6  # h_des (J/kg)
H_FG_J_PER_KG: float = 2.256e6  # h_fg condensation (J/kg)
K_AIR_W_M_K: float = 0.0286  # k_air (W/m·K)
K_AL_W_M_K: float = 167.0  # k_al (W/m·K) — Table S3
K_SILICONE_W_M_K: float = 0.2  # k_silicone (W/m·K)
K_GEL_W_M_K: float = 0.6  # k_w hydrogel (W/m·K) — Table S3
K_GLASS_W_M_K: float = 1.2  # k_glass (W/m·K)
RHO_AL_KG_M3: float = 2700.0
CP_AL_J_KG_K: float = 900.0
RHO_GLASS_KG_M3: float = 2230.0  # borosilicate cover
CP_GLASS_J_KG_K: float = 830.0
CP_GEL_J_KG_K: float = 3500.0  # hydrated PAM-LiCl composite (water-dominated)

# Optical / radiative
EPS_GEL: float = 1.0
EPS_AL: float = 0.05
EPS_ABS: float = 0.95
EPS_GLASS: float = 0.9  # not used in Wilson Eqs 3/4 (blackbody IR); reserved
TAU_GLASS: float = 0.9

# Device orientation / condenser fins
TILT_DEG: float = 30.0
FIN_AREA_RATIO: float = 7.1  # A_r

# Backward-compatible aliases
L_AL_M: float = L_C_M


def u_gel_w_m2_k(h_m: float) -> float:
    """Note S1 cumulative gel–absorber conductance (series resistances).

    1/U_gel = L_al/k_al + L_silicone/k_silicone + H(t)/k_hydrogel
    """
    h = max(float(h_m), H0_M * 0.25)
    resistance = (
        L_AL_STACK_M / K_AL_W_M_K
        + L_SILICONE_M / K_SILICONE_W_M_K
        + h / K_GEL_W_M_K
    )
    return 1.0 / resistance


# Reference value at fabrication thickness H₀ (for tests / docs)
U_GEL_W_M2_K: float = u_gel_w_m2_k(H0_M)

# Condenser thermal mass per footprint area (ρ_al c_p L_c)
CONDENSER_THERMAL_MASS_J_M2_K: float = RHO_AL_KG_M3 * CP_AL_J_KG_K * L_C_M

# Lumped thermal capacitances per footprint area (J/m²K) for transient desorption
# solvers. Physical (ρ c_p L) values from Table S3 — no calibration factors.
GLASS_THERMAL_MASS_J_M2_K: float = RHO_GLASS_KG_M3 * CP_GLASS_J_KG_K * L_GLASS_M
ABSORBER_THERMAL_MASS_J_M2_K: float = RHO_AL_KG_M3 * CP_AL_J_KG_K * L_AL_STACK_M


def gel_thermal_mass_j_m2_k(h_m: float) -> float:
    """(ρ c_p H)_gel — Note S1 Eq. S1 hydrogel thermal storage per footprint area."""
    return RHO_COMPOSITE_KG_M3 * CP_GEL_J_KG_K * max(float(h_m), H0_M * 0.25)

# =============================================================================
# Default device parameters for fluid-heated daily-cycle SAWH
# =============================================================================

GEL_THERMAL_MASS_J_M2_K: float = 1.5e5
GEL_EMISSIVITY: float = 1.0

# Loop fluid → gel HX (fixed setpoints during desorption)
T_F_C: float = 58.0
M_DOT_F_KG_S_M2: float = 0.25
UA_GEL_W_K: float = 800.0
FLUID_CP_J_KG_K: float = 4180.0

# Condenser (finned aluminum, Wilson-style)
CONDENSER_THICKNESS_M: float = 0.125 * 0.0254
CONDENSER_RHO_KG_M3: float = 2700.0
CONDENSER_CP_J_KG_K: float = 900.0
CONDENSER_EMISSIVITY: float = 0.05

# Sorbent / geometry (Wilson Table S3) -- H0_M, G_CHAMBER_M_S, RHO_COMPOSITE_KG_M3,
# TILT_DEG, FIN_AREA_RATIO, H_FG_J_PER_KG already defined above (Table S3 section);
# same values, kept as single definitions.
DEFAULT_SALT_NAME: str = "LiCl"
SALT_TO_POLYMER_RATIO: float = 4.0
VAPOR_GAP_M: float = 0.04

# Data-center process air
T_AMB_C: float = 32.0
RH_AMB: float = 0.45
H_AMB_W_M2_K: float = 10.0

# =============================================================================
# Conde (2004) aqueous LiCl and CaCl2 solution properties
# =============================================================================

T_CRIT_H2O_K: float = 647.096
P_CRIT_H2O_PA: float = 22.064e6

# Valid brine salt mass-fraction ranges (Conde § Density)
XI_MAX_LICL: float = 0.56
XI_MAX_CACL2: float = 0.60

_BRACKET_LO: float = 0.01
_BRACKET_HI: float = 0.75


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
    xi_max: float


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
    xi_max=XI_MAX_LICL,
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
    xi_max=XI_MAX_CACL2,
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


def reduced_temperature(temperature_c: float) -> float:
    """θ = T / T_c,H2O."""
    return (float(temperature_c) + 273.15) / T_CRIT_H2O_K


def _pi_25(xi: float, params: VaporPressureParams) -> float:
    return (
        1.0
        - (1.0 + (xi / params.pi6) ** params.pi7) ** params.pi8
        - params.pi9 * math.exp(-((xi - 0.1) ** 2) / 0.005)
    )


def _f_xi_theta(xi: float, theta: float, params: VaporPressureParams) -> float:
    a_term = 2.0 - (1.0 + (xi / params.pi0) ** params.pi1) ** params.pi2
    b_term = (1.0 + (xi / params.pi3) ** params.pi4) ** params.pi5 - 1.0
    return a_term + b_term * theta


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
    if xi > params.xi_max:
        return float("nan")
    theta = reduced_temperature(temperature_c)
    pi = _pi_25(xi, params) * _f_xi_theta(xi, theta, params)
    return max(0.0, min(1.0, float(pi)))


def water_activity_licl(
    salt_mass_fraction: float,
    temperature_c: float = 25.0,
) -> float:
    """LiCl–H2O brine water activity (Conde 2004 Table 3)."""
    return vapor_pressure_ratio(salt_mass_fraction, temperature_c, LICL_VAPOR_PRESSURE)


def water_activity_cacl2(
    salt_mass_fraction: float,
    temperature_c: float = 25.0,
) -> float:
    """CaCl2–H2O brine water activity (Conde 2004 Table 3)."""
    return vapor_pressure_ratio(salt_mass_fraction, temperature_c, CACL2_VAPOR_PRESSURE)


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
    if rh >= 0.99:
        return _BRACKET_LO
    if temperature_c > temperature_max_c:
        return float("nan")

    def residual(xi: float) -> float:
        return rh - vapor_pressure_ratio(xi, temperature_c, params)

    hi = min(_BRACKET_HI, params.xi_max)
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

# =============================================================================
# Heat-transfer correlations (Hollands et al. 1976; Wilson Note S1)
# =============================================================================

STEFAN_BOLTZMANN_W_M2_K4: float = 5.670374419e-8
D_AIR_M2_S: float = 2.62e-5  # H2O in air ~25 °C (Note S1 Sh = Nu analogy)
GRAVITY_M_S2: float = 9.81
BETA_AIR_K: float = 1.0 / 300.0
NU_AIR_M2_S: float = 1.5e-5
PR_AIR: float = 0.71
RHO_AIR_KG_M3: float = 1.2
CP_AIR_J_KG_K: float = 1005.0
ALPHA_AIR_M2_S: float = K_AIR_W_M_K / (RHO_AIR_KG_M3 * CP_AIR_J_KG_K)


def hx_effectiveness_q(
    m_dot_cp_w_k: float,
    ua_w_k: float,
    delta_t_k: float,
) -> float:
    """Q = m_dot cp ΔT (1 - exp(-NTU)); NTU = UA/(m_dot cp)."""
    if abs(delta_t_k) < 1e-12:
        return 0.0
    mdot_cp = max(m_dot_cp_w_k, 1e-12)
    ntu = ua_w_k / mdot_cp
    return mdot_cp * delta_t_k * (1.0 - math.exp(-ntu))


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


def conduction_air_gap_w_m2(t_hot_c: float, t_cold_c: float, gap_m: float) -> float:
    if gap_m <= 0.0:
        return 0.0
    return K_AIR_W_M_K / gap_m * (t_hot_c - t_cold_c)


def _rayleigh_vapor_gap(
    gap_m: float,
    t_hot_c: float,
    t_cold_c: float,
) -> float:
    """Rayleigh number for the vapor-gap cavity (properties at mean film temperature)."""
    if gap_m <= 0.0:
        return 0.0
    delta_t = max(abs(t_hot_c - t_cold_c), 1e-6)
    return (
        GRAVITY_M_S2
        * BETA_AIR_K
        * delta_t
        * gap_m**3
        / (NU_AIR_M2_S * ALPHA_AIR_M2_S)
    )


def hollands_nu_eq_s3(ra: float, *, tilt_deg: float) -> float:
    """Wilson Note S1 Eq. S3 — Hollands et al. 1976 tilted parallel plates.

    Nu = 1 + 1.44 * [1 − 1708 sin(1.8θ)^1.6 / Ra cosθ]* [1 − 1708 / Ra cosθ]*
           + [(Ra cosθ / 5830)^(1/3) − 1]*

    where []* = max(0, ...).  Single expression valid for all Ra.
    """
    cos_t = max(math.cos(math.radians(tilt_deg)), 1e-6)
    ra_cos = ra * cos_t
    if ra_cos <= 0.0:
        return 1.0
    sin_18t_16 = math.sin(math.radians(1.8 * tilt_deg)) ** 1.6
    f1 = max(0.0, 1.0 - 1708.0 * sin_18t_16 / ra_cos)
    f2 = max(0.0, 1.0 - 1708.0 / ra_cos)
    f3 = max(0.0, (ra_cos / 5830.0) ** (1.0 / 3.0) - 1.0)
    return 1.0 + 1.44 * f1 * f2 + f3


def hollands_vapor_gap_h_conv_w_m2_k(
    gap_m: float,
    t_hot_c: float,
    t_cold_c: float,
    *,
    tilt_deg: float = 35.0,
) -> float:
    """Note S1 Eqs. S3–S4: h_conv,g = Nu · k_air / (L_g − H)."""
    if gap_m <= 0.0:
        return 0.0
    ra = _rayleigh_vapor_gap(gap_m, t_hot_c, t_cold_c)
    nu = hollands_nu_eq_s3(ra, tilt_deg=tilt_deg)
    return nu * K_AIR_W_M_K / gap_m


def vapor_gap_mass_transfer_inhibited(gap_m: float) -> bool:
    """True when gap is below Wilson's ~7 mm thermobuoyancy / transport limit."""
    return gap_m < VAPOR_GAP_TRANSPORT_MIN_M


def wind_to_h_amb_w_m2_k(wind_speed_m_s: float, *, base: float = 10.0) -> float:
    """Map 10 m wind speed to external convection coefficient (paper: ~10 at 0.5 m/s)."""
    w = max(0.0, float(wind_speed_m_s))
    return base * (0.5 + w) / 1.0


def condenser_h_conv_w_m2_k(h_amb: float, *, fin_area_ratio: float = 7.0) -> float:
    return fin_area_ratio * h_amb

# =============================================================================
# Salt catalog and PAM-LiCl water-activity models for Wilson Eq. 5
# =============================================================================

WATER_MOLAR_MASS_KG_MOL: float = 0.018015
GAS_CONSTANT_J_MOL_K: float = 8.314462618
C_W_MAX_MOL_M3: float = 400000.0
C_W_MIN_MOL_M3: float = 100.0


CANDIDATE_SALTS: tuple[str, ...] = ("LiCl", "NaCl", "CaCl2", "MgCl2")


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


def _salt_catalog_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "materials" / "salt_catalog.csv"


def _heat_of_desorption_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "data"
        / "materials"
        / "salt_heat_of_desorption.csv"
    )


@lru_cache(maxsize=1)
def _load_heat_of_desorption() -> dict[str, float]:
    path = _heat_of_desorption_path()
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row["salt_name"]).strip()
        try:
            h = float(row["heat_of_desorption_j_per_kg"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(h) and h > 0.0:
            out[name] = h
    return out


@lru_cache(maxsize=1)
def _load_salt_catalog() -> dict[str, SaltProperties]:
    df = pd.read_csv(_salt_catalog_path())
    h_des_table = _load_heat_of_desorption()
    out: dict[str, SaltProperties] = {}
    for _, row in df.iterrows():
        name = str(row["salt"]).strip()
        h_des = h_des_table.get(name, float(row["h_des_j_per_kg"]))
        out[name] = SaltProperties(
            name=name,
            formula_weight_g_mol=float(row["formula_weight_g_mol"]),
            ions_per_formula=int(row["ions_per_formula"]),
            price_usd_per_kg=float(row["price_usd_per_kg"]),
            h_des_j_per_kg=float(h_des),
            rho_solution_kg_m3=float(row["rho_solution_kg_m3"]),
            default_sl=float(row["default_sl"]),
            rh_min=float(row["rh_min"]),
            rh_max=float(row["rh_max"]),
        )
    return out


def get_salt(name: str) -> SaltProperties:
    catalog = _load_salt_catalog()
    if name not in catalog:
        raise KeyError(f"Unknown salt {name!r}; available: {sorted(catalog)}")
    return catalog[name]


def get_salt_price_usd_per_kg(name: str) -> float:
    return get_salt(name).price_usd_per_kg


TEMPERATURE_CLAMP_C: tuple[float, float] = (-40.0, 120.0)


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
_CHAMBER_DISH_DIAMETER_M: float = 0.060
_SYNTHESIS_BATCH_ML: float = 50.0
_PAM_LICL_STANDARD_POUR_ML: float = 8.0
_PAM_LICL_2GG_CHAMBER_POUR_ML: float = 12.8  # thicker pour for similar H₀ at 20 % RH
_LICL_BATCH_G_BY_SL: dict[int, float] = {
    1: 4.18,
    2: 8.36,
    4: 16.72,
    8: 33.44,
}
# Table S3 reference for anchoring synthesis c_s to the 4 g/g DVS dry-basis density.
_CHAMBER_CS_CALIB_SL: float = 4.0
_CHAMBER_CS_CALIB_H0_MM: float = 2.34
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
    """Fixed c_s for Díaz-Marín Eq. 8 from Methods pour inventory at Table S3 H₀.

    LiCl moles in the poured solution are spread over the gel footprint area times
    the measured initial thickness H₀ (SI Note S9). By default the 4 g/g PAM-LiCl
    reference (8 mL pour, H₀ = 2.34 mm) is scaled to match ``DRY_COMPOSITE_DENSITY``
    so panel 5c equilibria stay aligned with the DVS isotherm calibration.
    """
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
    """``c_s`` for Eq. 8 with SI Note S7 constant solution density at 20 % RH.

    Salt moles come from the Methods pour inventory (``chamber_c_s_from_synthesis``).
    For PAM--LiCl 2 g/g chamber samples the paper poured 12.8 mL (vs 8 mL) to match
    the ~2.34 mm thickness of the 4 g/g reference at 20 % RH; measured H₀ can still
    differ (Table S3: 2.16 mm). Holding ``c_s`` at the 4 g/g calibration thickness
    while ``H₀`` in ``g/H₀`` uses the measured value matches the digitized model
    curves (panel 5d) without changing equilibrium plateaus.
    """
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


def _pam_licl_dvs_isotherm_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "data"
        / "materials"
        / "PAM-LiCL_isotherm.csv"
    )


@lru_cache(maxsize=1)
def _load_pam_licl_dvs_isotherm() -> tuple[np.ndarray, np.ndarray]:
    """Note S2 DVS isotherm: RH (%), gravimetric uptake (g water / g dry composite)."""
    path = _pam_licl_dvs_isotherm_path()
    rh_pct: list[float] = []
    uptake_g_g: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            rh_pct.append(float(parts[0].strip()))
            uptake_g_g.append(float(parts[1].strip()))
    if not rh_pct:
        raise ValueError(f"No isotherm data in {path}")
    order = np.argsort(rh_pct)
    rh = np.array(rh_pct, dtype=float)[order]
    uptake = np.array(uptake_g_g, dtype=float)[order]
    return rh, uptake


def pam_licl_uptake_g_g_at_rh(rh_fraction: float) -> float:
    """Forward DVS isotherm: equilibrium uptake (g/g) at relative humidity."""
    rh_pct, uptake = _load_pam_licl_dvs_isotherm()
    r = max(0.0, min(100.0, float(rh_fraction) * 100.0))
    return float(np.interp(r, rh_pct, uptake))


# Dry-basis composite density for all gravimetric-uptake <-> c_w conversions.
# Table S3 reports the composite density at fabrication (25 °C, 20% RH); the DVS
# isotherm uptake is gravimetric per gram of *dry* composite, so the dry-basis
# density is rho_composite(20% RH) / (1 + uptake(20% RH)). Using the wet (20% RH)
# density here would over-count the sorbent dry mass by (1 + u20) ≈ 2.26x and
# inflate the absolute water inventory / desorption swing accordingly.
DRY_COMPOSITE_DENSITY_KG_M3: float = RHO_COMPOSITE_KG_M3 / (
    1.0 + pam_licl_uptake_g_g_at_rh(0.20)
)


def pam_licl_water_activity_from_uptake_g_g(uptake_g_g: float) -> float:
    """Invert DVS isotherm: water activity from gravimetric uptake."""
    rh_pct, uptake = _load_pam_licl_dvs_isotherm()
    u = float(uptake_g_g)
    if u <= float(uptake[0]):
        return max(0.0, float(rh_pct[0]) / 100.0)
    if u >= float(uptake[-1]):
        return min(1.0, float(rh_pct[-1]) / 100.0)
    aw = float(np.interp(u, uptake, rh_pct)) / 100.0
    return max(0.0, min(1.0, aw))


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
    """Gravimetric moisture content m_w / m_dry (g/g) on a footprint basis.

    Water inventory is referenced to the fixed fabrication thickness H₀, not the
    swollen H(t): Wilson defines c_w and the desorption flux ṁ_des = MW·H₀·dc_w/dt
    (Note S1) on the H₀ basis, so the sorbate inventory per area is c_w·H₀. The
    swelling H(t) (Eq. 6) enters only the vapor-gap convection (L_g − H) and the
    U_gel conductance (H/k_gel), never the water inventory or its activity. Using
    the swollen H here would double-count dilution and break consistency with the
    yield integral. ``h_m`` is retained for signature compatibility but unused.

    When composite state is supplied, dry mass uses salt inventory with
    ``formula_weight_g_mol * salt_weight_factor`` in the uptake denominator only
    (``c_s`` and brine activity are unchanged). Otherwise falls back to the
    calibrated DVS dry-basis density.
    """
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


def composite_component_mass_densities_kg_m3(
    c_w: float,
    c_s: float,
    *,
    formula_weight_g_mol: float,
    salt_to_polymer_ratio: float,
) -> tuple[float, float, float]:
    """Water, salt, and polymer mass densities (kg/m³ gel) from molar state."""
    mass_water = max(0.0, c_w) * WATER_MOLAR_MASS_KG_MOL
    mass_salt = max(0.0, c_s) * formula_weight_g_mol / 1000.0
    mass_polymer = mass_salt / max(salt_to_polymer_ratio, 1e-9)
    return mass_water, mass_salt, mass_polymer


def brine_salt_fraction_from_composite(
    composite_salt_fraction: float,
    *,
    salt_to_polymer_ratio: float,
) -> float:
    """Map composite salt fraction (polymer in denominator) to LiCl brine fraction."""
    f_c = float(composite_salt_fraction)
    if not math.isfinite(f_c):
        return float("nan")
    spr = max(salt_to_polymer_ratio, 1e-9)
    denom = 1.0 - f_c / spr
    if denom <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, f_c / denom))


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


def pam_licl_composite_salt_fraction(
    c_w: float,
    *,
    c_s: float,
    h_m: float,
    h0_ref_m: float,
    formula_weight_g_mol: float,
    salt_to_polymer_ratio: float,
) -> float:
    """Salt mass fraction in wet PAM-LiCl: m_s / (m_w + m_s + m_p) on a footprint basis."""
    del h_m  # inventory referenced to H₀ (see pam_licl_gravimetric_uptake_g_g)
    salt_mol_m2 = c_s * h0_ref_m
    mass_salt = salt_mol_m2 * formula_weight_g_mol / 1000.0
    mass_polymer = mass_salt / max(salt_to_polymer_ratio, 1e-9)
    mass_water = max(0.0, c_w) * h0_ref_m * WATER_MOLAR_MASS_KG_MOL
    total = mass_water + mass_salt + mass_polymer
    if total <= 0.0:
        return 1.0
    return mass_salt / total


def licl_brine_salt_fraction_from_gel(
    c_w: float,
    *,
    c_s: float,
    h_m: float,
    h0_ref_m: float,
    formula_weight_g_mol: float,
    salt_weight_factor: float = 1.0,
) -> float:
    """Brine salt mass fraction m_s / (m_s + m_w) — LiCl solution a_w,s (Eq. 5)."""
    del h_m  # inventory referenced to H₀ (see pam_licl_gravimetric_uptake_g_g)
    salt_mol_m2 = c_s * h0_ref_m
    mass_salt = salt_mol_m2 * formula_weight_g_mol * salt_weight_factor / 1000.0
    mass_water = max(0.0, c_w) * h0_ref_m * WATER_MOLAR_MASS_KG_MOL
    total = mass_salt + mass_water
    if total <= 0.0:
        return 1.0
    return mass_salt / total


def water_activity_from_c_w(
    c_w: float,
    *,
    c_s: float,
    ions_per_formula: int,
    temperature_c: float = 25.0,
    salt_name: str = "LiCl",
    formula_weight_g_mol: float = 42.394,
    salt_to_polymer_ratio: float = 4.0,
    h_m: float | None = None,
    h0_ref_m: float | None = None,
    salt_weight_factor: float = 1.0,
) -> float:
    """Brine a_w,s in Eq. 5 (Wilson Device); activity of water in the salt solution."""
    del ions_per_formula, salt_to_polymer_ratio
    if c_w <= 0.0 or c_s <= 0.0:
        return 1.0
    h_ref = h0_ref_m if h0_ref_m is not None else 0.004
    h = h_m if h_m is not None else h_ref
    mw_eff = formula_weight_g_mol * salt_weight_factor
    if salt_name == "LiCl":
        f_b = licl_brine_salt_fraction_from_gel(
            c_w,
            c_s=c_s,
            h_m=h,
            h0_ref_m=h_ref,
            formula_weight_g_mol=formula_weight_g_mol,
            salt_weight_factor=salt_weight_factor,
        )
        aw = licl_water_activity_at_brine_fraction(f_b, temperature_c)
        if math.isfinite(aw):
            return aw
        return float("nan")

    f_b = brine_salt_fraction_from_c_w(c_w, c_s, mw_eff)
    aw = water_activity_at_brine_fraction(salt_name, f_b, temperature_c)
    if math.isfinite(aw):
        return aw
    return float("nan")


def equilibrium_c_w_at_rh(
    rh: float,
    *,
    c_s: float,
    ions_per_formula: int,
    temperature_c: float = 25.0,
    salt_name: str = "LiCl",
    formula_weight_g_mol: float = 42.394,
    salt_to_polymer_ratio: float = 4.0,
    h_m: float | None = None,
    h0_ref_m: float | None = None,
) -> float:
    """Invert a_w(RH) to c_w at reference hydrogel thickness H₀."""
    del ions_per_formula
    if rh <= 0.0:
        return 0.0
    if rh >= 0.99:
        return C_W_MAX_MOL_M3

    h_ref = h0_ref_m if h0_ref_m is not None else 0.004
    del h_m  # inventory referenced to H₀ (see pam_licl_gravimetric_uptake_g_g)

    if salt_name == "LiCl":
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
    c_w = mass_water / (h_ref * WATER_MOLAR_MASS_KG_MOL)
    return max(C_W_MIN_MOL_M3, min(C_W_MAX_MOL_M3, c_w))


# Methods: hydrogel cast at equilibrium with ~20% RH ambient.
FABRICATION_EQUILIBRIUM_RH: float = 0.20


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
) -> float:
    """Initial gel water state after fabrication at ~20% RH ambient."""
    h0 = hydrogel_thickness_m
    if salt_name == "LiCl":
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
    return equilibrium_c_w_at_rh(
        FABRICATION_EQUILIBRIUM_RH,
        c_s=c_s,
        ions_per_formula=s.ions_per_formula,
        salt_name=salt_name,
        formula_weight_g_mol=fw,
        salt_to_polymer_ratio=salt_to_polymer_ratio,
        h_m=h0,
        h0_ref_m=h0,
    )

# =============================================================================
# Equilibrium brine isotherms for NaCl, LiCl, CaCl2, and MgCl2
# =============================================================================


def mf_NaCl(relative_humidity: float) -> float:
    """Equilibrium brine salt fraction for NaCl at 25°C."""
    if not (0.0 < relative_humidity < 1.0):
        return float("nan")
    a4, a3, a2, a1, a0 = 5.863, -5.545, -0.332, -0.5597, 0.9998

    def residual(salt_fraction: float) -> float:
        return (
            relative_humidity
            - a0
            - a1 * salt_fraction
            - a2 * salt_fraction**2
            - a3 * salt_fraction**3
            - a4 * salt_fraction**4
        )

    return find_root_bracketed(residual, 0.0116, 0.264)


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
    a4, a3, a2, a1, a0 = 186.32487108, -153.67496570, 38.21982328, -4.86704441, 1.16231287

    def residual(salt_fraction: float) -> float:
        return (
            relative_humidity
            - a0
            - a1 * salt_fraction
            - a2 * salt_fraction**2
            - a3 * salt_fraction**3
            - a4 * salt_fraction**4
        )

    return find_root_bracketed(residual, 0.01, 0.75, scan=True, n_intervals=19)


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


def _aw_polynomial(salt_fraction: float, coeffs: tuple[float, ...]) -> float:
    if not (0.0 <= salt_fraction < 1.0) or not math.isfinite(salt_fraction):
        return float("nan")
    a_w = 0.0
    for k, coeff in enumerate(coeffs):
        a_w += coeff * (salt_fraction**k)
    return float(a_w)


def water_activity_at_brine_fraction(
    salt_name: str,
    brine_salt_fraction: float,
    temperature_c: float = 25.0,
) -> float:
    """Forward isotherm: brine water activity at salt mass fraction and temperature."""
    rec = get_salt(salt_name)
    f = float(brine_salt_fraction)
    if not (0.0 <= f < 1.0) or not math.isfinite(f):
        return float("nan")
    if rec.name == "NaCl":
        return _aw_polynomial(f, (0.9998, -0.5597, -0.332, -5.545, 5.863))
    if rec.name == "MgCl2":
        return _aw_polynomial(
            f, (1.16231287, -4.86704441, 38.21982328, -153.67496570, 186.32487108)
        )
    if rec.name == "LiCl":
        if temperature_c > 150.0:
            return float("nan")
        return water_activity_licl(f, min(temperature_c, 150.0))
    if rec.name == "CaCl2":
        if temperature_c > 100.0:
            return float("nan")
        return water_activity_cacl2(f, temperature_c)
    return float("nan")


def brine_salt_fraction_from_c_w(
    c_w_mol_m3: float,
    c_s_mol_m3: float,
    effective_formula_weight_g_per_mol: float,
) -> float:
    """Brine salt mass fraction from gel water/salt molarities (mol/m³ gel)."""
    if not all(map(math.isfinite, (c_w_mol_m3, c_s_mol_m3, effective_formula_weight_g_per_mol))):
        return float("nan")
    if c_w_mol_m3 < 0.0 or c_s_mol_m3 < 0.0:
        return float("nan")
    mass_water = c_w_mol_m3 * 18.015 / 1000.0
    mass_salt = c_s_mol_m3 * effective_formula_weight_g_per_mol / 1000.0
    total = mass_water + mass_salt
    if total <= 0.0:
        return float("nan")
    return float(mass_salt / total)

# =============================================================================
# Gel energy balance with fixed-T_f loop-fluid HX (replaces Wilson solar stack)
# =============================================================================

@dataclass(frozen=True, slots=True)
class ThermalState:
    t_gel_c: float
    h_conv_g: float
    m_des_kg_s_m2: float
    q_f_to_gel_w_m2: float


@dataclass(frozen=True, slots=True)
class DeviceThermalParams:
    vapor_gap_m: float = VAPOR_GAP_M
    eps_gel: float = GEL_EMISSIVITY
    eps_al: float = CONDENSER_EMISSIVITY
    tilt_deg: float = TILT_DEG
    h_des_j_per_kg: float = H_DES_J_PER_KG
    gel_thermal_mass_j_m2_k: float = GEL_THERMAL_MASS_J_M2_K
    t_f_c: float = T_F_C
    m_dot_f_kg_s_m2: float = M_DOT_F_KG_S_M2
    ua_gel_w_k: float = UA_GEL_W_K
    fluid_cp_j_kg_k: float = FLUID_CP_J_KG_K


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

# =============================================================================
# Wilson et al. 2025 Eqs. 5-6 -- convection-limited mass transfer
# =============================================================================

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

# =============================================================================
# MOF adsorbent isotherm and mass-transfer rates (tabulated MIL-100(Fe) @ 303 K)
# =============================================================================

DEFAULT_MOF_NAME: str = "MIL-100_Fe"
Q_MIN_KG_KG: float = 0.0
Q_REGEN_KG_KG: float = 0.08


@dataclass(frozen=True, slots=True)
class MofProperties:
    name: str
    isotherm_file: str
    q_max_kg_kg: float
    h_ads_j_per_kg: float
    h_des_j_per_kg: float
    m_ads_kg_m2: float
    g_conv_m_s: float
    price_usd_per_kg: float


def _materials_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "materials"


def _mof_catalog_path() -> Path:
    return _materials_dir() / "mof_catalog.csv"


def _isotherm_path(filename: str) -> Path:
    return _materials_dir() / filename


@lru_cache(maxsize=8)
def _load_isotherm(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load tabulated isotherm: RH fraction, equilibrium loading q (kg water / kg MOF).

    Source columns: relative pressure (%), H2O uptake (mol/kg). Relative pressure is
    treated as RH at the measurement temperature (303 K).
    """
    path = _isotherm_path(filename)
    rh_pct: list[float] = []
    mol_per_kg: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            rh_pct.append(float(parts[0]))
            mol_per_kg.append(float(parts[1]))
    if not rh_pct:
        raise ValueError(f"No isotherm data in {path}")
    order = np.argsort(rh_pct)
    rh_frac = np.array(rh_pct, dtype=float)[order] / 100.0
    q_kg_kg = np.array(mol_per_kg, dtype=float)[order] * WATER_MOLAR_MASS_KG_MOL
    return rh_frac, q_kg_kg


@lru_cache(maxsize=1)
def _load_mof_catalog() -> dict[str, MofProperties]:
    df = pd.read_csv(_mof_catalog_path())
    out: dict[str, MofProperties] = {}
    for _, row in df.iterrows():
        name = str(row["mof"]).strip()
        iso_file = str(row["isotherm_file"]).strip()
        _, q_tab = _load_isotherm(iso_file)
        out[name] = MofProperties(
            name=name,
            isotherm_file=iso_file,
            q_max_kg_kg=float(np.max(q_tab)),
            h_ads_j_per_kg=float(row["h_ads_j_per_kg"]),
            h_des_j_per_kg=float(row["h_des_j_per_kg"]),
            m_ads_kg_m2=float(row["m_ads_kg_m2"]),
            g_conv_m_s=float(row["g_conv_m_s"]),
            price_usd_per_kg=float(row["price_usd_per_kg"]),
        )
    return out


def get_mof(name: str) -> MofProperties:
    catalog = _load_mof_catalog()
    if name not in catalog:
        raise KeyError(f"Unknown MOF {name!r}; available: {sorted(catalog)}")
    return catalog[name]


def loading_at_rh(
    rh_fraction: float,
    *,
    props: MofProperties,
) -> float:
    """Forward isotherm q(RH) from tabulated MIL-100(Fe) data at 303 K."""
    rh_tab, q_tab = _load_isotherm(props.isotherm_file)
    rh = max(0.0, min(1.0, float(rh_fraction)))
    return float(np.interp(rh, rh_tab, q_tab))


def water_activity_from_loading(
    q_kg_kg: float,
    *,
    temperature_c: float,
    props: MofProperties,
) -> float:
    """Invert tabulated q(RH): water activity (≈ RH) at equilibrium loading."""
    del temperature_c  # isotherm measured at 303 K
    q = max(0.0, min(props.q_max_kg_kg, float(q_kg_kg)))
    if q <= 1e-12:
        return 0.0
    rh_tab, q_tab = _load_isotherm(props.isotherm_file)
    if q >= float(q_tab[-1]) - 1e-12:
        return float(rh_tab[-1])
    if q <= float(q_tab[0]):
        return float(rh_tab[0])
    return float(np.interp(q, q_tab, rh_tab))


def equilibrium_loading_at_rh(
    rh: float,
    *,
    temperature_c: float,
    props: MofProperties,
) -> float:
    del temperature_c  # isotherm measured at 303 K
    return loading_at_rh(rh, props=props)


def fabrication_q_initial(*, props: MofProperties, temperature_c: float = 25.0) -> float:
    """Initial MOF loading after fabrication at ~20% RH ambient."""
    return equilibrium_loading_at_rh(
        FABRICATION_EQUILIBRIUM_RH,
        temperature_c=temperature_c,
        props=props,
    )


def dq_dt(
    q_kg_kg: float,
    *,
    t_gel_c: float,
    driving: float,
    props: MofProperties,
    g_m_s: float,
    phase: MassTransferPhase,
) -> float:
    """dq/dt (kg/kg/s) — Wilson Eq. 5 analog for a fixed MOF coating inventory."""
    q = max(Q_MIN_KG_KG, min(props.q_max_kg_kg, float(q_kg_kg)))
    aw = water_activity_from_loading(q, temperature_c=t_gel_c, props=props)
    delta = driving - aw
    if phase == "absorption":
        if delta <= 0.0:
            return 0.0
    elif delta <= 0.0:
        return 0.0

    t_k = max(clamp_temperature_c(t_gel_c) + 273.15, 200.0)
    p_sat = saturation_vapor_pressure_pa(t_gel_c)
    rate_mol_m3_s = g_m_s * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * abs(delta)
    dq = rate_mol_m3_s * WATER_MOLAR_MASS_KG_MOL / props.m_ads_kg_m2
    if phase == "desorption":
        dq = -dq
        if q + dq < Q_MIN_KG_KG:
            return max(dq, -q)
        return dq

    q_cap = props.q_max_kg_kg - q
    if dq > q_cap:
        return max(0.0, q_cap)
    return dq if q < props.q_max_kg_kg else 0.0


def m_flux_kg_s_m2_from_dq(dq_dt_val: float, *, m_ads_kg_m2: float) -> float:
    """Mass flux (kg/m²/s) from loading rate on a fixed MOF inventory."""
    if dq_dt_val >= 0.0:
        return max(0.0, dq_dt_val * m_ads_kg_m2)
    return max(0.0, -dq_dt_val * m_ads_kg_m2)


def mof_mass_transfer_g_m_s(
    *,
    phase: MassTransferPhase,
    props: MofProperties,
    h_m: float,
    t_gel_c: float,
    t_cond_c: float | None = None,
    vapor_gap_m: float,
    tilt_deg: float,
) -> float:
    """Open-bed g_conv (absorption) or heat–mass analogy g (desorption)."""
    if phase == "absorption":
        return props.g_conv_m_s
    if t_cond_c is None:
        raise ValueError("t_cond_c required for MOF desorption mass transfer")
    return mass_transfer_g_m_s(
        phase="desorption",
        params=_MofMassBridge(
            g_conv_m_s=props.g_conv_m_s,
            h0_ref_m=h_m,
            vapor_gap_m=vapor_gap_m,
            tilt_deg=tilt_deg,
        ),
        h_m=h_m,
        t_gel_c=t_gel_c,
        t_cond_c=t_cond_c,
    )


@dataclass(frozen=True, slots=True)
class _MofMassBridge:
    """Minimal MassTransferParams stand-in for vapor-gap g during MOF desorption."""

    g_conv_m_s: float
    h0_ref_m: float
    vapor_gap_m: float
    tilt_deg: float
    c_s_mol_m3: float = 0.0
    ions_per_formula: int = 1
    rho_solution_kg_m3: float = 1000.0
    salt_name: str = "MOF"
    formula_weight_g_mol: float = 1.0
    salt_to_polymer_ratio: float = 1.0


def water_kg_m2(q_kg_kg: float, *, props: MofProperties) -> float:
    return q_kg_kg * props.m_ads_kg_m2


SorbentKind = Literal["hydrogel", "mof"]

# =============================================================================
# Unified sorbent interface: PAM-salt hydrogel (default) or MOF coating
# =============================================================================

if TYPE_CHECKING:
    from waste_heat_lumped.simulation import DeviceConfig

# PhaseResult.c_w stores mol/m³ for hydrogel or kg/kg loading for MOF.
LOADING_MIN = Q_MIN_KG_KG


def is_hydrogel(config: DeviceConfig) -> bool:
    return config.sorbent == "hydrogel"


def is_mof(config: DeviceConfig) -> bool:
    return config.sorbent == "mof"


def inventory_label(config: DeviceConfig) -> str:
    return "gel" if is_hydrogel(config) else "mof"


def inventory_ylabel(config: DeviceConfig) -> str:
    return "Water in gel (L/m²)" if is_hydrogel(config) else "Water in MOF (L/m²)"


def inventory_prefix(config: DeviceConfig) -> str:
    return "water_in_gel" if is_hydrogel(config) else "water_in_mof"


def initial_loading(config: DeviceConfig) -> float:
    if is_hydrogel(config):
        return fabrication_c_w_initial(
            salt_name=config.salt_name,
            salt_to_polymer_ratio=config.salt_to_polymer_ratio,
            hydrogel_thickness_m=config.hydrogel_thickness_m,
            hydrogel_density_kg_m3=config.hydrogel_density_kg_m3,
        )
    return fabrication_q_initial(props=config.mof())


def water_in_gel_l_m2(
    c_w: float,
    h_m: float,
    *,
    h0_ref_m: float = 0.004,
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


def water_in_sorbent_l_m2(
    loading: float,
    h_m: float,
    *,
    config: DeviceConfig,
) -> float:
    if is_hydrogel(config):
        return water_in_gel_l_m2(loading, h_m, h0_ref_m=config.hydrogel_thickness_m)
    return water_kg_m2(loading, props=config.mof())


def clip_loading(loading: float, *, config: DeviceConfig) -> float:
    if is_hydrogel(config):
        return max(C_W_MIN_MOL_M3, min(C_W_MAX_MOL_M3, loading))
    props = config.mof()
    return max(Q_MIN_KG_KG, min(props.q_max_kg_kg, loading))


def evaluate_mass_rates(
    *,
    loading: float,
    h_m: float,
    t_gel_c: float,
    t_cond_c: float | None,
    rh: float,
    phase: str,
    mass: MassTransferParams,
    config: DeviceConfig,
    vapor_gap_m: float,
) -> tuple[float, float, float]:
    """Return (dloading/dt, dH/dt, m_des_kg_s_m2)."""
    if is_hydrogel(config):
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

    props: MofProperties = config.mof()
    thermal = config.thermal_params()
    if phase == "absorption":
        g = mof_mass_transfer_g_m_s(
            phase="absorption",
            props=props,
            h_m=h_m,
            t_gel_c=t_gel_c,
            vapor_gap_m=vapor_gap_m,
            tilt_deg=thermal.tilt_deg,
        )
        dq = dq_dt(
            loading,
            t_gel_c=t_gel_c,
            driving=rh,
            props=props,
            g_m_s=g,
            phase="absorption",
        )
        return dq, 0.0, m_flux_kg_s_m2_from_dq(dq, m_ads_kg_m2=props.m_ads_kg_m2)

    assert t_cond_c is not None
    g = mof_mass_transfer_g_m_s(
        phase="desorption",
        props=props,
        h_m=h_m,
        t_gel_c=t_gel_c,
        t_cond_c=t_cond_c,
        vapor_gap_m=vapor_gap_m,
        tilt_deg=thermal.tilt_deg,
    )
    c_r = concentration_ratio_desorption(t_gel_c, t_cond_c)
    dq = dq_dt(
        loading,
        t_gel_c=t_gel_c,
        driving=c_r,
        props=props,
        g_m_s=g,
        phase="desorption",
    )
    if dq > 0.0:
        dq = 0.0
    m_des = m_flux_kg_s_m2_from_dq(dq, m_ads_kg_m2=props.m_ads_kg_m2)
    return dq, 0.0, m_des


__all__ = [
    "DEFAULT_MOF_NAME",
    "LOADING_MIN",
    "SorbentKind",
    "clip_loading",
    "evaluate_mass_rates",
    "get_mof",
    "initial_loading",
    "inventory_label",
    "inventory_prefix",
    "inventory_ylabel",
    "is_hydrogel",
    "is_mof",
    "water_in_sorbent_l_m2",
]
