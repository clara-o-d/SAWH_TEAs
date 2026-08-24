"""System physics for the two-bed waste-heat SAWH with direct waste-heat coupling (no HTF
loop): heat-transfer correlations, system defaults, brine/salt thermodynamics,
contactor/condenser energy balances, mass transfer, and the hydrogel sorbent model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from solar_lumped._parameters_xlsx import SALTS, physics_value as _pv

if TYPE_CHECKING:
    from waste_heat.simulation import SystemConfig

# Every constant below comes from solar_lumped/docs/parameters.xlsx -- the repo-wide
# single source of truth. Rows without a "Waste-heat:" prefix are shared verbatim with
# the solar device; prefixed rows are this two-bed system's own operating point.
STEFAN_BOLTZMANN_W_M2_K4: float = _pv("Stefan-Boltzmann constant (sigma)")
K_AIR_W_M_K: float = _pv("Air thermal conductivity (k_air)")


def parallel_plate_emissivity(eps_a: float, eps_b: float) -> float:
    if eps_a <= 0.0 or eps_b <= 0.0:
        return 0.0
    return 1.0 / (1.0 / eps_a + 1.0 / eps_b - 1.0)


def radiative_exchange_w_m2(t_hot_c: float, t_cold_c: float, *, emissivity: float = 0.9) -> float:
    t_hot_k = t_hot_c + 273.15
    t_cold_k = t_cold_c + 273.15
    return emissivity * STEFAN_BOLTZMANN_W_M2_K4 * (t_hot_k**4 - t_cold_k**4)


def rarefied_gap_h_w_m2_k(
    gap_m: float,
    t_hot_c: float,
    t_cold_c: float,
    *,
    p_total_pa: float,
    t_mean_c: float | None = None,
) -> float:
    """Gap conductance at partial vacuum (Knudsen + continuum blend): parallel-plate
    molecular conduction h ≈ k_eff/gap with k_eff = k_0/(1 + Kn), Kn = λ/gap."""
    if gap_m <= 0.0:
        return 0.0
    p = max(p_total_pa, 1.0)
    # Mean free path of water vapor ~ 2e-3 / p(Pa) m (order-of-magnitude at 300 K)
    mean_free_path_m = 2.0e-3 * (P_REF_PA / p)
    kn = mean_free_path_m / gap_m
    k_eff = K_AIR_W_M_K / (1.0 + kn)
    return max(k_eff / gap_m, K_AIR_W_M_K / (10.0 * gap_m))


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


def condenser_h_conv_w_m2_k(h_amb: float, *, fin_area_ratio: float = 7.0) -> float:
    return fin_area_ratio * h_amb


CONTACTOR_THERMAL_MASS_J_M2_K: float = _pv("Waste-heat: contactor thermal mass")
CONTACTOR_AREA_M2: float = _pv("Waste-heat: contactor area")
CONTACTOR_EMISSIVITY: float = _pv("Waste-heat: contactor emissivity")

# Vacuum gap (desorbing contactor to condenser)
VACUUM_GAP_M: float = _pv("Waste-heat: vacuum gap")
P_COND_PA: float = _pv("Waste-heat: chamber condensing pressure")

# Contactor-side UA (desorbing contactor <-> its internal heat-transfer surface): the
# contactor-side leg of the direct waste-heat coupling below.
CONTACTOR_UA_W_K: float = _pv("Waste-heat: contactor UA")

# Waste heat (liquid-cooled data center)
T_WH_IN_C: float = _pv("Waste-heat: stream inlet temperature (T_wh,in)")
CP_WH_J_KG_K: float = _pv("Waste-heat: stream specific heat (cp_wh)")
M_WH_KG_S_M2: float = _pv("Waste-heat: stream mass flow (m_wh)")
WH_HX_UA_W_K: float = _pv("Waste-heat: HX UA (UA_hx)")

# Direct waste-heat-to-desorber equivalent UA: WH_HX_UA_W_K in series with
# CONTACTOR_UA_W_K (these used to sandwich the removed pumped HTF loop).
UA_WH_DESORBER_W_K: float = (WH_HX_UA_W_K * CONTACTOR_UA_W_K) / (WH_HX_UA_W_K + CONTACTOR_UA_W_K)

# Vacuum pump conductance (kg / s / Pa / m²)
C_VAC_BASE_KG_S_PA_M2: float = _pv("Waste-heat: vacuum conductance base (C_vac)")
C_VAC_MIN_KG_S_PA_M2: float = _pv("Waste-heat: vacuum conductance lower bound")
C_VAC_MAX_KG_S_PA_M2: float = _pv("Waste-heat: vacuum conductance upper bound")

# Condenser (finned aluminum, Wilson-style)
FIN_AREA_RATIO: float = _pv("Condenser fin area ratio (A_r)")
CONDENSER_THICKNESS_M: float = _pv("Waste-heat: condenser plate thickness")
CONDENSER_RHO_KG_M3: float = _pv("Aluminum density (rho_Al)")
CONDENSER_CP_J_KG_K: float = _pv("Aluminum specific heat (cp_Al)")
CONDENSER_EMISSIVITY: float = _pv("Condenser (Al) emissivity (eps_Al)")
H_FG_J_PER_KG: float = _pv("Condensation enthalpy (h_fg)")

# Adsorbing-contactor finned heat sink. Plain ambient convection gives a ~167 min cooldown
# constant -- slower than a 90-350 min half-cycle, so the bed overheats past the source
# temperature over repeated swaps. Condenser-style fins cut that to ~23 min.
CONTACTOR_FIN_AREA_RATIO: float = FIN_AREA_RATIO

# Cycle / control
# end half-cycle when vapor-gap RH outside desorber <= this
RH_DESORBER_SWITCH: float = _pv("Waste-heat: desorber RH switch threshold")
# max half-cycle duration (s); RH threshold ends early
TAU_HALF_S: float = _pv("Waste-heat: maximum half-cycle duration (tau_half)")
K_M_PER_KG_M2: float = _pv("Waste-heat: mass-transfer resistance coefficient (k_m)")
K_P_PER_KG_S_M2: float = _pv("Waste-heat: pressure-drop coefficient (k_p)")

# Data-center process air
T_AMB_C: float = _pv("Waste-heat: ambient temperature (T_amb)")
RH_AMB: float = _pv("Waste-heat: ambient RH (RH_amb)")
H_AMB_W_M2_K: float = _pv("Ambient convection coefficient (h_amb)")

# Sorbent defaults
DEFAULT_SALT_NAME: str = "LiCl"
SALT_LOADING: float = _pv("Salt loading (SL)")
H0_M: float = _pv("Hydrogel reference thickness (H0)", mm_to_m=True)
G_CHAMBER_M_S: float = _pv("Chamber convection coefficient, absorption (g_chamber)")
RHO_COMPOSITE_KG_M3: float = _pv("Composite (hydrogel) density at 20% RH (rho_gel)")
VAPOR_GAP_M: float = _pv("Vapor gap (L_g)", mm_to_m=True)
TILT_DEG: float = _pv("Tilt angle (theta)")
HYDROGEL_MAX_DEPLETION_S: float = _pv("Waste-heat: hydrogel depletion time constant")

WATER_MOLAR_MASS_KG_MOL: float = _pv("Water molar mass (MW_w)")
GAS_CONSTANT_J_MOL_K: float = _pv("Universal gas constant (R)")
C_W_MAX_MOL_M3: float = _pv("Gel water concentration numerical backstop, upper")
C_W_MIN_MOL_M3: float = _pv("Gel water concentration numerical backstop, lower")


@dataclass(frozen=True, slots=True)
class SaltProperties:
    name: str
    formula_weight_g_mol: float
    ions_per_formula: int
    price_usd_per_kg: float
    h_des_j_per_kg: float
    rho_solution_kg_m3: float
    default_sl: float
    hydrate_h2o_per_formula: float


@lru_cache(maxsize=1)
def _load_salt_catalog() -> dict[str, SaltProperties]:
    """The parameters.xlsx Salts sheet. Replaces the former salt_catalog.csv, which was
    a fork of that sheet whose prices and h_des had all drifted away from it."""
    return {
        name: SaltProperties(
            name=name,
            formula_weight_g_mol=float(row["formula_weight_g_mol"]),
            ions_per_formula=int(row["ions_per_formula"]),
            price_usd_per_kg=float(row["price_usd_per_kg"]),
            h_des_j_per_kg=float(row["h_des_j_per_kg"]),
            rho_solution_kg_m3=float(row["rho_solution_kg_m3"]),
            default_sl=float(row["default_sl"]),
            hydrate_h2o_per_formula=float(row["hydrate_h2o_per_formula"]),
        )
        for name, row in SALTS.items()
    }


def get_salt(name: str) -> SaltProperties:
    catalog = _load_salt_catalog()
    if name not in catalog:
        raise KeyError(f"Unknown salt {name!r}; available: {sorted(catalog)}")
    return catalog[name]


def get_salt_price_usd_per_kg(name: str) -> float:
    return get_salt(name).price_usd_per_kg


TEMPERATURE_CLAMP_C: tuple[float, float] = (
    _pv("Temperature clamp lower bound"),
    _pv("Waste-heat: temperature clamp upper bound"),
)


def clamp_temperature_c(temperature_c: float) -> float:
    lo, hi = TEMPERATURE_CLAMP_C
    if not math.isfinite(temperature_c):
        return 25.0
    return max(lo, min(hi, float(temperature_c)))


def saturation_vapor_pressure_pa(temperature_c: float) -> float:
    """Tetens (Magnus) formula, Pa."""
    t = clamp_temperature_c(temperature_c)
    return 1000.0 * 0.61078 * math.exp(17.27 * t / (t + 237.3))


# Dry-basis composite density, measured: rho_composite(20% RH) / (1 + u(20% RH)). The wet
# density would over-count dry sorbent mass by (1 + u20) ~ 2.26x. A mass/density
# calibration, NOT a sorption model -- water uptake comes from the salt-in-water activity
# model (water_activity_from_c_w) alone, here as in solar_lumped.
DRY_COMPOSITE_DENSITY_KG_M3: float = _pv("Dry composite density (rho_dry)")


def pam_licl_dry_mass_kg_m2(
    h0_ref_m: float,
    *,
    dry_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3,
) -> float:
    """Dry PAM-LiCl composite mass per m² at reference thickness H₀ (dry-basis density)."""
    return dry_density_kg_m3 * h0_ref_m


def pam_licl_gravimetric_uptake_g_g(
    c_w: float,
    h_m: float,
    *,
    h0_ref_m: float,
    dry_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3,
) -> float:
    """Gravimetric moisture content m_w / m_dry (g/g) on a footprint basis."""
    m_dry = pam_licl_dry_mass_kg_m2(h0_ref_m, dry_density_kg_m3=dry_density_kg_m3)
    if m_dry <= 0.0:
        return 0.0
    mass_water = max(0.0, c_w) * h_m * WATER_MOLAR_MASS_KG_MOL
    return mass_water / m_dry


def licl_water_activity_at_brine_fraction(
    brine_salt_fraction: float,
    temperature_c: float,
) -> float:
    """Forward LiCl isotherm a_w vs brine salt mass fraction."""
    f = float(brine_salt_fraction)
    if not (0.0 <= f < 1.0) or not math.isfinite(f):
        return float("nan")
    if temperature_c > 100.0:
        return float("nan")
    tr = (temperature_c + 273.15) / 647.0
    p0, p1, p2 = 0.28, 4.3, 0.60
    p3, p4, p5 = 0.21, 5.10, 0.49
    p6, p7, p8, p9 = 0.362, -4.75, -0.40, 0.03
    concentration_term = (
        1.0
        - (1.0 + (f / p6) ** p7) ** p8
        - p9 * math.exp(-((f - 0.1) ** 2) / 0.005)
    )
    temperature_term = (
        2.0
        - (1.0 + (f / p0) ** p1) ** p2
        + ((1.0 + (f / p3) ** p4) ** p5 - 1.0) * tr
    )
    return max(0.0, min(1.0, float(concentration_term * temperature_term)))


def water_activity_from_c_w(
    c_w: float,
    *,
    c_s: float,
    ions_per_formula: int,
    temperature_c: float = 25.0,
    salt_name: str = "LiCl",
    formula_weight_g_mol: float = 42.394,
    salt_loading: float = 4.0,
    h_m: float | None = None,
    h0_ref_m: float | None = None,
) -> float:
    """LiCl brine a_w,s in Eq. 5 (Wilson Device); activity of water in the salt solution."""
    del ions_per_formula
    if c_w <= 0.0 or c_s <= 0.0:
        return 1.0
    h_ref = h0_ref_m if h0_ref_m is not None else 0.004
    h = h_m if h_m is not None else h_ref
    if salt_name == "LiCl":
        # Brine salt mass fraction m_s / (m_s + m_w) — LiCl solution a_w,s (Eq. 5).
        h_floor = max(h, h_ref * 0.25)
        salt_mol_m2 = c_s * h_ref
        mass_salt = salt_mol_m2 * formula_weight_g_mol / 1000.0
        mass_water = max(0.0, c_w) * h_floor * WATER_MOLAR_MASS_KG_MOL
        total = mass_salt + mass_water
        f_b = 1.0 if total <= 0.0 else mass_salt / total
        aw = licl_water_activity_at_brine_fraction(f_b, temperature_c)
        if math.isfinite(aw):
            return aw
    n_w = c_w
    n_s = c_s * 2
    x_w = n_w / (n_w + n_s + 1e-30)
    return max(0.0, min(1.0, x_w))


def equilibrium_c_w_at_rh(
    rh: float,
    *,
    c_s: float,
    ions_per_formula: int,
    temperature_c: float = 25.0,
    salt_name: str = "LiCl",
    formula_weight_g_mol: float = 42.394,
    salt_loading: float = 4.0,
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
    h = h_m if h_m is not None else h_ref

    if salt_name == "LiCl":
        # Invert LiCl isotherm: brine salt fraction at equilibrium with RH (bisection).
        if rh >= 0.99:
            f_b = 0.01
        else:
            lo, hi = 0.01, 0.75
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                aw = licl_water_activity_at_brine_fraction(mid, temperature_c)
                if not math.isfinite(aw) or aw < rh:
                    hi = mid
                else:
                    lo = mid
            f_b = 0.5 * (lo + hi)
        salt_mol_m2 = c_s * h_ref
        mass_salt = salt_mol_m2 * formula_weight_g_mol / 1000.0
        if f_b <= 0.0:
            return C_W_MAX_MOL_M3
        mass_water = mass_salt * (1.0 - f_b) / f_b
        if mass_water <= 0.0:
            return C_W_MIN_MOL_M3
        c_w = mass_water / (h * WATER_MOLAR_MASS_KG_MOL)
        return max(C_W_MIN_MOL_M3, min(C_W_MAX_MOL_M3, c_w))

    lo, hi = C_W_MIN_MOL_M3, C_W_MAX_MOL_M3
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        aw = water_activity_from_c_w(
            mid,
            c_s=c_s,
            ions_per_formula=2,
            temperature_c=temperature_c,
            salt_name=salt_name,
            formula_weight_g_mol=formula_weight_g_mol,
            salt_loading=salt_loading,
            h_m=h,
            h0_ref_m=h_ref,
        )
        if aw < rh:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Methods: hydrogel cast at equilibrium with ~20% RH ambient.
FABRICATION_EQUILIBRIUM_RH: float = _pv("Fabrication equilibrium RH")


def m_des_kg_s_m2(
    *,
    temperature_c: float,
    t_cond_c: float,
    c_vac_kg_s_pa_m2: float,
    q_kg_kg: float | None = None,
    m_ads_kg_m2: float | None = None,
    max_depletion_s: float = 600.0,
) -> float:
    """Vacuum desorption flux (Eq. massdes in governing_eq.tex), driven by the condenser
    saturation pressure difference ΔP = P_sat(T_d) − P_sat(T_cond)."""
    p_sat_des = saturation_vapor_pressure_pa(temperature_c)
    p_sat_cond = saturation_vapor_pressure_pa(t_cond_c)
    delta_p = max(0.0, p_sat_des - p_sat_cond)
    raw = max(0.0, c_vac_kg_s_pa_m2 * delta_p)
    if q_kg_kg is not None and m_ads_kg_m2 is not None and max_depletion_s > 0.0:
        avail_kg_m2 = max(0.0, q_kg_kg) * m_ads_kg_m2
        raw = min(raw, avail_kg_m2 / max_depletion_s)
    return raw


@dataclass(frozen=True, slots=True)
class ThermalEnvironment:
    t_amb_c: float
    rh_amb: float
    h_amb_w_m2_k: float
    t_wh_in_c: float
    m_dot_wh_kg_s_m2: float


@dataclass(frozen=True, slots=True)
class ContactorThermalParams:
    contactor_thermal_mass_j_m2_k: float = CONTACTOR_THERMAL_MASS_J_M2_K
    contactor_area_m2: float = CONTACTOR_AREA_M2
    contactor_emissivity: float = CONTACTOR_EMISSIVITY
    cp_wh_j_kg_k: float = CP_WH_J_KG_K
    ua_wh_desorber_w_k: float = UA_WH_DESORBER_W_K
    contactor_fin_area_ratio: float = CONTACTOR_FIN_AREA_RATIO
    vacuum_gap_m: float = VACUUM_GAP_M
    p_vacuum_pa: float = P_COND_PA
    fin_area_ratio: float = FIN_AREA_RATIO
    condenser_thermal_mass_j_m2_k: float = (
        CONDENSER_RHO_KG_M3 * CONDENSER_CP_J_KG_K * CONDENSER_THICKNESS_M
    )
    condenser_emissivity: float = CONDENSER_EMISSIVITY
    h_fg_j_per_kg: float = H_FG_J_PER_KG


def dT_a_dt(
    *,
    t_a_c: float,
    m_ads_kg_s_m2: float,
    h_ads_j_per_kg: float,
    params: ContactorThermalParams,
    env: ThermalEnvironment,
) -> float:
    """Adsorbing-contactor energy balance: heat of adsorption rejected to ambient through a
    condenser-style finned heat sink (plain convection is too slow -- see
    ``CONTACTOR_FIN_AREA_RATIO``)."""
    q_gen = m_ads_kg_s_m2 * h_ads_j_per_kg
    h_conv = condenser_h_conv_w_m2_k(env.h_amb_w_m2_k, fin_area_ratio=params.contactor_fin_area_ratio)
    q_conv = h_conv * params.contactor_area_m2 * (t_a_c - env.t_amb_c)
    rhs = q_gen - q_conv
    tmass = max(params.contactor_thermal_mass_j_m2_k, 1.0)
    return rhs / tmass


def dT_d_dt(
    *,
    t_d_c: float,
    t_cond_c: float,
    m_des_kg_s_m2: float,
    h_des_j_per_kg: float,
    params: ContactorThermalParams,
    env: ThermalEnvironment,
) -> float:
    """Desorbing-contactor energy balance: direct waste-heat coupling via UA_eq."""
    q_wh_to_d = hx_effectiveness_q(
        env.m_dot_wh_kg_s_m2 * params.cp_wh_j_kg_k,
        params.ua_wh_desorber_w_k,
        env.t_wh_in_c - t_d_c,
    )
    h_gap = rarefied_gap_h_w_m2_k(
        params.vacuum_gap_m,
        t_d_c,
        t_cond_c,
        p_total_pa=params.p_vacuum_pa,
    )
    eps = parallel_plate_emissivity(params.contactor_emissivity, params.condenser_emissivity)
    q_rad = radiative_exchange_w_m2(t_d_c, t_cond_c, emissivity=eps)
    q_gap = h_gap * params.contactor_area_m2 * (t_d_c - t_cond_c)
    rhs = (
        q_wh_to_d
        - m_des_kg_s_m2 * h_des_j_per_kg
        - q_gap
        - q_rad
    )
    tmass = max(params.contactor_thermal_mass_j_m2_k, 1.0)
    return rhs / tmass


def dT_cond_dt(
    *,
    t_d_c: float,
    t_cond_c: float,
    t_amb_c: float,
    m_des_kg_s_m2: float,
    h_amb_w_m2_k: float,
    params: ContactorThermalParams,
) -> float:
    t_d = clamp_temperature_c(t_d_c)
    t_cond = clamp_temperature_c(t_cond_c)
    h_gap = rarefied_gap_h_w_m2_k(
        params.vacuum_gap_m,
        t_d,
        t_cond,
        p_total_pa=params.p_vacuum_pa,
    )
    eps = parallel_plate_emissivity(params.contactor_emissivity, params.condenser_emissivity)
    q_rad = radiative_exchange_w_m2(t_d, t_cond, emissivity=eps)
    q_gap = h_gap * params.contactor_area_m2 * (t_d - t_cond)
    h_conv_cond = condenser_h_conv_w_m2_k(h_amb_w_m2_k, fin_area_ratio=params.fin_area_ratio)
    q_conv = h_conv_cond * (t_cond - t_amb_c)
    rhs = q_gap + m_des_kg_s_m2 * params.h_fg_j_per_kg + q_rad - q_conv
    tmass = max(params.condenser_thermal_mass_j_m2_k, 1.0)
    return rhs / tmass
MassTransferPhase = Literal["absorption", "desorption"]

# K_AIR_W_M_K already defined above (correlations section); same value, single definition.
D_AIR_M2_S: float = _pv("Water-vapor-in-air diffusivity (D_air)")  # H2O in air, 1 atm / 25 C
GRAVITY_M_S2: float = _pv("Gravitational acceleration (g)")
NU_AIR_M2_S: float = _pv("Air kinematic viscosity (nu_air)")  # 1 atm / 25 C
RHO_AIR_KG_M3: float = _pv("Air density (rho_air)")
CP_AIR_J_KG_K: float = _pv("Air specific heat (cp_air)")
# Thermal diffusivity alpha = k/(rho*cp). Previously a hardcoded 1.8e-5 here, which is
# the *dynamic viscosity* of air (Pa*s), not a diffusivity -- so Ra was off by ~1.3x.
ALPHA_AIR_M2_S: float = K_AIR_W_M_K / (RHO_AIR_KG_M3 * CP_AIR_J_KG_K)
P_REF_PA: float = _pv("Waste-heat: reference pressure")  # nu/alpha/D are quoted here


def hollands_nu(
    gap_m: float,
    t_hot_c: float,
    t_cold_c: float,
    *,
    tilt_deg: float,
    p_gap_pa: float = P_COND_PA,
) -> float:
    """Nusselt number across a tilted parallel-plate gap, Hollands et al. (1976).

    Ra = g*(dRho/rho_air)*L^3/(nu*alpha) using the exact ideal-gas density difference
    (rho ~ 1/T, so the reference density cancels) with rho_air at the film temperature.
    Both nu and alpha scale as 1/p for an ideal gas, so Ra scales as (p/p_ref)^2: at this
    device's ~30 mbar gap that suppresses Ra by ~1100x versus 1 atm, pinning Nu to 1
    (conduction/diffusion limit), which is the physically correct answer for a vacuum gap.

    This replaces an invented `0.720*Ra^0.25*(1 + 0.1*cos(tilt))` fit that was labelled
    Hollands but is not, and which overstated Nu ~4x at 1 atm and ~6x at 30 mbar.

    Hollands is validated to Ra*cos(tilt) ~ 1e5; above that see ElSherbiny et al. (1982).
    """
    if gap_m <= 0.0:
        return 1.0
    # Floor above absolute zero: solver iterates can transiently pass unclamped temperatures.
    t_hot_k = max(t_hot_c + 273.15, 1.0)
    t_cold_k = max(t_cold_c + 273.15, 1.0)
    t_film_k = 0.5 * (t_hot_k + t_cold_k)
    delta_t = max(abs(t_hot_k - t_cold_k), 1e-6)
    d_rho_over_rho = t_film_k * delta_t / (t_hot_k * t_cold_k)
    p_scale = (max(p_gap_pa, 0.0) / P_REF_PA) ** 2
    ra = GRAVITY_M_S2 * d_rho_over_rho * gap_m**3 / (NU_AIR_M2_S * ALPHA_AIR_M2_S) * p_scale
    cos_t = max(math.cos(math.radians(tilt_deg)), 1e-6)
    ra_cos = ra * cos_t
    if ra_cos <= 0.0:
        return 1.0
    f1 = max(0.0, 1.0 - 1708.0 * math.sin(math.radians(1.8 * tilt_deg)) ** 1.6 / ra_cos)
    f2 = max(0.0, 1.0 - 1708.0 / ra_cos)
    f3 = max(0.0, (ra_cos / 5830.0) ** (1.0 / 3.0) - 1.0)
    # Nu >= 1 by construction, so h = Nu*k/gap needs no separate conduction floor.
    return 1.0 + 1.44 * f1 * f2 + f3


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
        salt_loading=params.salt_loading,
        h_m=h_m,
        h0_ref_m=params.h0_ref_m,
    )
    # No measured-uptake cap: water in the gel has the chemical potential of pure water
    # plus RT ln(x_w gamma_w), so the salt-in-water activity IS the isotherm. Mirrors
    # solar_lumped.physics._absorption_effective_water_activity.
    return aw_brine


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
        salt_loading=params.salt_loading,
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
    salt_loading: float = 4.0
    # Vapor-gap total pressure: sets the Ra ~ p^2 suppression in hollands_nu.
    p_gap_pa: float = P_COND_PA


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
    nu = hollands_nu(
        gap_m, t_gel_c, t_cond_c, tilt_deg=params.tilt_deg, p_gap_pa=params.p_gap_pa
    )
    h_conv = nu * K_AIR_W_M_K / gap_m
    # Le ~ 1 heat-to-mass analogy: Sh = Nu  =>  g = h_conv * D/k.
    # ponytail: D_AIR_M2_S is binary H2O-in-air at 1 atm, but this gap runs at
    # p_gap_pa (~30 mbar) and is mostly water vapour, not air. Binary diffusivity
    # scales as 1/p (D would be ~34x larger here), and in a near-pure-vapour gap the
    # Sh=Nu binary-diffusion picture breaks down in favour of Stefan/effusive flow.
    # Left at the 1 atm value deliberately -- revisit with a rarefied vapour-transport
    # model (cf. rarefied_gap_h_w_m2_k on the heat side) before trusting absolute yields.
    return h_conv * D_AIR_M2_S / K_AIR_W_M_K


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
    g = mass_transfer_g_m_s(
        phase=phase,
        params=params,
        h_m=h_m,
        t_gel_c=t_gel_c,
        t_cond_c=t_cond_c,
    )
    pref = g / params.h0_ref_m
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
    # Per-salt hydrate floor, not the flat C_W_MIN_MOL_M3 backstop (which is ~100x drier
    # than any hydrate and so never bound). See hydrate_floor_c_w.
    if c_w <= hydrate_floor_c_w(
        c_s_mol_m3=params.c_s_mol_m3, salt_name=params.salt_name
    ) and rate < 0.0:
        return 0.0
    return rate


def hydrate_floor_c_w(*, c_s_mol_m3: float, salt_name: str) -> float:
    """Lowest physically reachable gel water concentration (mol/m3), on the H0 basis.

    Mirrors solar_lumped.physics.hydrate_floor_c_w, reading the same Salts-sheet
    ``hydrate_h2o_per_formula`` column. Eq. 5's rate knows c_w only through a_w, and a_w
    floors at the salt's deliquescence RH, so desorption never self-terminates on its own
    and the ODE will drive c_w through zero. The missing bound is chemical: once every
    remaining water molecule is crystal-bound as salt.nH2O, removing more is a dehydration
    reaction costing far more than h_fg, which this model does not represent.

    This replaces the flat C_W_MIN_MOL_M3 backstop as the desorption floor. That constant
    is ~0.01-0.03 water per formula unit, roughly 100x drier than any hydrate, i.e. no
    constraint at all -- see its Source note in parameters.xlsx.
    """
    return max(0.0, get_salt(salt_name).hydrate_h2o_per_formula * float(c_s_mol_m3))


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
    """Eq. 6 hydrogel thickness rate, consistent with Eq. 5 dc_w/dt:
    dH/dt = g·(MW/ρ_sol)·(p_sat/RT)·driving (m/s)."""
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

# --- Sorbent interface: LiCl hydrogel ---


@dataclass(frozen=True, slots=True)
class BedState:
    loading: float
    h_m: float | None = None


@dataclass(frozen=True, slots=True)
class SorbentMassRates:
    d_loading_a: float
    d_loading_d: float
    d_h_a: float
    d_h_d: float
    m_ads_kg_s_m2: float
    m_des_kg_s_m2: float


def mass_state_size(config: SystemConfig) -> int:
    return 4


def inventory_label(config: SystemConfig) -> str:
    return "gel"


def inventory_column(config: SystemConfig) -> str:
    return "water_in_gel_l_m2"


def inventory_ylabel(config: SystemConfig) -> str:
    return "Water in gel (L/m²)"


def h_ads_j_per_kg(config: SystemConfig) -> float:
    return get_salt(config.salt_name).h_des_j_per_kg


def h_des_j_per_kg(config: SystemConfig) -> float:
    return get_salt(config.salt_name).h_des_j_per_kg


def mass_transfer_params(config: SystemConfig) -> MassTransferParams:
    s = get_salt(config.salt_name)
    return MassTransferParams(
        g_conv_m_s=config.g_conv_m_s,
        h0_ref_m=config.hydrogel_thickness_m,
        vapor_gap_m=config.vapor_gap_m,
        tilt_deg=config.tilt_deg,
        c_s_mol_m3=(
            config.hydrogel_density_kg_m3
            * (config.salt_loading / (1.0 + config.salt_loading))
            / (s.formula_weight_g_mol / 1000.0)
        ),
        ions_per_formula=s.ions_per_formula,
        rho_solution_kg_m3=s.rho_solution_kg_m3,
        salt_name=s.name,
        formula_weight_g_mol=s.formula_weight_g_mol,
        salt_loading=config.salt_loading,
        p_gap_pa=config.p_cond_pa,
    )


def water_in_gel_l_m2(
    loading: float,
    h_m: float,
    *,
    config: SystemConfig,
) -> float:
    """Water in gel (L/m²) on Wilson Fig. S1's gravimetric basis."""
    u = pam_licl_gravimetric_uptake_g_g(loading, h_m, h0_ref_m=config.hydrogel_thickness_m)
    return u * pam_licl_dry_mass_kg_m2(config.hydrogel_thickness_m)


def initial_bed_states(config: SystemConfig) -> tuple[BedState, BedState]:
    h0 = config.hydrogel_thickness_m
    # Brine equilibrium at each bed's RH, the same route solar_lumped's
    # fabrication_c_w_initial takes.
    mp = mass_transfer_params(config)
    c_ads = equilibrium_c_w_at_rh(
        RH_AMB * 0.65, c_s=mp.c_s_mol_m3, ions_per_formula=mp.ions_per_formula,
        salt_name=mp.salt_name, formula_weight_g_mol=mp.formula_weight_g_mol,
        salt_loading=config.salt_loading, h0_ref_m=h0,
    )
    c_regen = equilibrium_c_w_at_rh(
        FABRICATION_EQUILIBRIUM_RH, c_s=mp.c_s_mol_m3, ions_per_formula=mp.ions_per_formula,
        salt_name=mp.salt_name, formula_weight_g_mol=mp.formula_weight_g_mol,
        salt_loading=config.salt_loading, h0_ref_m=h0,
    )
    return BedState(c_ads, h0), BedState(c_regen, h0)


def _hydrogel_adsorption_rates(
    c_w: float,
    h_m: float,
    *,
    t_c: float,
    rh: float,
    params: MassTransferParams,
    h_min: float,
) -> tuple[float, float, float]:
    c_r = concentration_ratio_absorption(rh)
    dc = dc_w_dt(c_w, t_gel_c=t_c, c_r=c_r, params=params, h_m=h_m, phase="absorption")
    dh = dH_dt(c_w, t_gel_c=t_c, c_r=c_r, params=params, h_m=h_m, phase="absorption")
    if h_m <= h_min + 1e-12:
        dh = max(0.0, dh)
    m_ads = m_ads_kg_s_m2_from_state(c_w, h_m, dc, dh)
    return dc, dh, m_ads


def _hydrogel_desorption_rates(
    c_w: float,
    h_m: float,
    *,
    t_c: float,
    t_cond_c: float,
    c_vac: float,
    params: MassTransferParams,
) -> tuple[float, float, float]:
    avail = WATER_MOLAR_MASS_KG_MOL * max(0.0, c_w) * h_m
    m_vac = m_des_kg_s_m2(
        temperature_c=t_c,
        t_cond_c=t_cond_c,
        c_vac_kg_s_pa_m2=c_vac,
        q_kg_kg=None,
        m_ads_kg_m2=None,
    )
    m_vac = min(m_vac, avail / HYDROGEL_MAX_DEPLETION_S if avail > 0 else 0.0)

    c_r = concentration_ratio_desorption(t_c, t_cond_c)
    dc_w = dc_w_dt(
        c_w,
        t_gel_c=t_c,
        c_r=c_r,
        params=params,
        h_m=h_m,
        phase="desorption",
        t_cond_c=t_cond_c,
    )
    dh = dH_dt(
        c_w,
        t_gel_c=t_c,
        c_r=c_r,
        params=params,
        h_m=h_m,
        phase="desorption",
        t_cond_c=t_cond_c,
    )
    m_wilson = m_des_kg_s_m2_from_state(c_w, h_m, dc_w, dh)
    if m_wilson > 1e-14:
        scale = min(1.0, m_vac / m_wilson)
        dc_w *= scale
        dh *= scale
    elif m_vac > 0.0 and h_m > 1e-12:
        dc_w = -m_vac / (WATER_MOLAR_MASS_KG_MOL * h_m)
        dh = 0.0
    else:
        dc_w = 0.0
        dh = 0.0
    if dc_w > 0.0:
        dc_w = 0.0
    if dh > 0.0:
        dh = 0.0
    # The vacuum branch above sets dc_w straight from m_vac, bypassing dc_w_dt's clip, so
    # the hydrate floor has to be reapplied here or it would not bind on that path at all.
    # Limits removal to the water above the floor rather than to all of it (the old
    # -c_w), since crystal-bound water cannot be driven off by pumping either.
    floor = hydrate_floor_c_w(c_s_mol_m3=params.c_s_mol_m3, salt_name=params.salt_name)
    if c_w + dc_w * 1.0 < floor:
        dc_w = min(0.0, floor - c_w)
    return dc_w, dh, m_vac


def mass_rates(
    *,
    loading_a: float,
    loading_d: float,
    h_a: float,
    h_d: float,
    t_a_c: float,
    t_d_c: float,
    t_cond_c: float,
    rh_amb: float,
    c_vac_kg_s_pa_m2: float,
    config: SystemConfig,
    equalize: bool = True,
) -> SorbentMassRates:
    params = mass_transfer_params(config)
    h_min = config.hydrogel_thickness_m
    dc_a, dh_a, m_ads = _hydrogel_adsorption_rates(
        loading_a, max(h_a, h_min), t_c=t_a_c, rh=rh_amb, params=params, h_min=h_min
    )
    dc_d, dh_d, m_des = _hydrogel_desorption_rates(
        loading_d,
        max(h_d, h_min),
        t_c=t_d_c,
        t_cond_c=t_cond_c,
        c_vac=c_vac_kg_s_pa_m2,
        params=params,
    )
    rates = SorbentMassRates(dc_a, dc_d, dh_a, dh_d, m_ads, m_des)

    if not equalize:
        return rates

    # Scale bed rates so ṁ_ads = ṁ_des = min(natural fluxes) each step.
    m_eq = min(rates.m_ads_kg_s_m2, rates.m_des_kg_s_m2)
    if m_eq <= 0.0:
        return SorbentMassRates(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    s_ads = m_eq / rates.m_ads_kg_s_m2 if rates.m_ads_kg_s_m2 > 1e-14 else 0.0
    s_des = m_eq / rates.m_des_kg_s_m2 if rates.m_des_kg_s_m2 > 1e-14 else 0.0
    return SorbentMassRates(
        rates.d_loading_a * s_ads,
        rates.d_loading_d * s_des,
        rates.d_h_a * s_ads,
        rates.d_h_d * s_des,
        m_eq,
        m_eq,
    )


def fluxes_for_control(
    *,
    loading_a: float,
    loading_d: float,
    h_a: float,
    h_d: float,
    t_a_c: float,
    t_d_c: float,
    t_cond_c: float,
    rh_amb: float,
    c_vac_kg_s_pa_m2: float,
    config: SystemConfig,
) -> tuple[float, float]:
    rates = mass_rates(
        loading_a=loading_a,
        loading_d=loading_d,
        h_a=h_a,
        h_d=h_d,
        t_a_c=t_a_c,
        t_d_c=t_d_c,
        t_cond_c=t_cond_c,
        rh_amb=rh_amb,
        c_vac_kg_s_pa_m2=c_vac_kg_s_pa_m2,
        config=config,
        equalize=False,
    )
    return rates.m_ads_kg_s_m2, rates.m_des_kg_s_m2
