"""Device physics: heat-transfer correlations, default device parameters, brine/salt
thermodynamics, MOF adsorbent isotherms, contactor/loop/condenser energy balances,
mass transfer, and the unified sorbent interface for the two-bed waste-heat SAWH.

Consolidated from the former physics/{correlations, device_defaults, salt_properties,
adsorbent, contactor_balances, mass_transfer, sorbent}.py. Section headers below mark
each former module's boundary for traceability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from waste_heat_cycle_lumped.simulation import DeviceConfig


# =============================================================================
# Heat-transfer correlations for waste-heat two-bed SAWH
# =============================================================================

STEFAN_BOLTZMANN_W_M2_K4: float = 5.670374419e-8
K_AIR_W_M_K: float = 0.0286
MOLAR_MASS_WATER_KG_MOL: float = 0.018015
R_UNIVERSAL_J_MOL_K: float = 8.314462618


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
    """Gap conductance at partial vacuum (Knudsen + continuum blend).

    Uses parallel-plate molecular conduction h ≈ k_eff / gap with
    k_eff = k_0 / (1 + Kn) and Kn = λ / gap.
    """
    if gap_m <= 0.0:
        return 0.0
    t_m = t_mean_c if t_mean_c is not None else 0.5 * (t_hot_c + t_cold_c)
    t_k = max(t_m + 273.15, 200.0)
    p = max(p_total_pa, 1.0)
    # Mean free path of water vapor ~ 2e-3 / p(Pa) m (order-of-magnitude at 300 K)
    mean_free_path_m = 2.0e-3 * (101325.0 / p)
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


def waste_heat_to_loop_q_w(
    *,
    m_dot_wh_kg_s: float,
    cp_wh_j_kg_k: float,
    t_wh_in_c: float,
    t_f_c: float,
    ua_wh_w_k: float,
) -> tuple[float, float]:
    """NTU-epsilon HX from waste-heat stream to loop fluid; returns (Q_wh→f, T_wh_out)."""
    delta = t_wh_in_c - t_f_c
    q = hx_effectiveness_q(m_dot_wh_kg_s * cp_wh_j_kg_k, ua_wh_w_k, delta)
    if m_dot_wh_kg_s <= 1e-12:
        return 0.0, t_wh_in_c
    t_out = t_wh_in_c - q / (m_dot_wh_kg_s * cp_wh_j_kg_k)
    return q, t_out


def vacuum_conductance_kg_s_pa_m2(c_vac: float) -> float:
    """Identity map — C_vac setpoint already in kg/(s·Pa·m²)."""
    return max(0.0, float(c_vac))

# =============================================================================
# Default device parameters for waste-heat two-bed SAWH (data-center baseline)
# =============================================================================

CONTACTOR_THERMAL_MASS_J_M2_K: float = 1.5e5
CONTACTOR_AREA_M2: float = 1.0
CONTACTOR_EMISSIVITY: float = 0.90

# Vacuum gap (desorbing contactor to condenser)
VACUUM_GAP_M: float = 0.04
P_COND_PA: float = 3000.0  # ~30 mbar

# HTF coupling loop
FLUID_THERMAL_MASS_J_M2_K: float = 2.0e4
FLUID_CP_J_KG_K: float = 4180.0
FLUID_RHO_KG_M3: float = 1000.0
UA_ADSORBER_W_K: float = 800.0
UA_DESORBER_W_K: float = 800.0
M_F_BASE_KG_S_M2: float = 0.25
M_F_MIN_KG_S_M2: float = 0.02
M_F_MAX_KG_S_M2: float = 2.0
LOOP_LOSS_FRACTION: float = 0.05

# Waste heat (liquid-cooled data center)
T_WH_IN_C: float = 58.0
CP_WH_J_KG_K: float = 4180.0
M_WH_KG_S_M2: float = 0.15
WH_HX_UA_W_K: float = 1200.0

# Vacuum pump conductance (kg / s / Pa / m²)
C_VAC_BASE_KG_S_PA_M2: float = 8.0e-9
C_VAC_MIN_KG_S_PA_M2: float = 1.0e-10
C_VAC_MAX_KG_S_PA_M2: float = 5.0e-6

# Condenser (finned aluminum, Wilson-style)
CONDENSER_GAP_M: float = VACUUM_GAP_M
FIN_AREA_RATIO: float = 7.1
CONDENSER_THICKNESS_M: float = 0.125 * 0.0254
CONDENSER_RHO_KG_M3: float = 2700.0
CONDENSER_CP_J_KG_K: float = 900.0
CONDENSER_EMISSIVITY: float = 0.05
H_FG_J_PER_KG: float = 2.256e6

# Cycle / control
RH_DESORBER_SWITCH: float = 0.35  # end half-cycle when vapor-gap RH outside desorber ≤ this
TAU_HALF_S: float = 21600.0  # max half-cycle duration (s); RH threshold ends early
K_T_PER_K: float = 0.08
K_M_PER_KG_M2: float = 2.0e4
K_P_PER_KG_S_M2: float = 5.0e3

# Data-center process air
T_AMB_C: float = 32.0
RH_AMB: float = 0.45
H_AMB_W_M2_K: float = 10.0

# Sorbent defaults
DEFAULT_SORBENT: str = "hydrogel"
DEFAULT_MOF_NAME: str = "MIL-100_Fe"
DEFAULT_SALT_NAME: str = "LiCl"
SALT_TO_POLYMER_RATIO: float = 4.0
H0_M: float = 0.004
G_CHAMBER_M_S: float = 0.015
RHO_COMPOSITE_KG_M3: float = 1250.2
VAPOR_GAP_M: float = 0.04
TILT_DEG: float = 30.0
HYDROGEL_MAX_DEPLETION_S: float = 600.0
C_W_MIN_HYDROGEL: float = 100.0

# MOF placeholder
Q_MIN_KG_KG: float = 0.0
Q_MAX_KG_KG: float = 0.53  # MIL-100(Fe) tabulated maximum @ ~99 % RH
Q_REGEN_KG_KG: float = 0.08

# =============================================================================
# Salt catalog and PAM-LiCl water-activity models for Wilson Eq. 5
# =============================================================================

WATER_MOLAR_MASS_KG_MOL: float = 0.018015
GAS_CONSTANT_J_MOL_K: float = 8.314462618
C_W_MAX_MOL_M3: float = 400000.0
C_W_MIN_MOL_M3: float = 100.0
DRY_COMPOSITE_DENSITY_KG_M3: float = 1000.0


@dataclass(frozen=True, slots=True)
class SaltProperties:
    name: str
    formula_weight_g_mol: float
    ions_per_formula: int
    price_usd_per_kg: float
    h_des_j_per_kg: float
    rho_solution_kg_m3: float
    default_sl: float


@lru_cache(maxsize=1)
def _load_salt_catalog() -> dict[str, SaltProperties]:
    df = pd.read_csv(Path(__file__).resolve().parent / "data" / "materials" / "salt_catalog.csv")
    out: dict[str, SaltProperties] = {}
    for _, row in df.iterrows():
        name = str(row["salt"]).strip()
        out[name] = SaltProperties(
            name=name,
            formula_weight_g_mol=float(row["formula_weight_g_mol"]),
            ions_per_formula=int(row["ions_per_formula"]),
            price_usd_per_kg=float(row["price_usd_per_kg"]),
            h_des_j_per_kg=float(row["h_des_j_per_kg"]),
            rho_solution_kg_m3=float(row["rho_solution_kg_m3"]),
            default_sl=float(row["default_sl"]),
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
    """Tetens (Magnus) formula, Pa."""
    t = clamp_temperature_c(temperature_c)
    return 1000.0 * 0.61078 * math.exp(17.27 * t / (t + 237.3))



@lru_cache(maxsize=1)
def _load_pam_licl_dvs_isotherm() -> tuple[np.ndarray, np.ndarray]:
    """Note S2 DVS isotherm: RH (%), gravimetric uptake (g water / g dry composite)."""
    path = Path(__file__).resolve().parent / "data" / "materials" / "PAM-LiCL_isotherm.csv"
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
    h = max(h_m, h0_ref_m * 0.25)
    salt_mol_m2 = c_s * h0_ref_m
    mass_salt = salt_mol_m2 * formula_weight_g_mol / 1000.0
    mass_polymer = mass_salt / max(salt_to_polymer_ratio, 1e-9)
    mass_water = max(0.0, c_w) * h * WATER_MOLAR_MASS_KG_MOL
    total = mass_water + mass_salt + mass_polymer
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
            salt_to_polymer_ratio=salt_to_polymer_ratio,
            h_m=h,
            h0_ref_m=h_ref,
        )
        if aw < rh:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
    if rh <= 0.0:
        return C_W_MIN_MOL_M3
    # Forward DVS isotherm: equilibrium uptake (g/g) at relative humidity.
    rh_pct, uptake = _load_pam_licl_dvs_isotherm()
    r = max(0.0, min(100.0, float(rh) * 100.0))
    u = float(np.interp(r, rh_pct, uptake))
    m_dry = dry_density_kg_m3 * h0_ref_m
    mass_water_kg_m2 = u * m_dry
    h = max(h_m, h0_ref_m * 0.25)
    c_w = mass_water_kg_m2 / (h * WATER_MOLAR_MASS_KG_MOL)
    return max(C_W_MIN_MOL_M3, min(C_W_MAX_MOL_M3, c_w))

# =============================================================================
# MOF adsorbent isotherm and mass-transfer rates (tabulated MIL-100(Fe) @ 303 K)
# =============================================================================

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


@lru_cache(maxsize=8)
def _load_isotherm(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load tabulated isotherm: RH fraction, equilibrium loading q (kg water / kg MOF).

    Source columns: relative pressure (%), H2O uptake (mol/kg). Relative pressure is
    treated as RH at the measurement temperature (303 K).
    """
    path = _materials_dir() / filename
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
    df = pd.read_csv(_materials_dir() / "mof_catalog.csv")
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
    """Forward isotherm q(RH) from tabulated MIL-100(Fe) data at 303 K."""
    del temperature_c  # isotherm measured at 303 K
    rh_tab, q_tab = _load_isotherm(props.isotherm_file)
    rh_clamped = max(0.0, min(1.0, float(rh)))
    return float(np.interp(rh_clamped, rh_tab, q_tab))


def m_ads_kg_s_m2(
    q_kg_kg: float,
    *,
    temperature_c: float,
    rh_amb: float,
    props: MofProperties,
) -> float:
    """Adsorption mass flux (kg/m²/s) — Wilson Eq. 5 analog (open bed)."""
    t_c = clamp_temperature_c(temperature_c)
    t_k = max(t_c + 273.15, 200.0)
    p_sat = saturation_vapor_pressure_pa(t_c)
    aw_eq = water_activity_from_loading(q_kg_kg, temperature_c=t_c, props=props)
    driving = rh_amb - aw_eq
    if driving <= 0.0:
        return 0.0
    rate_mol_m3_s = props.g_conv_m_s * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving
    dq_dt = rate_mol_m3_s * WATER_MOLAR_MASS_KG_MOL / props.m_ads_kg_m2
    return max(0.0, dq_dt * props.m_ads_kg_m2)


def m_des_kg_s_m2(
    *,
    temperature_c: float,
    t_cond_c: float,
    c_vac_kg_s_pa_m2: float,
    q_kg_kg: float | None = None,
    m_ads_kg_m2: float | None = None,
    max_depletion_s: float = 600.0,
) -> float:
    """Vacuum desorption flux — Eq. massdes in governing_eq.tex.

    Driving potential uses saturation pressure at the condenser surface:
    ΔP = P_sat(T_d) − P_sat(T_cond).
    """
    p_sat_des = saturation_vapor_pressure_pa(temperature_c)
    p_sat_cond = saturation_vapor_pressure_pa(t_cond_c)
    delta_p = max(0.0, p_sat_des - p_sat_cond)
    raw = max(0.0, c_vac_kg_s_pa_m2 * delta_p)
    if q_kg_kg is not None and m_ads_kg_m2 is not None and max_depletion_s > 0.0:
        avail_kg_m2 = max(0.0, q_kg_kg) * m_ads_kg_m2
        raw = min(raw, avail_kg_m2 / max_depletion_s)
    return raw


def dq_dt_adsorption(
    q_kg_kg: float,
    *,
    temperature_c: float,
    rh_amb: float,
    props: MofProperties,
) -> float:
    """d q / dt (kg/kg/s) on adsorbing contactor."""
    m_ads = m_ads_kg_s_m2(q_kg_kg, temperature_c=temperature_c, rh_amb=rh_amb, props=props)
    if props.m_ads_kg_m2 <= 0.0:
        return 0.0
    dq = m_ads / props.m_ads_kg_m2
    q_cap = props.q_max_kg_kg - q_kg_kg
    if dq > 0.0 and dq * 1.0 > q_cap:
        return max(0.0, q_cap)
    return dq if q_kg_kg < props.q_max_kg_kg else 0.0


def water_kg_m2(q_kg_kg: float, *, props: MofProperties) -> float:
    return q_kg_kg * props.m_ads_kg_m2

# =============================================================================
# Contactor, loop, and condenser energy balances (governing_eq.tex)
# =============================================================================

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
    fluid_thermal_mass_j_m2_k: float = FLUID_THERMAL_MASS_J_M2_K
    fluid_cp_j_kg_k: float = FLUID_CP_J_KG_K
    ua_adsorber_w_k: float = UA_ADSORBER_W_K
    ua_desorber_w_k: float = UA_DESORBER_W_K
    cp_wh_j_kg_k: float = CP_WH_J_KG_K
    wh_hx_ua_w_k: float = WH_HX_UA_W_K
    loop_loss_fraction: float = LOOP_LOSS_FRACTION
    vacuum_gap_m: float = VACUUM_GAP_M
    p_vacuum_pa: float = P_COND_PA
    fin_area_ratio: float = FIN_AREA_RATIO
    condenser_thermal_mass_j_m2_k: float = (
        CONDENSER_RHO_KG_M3 * CONDENSER_CP_J_KG_K * CONDENSER_THICKNESS_M
    )
    condenser_emissivity: float = CONDENSER_EMISSIVITY
    h_fg_j_per_kg: float = H_FG_J_PER_KG


@dataclass(frozen=True, slots=True)
class ThermalFluxes:
    q_a_to_f_w_m2: float
    q_f_to_d_w_m2: float
    q_wh_to_f_w_m2: float
    q_f_loss_w_m2: float
    q_gap_w_m2: float
    q_rad_d_w_m2: float
    q_rad_cond_w_m2: float
    q_conv_amb_a_w_m2: float
    q_conv_cond_w_m2: float


def loop_heat_fluxes(
    *,
    t_a_c: float,
    t_d_c: float,
    t_f_c: float,
    m_dot_f_kg_s_m2: float,
    params: ContactorThermalParams,
    env: ThermalEnvironment,
) -> ThermalFluxes:
    """Compute Q_a→f, Q_f→d, Q_wh→f, and loss terms (W/m²)."""
    cp_f = params.fluid_cp_j_kg_k
    mdot_cp = m_dot_f_kg_s_m2 * cp_f
    q_a_to_f = hx_effectiveness_q(mdot_cp, params.ua_adsorber_w_k, t_a_c - t_f_c)
    q_f_to_d = hx_effectiveness_q(mdot_cp, params.ua_desorber_w_k, t_f_c - t_d_c)
    q_wh_to_f, _ = waste_heat_to_loop_q_w(
        m_dot_wh_kg_s=env.m_dot_wh_kg_s_m2,
        cp_wh_j_kg_k=params.cp_wh_j_kg_k,
        t_wh_in_c=env.t_wh_in_c,
        t_f_c=t_f_c,
        ua_wh_w_k=params.wh_hx_ua_w_k,
    )
    q_loss = params.loop_loss_fraction * (abs(q_a_to_f) + abs(q_f_to_d))
    return ThermalFluxes(
        q_a_to_f_w_m2=q_a_to_f,
        q_f_to_d_w_m2=q_f_to_d,
        q_wh_to_f_w_m2=q_wh_to_f,
        q_f_loss_w_m2=q_loss,
        q_gap_w_m2=0.0,
        q_rad_d_w_m2=0.0,
        q_rad_cond_w_m2=0.0,
        q_conv_amb_a_w_m2=0.0,
        q_conv_cond_w_m2=0.0,
    )


def dT_a_dt(
    *,
    t_a_c: float,
    t_f_c: float,
    m_ads_kg_s_m2: float,
    h_ads_j_per_kg: float,
    m_dot_f_kg_s_m2: float,
    params: ContactorThermalParams,
    env: ThermalEnvironment,
) -> float:
    flux = loop_heat_fluxes(
        t_a_c=t_a_c,
        t_d_c=t_a_c,
        t_f_c=t_f_c,
        m_dot_f_kg_s_m2=m_dot_f_kg_s_m2,
        params=params,
        env=env,
    )
    q_gen = m_ads_kg_s_m2 * h_ads_j_per_kg
    q_conv = env.h_amb_w_m2_k * params.contactor_area_m2 * (t_a_c - env.t_amb_c)
    rhs = q_gen - q_conv - flux.q_a_to_f_w_m2
    tmass = max(params.contactor_thermal_mass_j_m2_k, 1.0)
    return rhs / tmass


def dT_d_dt(
    *,
    t_d_c: float,
    t_f_c: float,
    t_cond_c: float,
    m_des_kg_s_m2: float,
    h_des_j_per_kg: float,
    m_dot_f_kg_s_m2: float,
    params: ContactorThermalParams,
    env: ThermalEnvironment,
) -> float:
    flux = loop_heat_fluxes(
        t_a_c=t_d_c,
        t_d_c=t_d_c,
        t_f_c=t_f_c,
        m_dot_f_kg_s_m2=m_dot_f_kg_s_m2,
        params=params,
        env=env,
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
        flux.q_f_to_d_w_m2
        - m_des_kg_s_m2 * h_des_j_per_kg
        - q_gap
        - q_rad
    )
    tmass = max(params.contactor_thermal_mass_j_m2_k, 1.0)
    return rhs / tmass


def dT_f_dt(
    *,
    t_a_c: float,
    t_d_c: float,
    t_f_c: float,
    m_dot_f_kg_s_m2: float,
    params: ContactorThermalParams,
    env: ThermalEnvironment,
) -> float:
    flux = loop_heat_fluxes(
        t_a_c=t_a_c,
        t_d_c=t_d_c,
        t_f_c=t_f_c,
        m_dot_f_kg_s_m2=m_dot_f_kg_s_m2,
        params=params,
        env=env,
    )
    rhs = flux.q_a_to_f_w_m2 + flux.q_wh_to_f_w_m2 - flux.q_f_to_d_w_m2 - flux.q_f_loss_w_m2
    tmass = max(params.fluid_thermal_mass_j_m2_k, 1.0)
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

# =============================================================================
# Wilson et al. 2025 Eqs. 5-6 -- PAM-LiCl hydrogel mass transfer
# =============================================================================

MassTransferPhase = Literal["absorption", "desorption"]

# K_AIR_W_M_K already defined above (correlations section); same value, single definition.
D_AIR_M2_S: float = 2.62e-5
GRAVITY_M_S2: float = 9.81
BETA_AIR_K: float = 1.0 / 300.0
NU_AIR_M2_S: float = 1.5e-5
PR_AIR: float = 0.71


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
    # Invert DVS isotherm: water activity from gravimetric uptake.
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
    # Hollands natural-convection correlation between parallel plates (Wilson desorption g ratio).
    if gap_m <= 0.0:
        h_conv = 50.0
    else:
        delta_t = max(abs(t_gel_c - t_cond_c), 0.1)
        ra = GRAVITY_M_S2 * BETA_AIR_K * delta_t * gap_m**3 / (NU_AIR_M2_S * 1.8e-5) * PR_AIR
        if ra < 1708.0:
            h_conv = max(K_AIR_W_M_K / gap_m, 0.5)
        else:
            nu = 0.720 * ra**0.25 * (1.0 + math.cos(math.radians(params.tilt_deg)) * 0.1)
            h_conv = max(nu * K_AIR_W_M_K / gap_m, K_AIR_W_M_K / gap_m)
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

# =============================================================================
# Unified sorbent interface: LiCl hydrogel (default) or MOF placeholder
# =============================================================================

SorbentKind = Literal["hydrogel", "mof"]


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


def is_hydrogel(config: DeviceConfig) -> bool:
    return config.sorbent == "hydrogel"


def mass_state_size(config: DeviceConfig) -> int:
    return 4 if is_hydrogel(config) else 2


def inventory_label(config: DeviceConfig) -> str:
    return "gel" if is_hydrogel(config) else "mof"


def inventory_column(config: DeviceConfig) -> str:
    return "water_in_gel_l_m2" if is_hydrogel(config) else "water_in_mof_l_m2"


def inventory_ylabel(config: DeviceConfig) -> str:
    return "Water in gel (L/m²)" if is_hydrogel(config) else "Water in MOF (L/m²)"


def h_ads_j_per_kg(config: DeviceConfig) -> float:
    if is_hydrogel(config):
        return get_salt(config.salt_name).h_des_j_per_kg
    return config.mof().h_ads_j_per_kg


def h_des_j_per_kg(config: DeviceConfig) -> float:
    if is_hydrogel(config):
        return get_salt(config.salt_name).h_des_j_per_kg
    return config.mof().h_des_j_per_kg


def mass_transfer_params(config: DeviceConfig) -> MassTransferParams:
    s = get_salt(config.salt_name)
    return MassTransferParams(
        g_conv_m_s=config.g_conv_m_s,
        h0_ref_m=config.hydrogel_thickness_m,
        vapor_gap_m=config.vapor_gap_m,
        tilt_deg=config.tilt_deg,
        c_s_mol_m3=(
            config.hydrogel_density_kg_m3
            * (config.salt_to_polymer_ratio / (1.0 + config.salt_to_polymer_ratio))
            / (s.formula_weight_g_mol / 1000.0)
        ),
        ions_per_formula=s.ions_per_formula,
        rho_solution_kg_m3=s.rho_solution_kg_m3,
        salt_name=s.name,
        formula_weight_g_mol=s.formula_weight_g_mol,
        salt_to_polymer_ratio=config.salt_to_polymer_ratio,
    )


def water_kg_m2_bed(loading: float, *, config: DeviceConfig, h_m: float | None = None) -> float:
    if is_hydrogel(config):
        h = h_m if h_m is not None else config.hydrogel_thickness_m
        return WATER_MOLAR_MASS_KG_MOL * loading * h
    return water_kg_m2(loading, props=config.mof())


def water_in_gel_l_m2(
    loading: float,
    h_m: float,
    *,
    config: DeviceConfig,
) -> float:
    """Water in gel (L/m²) on Wilson Fig. S1 DVS gravimetric basis."""
    u = pam_licl_gravimetric_uptake_g_g(loading, h_m, h0_ref_m=config.hydrogel_thickness_m)
    return u * pam_licl_dry_mass_kg_m2(config.hydrogel_thickness_m)


def initial_bed_states(config: DeviceConfig) -> tuple[BedState, BedState]:
    if is_hydrogel(config):
        h0 = config.hydrogel_thickness_m
        c_ads = equilibrium_c_w_from_dvs_at_rh(
            RH_AMB * 0.65,
            h_m=h0,
            h0_ref_m=h0,
        )
        c_regen = equilibrium_c_w_from_dvs_at_rh(
            FABRICATION_EQUILIBRIUM_RH,
            h_m=h0,
            h0_ref_m=h0,
        )
        return BedState(c_ads, h0), BedState(c_regen, h0)
    props = config.mof()
    q_ads = equilibrium_loading_at_rh(RH_AMB * 0.65, temperature_c=T_AMB_C, props=props)
    return BedState(q_ads), BedState(Q_REGEN_KG_KG)


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
    if c_w + dc_w * 1.0 < C_W_MIN_HYDROGEL:
        dc_w = max(dc_w, -c_w)
    return dc_w, dh, m_vac


def _mof_mass_rates(
    q_a: float,
    q_d: float,
    *,
    t_a: float,
    t_d: float,
    t_cond_c: float,
    rh: float,
    c_vac: float,
    props: MofProperties,
) -> tuple[float, float, float, float]:
    m_ads = m_ads_kg_s_m2(q_a, temperature_c=t_a, rh_amb=rh, props=props)
    dq_a = dq_dt_adsorption(q_a, temperature_c=t_a, rh_amb=rh, props=props)
    m_des = m_des_kg_s_m2(
        temperature_c=t_d,
        t_cond_c=t_cond_c,
        c_vac_kg_s_pa_m2=c_vac,
        q_kg_kg=q_d,
        m_ads_kg_m2=props.m_ads_kg_m2,
    )
    # d q / dt (kg/kg/s) on desorbing contactor (negative when desorbing).
    dq_d = 0.0 if props.m_ads_kg_m2 <= 0.0 else -m_des / props.m_ads_kg_m2
    if q_d + dq_d < Q_MIN_KG_KG:
        dq_d = max(dq_d, -q_d)
    return dq_a, dq_d, m_ads, m_des


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
    config: DeviceConfig,
    equalize: bool = True,
) -> SorbentMassRates:
    if is_hydrogel(config):
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
    else:
        props = config.mof()
        dq_a, dq_d, m_ads, m_des = _mof_mass_rates(
            loading_a,
            loading_d,
            t_a=t_a_c,
            t_d=t_d_c,
            t_cond_c=t_cond_c,
            rh=rh_amb,
            c_vac=c_vac_kg_s_pa_m2,
            props=props,
        )
        rates = SorbentMassRates(dq_a, dq_d, 0.0, 0.0, m_ads, m_des)

    if not equalize:
        return rates

    # Scale bed rates so ṁ_ads = ṁ_des = min(natural fluxes) each step.
    m_eq = min(rates.m_ads_kg_s_m2, rates.m_des_kg_s_m2)
    if m_eq <= 0.0:
        return SorbentMassRates(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    s_ads = m_eq / rates.m_ads_kg_s_m2 if rates.m_ads_kg_s_m2 > 1e-14 else 0.0
    s_des = m_eq / rates.m_des_kg_s_m2 if rates.m_des_kg_s_m2 > 1e-14 else 0.0
    if is_hydrogel(config):
        return SorbentMassRates(
            rates.d_loading_a * s_ads,
            rates.d_loading_d * s_des,
            rates.d_h_a * s_ads,
            rates.d_h_d * s_des,
            m_eq,
            m_eq,
        )
    return SorbentMassRates(
        rates.d_loading_a * s_ads,
        rates.d_loading_d * s_des,
        0.0,
        0.0,
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
    config: DeviceConfig,
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
