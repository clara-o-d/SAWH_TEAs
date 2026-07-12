"""Salt catalog and PAM-LiCl water-activity models for Wilson Eq. 5."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

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


def _catalog_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "materials" / "salt_catalog.csv"


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, SaltProperties]:
    df = pd.read_csv(_catalog_path())
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
    catalog = _load_catalog()
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


def salt_molarity_from_composite(
    salt_to_polymer_ratio: float,
    hydrogel_density_kg_m3: float,
    formula_weight_g_mol: float,
) -> float:
    """Fixed salt molar concentration c_s (mol/m³ gel) in desorbed composite."""
    f_salt = salt_to_polymer_ratio / (1.0 + salt_to_polymer_ratio)
    mass_salt_kg_m3 = hydrogel_density_kg_m3 * f_salt
    return mass_salt_kg_m3 / (formula_weight_g_mol / 1000.0)


def _pam_licl_dvs_isotherm_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
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
) -> float:
    """Gravimetric moisture content m_w / m_dry (g/g) on a footprint basis."""
    m_dry = pam_licl_dry_mass_kg_m2(h0_ref_m, dry_density_kg_m3=dry_density_kg_m3)
    if m_dry <= 0.0:
        return 0.0
    mass_water = max(0.0, c_w) * h_m * WATER_MOLAR_MASS_KG_MOL
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


def licl_equilibrium_brine_salt_fraction(
    relative_humidity: float,
    temperature_c: float = 25.0,
) -> float:
    """Invert LiCl isotherm: brine salt fraction at equilibrium with RH."""
    rh = float(relative_humidity)
    if rh <= 0.0:
        return 1.0
    if rh >= 0.99:
        return 0.01
    lo, hi = 0.01, 0.75
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        aw = licl_water_activity_at_brine_fraction(mid, temperature_c)
        if not math.isfinite(aw) or aw < rh:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


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


def licl_brine_salt_fraction_from_gel(
    c_w: float,
    *,
    c_s: float,
    h_m: float,
    h0_ref_m: float,
    formula_weight_g_mol: float,
) -> float:
    """Brine salt mass fraction m_s / (m_s + m_w) — LiCl solution a_w,s (Eq. 5)."""
    h = max(h_m, h0_ref_m * 0.25)
    salt_mol_m2 = c_s * h0_ref_m
    mass_salt = salt_mol_m2 * formula_weight_g_mol / 1000.0
    mass_water = max(0.0, c_w) * h * WATER_MOLAR_MASS_KG_MOL
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
) -> float:
    """LiCl brine a_w,s in Eq. 5 (Wilson Device); activity of water in the salt solution."""
    del ions_per_formula
    if c_w <= 0.0 or c_s <= 0.0:
        return 1.0
    h_ref = h0_ref_m if h0_ref_m is not None else 0.004
    h = h_m if h_m is not None else h_ref
    if salt_name == "LiCl":
        f_b = licl_brine_salt_fraction_from_gel(
            c_w,
            c_s=c_s,
            h_m=h,
            h0_ref_m=h_ref,
            formula_weight_g_mol=formula_weight_g_mol,
        )
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
        f_b = licl_equilibrium_brine_salt_fraction(rh, temperature_c)
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
    u = pam_licl_uptake_g_g_at_rh(rh)
    m_dry = dry_density_kg_m3 * h0_ref_m
    mass_water_kg_m2 = u * m_dry
    h = max(h_m, h0_ref_m * 0.25)
    c_w = mass_water_kg_m2 / (h * WATER_MOLAR_MASS_KG_MOL)
    return max(C_W_MIN_MOL_M3, min(C_W_MAX_MOL_M3, c_w))
