"""Salt catalog and PAM-LiCl water-activity models for Wilson Eq. 5."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from solar_lumped.physics import table_s3
from solar_lumped.physics.conde2004 import (
    equilibrium_salt_mass_fraction_licl,
    water_activity_licl,
    water_vapor_pressure_pa as conde_water_vapor_pressure_pa,
)

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


def _catalog_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "materials" / "salt_catalog.csv"


def _heat_of_desorption_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
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
def _load_catalog() -> dict[str, SaltProperties]:
    df = pd.read_csv(_catalog_path())
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
    """Pure-water vapour pressure (Pa): Conde (2004) Saul–Wagner, Tetens fallback."""
    t = clamp_temperature_c(temperature_c)
    p = conde_water_vapor_pressure_pa(t)
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


# Dry-basis composite density for all gravimetric-uptake <-> c_w conversions.
# Table S3 reports the composite density at fabrication (25 °C, 20% RH); the DVS
# isotherm uptake is gravimetric per gram of *dry* composite, so the dry-basis
# density is rho_composite(20% RH) / (1 + uptake(20% RH)). Using the wet (20% RH)
# density here would over-count the sorbent dry mass by (1 + u20) ≈ 2.26x and
# inflate the absolute water inventory / desorption swing accordingly.
DRY_COMPOSITE_DENSITY_KG_M3: float = table_s3.RHO_COMPOSITE_KG_M3 / (
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
) -> float:
    """Gravimetric moisture content m_w / m_dry (g/g) on a footprint basis.

    Water inventory is referenced to the fixed fabrication thickness H₀, not the
    swollen H(t): Wilson defines c_w and the desorption flux ṁ_des = MW·H₀·dc_w/dt
    (Note S1) on the H₀ basis, so the sorbate inventory per area is c_w·H₀. The
    swelling H(t) (Eq. 6) enters only the vapor-gap convection (L_g − H) and the
    U_gel conductance (H/k_gel), never the water inventory or its activity. Using
    the swollen H here would double-count dilution and break consistency with the
    yield integral. ``h_m`` is retained for signature compatibility but unused.
    """
    del h_m
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
) -> float:
    """Brine salt mass fraction m_s / (m_s + m_w) — LiCl solution a_w,s (Eq. 5)."""
    del h_m  # inventory referenced to H₀ (see pam_licl_gravimetric_uptake_g_g)
    salt_mol_m2 = c_s * h0_ref_m
    mass_salt = salt_mol_m2 * formula_weight_g_mol / 1000.0
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
) -> float:
    """Brine a_w,s in Eq. 5 (Wilson Device); activity of water in the salt solution."""
    del ions_per_formula, salt_to_polymer_ratio
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
        return float("nan")

    from solar_lumped.physics.brine_equilibrium import (
        brine_salt_fraction_from_c_w,
        water_activity_at_brine_fraction,
    )

    f_b = brine_salt_fraction_from_c_w(c_w, c_s, formula_weight_g_mol)
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
        from solar_lumped.physics.brine_equilibrium import equilibrate_salt_mf

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
