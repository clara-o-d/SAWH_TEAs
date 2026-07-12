"""MOF adsorbent isotherm and mass-transfer rates (tabulated MIL-100(Fe) @ 303 K)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from waste_heat_cycle_lumped.physics import device_defaults as dd
from waste_heat_cycle_lumped.physics.salt_properties import (
    GAS_CONSTANT_J_MOL_K,
    WATER_MOLAR_MASS_KG_MOL,
    clamp_temperature_c,
    saturation_vapor_pressure_pa,
)


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
    return Path(__file__).resolve().parent.parent / "data" / "materials"


def _catalog_path() -> Path:
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
def _load_catalog() -> dict[str, MofProperties]:
    df = pd.read_csv(_catalog_path())
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
    catalog = _load_catalog()
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
    p_cond_pa: float,
    c_vac_kg_s_pa_m2: float,
    q_kg_kg: float | None = None,
    m_ads_kg_m2: float | None = None,
    max_depletion_s: float = 600.0,
) -> float:
    """Vacuum desorption flux — Eq. massdes in governing_eq.tex."""
    p_sat = saturation_vapor_pressure_pa(temperature_c)
    delta_p = max(0.0, p_sat - p_cond_pa)
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


def dq_dt_desorption(
    q_kg_kg: float,
    *,
    temperature_c: float,
    p_cond_pa: float,
    c_vac_kg_s_pa_m2: float,
    props: MofProperties,
) -> float:
    """d q / dt (kg/kg/s) on desorbing contactor (negative when desorbing)."""
    m_des = m_des_kg_s_m2(
        temperature_c=temperature_c,
        p_cond_pa=p_cond_pa,
        c_vac_kg_s_pa_m2=c_vac_kg_s_pa_m2,
        q_kg_kg=q_kg_kg,
        m_ads_kg_m2=props.m_ads_kg_m2,
    )
    if props.m_ads_kg_m2 <= 0.0:
        return 0.0
    dq = -m_des / props.m_ads_kg_m2
    if q_kg_kg + dq < dd.Q_MIN_KG_KG:
        return max(dq, -q_kg_kg)
    return dq


def water_kg_m2(q_kg_kg: float, *, props: MofProperties) -> float:
    return q_kg_kg * props.m_ads_kg_m2
