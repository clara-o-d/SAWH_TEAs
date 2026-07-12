"""MOF adsorbent isotherm and mass-transfer rates (tabulated MIL-100(Fe) @ 303 K)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from waste_heat_lumped.physics.mass_transfer import (
    MassTransferPhase,
    mass_transfer_g_m_s,
)
from waste_heat_lumped.physics.salt_properties import (
    FABRICATION_EQUILIBRIUM_RH,
    GAS_CONSTANT_J_MOL_K,
    WATER_MOLAR_MASS_KG_MOL,
    clamp_temperature_c,
    saturation_vapor_pressure_pa,
)

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
