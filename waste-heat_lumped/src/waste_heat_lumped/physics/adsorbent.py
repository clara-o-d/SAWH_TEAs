"""MOF / zeolite adsorbent isotherm and mass-transfer rates (governing_eq.tex Eqs. mass)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from waste_heat_lumped.physics import device_defaults as dd
from waste_heat_lumped.physics.salt_properties import (
    GAS_CONSTANT_J_MOL_K,
    clamp_temperature_c,
    saturation_vapor_pressure_pa,
)


@dataclass(frozen=True, slots=True)
class MofProperties:
    name: str
    q_max_kg_kg: float
    q1_max_kg_kg: float
    k1_pa_inv: float
    q2_max_kg_kg: float
    k2_pa_inv: float
    h_ads_j_per_kg: float
    h_des_j_per_kg: float
    m_ads_kg_m2: float
    g_conv_m_s: float
    price_usd_per_kg: float


def _catalog_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "materials" / "mof_catalog.csv"


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, MofProperties]:
    df = pd.read_csv(_catalog_path())
    out: dict[str, MofProperties] = {}
    for _, row in df.iterrows():
        name = str(row["mof"]).strip()
        out[name] = MofProperties(
            name=name,
            q_max_kg_kg=float(row["q_max_kg_kg"]),
            q1_max_kg_kg=float(row["q1_max_kg_kg"]),
            k1_pa_inv=float(row["K1_pa_inv"]),
            q2_max_kg_kg=float(row["q2_max_kg_kg"]),
            k2_pa_inv=float(row["K2_pa_inv"]),
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


def loading_from_aw_dual_site(
    aw: float,
    *,
    props: MofProperties,
    p_sat_pa: float,
) -> float:
    """Dual-site Langmuir uptake q(aw) at fixed T (K1,K2 referenced to P_sat)."""
    aw = max(0.0, min(1.0, float(aw)))
    k1 = props.k1_pa_inv * p_sat_pa
    k2 = props.k2_pa_inv * p_sat_pa
    q1 = props.q1_max_kg_kg * (k1 * aw) / (1.0 + k1 * aw)
    q2 = props.q2_max_kg_kg * (k2 * aw) / (1.0 + k2 * aw)
    return min(props.q_max_kg_kg, q1 + q2)


def water_activity_from_loading(
    q_kg_kg: float,
    *,
    temperature_c: float,
    props: MofProperties,
) -> float:
    """Invert q(aw) via bisection; aw decreases with loading at fixed T."""
    q = max(0.0, min(props.q_max_kg_kg, float(q_kg_kg)))
    if q <= 1e-12:
        return 0.0
    p_sat = saturation_vapor_pressure_pa(temperature_c)

    def q_at(aw: float) -> float:
        return loading_from_aw_dual_site(aw, props=props, p_sat_pa=p_sat)

    if q >= q_at(1.0) - 1e-10:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if q_at(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def equilibrium_loading_at_rh(
    rh: float,
    *,
    temperature_c: float,
    props: MofProperties,
) -> float:
    p_sat = saturation_vapor_pressure_pa(temperature_c)
    return loading_from_aw_dual_site(rh, props=props, p_sat_pa=p_sat)


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
    # Convert volumetric rate to kg/m²/s using coating inventory scale
    dq_dt = rate_mol_m3_s * 0.018015 / props.m_ads_kg_m2
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
