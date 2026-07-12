"""Unified sorbent interface: PAM-salt hydrogel (default) or MOF coating."""

from __future__ import annotations

from typing import TYPE_CHECKING

from waste_heat_lumped.physics.adsorbent import (
    DEFAULT_MOF_NAME,
    MofProperties,
    Q_MIN_KG_KG,
    SorbentKind,
    fabrication_q_initial,
    get_mof,
    m_flux_kg_s_m2_from_dq,
    water_kg_m2,
)
from waste_heat_lumped.physics.mass_transfer import (
    MassTransferParams,
    concentration_ratio_absorption,
    concentration_ratio_desorption,
    dH_dt,
    dc_w_dt,
    m_des_kg_s_m2_from_dc_w,
)
from waste_heat_lumped.physics.salt_properties import fabrication_c_w_initial
from waste_heat_lumped.weather.fig_s1 import water_in_gel_l_m2

if TYPE_CHECKING:
    from waste_heat_lumped.simulation.device_config import DeviceConfig

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
        from waste_heat_lumped.physics.mass_transfer import C_W_MAX_MOL_M3, C_W_MIN_MOL_M3

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

    from waste_heat_lumped.physics.adsorbent import dq_dt, mof_mass_transfer_g_m_s

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
