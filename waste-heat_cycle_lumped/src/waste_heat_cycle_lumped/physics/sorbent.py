"""Unified sorbent interface: LiCl hydrogel (default) or MOF placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from waste_heat_cycle_lumped.physics import device_defaults as dd
from waste_heat_cycle_lumped.physics.adsorbent import (
    MofProperties,
    dq_dt_adsorption,
    dq_dt_desorption,
    equilibrium_loading_at_rh,
    get_mof,
    m_ads_kg_s_m2 as mof_m_ads_kg_s_m2,
    m_des_kg_s_m2 as vacuum_m_des_kg_s_m2,
    water_kg_m2 as mof_water_kg_m2,
)
from waste_heat_cycle_lumped.physics.mass_transfer import (
    MassTransferParams,
    concentration_ratio_absorption,
    concentration_ratio_desorption,
    dc_w_dt,
    dH_dt,
    m_ads_kg_s_m2_from_state,
    m_des_kg_s_m2_from_state,
)
from waste_heat_cycle_lumped.physics.salt_properties import (
    FABRICATION_EQUILIBRIUM_RH,
    WATER_MOLAR_MASS_KG_MOL,
    equilibrium_c_w_from_dvs_at_rh,
    get_salt,
    pam_licl_dry_mass_kg_m2,
    pam_licl_gravimetric_uptake_g_g,
    salt_molarity_from_composite,
)

if TYPE_CHECKING:
    from waste_heat_cycle_lumped.simulation.device_config import DeviceConfig

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
        c_s_mol_m3=salt_molarity_from_composite(
            config.salt_to_polymer_ratio,
            config.hydrogel_density_kg_m3,
            s.formula_weight_g_mol,
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
    return mof_water_kg_m2(loading, props=config.mof())


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
            dd.RH_AMB * 0.65,
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
    q_ads = equilibrium_loading_at_rh(dd.RH_AMB * 0.65, temperature_c=dd.T_AMB_C, props=props)
    return BedState(q_ads), BedState(dd.Q_REGEN_KG_KG)


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
    m_vac = vacuum_m_des_kg_s_m2(
        temperature_c=t_c,
        t_cond_c=t_cond_c,
        c_vac_kg_s_pa_m2=c_vac,
        q_kg_kg=None,
        m_ads_kg_m2=None,
    )
    m_vac = min(m_vac, avail / dd.HYDROGEL_MAX_DEPLETION_S if avail > 0 else 0.0)

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
    if c_w + dc_w * 1.0 < dd.C_W_MIN_HYDROGEL:
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
    m_ads = mof_m_ads_kg_s_m2(q_a, temperature_c=t_a, rh_amb=rh, props=props)
    dq_a = dq_dt_adsorption(q_a, temperature_c=t_a, rh_amb=rh, props=props)
    dq_d = dq_dt_desorption(
        q_d,
        temperature_c=t_d,
        t_cond_c=t_cond_c,
        c_vac_kg_s_pa_m2=c_vac,
        props=props,
    )
    m_des = vacuum_m_des_kg_s_m2(
        temperature_c=t_d,
        t_cond_c=t_cond_c,
        c_vac_kg_s_pa_m2=c_vac,
        q_kg_kg=q_d,
        m_ads_kg_m2=props.m_ads_kg_m2,
    )
    return dq_a, dq_d, m_ads, m_des


def _natural_mass_rates(
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
        return SorbentMassRates(dc_a, dc_d, dh_a, dh_d, m_ads, m_des)

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
    return SorbentMassRates(dq_a, dq_d, 0.0, 0.0, m_ads, m_des)


def _equalize_mass_rates(rates: SorbentMassRates, *, config: DeviceConfig) -> SorbentMassRates:
    """Scale bed rates so ṁ_ads = ṁ_des = min(natural fluxes) each step."""
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
    rates = _natural_mass_rates(
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
    )
    if equalize:
        return _equalize_mass_rates(rates, config=config)
    return rates


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
