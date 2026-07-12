"""Contactor, loop, and condenser energy balances (governing_eq.tex)."""

from __future__ import annotations

from dataclasses import dataclass

from waste_heat_cycle_lumped.physics import device_defaults as dd
from waste_heat_cycle_lumped.physics.correlations import (
    condenser_h_conv_w_m2_k,
    hx_effectiveness_q,
    parallel_plate_emissivity,
    radiative_exchange_w_m2,
    rarefied_gap_h_w_m2_k,
    waste_heat_to_loop_q_w,
)
from waste_heat_cycle_lumped.physics.salt_properties import clamp_temperature_c


@dataclass(frozen=True, slots=True)
class ThermalEnvironment:
    t_amb_c: float
    rh_amb: float
    h_amb_w_m2_k: float
    t_wh_in_c: float
    m_dot_wh_kg_s_m2: float


@dataclass(frozen=True, slots=True)
class ContactorThermalParams:
    contactor_thermal_mass_j_m2_k: float = dd.CONTACTOR_THERMAL_MASS_J_M2_K
    contactor_area_m2: float = dd.CONTACTOR_AREA_M2
    contactor_emissivity: float = dd.CONTACTOR_EMISSIVITY
    fluid_thermal_mass_j_m2_k: float = dd.FLUID_THERMAL_MASS_J_M2_K
    fluid_cp_j_kg_k: float = dd.FLUID_CP_J_KG_K
    ua_adsorber_w_k: float = dd.UA_ADSORBER_W_K
    ua_desorber_w_k: float = dd.UA_DESORBER_W_K
    cp_wh_j_kg_k: float = dd.CP_WH_J_KG_K
    wh_hx_ua_w_k: float = dd.WH_HX_UA_W_K
    loop_loss_fraction: float = dd.LOOP_LOSS_FRACTION
    vacuum_gap_m: float = dd.VACUUM_GAP_M
    p_vacuum_pa: float = dd.P_COND_PA
    fin_area_ratio: float = dd.FIN_AREA_RATIO
    condenser_thermal_mass_j_m2_k: float = (
        dd.CONDENSER_RHO_KG_M3 * dd.CONDENSER_CP_J_KG_K * dd.CONDENSER_THICKNESS_M
    )
    condenser_emissivity: float = dd.CONDENSER_EMISSIVITY
    h_fg_j_per_kg: float = dd.H_FG_J_PER_KG


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
