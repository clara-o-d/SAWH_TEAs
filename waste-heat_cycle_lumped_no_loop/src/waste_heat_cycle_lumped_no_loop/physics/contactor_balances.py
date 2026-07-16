"""Contactor and condenser energy balances (governing_eq.tex).

The pumped HTF loop that used to couple the waste-heat stream to the
adsorbing/desorbing contactors has been removed. The desorbing contactor is
now coupled directly to the fixed waste-heat source through a single
equivalent UA (the series combination of the old waste-heat-to-loop HX and
loop-to-desorber HX; see ``device_defaults.UA_WH_DESORBER_W_K``).
"""

from __future__ import annotations

from dataclasses import dataclass

from waste_heat_cycle_lumped_no_loop.physics import device_defaults as dd
from waste_heat_cycle_lumped_no_loop.physics.correlations import (
    condenser_h_conv_w_m2_k,
    hx_effectiveness_q,
    parallel_plate_emissivity,
    radiative_exchange_w_m2,
    rarefied_gap_h_w_m2_k,
)
from waste_heat_cycle_lumped_no_loop.physics.salt_properties import clamp_temperature_c


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
    cp_wh_j_kg_k: float = dd.CP_WH_J_KG_K
    ua_wh_desorber_w_k: float = dd.UA_WH_DESORBER_W_K
    contactor_fin_area_ratio: float = dd.CONTACTOR_FIN_AREA_RATIO
    vacuum_gap_m: float = dd.VACUUM_GAP_M
    p_vacuum_pa: float = dd.P_COND_PA
    fin_area_ratio: float = dd.FIN_AREA_RATIO
    condenser_thermal_mass_j_m2_k: float = (
        dd.CONDENSER_RHO_KG_M3 * dd.CONDENSER_CP_J_KG_K * dd.CONDENSER_THICKNESS_M
    )
    condenser_emissivity: float = dd.CONDENSER_EMISSIVITY
    h_fg_j_per_kg: float = dd.H_FG_J_PER_KG


def dT_a_dt(
    *,
    t_a_c: float,
    m_ads_kg_s_m2: float,
    h_ads_j_per_kg: float,
    params: ContactorThermalParams,
    env: ThermalEnvironment,
) -> float:
    """Adsorbing-contactor energy balance: heat of adsorption rejected to ambient.

    Rejected through a finned heat sink (same fin sizing as the condenser),
    since ambient convection is now this bed's only heat-rejection path and
    plain convection alone is too slow to shed the heat of adsorption plus
    the bed's residual heat from its prior desorbing role within one
    half-cycle -- see the ``CONTACTOR_FIN_AREA_RATIO`` comment in
    ``device_defaults.py``.
    """
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
