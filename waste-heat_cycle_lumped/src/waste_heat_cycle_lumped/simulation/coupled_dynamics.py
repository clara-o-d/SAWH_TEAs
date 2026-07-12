"""Coupled two-bed dynamics (governing_eq.tex)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waste_heat_lumped.physics.contactor_balances import (
    ThermalEnvironment,
    dT_a_dt,
    dT_cond_dt,
    dT_d_dt,
    dT_f_dt,
    loop_heat_fluxes,
)
from waste_heat_lumped.physics.salt_properties import clamp_temperature_c
from waste_heat_lumped.physics.sorbent import (
    fluxes_for_control,
    h_ads_j_per_kg,
    h_des_j_per_kg,
    is_hydrogel,
    mass_rates,
)
from waste_heat_lumped.simulation.control import ControlOutputs, compute_controls
from waste_heat_lumped.simulation.device_config import DeviceConfig


@dataclass(frozen=True, slots=True)
class CoupledRates:
    dy_mass: np.ndarray
    dT_a_dt: float
    dT_d_dt: float
    dT_f_dt: float
    dT_cond_dt: float
    m_ads_kg_s_m2: float
    m_des_kg_s_m2: float
    controls: ControlOutputs
    fluxes: object


def _parse_mass_state(
    state: np.ndarray,
    *,
    config: DeviceConfig,
) -> tuple[float, float, float, float]:
    if is_hydrogel(config):
        return float(state[0]), float(state[1]), float(state[2]), float(state[3])
    return float(state[0]), float(state[1]), config.hydrogel_thickness_m, config.hydrogel_thickness_m


def evaluate_coupled_rates(
    *,
    mass_state: np.ndarray,
    t_a_c: float,
    t_d_c: float,
    t_f_c: float,
    t_cond_c: float,
    env: ThermalEnvironment,
    config: DeviceConfig,
    controls: ControlOutputs,
) -> CoupledRates:
    """Rates for half-cycle: contactor A adsorbs, contactor B desorbs."""
    thermal = config.thermal_params()
    loading_a, h_a, loading_d, h_d = _parse_mass_state(mass_state, config=config)

    t_a = clamp_temperature_c(t_a_c)
    t_d = clamp_temperature_c(t_d_c)
    t_f = clamp_temperature_c(t_f_c)
    t_cond = clamp_temperature_c(t_cond_c)

    sorbent = mass_rates(
        loading_a=loading_a,
        loading_d=loading_d,
        h_a=h_a,
        h_d=h_d,
        t_a_c=t_a,
        t_d_c=t_d,
        t_cond_c=t_cond,
        rh_amb=env.rh_amb,
        p_cond_pa=config.p_cond_pa,
        c_vac_kg_s_pa_m2=controls.c_vac_kg_s_pa_m2,
        config=config,
    )

    if is_hydrogel(config):
        dy_mass = np.array(
            [sorbent.d_loading_a, sorbent.d_h_a, sorbent.d_loading_d, sorbent.d_h_d],
            dtype=float,
        )
    else:
        dy_mass = np.array([sorbent.d_loading_a, sorbent.d_loading_d], dtype=float)

    dta = dT_a_dt(
        t_a_c=t_a,
        t_f_c=t_f,
        m_ads_kg_s_m2=sorbent.m_ads_kg_s_m2,
        h_ads_j_per_kg=h_ads_j_per_kg(config),
        m_dot_f_kg_s_m2=controls.m_dot_f_kg_s_m2,
        params=thermal,
        env=env,
    )
    dtd = dT_d_dt(
        t_d_c=t_d,
        t_f_c=t_f,
        t_cond_c=t_cond,
        m_des_kg_s_m2=sorbent.m_des_kg_s_m2,
        h_des_j_per_kg=h_des_j_per_kg(config),
        m_dot_f_kg_s_m2=controls.m_dot_f_kg_s_m2,
        params=thermal,
        env=env,
    )
    dtf = dT_f_dt(
        t_a_c=t_a,
        t_d_c=t_d,
        t_f_c=t_f,
        m_dot_f_kg_s_m2=controls.m_dot_f_kg_s_m2,
        params=thermal,
        env=env,
    )
    dtcond = dT_cond_dt(
        t_d_c=t_d,
        t_cond_c=t_cond,
        t_amb_c=env.t_amb_c,
        m_des_kg_s_m2=sorbent.m_des_kg_s_m2,
        h_amb_w_m2_k=env.h_amb_w_m2_k,
        params=thermal,
    )
    fluxes = loop_heat_fluxes(
        t_a_c=t_a,
        t_d_c=t_d,
        t_f_c=t_f,
        m_dot_f_kg_s_m2=controls.m_dot_f_kg_s_m2,
        params=thermal,
        env=env,
    )

    return CoupledRates(
        dy_mass=dy_mass,
        dT_a_dt=dta,
        dT_d_dt=dtd,
        dT_f_dt=dtf,
        dT_cond_dt=dtcond,
        m_ads_kg_s_m2=sorbent.m_ads_kg_s_m2,
        m_des_kg_s_m2=sorbent.m_des_kg_s_m2,
        controls=controls,
        fluxes=fluxes,
    )


def controls_for_state(
    *,
    mass_state: np.ndarray,
    t_a_c: float,
    t_d_c: float,
    t_cond_c: float,
    env: ThermalEnvironment,
    config: DeviceConfig,
    integral_ads_kg_m2: float,
    integral_des_kg_m2: float,
) -> ControlOutputs:
    loading_a, h_a, loading_d, h_d = _parse_mass_state(mass_state, config=config)
    ctrl_p = config.controller_params()
    t_a = clamp_temperature_c(t_a_c)
    t_d = clamp_temperature_c(t_d_c)
    m_ads, m_des = fluxes_for_control(
        loading_a=loading_a,
        loading_d=loading_d,
        h_a=h_a,
        h_d=h_d,
        t_a_c=t_a,
        t_d_c=t_d,
        t_cond_c=t_cond_c,
        rh_amb=env.rh_amb,
        p_cond_pa=config.p_cond_pa,
        c_vac_kg_s_pa_m2=ctrl_p.c_vac_base_kg_s_pa_m2,
        config=config,
    )
    return compute_controls(
        t_a_c=t_a,
        t_d_c=t_d,
        m_ads_kg_s_m2=m_ads,
        m_des_kg_s_m2=m_des,
        params=ctrl_p,
        integral_ads_kg_m2=integral_ads_kg_m2,
        integral_des_kg_m2=integral_des_kg_m2,
    )
