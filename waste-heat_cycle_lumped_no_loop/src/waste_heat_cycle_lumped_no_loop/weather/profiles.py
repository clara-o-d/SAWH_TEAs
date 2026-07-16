"""Operating profiles for waste-heat two-bed SAWH (data-center scenarios)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from waste_heat_cycle_lumped_no_loop.physics import device_defaults as dd
from waste_heat_cycle_lumped_no_loop.physics.sorbent import initial_bed_states

PROFILE_DT_S = 60.0


@dataclass(frozen=True, slots=True)
class HalfCycleProfile:
    """Weather / boundary conditions for one half-cycle (up to max duration τ_1/2,max)."""

    temperature_c: tuple[float, ...]
    relative_humidity: tuple[float, ...]
    h_amb_w_m2_k: tuple[float, ...]
    t_wh_in_c: tuple[float, ...]
    m_dot_wh_kg_s_m2: tuple[float, ...]
    dt_s: float = PROFILE_DT_S


def _steps_for_tau(tau_half_s: float, dt_s: float = PROFILE_DT_S) -> int:
    return max(4, int(round(tau_half_s / dt_s)))


def _constant_profile(
    *,
    n: int,
    t_amb_c: float,
    rh: float,
    h_amb: float,
    t_wh_in_c: float,
    m_dot_wh: float,
    dt_s: float,
) -> HalfCycleProfile:
    return HalfCycleProfile(
        temperature_c=(t_amb_c,) * n,
        relative_humidity=(rh,) * n,
        h_amb_w_m2_k=(h_amb,) * n,
        t_wh_in_c=(t_wh_in_c,) * n,
        m_dot_wh_kg_s_m2=(m_dot_wh,) * n,
        dt_s=dt_s,
    )


def datacenter_baseline_profile(
    *,
    tau_half_s: float | None = None,
    dt_s: float = PROFILE_DT_S,
    t_amb_c: float = dd.T_AMB_C,
    rh: float = dd.RH_AMB,
    h_amb: float = dd.H_AMB_W_M2_K,
    t_wh_in_c: float = dd.T_WH_IN_C,
    m_dot_wh_kg_s_m2: float = dd.M_WH_KG_S_M2,
) -> HalfCycleProfile:
    tau = tau_half_s if tau_half_s is not None else dd.TAU_HALF_S
    n = _steps_for_tau(tau, dt_s)
    return _constant_profile(
        n=n,
        t_amb_c=t_amb_c,
        rh=rh,
        h_amb=h_amb,
        t_wh_in_c=t_wh_in_c,
        m_dot_wh=m_dot_wh_kg_s_m2,
        dt_s=dt_s,
    )


def datacenter_diurnal_profile(
    *,
    tau_half_s: float | None = None,
    dt_s: float = PROFILE_DT_S,
    t_amb_mean_c: float = dd.T_AMB_C,
    t_amb_amp_c: float = 3.0,
    rh_mean: float = dd.RH_AMB,
    rh_amp: float = 0.08,
    h_amb: float = dd.H_AMB_W_M2_K,
    t_wh_in_c: float = dd.T_WH_IN_C,
    m_dot_wh_kg_s_m2: float = dd.M_WH_KG_S_M2,
) -> HalfCycleProfile:
    tau = tau_half_s if tau_half_s is not None else dd.TAU_HALF_S
    n = _steps_for_tau(tau, dt_s)
    temps: list[float] = []
    rhs: list[float] = []
    for i in range(n):
        phase = 2.0 * math.pi * i / n
        temps.append(t_amb_mean_c + t_amb_amp_c * math.sin(phase))
        rhs.append(max(0.05, min(0.95, rh_mean + rh_amp * math.cos(phase))))
    return HalfCycleProfile(
        temperature_c=tuple(temps),
        relative_humidity=tuple(rhs),
        h_amb_w_m2_k=(h_amb,) * n,
        t_wh_in_c=(t_wh_in_c,) * n,
        m_dot_wh_kg_s_m2=(m_dot_wh_kg_s_m2,) * n,
        dt_s=dt_s,
    )


def initial_loadings(config) -> tuple[float, float]:
    """Default (loading_adsorbing, loading_desorbing) at cycle start."""
    bed_a, bed_d = initial_bed_states(config)
    return bed_a.loading, bed_d.loading
