"""Feedback control for cycle matching (governing_eq.tex Eq. cycle)."""

from __future__ import annotations

from dataclasses import dataclass

from waste_heat_lumped.simulation.device_config import ControllerParams


@dataclass
class ControllerState:
    integral_ads_kg_m2: float = 0.0
    integral_des_kg_m2: float = 0.0


@dataclass(frozen=True, slots=True)
class ControlOutputs:
    m_dot_f_kg_s_m2: float
    c_vac_kg_s_pa_m2: float


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_controls(
    *,
    t_a_c: float,
    t_d_c: float,
    m_ads_kg_s_m2: float,
    m_des_kg_s_m2: float,
    params: ControllerParams,
    integral_ads_kg_m2: float,
    integral_des_kg_m2: float,
) -> ControlOutputs:
    """Temperature matching → HTF flow; vacuum tracks adsorption-limited desorption.

    Half-cycles end when vapor-gap RH outside the desorber falls below the switch
    threshold. Instantaneous ṁ_ads = ṁ_des is enforced in ``sorbent.mass_rates``;
    vacuum conductance is scaled so the natural desorption capacity follows uptake.
    """
    del integral_ads_kg_m2, integral_des_kg_m2
    delta_t = t_a_c - t_d_c
    m_f = clip(
        params.m_f_base_kg_s_m2 * (1.0 + params.k_t_per_k * delta_t),
        params.m_f_min_kg_s_m2,
        params.m_f_max_kg_s_m2,
    )
    c_vac = clip(
        params.c_vac_base_kg_s_pa_m2
        * max(0.1, m_ads_kg_s_m2 / max(m_des_kg_s_m2, 1e-12)),
        params.c_vac_min_kg_s_pa_m2,
        params.c_vac_max_kg_s_pa_m2,
    )
    return ControlOutputs(m_dot_f_kg_s_m2=m_f, c_vac_kg_s_pa_m2=c_vac)


def advance_controller_integrals(
    state: ControllerState,
    *,
    m_ads_kg_s_m2: float,
    m_des_kg_s_m2: float,
    dt_s: float,
) -> None:
    state.integral_ads_kg_m2 += max(0.0, m_ads_kg_s_m2) * dt_s
    state.integral_des_kg_m2 += max(0.0, m_des_kg_s_m2) * dt_s


def reset_controller_state(state: ControllerState) -> None:
    state.integral_ads_kg_m2 = 0.0
    state.integral_des_kg_m2 = 0.0
