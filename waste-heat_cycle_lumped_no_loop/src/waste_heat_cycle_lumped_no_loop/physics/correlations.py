"""Heat-transfer correlations for waste-heat two-bed SAWH."""

from __future__ import annotations

import math

STEFAN_BOLTZMANN_W_M2_K4: float = 5.670374419e-8
K_AIR_W_M_K: float = 0.0286
MOLAR_MASS_WATER_KG_MOL: float = 0.018015
R_UNIVERSAL_J_MOL_K: float = 8.314462618


def parallel_plate_emissivity(eps_a: float, eps_b: float) -> float:
    if eps_a <= 0.0 or eps_b <= 0.0:
        return 0.0
    return 1.0 / (1.0 / eps_a + 1.0 / eps_b - 1.0)


def radiative_exchange_w_m2(t_hot_c: float, t_cold_c: float, *, emissivity: float = 0.9) -> float:
    t_hot_k = t_hot_c + 273.15
    t_cold_k = t_cold_c + 273.15
    return emissivity * STEFAN_BOLTZMANN_W_M2_K4 * (t_hot_k**4 - t_cold_k**4)


def rarefied_gap_h_w_m2_k(
    gap_m: float,
    t_hot_c: float,
    t_cold_c: float,
    *,
    p_total_pa: float,
    t_mean_c: float | None = None,
) -> float:
    """Gap conductance at partial vacuum (Knudsen + continuum blend).

    Uses parallel-plate molecular conduction h ≈ k_eff / gap with
    k_eff = k_0 / (1 + Kn) and Kn = λ / gap.
    """
    if gap_m <= 0.0:
        return 0.0
    t_m = t_mean_c if t_mean_c is not None else 0.5 * (t_hot_c + t_cold_c)
    t_k = max(t_m + 273.15, 200.0)
    p = max(p_total_pa, 1.0)
    # Mean free path of water vapor ~ 2e-3 / p(Pa) m (order-of-magnitude at 300 K)
    mean_free_path_m = 2.0e-3 * (101325.0 / p)
    kn = mean_free_path_m / gap_m
    k_eff = K_AIR_W_M_K / (1.0 + kn)
    return max(k_eff / gap_m, K_AIR_W_M_K / (10.0 * gap_m))


def hx_effectiveness_q(
    m_dot_cp_w_k: float,
    ua_w_k: float,
    delta_t_k: float,
) -> float:
    """Q = m_dot cp ΔT (1 - exp(-NTU)); NTU = UA/(m_dot cp)."""
    if abs(delta_t_k) < 1e-12:
        return 0.0
    mdot_cp = max(m_dot_cp_w_k, 1e-12)
    ntu = ua_w_k / mdot_cp
    return mdot_cp * delta_t_k * (1.0 - math.exp(-ntu))


def condenser_h_conv_w_m2_k(h_amb: float, *, fin_area_ratio: float = 7.0) -> float:
    return fin_area_ratio * h_amb


def vacuum_conductance_kg_s_pa_m2(c_vac: float) -> float:
    """Identity map — C_vac setpoint already in kg/(s·Pa·m²)."""
    return max(0.0, float(c_vac))
