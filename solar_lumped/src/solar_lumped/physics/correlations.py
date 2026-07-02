"""Heat-transfer correlations (Hollands et al. 1976; Wilson Note S1)."""

from __future__ import annotations

import math

from solar_lumped.physics import table_s3

STEFAN_BOLTZMANN_W_M2_K4: float = 5.670374419e-8
K_AIR_W_M_K: float = table_s3.K_AIR_W_M_K
D_AIR_M2_S: float = 2.62e-5  # H2O in air ~25 °C (Note S1 Sh = Nu analogy)
GRAVITY_M_S2: float = 9.81
BETA_AIR_K: float = 1.0 / 300.0
NU_AIR_M2_S: float = 1.5e-5
PR_AIR: float = 0.71


def parallel_plate_emissivity(eps_a: float, eps_b: float) -> float:
    """Note S1 Eq. S2 — infinite parallel plates."""
    if eps_a <= 0.0 or eps_b <= 0.0:
        return 0.0
    return 1.0 / (1.0 / eps_a + 1.0 / eps_b - 1.0)


def mass_transfer_g_from_h_conv_m_s(h_conv_w_m2_k: float) -> float:
    """Note S1 Eq. S5 (Le ≈ 1): g = h_conv · D_air / k_air."""
    return h_conv_w_m2_k * D_AIR_M2_S / K_AIR_W_M_K


def radiative_exchange_w_m2(t_hot_c: float, t_cold_c: float, *, emissivity: float = 0.9) -> float:
    t_hot_k = t_hot_c + 273.15
    t_cold_k = t_cold_c + 273.15
    return emissivity * STEFAN_BOLTZMANN_W_M2_K4 * (t_hot_k**4 - t_cold_k**4)


def conduction_air_gap_w_m2(t_hot_c: float, t_cold_c: float, gap_m: float) -> float:
    if gap_m <= 0.0:
        return 0.0
    return K_AIR_W_M_K / gap_m * (t_hot_c - t_cold_c)


def hollands_vapor_gap_h_conv_w_m2_k(
    gap_m: float,
    t_hot_c: float,
    t_cold_c: float,
    *,
    tilt_deg: float = 35.0,
) -> float:
    """Natural convection between parallel plates (Hollands et al. 1976 approximation)."""
    if gap_m <= 0.0:
        return 50.0
    delta_t = max(abs(t_hot_c - t_cold_c), 0.1)
    t_mean_k = 0.5 * (t_hot_c + t_cold_c) + 273.15
    ra = GRAVITY_M_S2 * BETA_AIR_K * delta_t * gap_m**3 / (NU_AIR_M2_S * 1.8e-5) * PR_AIR
    ra_crit = 1708.0
    if ra < ra_crit:
        return max(K_AIR_W_M_K / gap_m, 0.5)
    nu = 0.720 * ra**0.25 * (1.0 + math.cos(math.radians(tilt_deg)) * 0.1)
    k_eff = nu * K_AIR_W_M_K / gap_m
    return max(k_eff, K_AIR_W_M_K / gap_m)


def wind_to_h_amb_w_m2_k(wind_speed_m_s: float, *, base: float = 10.0) -> float:
    """Map 10 m wind speed to external convection coefficient (paper: ~10 at 0.5 m/s)."""
    w = max(0.0, float(wind_speed_m_s))
    return base * (0.5 + w) / 1.0


def condenser_h_conv_w_m2_k(h_amb: float, *, fin_area_ratio: float = 7.0) -> float:
    return fin_area_ratio * h_amb
