"""Wilson Note S1 Fig. S1D mass-transfer validation profile (24 h cycle)."""

from __future__ import annotations

from waste_heat_lumped.physics.salt_properties import (
    WATER_MOLAR_MASS_KG_MOL,
    pam_licl_dry_mass_kg_m2,
    pam_licl_gravimetric_uptake_g_g,
)
from waste_heat_lumped.weather.profiles import (
    PHASE_DT_S,
    DailyWeatherProfile,
    PhaseProfile,
)

# Fig. S1D: 50% RH absorption then 800 W/m² desorption (24 h total).
# Note S1 Fig. S1D: paper fit used measured T_gel and T_cond as model inputs;
# this replay self-consistently solves thermal fields (expect some deviation).
FIG_S1_TEMPERATURE_C = 25.0
FIG_S1_ABSORPTION_RH = 0.5
FIG_S1_DESORPTION_SOLAR_W_M2 = 800.0
FIG_S1_H_AMB_W_M2_K = 10.0
FIG_S1_ABSORPTION_HOURS = 16.0
FIG_S1_DESORPTION_HOURS = 8.0
FIG_S1_DT_S = PHASE_DT_S
FIG_S1_ABSORPTION_STEPS = int(round(FIG_S1_ABSORPTION_HOURS * 3600.0 / PHASE_DT_S))
FIG_S1_DESORPTION_STEPS = int(round(FIG_S1_DESORPTION_HOURS * 3600.0 / PHASE_DT_S))

# Experimental endpoints from Fig. S1D (water in gel, L/m²).
FIG_S1_INITIAL_WATER_L_M2 = 1.2
FIG_S1_PEAK_WATER_L_M2 = 2.2
FIG_S1_FINAL_WATER_L_M2 = 1.2


def water_in_gel_l_m2(
    c_w: float,
    h_m: float,
    *,
    h0_ref_m: float = 0.004,
    dvs_basis: bool = True,
) -> float:
    """Water in gel (L/m²). Paper Fig. S1D uses DVS gravimetric basis (g/g × m_dry)."""
    if dvs_basis:
        u = pam_licl_gravimetric_uptake_g_g(c_w, h_m, h0_ref_m=h0_ref_m)
        return u * pam_licl_dry_mass_kg_m2(h0_ref_m)
    return max(0.0, c_w) * h_m * WATER_MOLAR_MASS_KG_MOL


def c_w_from_water_in_gel_l_m2(water_l_m2: float, h_m: float) -> float:
    """Invert water-in-gel inventory to uniform c_w (mol/m³) at thickness h_m."""
    if h_m <= 0.0:
        return 0.0
    return max(0.0, water_l_m2) / (h_m * WATER_MOLAR_MASS_KG_MOL)


def fig_s1_initial_c_w(*, h_m: float = 0.004) -> float:
    """Initial brine state matching Fig. S1D start (~1.2 L/m² at H₀)."""
    return c_w_from_water_in_gel_l_m2(FIG_S1_INITIAL_WATER_L_M2, h_m)


def fig_s1_profile() -> DailyWeatherProfile:
    """Build Note S1 Fig. S1D replay: 16 h @ 50% RH, 8 h @ 800 W/m² solar."""
    n_abs = FIG_S1_ABSORPTION_STEPS
    n_des = FIG_S1_DESORPTION_STEPS
    t = FIG_S1_TEMPERATURE_C
    return DailyWeatherProfile(
        absorption=PhaseProfile(
            temperature_c=(t,) * n_abs,
            relative_humidity=(FIG_S1_ABSORPTION_RH,) * n_abs,
            solar_w_m2=(0.0,) * n_abs,
            h_amb_w_m2_k=(FIG_S1_H_AMB_W_M2_K,) * n_abs,
            dt_s=FIG_S1_DT_S,
        ),
        desorption=PhaseProfile(
            temperature_c=(t,) * n_des,
            relative_humidity=(FIG_S1_ABSORPTION_RH,) * n_des,
            solar_w_m2=(FIG_S1_DESORPTION_SOLAR_W_M2,) * n_des,
            h_amb_w_m2_k=(FIG_S1_H_AMB_W_M2_K,) * n_des,
            dt_s=FIG_S1_DT_S,
        ),
    )
