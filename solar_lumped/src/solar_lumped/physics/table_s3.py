"""Wilson & Díaz-Marín (*Device* 2025) Table S3 / Note S1 COMSOL device parameters."""

from __future__ import annotations

# Geometry
H0_M: float = 0.004  # hydrogel reference thickness H₀ (m)
L_G_M: float = 0.04  # vapor gap L_g (m)
L_INS_M: float = 0.005  # insulation gap L_ins (m)
L_C_M: float = 0.005  # condenser aluminum plate thickness L_c (m)
L_GLASS_IN: float = 0.125  # cover glass thickness (in)
L_GLASS_M: float = L_GLASS_IN * 0.0254
L_AL_STACK_M: float = L_C_M  # aluminum in gel–absorber stack (Table S3 L_c)
L_SILICONE_M: float = 0.001  # silicone coating (m)

# Wilson §2.2 / Note S1: thermobuoyancy and mass transport inhibited below ~7 mm gap
VAPOR_GAP_TRANSPORT_MIN_M: float = 0.007

# Materials / transport
G_CHAMBER_M_S: float = 0.0085  # g_chamber (m/s)
RHO_SOL_KG_M3: float = 1250.2  # ρ_sol brine solution density (kg/m³)
RHO_COMPOSITE_KG_M3: float = 1250.2  # composite density at fabrication (25 °C, 20% RH), Table S3
H_DES_J_PER_KG: float = 2.32e6  # h_des (J/kg)
H_FG_J_PER_KG: float = 2.256e6  # h_fg condensation (J/kg)
K_AIR_W_M_K: float = 0.0286  # k_air (W/m·K)
K_AL_W_M_K: float = 167.0  # k_al (W/m·K) — Table S3
K_SILICONE_W_M_K: float = 0.2  # k_silicone (W/m·K)
K_GEL_W_M_K: float = 0.6  # k_w hydrogel (W/m·K) — Table S3
K_GLASS_W_M_K: float = 1.2  # k_glass (W/m·K)
RHO_AL_KG_M3: float = 2700.0
CP_AL_J_KG_K: float = 900.0
RHO_GLASS_KG_M3: float = 2230.0  # borosilicate cover
CP_GLASS_J_KG_K: float = 830.0
CP_GEL_J_KG_K: float = 3500.0  # hydrated PAM-LiCl composite (water-dominated)

# Optical / radiative
EPS_GEL: float = 1.0
EPS_AL: float = 0.05
EPS_ABS: float = 0.95
EPS_GLASS: float = 0.9  # outer glass (not tabulated; typical low-e cover)
TAU_GLASS: float = 0.9

# Device orientation / condenser fins
TILT_DEG: float = 30.0
FIN_AREA_RATIO: float = 7.1  # A_r

# Backward-compatible aliases
L_AL_M: float = L_C_M


def u_gel_w_m2_k(h_m: float) -> float:
    """Note S1 cumulative gel–absorber conductance (series resistances).

    1/U_gel = L_al/k_al + L_silicone/k_silicone + H(t)/k_hydrogel
    """
    h = max(float(h_m), H0_M * 0.25)
    resistance = (
        L_AL_STACK_M / K_AL_W_M_K
        + L_SILICONE_M / K_SILICONE_W_M_K
        + h / K_GEL_W_M_K
    )
    return 1.0 / resistance


# Reference value at fabrication thickness H₀ (for tests / docs)
U_GEL_W_M2_K: float = u_gel_w_m2_k(H0_M)

# Condenser thermal mass per footprint area (ρ_al c_p L_c)
CONDENSER_THERMAL_MASS_J_M2_K: float = RHO_AL_KG_M3 * CP_AL_J_KG_K * L_C_M

# Lumped thermal capacitances per footprint area (J/m²K) for transient desorption
# solvers. Physical (ρ c_p L) values from Table S3 — no calibration factors.
GLASS_THERMAL_MASS_J_M2_K: float = RHO_GLASS_KG_M3 * CP_GLASS_J_KG_K * L_GLASS_M
ABSORBER_THERMAL_MASS_J_M2_K: float = RHO_AL_KG_M3 * CP_AL_J_KG_K * L_AL_STACK_M


def gel_thermal_mass_j_m2_k(h_m: float) -> float:
    """(ρ c_p H)_gel — Note S1 Eq. S1 hydrogel thermal storage per footprint area."""
    return RHO_COMPOSITE_KG_M3 * CP_GEL_J_KG_K * max(float(h_m), H0_M * 0.25)
