"""Wilson & Díaz-Marín (*Device* 2025) Table S3 / Note S1 COMSOL device parameters."""

from __future__ import annotations

# Geometry
H0_M: float = 0.004  # hydrogel reference thickness H₀ (m)
L_G_M: float = 0.04  # vapor gap L_g (m)
L_INS_M: float = 0.005  # insulation gap (m)
L_AL_IN: float = 0.125  # aluminum condenser / absorber plate thickness (in)
L_AL_M: float = L_AL_IN * 0.0254

# Materials / transport
G_CHAMBER_M_S: float = 0.0085  # g_chamber (m/s)
RHO_SOL_KG_M3: float = 1250.2  # ρ_sol (kg/m³)
RHO_COMPOSITE_KG_M3: float = 1250.2  # dry composite density (kg/m³)
H_DES_J_PER_KG: float = 2.32e6  # h_des (J/kg)
H_FG_J_PER_KG: float = 2.256e6  # h_fg condensation (J/kg)
K_AIR_W_M_K: float = 0.0286  # k_air (W/m·K)
K_AL_W_M_K: float = 237.0  # aluminum thermal conductivity (W/m·K)
RHO_AL_KG_M3: float = 2700.0
CP_AL_J_KG_K: float = 900.0

# Optical / radiative
EPS_GEL: float = 1.0
EPS_AL: float = 0.05
EPS_ABS: float = 0.95
TAU_GLASS: float = 0.9

# Device orientation / condenser fins
TILT_DEG: float = 30.0
FIN_AREA_RATIO: float = 7.1  # A_r

# Gel–absorber coupling: U_gel = k_al / L_al (Table S3)
U_GEL_W_M2_K: float = K_AL_W_M_K / L_AL_M

# Condenser thermal mass per footprint area (ρ_al c_p L_al)
CONDENSER_THERMAL_MASS_J_M2_K: float = RHO_AL_KG_M3 * CP_AL_J_KG_K * L_AL_M
