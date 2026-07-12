"""Default device parameters for fluid-heated daily-cycle SAWH."""

from __future__ import annotations

# Gel / contactor thermal (per m² footprint)
GEL_THERMAL_MASS_J_M2_K: float = 1.5e5
GEL_EMISSIVITY: float = 1.0

# Loop fluid → gel HX (fixed setpoints during desorption)
T_F_C: float = 58.0
M_DOT_F_KG_S_M2: float = 0.25
UA_GEL_W_K: float = 800.0
FLUID_CP_J_KG_K: float = 4180.0

# Condenser (finned aluminum, Wilson-style)
FIN_AREA_RATIO: float = 7.1
CONDENSER_THICKNESS_M: float = 0.125 * 0.0254
CONDENSER_RHO_KG_M3: float = 2700.0
CONDENSER_CP_J_KG_K: float = 900.0
CONDENSER_EMISSIVITY: float = 0.05
H_FG_J_PER_KG: float = 2.256e6

# Sorbent / geometry (Wilson Table S3)
DEFAULT_SALT_NAME: str = "LiCl"
SALT_TO_POLYMER_RATIO: float = 4.0
H0_M: float = 0.004
VAPOR_GAP_M: float = 0.04
G_CHAMBER_M_S: float = 0.0085
RHO_COMPOSITE_KG_M3: float = 1250.2
TILT_DEG: float = 30.0

# Data-center process air
T_AMB_C: float = 32.0
RH_AMB: float = 0.45
H_AMB_W_M2_K: float = 15.0
