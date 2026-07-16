"""Default device parameters for waste-heat two-bed SAWH (data-center baseline)."""

from __future__ import annotations

# Contactor geometry / thermal (per m² footprint)
CONTACTOR_THERMAL_MASS_J_M2_K: float = 1.5e5
CONTACTOR_AREA_M2: float = 1.0
CONTACTOR_EMISSIVITY: float = 0.90

# Vacuum gap (desorbing contactor to condenser)
VACUUM_GAP_M: float = 0.04
P_COND_PA: float = 3000.0  # ~30 mbar

# Contactor-side UA (desorbing contactor <-> its internal heat-transfer surface).
# Formerly the loop-to-desorber UA of the pumped HTF loop; kept as the
# contactor-side leg of the direct waste-heat coupling below.
CONTACTOR_UA_W_K: float = 800.0

# Waste heat (liquid-cooled data center)
T_WH_IN_C: float = 58.0
CP_WH_J_KG_K: float = 4180.0
M_WH_KG_S_M2: float = 0.15
WH_HX_UA_W_K: float = 1200.0

# Direct waste-heat-to-desorber equivalent UA: series combination of the
# waste-heat-stream HX (WH_HX_UA_W_K) and the contactor-side UA
# (CONTACTOR_UA_W_K) that used to sandwich the now-removed pumped HTF loop.
UA_WH_DESORBER_W_K: float = (WH_HX_UA_W_K * CONTACTOR_UA_W_K) / (WH_HX_UA_W_K + CONTACTOR_UA_W_K)

# Vacuum pump conductance (kg / s / Pa / m²)
C_VAC_BASE_KG_S_PA_M2: float = 8.0e-9
C_VAC_MIN_KG_S_PA_M2: float = 1.0e-10
C_VAC_MAX_KG_S_PA_M2: float = 5.0e-6

# Condenser (finned aluminum, Wilson-style)
CONDENSER_GAP_M: float = VACUUM_GAP_M
FIN_AREA_RATIO: float = 7.1
CONDENSER_THICKNESS_M: float = 0.125 * 0.0254
CONDENSER_RHO_KG_M3: float = 2700.0
CONDENSER_CP_J_KG_K: float = 900.0
CONDENSER_EMISSIVITY: float = 0.05
H_FG_J_PER_KG: float = 2.256e6

# Adsorbing-contactor finned heat sink. Without the pumped HTF loop, the
# adsorbing bed's only heat-rejection path is ambient convection, but plain
# convection (H_AMB_W_M2_K=15) gives a cooldown time constant of
# CONTACTOR_THERMAL_MASS_J_M2_K/(H_AMB_W_M2_K*CONTACTOR_AREA_M2) ~= 167 min --
# far slower than a ~90-350 min half-cycle, so the bed can't shed the heat of
# adsorption plus its residual heat from the prior desorbing role before it
# must swap again, and overheats past the waste-heat source temperature over
# repeated swaps. A finned heat sink sized like the device's existing
# condenser fin stack (same FIN_AREA_RATIO) cuts that time constant to
# ~23 min, fast enough to re-equilibrate near ambient within a half-cycle.
CONTACTOR_FIN_AREA_RATIO: float = FIN_AREA_RATIO

# Cycle / control
RH_DESORBER_SWITCH: float = 0.35  # end half-cycle when vapor-gap RH outside desorber ≤ this
TAU_HALF_S: float = 21600.0  # max half-cycle duration (s); RH threshold ends early
K_M_PER_KG_M2: float = 2.0e4
K_P_PER_KG_S_M2: float = 5.0e3

# Data-center process air
T_AMB_C: float = 32.0
RH_AMB: float = 0.45
H_AMB_W_M2_K: float = 15.0

# Sorbent defaults
DEFAULT_SORBENT: str = "hydrogel"
DEFAULT_MOF_NAME: str = "MIL-100_Fe"
DEFAULT_SALT_NAME: str = "LiCl"
SALT_TO_POLYMER_RATIO: float = 4.0
H0_M: float = 0.004
G_CHAMBER_M_S: float = 0.0085
RHO_COMPOSITE_KG_M3: float = 1250.2
VAPOR_GAP_M: float = 0.04
TILT_DEG: float = 30.0
HYDROGEL_MAX_DEPLETION_S: float = 600.0
C_W_MIN_HYDROGEL: float = 100.0

# MOF placeholder
Q_MIN_KG_KG: float = 0.0
Q_MAX_KG_KG: float = 0.53  # MIL-100(Fe) tabulated maximum @ ~99 % RH
Q_REGEN_KG_KG: float = 0.08
