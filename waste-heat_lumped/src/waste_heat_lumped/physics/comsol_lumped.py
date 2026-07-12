"""Wilson COMSOL lumped 0D model parameters (Model_Lumped_hydrogel_*.mph)."""

from __future__ import annotations

import math

from waste_heat_lumped.physics.salt_properties import (
    GAS_CONSTANT_J_MOL_K,
    WATER_MOLAR_MASS_KG_MOL,
    saturation_vapor_pressure_pa,
)

# --- Geometry (COMSOL defaults) ---
H0_M: float = 0.002
L_G_M: float = 0.04
L_COND_M: float = 0.01  # copper condenser plate
L_HEAT_M: float = 0.001  # absorber / heater plate
L_SILICONE_M: float = 0.0001
L_GLASS_GAP_M: float = 0.005
L_GLASS_M: float = 0.125 * 0.0254
DEVICE_WIDTH_M: float = 0.10  # D = 10 cm

# --- Environment / schedule ---
T_INT_C: float = 23.0
RH_HIGH: float = 0.5
RH_LOW: float = 0.2
Q_SOLAR_W_M2: float = 1000.0
H_FRONT_W_M2_K: float = 10.0
H_COND_W_M2_K: float = 100.0
ABSORPTION_H: float = 12.0
DESORPTION_H: float = 8.0
COOLING_H: float = 12.0

# --- Salt / fabrication IC (Matlab Equilibrium_LiCl @ 20% RH) ---
E0: float = 0.3929
SL: float = 4.0
MW_W_G_MOL: float = 18.0
MW_S_G_MOL: float = 42.4
RHO0: float = 1.0
RHO1: float = 0.540966
RHO2: float = -0.303792
RHO3: float = 0.100791
RHO_WATER_KG_M3: float = 1000.0

# --- Materials ---
G_CHAMBER_M_S: float = 0.0085
H_FG_J_PER_KG: float = 2.256e6
K_AIR_W_M_K: float = 0.0286
K_AL_W_M_K: float = 167.0
K_SILICONE_W_M_K: float = 0.2
K_WATER_W_M_K: float = 0.6
CP_WATER_J_KG_K: float = 4184.0
RHO_COPPER_KG_M3: float = 8933.0
CP_COPPER_J_KG_K: float = 385.0
RHO_AIR_KG_M3: float = 1.0677
NU_AIR_M2_S: float = 1.9984e-5
ALPHA_AIR_M2_S: float = 2.6611e-5
PR_AIR: float = 0.71
D_AIR_M2_S: float = 2.6637e-5
GRAVITY_M_S2: float = 9.8
P_AVG_PA: float = 101325.0
RA_DRY_J_KG_K: float = 287.05
RW_J_KG_K: float = GAS_CONSTANT_J_MOL_K / (MW_W_G_MOL / 1000.0)

# --- Optics (COMSOL glass / absorber) ---
EPS_ADS: float = 0.95
EPS_GLASS_IR: float = 0.8
REFLECT_GLASS: float = 0.2
SOLAR_GLASS_FRAC: float = 0.04
SOLAR_ADS_FRAC: float = 0.92 * 0.95
GLASS_EMIT_BACK_FRAC: float = 0.5

STEFAN_BOLTZMANN_W_M2_K4: float = 5.670374419e-8


def rho_sol_kg_m3(e: float = E0) -> float:
    x = e / (1.0 - e)
    return RHO_WATER_KG_M3 * (RHO0 + RHO1 * x + RHO2 * x**2 + RHO3 * x**3)


RHO_SOL_0_KG_M3: float = rho_sol_kg_m3(E0)


def cs0_mol_m3() -> float:
    return (RHO_SOL_0_KG_M3 / (MW_S_G_MOL / 1000.0)) / (1.0 / E0 + 1.0 / SL)


def cw0_mol_m3() -> float:
    return (
        (RHO_SOL_0_KG_M3 / (MW_W_G_MOL / 1000.0) * (1.0 - E0) / E0)
        / (1.0 / E0 + 1.0 / SL)
    )


CS0_MOL_M3: float = cs0_mol_m3()
CW0_MOL_M3: float = cw0_mol_m3()


def cs_mol_m3(c_w: float, h_m: float, *, h0_m: float = H0_M) -> float:
    lam = h_m / h0_m
    return (
        lam * (CS0_MOL_M3 * MW_S_G_MOL * (1.0 + 1.0 / SL) + CW0_MOL_M3 * MW_W_G_MOL)
        - c_w * MW_W_G_MOL
        - CS0_MOL_M3 * MW_S_G_MOL / SL
    ) / MW_S_G_MOL


def salt_mass_fraction(c_w: float, cs: float) -> float:
    denom = cs * MW_S_G_MOL + c_w * MW_W_G_MOL
    if denom <= 0.0:
        return E0
    return cs * MW_S_G_MOL / denom


def comsol_water_activity(c_w: float, h_m: float, t_gel_c: float, *, h0_m: float = H0_M) -> float:
    """COMSOL activity polynomial (Variables — Activity)."""
    cs = cs_mol_m3(c_w, h_m, h0_m=h0_m)
    e = salt_mass_fraction(c_w, cs)
    e = max(e, 1e-9)
    a = 2.0 - (1.0 + (e / 0.28) ** 4.3) ** 0.6
    b = (1.0 + (e / 0.21) ** 5.1) ** 0.49 - 1.0
    theta = (t_gel_c + 273.15) / (374.0 + 273.0)
    f = a + b * theta
    pi25 = (
        1.0
        - (1.0 + (e / 0.362) ** -4.75) ** -0.4
        - 0.03 * math.exp(-((e - 0.1) ** 2) / 0.005)
    )
    aw = pi25 * f
    return max(0.0, min(1.0, aw))


def humid_air_density_kg_m3(t_c: float) -> float:
    t_k = t_c + 273.15
    psat = saturation_vapor_pressure_pa(t_c)
    return (P_AVG_PA - psat) / RA_DRY_J_KG_K / t_k + psat / RW_J_KG_K / t_k


def comsol_rayleigh_l(gap_m: float, t_gel_c: float, t_cond_c: float) -> float:
    if gap_m <= 0.0:
        return 0.0
    rho_g = humid_air_density_kg_m3(t_gel_c)
    rho_c = humid_air_density_kg_m3(t_cond_c)
    delta_rho = abs(rho_c - rho_g)
    return (
        GRAVITY_M_S2
        * delta_rho
        * gap_m**3
        / (RHO_AIR_KG_M3 * NU_AIR_M2_S * ALPHA_AIR_M2_S)
    )


def comsol_nu_vapor_gap(gap_m: float, t_gel_c: float, t_cond_c: float) -> float:
    ra_l = comsol_rayleigh_l(gap_m, t_gel_c, t_cond_c)
    if ra_l <= 0.0:
        return 1.0
    return 0.22 * ((PR_AIR / (0.2 + PR_AIR)) * ra_l) ** 0.28 * (gap_m / DEVICE_WIDTH_M) ** 0.25


def comsol_h_conv_vapor_gap_w_m2_k(gap_m: float, t_gel_c: float, t_cond_c: float) -> float:
    if gap_m <= 0.0:
        return 0.0
    return comsol_nu_vapor_gap(gap_m, t_gel_c, t_cond_c) * K_AIR_W_M_K / gap_m


def comsol_g_conv_m_s(gap_m: float, t_gel_c: float, t_cond_c: float) -> float:
    h = comsol_h_conv_vapor_gap_w_m2_k(gap_m, t_gel_c, t_cond_c)
    return h * D_AIR_M2_S / K_AIR_W_M_K


def stage_conductance_w_m2_k() -> float:
    return 1.0 / (L_HEAT_M / K_AL_W_M_K + L_SILICONE_M / K_SILICONE_W_M_K)


def gel_thermal_mass_j_m2_k(h0_m: float = H0_M) -> float:
    return RHO_WATER_KG_M3 * CP_WATER_J_KG_K * h0_m


def condenser_thermal_mass_j_m2_k() -> float:
    return RHO_COPPER_KG_M3 * CP_COPPER_J_KG_K * L_COND_M


def concentration_ratio_rp(t_gel_c: float, t_cond_c: float) -> float:
    p_g = saturation_vapor_pressure_pa(t_gel_c)
    p_c = saturation_vapor_pressure_pa(t_cond_c)
    t_g_k = t_gel_c + 273.15
    t_c_k = t_cond_c + 273.15
    if p_g <= 0.0 or t_g_k <= 0.0 or t_c_k <= 0.0:
        return 0.0
    return (p_c / p_g) * (t_g_k / t_c_k)


def absorption_flux_mol_m3_s(
    c_w: float,
    h_m: float,
    t_gel_c: float,
    *,
    h0_m: float = H0_M,
    rh_high: float = RH_HIGH,
) -> float:
    aw = comsol_water_activity(c_w, h_m, t_gel_c, h0_m=h0_m)
    t_k = max(t_gel_c + 273.15, 200.0)
    p_sat = saturation_vapor_pressure_pa(t_gel_c)
    return (rh_high - aw) * p_sat / (GAS_CONSTANT_J_MOL_K * t_k)


def desorption_flux_mol_m3_s(
    c_w: float,
    h_m: float,
    t_gel_c: float,
    t_cond_c: float,
    *,
    h0_m: float = H0_M,
) -> float:
    aw = comsol_water_activity(c_w, h_m, t_gel_c, h0_m=h0_m)
    rp = concentration_ratio_rp(t_gel_c, t_cond_c)
    t_k = max(t_gel_c + 273.15, 200.0)
    p_sat = saturation_vapor_pressure_pa(t_gel_c)
    return p_sat / (GAS_CONSTANT_J_MOL_K * t_k) * (aw - rp)


def mass_rates(
    c_w: float,
    h_m: float,
    t_gel_c: float,
    t_cond_c: float,
    *,
    phase: str,
    h0_m: float = H0_M,
    vapor_gap_m: float = L_G_M,
    rh_high: float = RH_HIGH,
) -> tuple[float, float, float]:
    """Return (dc_w/dt, dH/dt, m_des kg/m²/s) using COMSOL mass ODEs."""
    gap = max(vapor_gap_m - h_m, 0.0)
    if phase == "absorption":
        gm1 = G_CHAMBER_M_S
        flux = absorption_flux_mol_m3_s(c_w, h_m, t_gel_c, h0_m=h0_m, rh_high=rh_high)
    else:
        gm1 = comsol_g_conv_m_s(gap, t_gel_c, t_cond_c)
        flux = -desorption_flux_mol_m3_s(c_w, h_m, t_gel_c, t_cond_c, h0_m=h0_m)

    dc = gm1 / h0_m * flux
    dh = WATER_MOLAR_MASS_KG_MOL / RHO_SOL_0_KG_M3 * gm1 * flux
    if phase == "desorption":
        m_des = max(0.0, -dc * WATER_MOLAR_MASS_KG_MOL * h0_m)
    else:
        m_des = 0.0
    return dc, dh, m_des
