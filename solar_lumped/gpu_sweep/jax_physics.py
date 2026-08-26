"""JAX port of the Wilson quasi-steady desorption RHS (Note S1 Eqs. 1-6 + Eq. 2).

Covers the note_s1/quasi_steady path only: scipy root/brentq solves become
fixed-iteration Newton/bisection, and only the LiCl brine branch is ported. All
functions are pure and vmap-safe (every branch is a jnp.where) and need float64 --
c_w is O(1e4) mol/m3, so float32 loses the precision atol=1e-7 assumes."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from solar_lumped._parameters_xlsx import physics_value as _pv
# Same m_des clamp the CPU path brackets its bisection with, so the joint Newton solve
# below can't wander outside the range evaluate_coupled_rates would have searched.
from solar_lumped.simulation import _M_DES_BRACKET_MAX
# Same g -> infinity scale as physics.mass_transfer_g_m_s, so both backends idealize
# by the identical factor.
from solar_lumped.physics import _INSTANT_EQUILIBRIUM_G_SCALE
# Gel/condenser clearance and its detection tolerance. Imported, not restated, so the
# two backends cannot drift apart on the number that caps swelling in both.
from solar_lumped.physics import GEL_CONDENSER_CLEARANCE_M, SWELLING_CAP_TOL_M
# Tsilingiris (2008) / Marrero & Mason (1972) coefficients and validity bounds are
# imported, not restated: a duplicated polynomial is a silent backend divergence waiting
# to happen, and physics.py carries the provenance comments for all of them.
from solar_lumped.physics import (
    _AIR_PROPS_T_HI_C,
    _AIR_PROPS_T_LO_C,
    _ATM_IN_PA,
    _CP_A_KJ_KG_K,
    _CP_V_KJ_KG_K,
    _D_H2O_AIR_A_ATM_CM2,
    _D_H2O_AIR_S,
    _D_H2O_AIR_T_HI_C,
    _D_H2O_AIR_T_LO_C,
    _K_A_W_M_K,
    _K_V_W_M_K,
    _M_AIR_KG_KMOL,
    _MU_A_NS_M2,
    _MU_V_NS_M2,
    _RHO_AIR_REF_KG_M3,
)

jax.config.update("jax_enable_x64", True)

# ---- Table S3 / Note S1 constants -- loaded from docs/parameters.xlsx, same
# values as solar_lumped/src/solar_lumped/physics.py. ----
H0_M = _pv("Hydrogel reference thickness (H0)", mm_to_m=True)
L_G_M = _pv("Vapor gap (L_g)", mm_to_m=True)
L_INS_M = _pv("Insulation gap (L_ins)", mm_to_m=True)
L_C_M = _pv("Condenser aluminum plate thickness (L_c)", mm_to_m=True)
VAPOR_GAP_TRANSPORT_MIN_M = _pv("Vapor-gap transport floor", mm_to_m=True)

RHO_GEL_KG_M3 = _pv("Composite (hydrogel) density at 20% RH (rho_gel)")
RHO_COMPOSITE_KG_M3 = RHO_GEL_KG_M3
H_DES_J_PER_KG = _pv("Desorption enthalpy, LiCl (h_des)")
H_FG_J_PER_KG = _pv("Condensation enthalpy (h_fg)")
K_GEL_W_M_K = _pv("Hydrogel thermal conductivity (k_gel)")
RABS_M2_K_W = _pv("Absorber-to-gel constant resistance (Rabs)")
RHO_AL_KG_M3 = _pv("Aluminum density (rho_Al)")
CP_AL_J_KG_K = _pv("Aluminum specific heat (cp_Al)")
# Table S3 k_Al, for complex mode's (B3) fin-efficiency parameter m = sqrt(2h/(k t)).
K_AL_W_M_K = _pv("Aluminum thermal conductivity (k_Al)")

EPS_GEL = _pv("Gel emissivity (eps_gel)")
EPS_AL = _pv("Condenser (Al) emissivity (eps_Al)")
# Case 2 (selective surface) IR emissivities -- same values as physics.py's
# EPS_ABS_IR_CASE2/EPS_GLASS_IR_CASE2.
EPS_ABS_IR_CASE2 = _pv("Absorber IR emissivity (eps_abs_ir)")
EPS_GLASS_IR_CASE2 = _pv("Glass IR emissivity (eps_glass_ir)")

STEFAN_BOLTZMANN = _pv("Stefan-Boltzmann constant (sigma)")
GRAVITY_M_S2 = _pv("Gravitational acceleration (g)")
P_ATM_SEA_LEVEL_PA = _pv("Atmospheric pressure (P0)")
# h_amb density derate; physics owns the exponent and the reference density.
H_AMB_DENSITY_EXPONENT = _pv("Ambient convection density exponent")
# ISO 15099 Eq. 52 aspect ratio A_g,i = height/gap; mirrors physics.CAVITY_HEIGHT_M.
CAVITY_HEIGHT_M = _pv("Cavity height (H_cav)")

WATER_MOLAR_MASS_KG_MOL = _pv("Water molar mass (MW_w)")
GAS_CONSTANT_J_MOL_K = _pv("Universal gas constant (R)")
# Tsilingiris works per kmol; mirrors the same two derived values in physics.py.
_M_H2O_KG_KMOL = WATER_MOLAR_MASS_KG_MOL * 1e3
_R_J_KMOL_K = GAS_CONSTANT_J_MOL_K * 1e3
# No C_W_MAX/C_W_MIN module constants: both gel-water bounds are per-instance
# (salt- and blend-dependent) and ride on MassParams. See physics.hydrate_floor_c_w
# and physics.dilution_ceiling_c_w.

CONDENSER_THERMAL_MASS_J_M2_K = RHO_AL_KG_M3 * CP_AL_J_KG_K * L_C_M

# Conde (2004) Table 3 LiCl vapour-pressure correlation parameters.
_PI0, _PI1, _PI2, _PI3, _PI4 = 0.28, 4.30, 0.60, 0.21, 5.10
_PI5, _PI6, _PI7, _PI8, _PI9 = 0.49, 0.362, -4.75, -0.40, 0.03
T_CRIT_H2O_K = _pv("Water critical temperature (T_crit,H2O)")
P_CRIT_H2O_PA = _pv("Water critical pressure (P_crit,H2O)")
_SAUL_WAGNER_A = jnp.array([-7.858230, 1.839910, -11.781100, 22.670500, -15.939300, 1.775160])
_SAUL_WAGNER_EXP = jnp.array([1.0, 1.5, 3.0, 3.5, 4.0, 7.5])

# LiCl saturation brine mass fraction: past saturation the excess salt is solid and a_w
# pins here. Temperature-resolved from Conde's crystallization line (physics owns the
# coefficients, imported rather than duplicated) because saturation concentration rises
# with temperature -- 0.458 at 25 C, 0.528 at 80 C -- and this clip is what sets the
# regime-2 activity plateau.
from solar_lumped.physics import _CRYSTALLIZATION_LINE as _CRYST
from solar_lumped.physics import BET_T_MAX_C, BET_T_MIN_C, CONDE_T_MAX_C
from solar_lumped.physics import LICL_BET as _BET


def xi_sat_licl(temperature_c):
    """Lowest xi at which any LiCl solid phase can crystallize, at this temperature.

    Mirrors physics.saturation_brine_salt_fraction("LiCl", T): the branch is whichever
    admits the smallest root, since the solution saturates as soon as some solid can
    form. The branch loop unrolls at trace time (coefficients are Python constants), so
    this stays one fused expression per call and is vmap-safe."""
    theta = (temperature_c + 273.15) / T_CRIT_H2O_K
    best = jnp.full_like(jnp.asarray(theta, dtype=float), jnp.inf)
    for a0, a1, a2 in _CRYST["LiCl"]:
        if a2 == 0.0:
            x = (theta - a0) / a1
            best = jnp.where((x > 0.0) & (x < 1.0), jnp.minimum(best, x), best)
            continue
        disc = a1 * a1 - 4.0 * a2 * (a0 - theta)
        root = jnp.sqrt(jnp.clip(disc, 0.0, None))
        for sign in (1.0, -1.0):
            x = (-a1 + sign * root) / (2.0 * a2)
            ok = (disc >= 0.0) & (x > 0.0) & (x < 1.0)
            best = jnp.where(ok, jnp.minimum(best, x), best)
    return best

TEMP_CLAMP_LO_C = _pv("Temperature clamp lower bound")
TEMP_CLAMP_HI_C = _pv("Temperature clamp upper bound")


def clamp_temperature_c(t_c):
    t_c = jnp.where(jnp.isfinite(t_c), t_c, 25.0)
    return jnp.clip(t_c, TEMP_CLAMP_LO_C, TEMP_CLAMP_HI_C)


def saturation_vapor_pressure_pa(temperature_c):
    """Saul-Wagner pure-water vapour pressure (Conde 2004 Appendix A)."""
    t_c = clamp_temperature_c(temperature_c)
    t_k = t_c + 273.15
    tau = 1.0 - t_k / T_CRIT_H2O_K
    numer = jnp.sum(_SAUL_WAGNER_A * tau**_SAUL_WAGNER_EXP)
    ln_p_pc = numer / (1.0 - tau)
    return P_CRIT_H2O_PA * jnp.exp(ln_p_pc)


def _bet_water_activity_licl(xi, temperature_c):
    """Zeng & Zhou (2006) eq 3 solved for a_w; mirrors physics.bet_water_activity."""
    t_k = jnp.clip(temperature_c, BET_T_MIN_C, BET_T_MAX_C) + 273.15
    m = 1000.0 * xi / (_BET.formula_weight_g_mol * (1.0 - xi))
    r = _BET.r0 + _BET.r1 * t_k
    c = jnp.exp(-(_BET.e0 + _BET.e1 * t_k) / (GAS_CONSTANT_J_MOL_K * t_k))
    n = 1000.0 / 18.015
    a, b, k = n * (c - 1.0), m * c * r - n * (c - 2.0), -n
    disc = jnp.clip(b * b - 4.0 * a * k, 0.0, None)
    return jnp.clip((-b + jnp.sqrt(disc)) / (2.0 * a), 0.0, 1.0)


def _bet_temperature_factor_licl(xi, temperature_c):
    """a_w(xi, T)/a_w(xi, 100 C) -- the >100 C slope Conde lacks. See physics."""
    t = jnp.minimum(temperature_c, BET_T_MAX_C)
    ref = _bet_water_activity_licl(xi, CONDE_T_MAX_C)
    factor = _bet_water_activity_licl(xi, t) / jnp.clip(ref, 1e-30, None)
    return jnp.where((ref > 0.0) & (temperature_c > CONDE_T_MAX_C), factor, 1.0)


def vapor_pressure_ratio_licl(xi, temperature_c):
    """pi = p_sol / p_H2O = brine water activity a_w (Conde 2004 Table 3, LiCl).

    Above CONDE_T_MAX_C the Conde part is held at 100 C and scaled by the Zeng & Zhou
    BET temperature ratio, mirroring physics.vapor_pressure_ratio."""
    xi = jnp.clip(xi, 1e-12, xi_sat_licl(temperature_c))
    theta = (jnp.minimum(temperature_c, CONDE_T_MAX_C) + 273.15) / T_CRIT_H2O_K
    pi25 = 1.0 - (1.0 + (xi / _PI6) ** _PI7) ** _PI8 - _PI9 * jnp.exp(-((xi - 0.1) ** 2) / 0.005)
    a_term = 2.0 - (1.0 + (xi / _PI0) ** _PI1) ** _PI2
    b_term = (1.0 + (xi / _PI3) ** _PI4) ** _PI5 - 1.0
    f = a_term + b_term * theta
    return jnp.clip(pi25 * f * _bet_temperature_factor_licl(xi, temperature_c), 0.0, 1.0)


def licl_brine_salt_fraction_from_gel(c_w, *, c_s, h0_ref_m, formula_weight_g_mol):
    """m_s / (m_s + m_w), footprint basis referenced to H0 (salt_properties.py)."""
    salt_mol_m2 = c_s * h0_ref_m
    mass_salt = salt_mol_m2 * formula_weight_g_mol / 1000.0
    mass_water = jnp.clip(c_w, 0.0, None) * h0_ref_m * WATER_MOLAR_MASS_KG_MOL
    total = mass_salt + mass_water
    return jnp.where(total <= 0.0, 1.0, mass_salt / jnp.clip(total, 1e-30, None))


def water_activity_licl_from_c_w(c_w, *, c_s, h0_ref_m, formula_weight_g_mol, temperature_c):
    f_b = licl_brine_salt_fraction_from_gel(
        c_w, c_s=c_s, h0_ref_m=h0_ref_m, formula_weight_g_mol=formula_weight_g_mol
    )
    aw = vapor_pressure_ratio_licl(f_b, jnp.minimum(temperature_c, BET_T_MAX_C))
    return jnp.where(c_w <= 0.0, 1.0, aw)


_ISOSTERIC_DT_K = _pv("Isosteric h_des finite-difference step")
_ISOSTERIC_T_MAX_C_LICL = BET_T_MAX_C


def isosteric_h_des_j_per_kg(xi, temperature_c):
    """Clausius-Clapeyron h_des at fixed composition; mirrors physics.isosteric_h_des_j_per_kg.

    LiCl only -- this backend ports no other brine. vapor_pressure_ratio_licl re-clips xi
    to xi_sat_licl(T) internally at each probe point, exactly as the CPU path does, so a
    saturated gel gets the saturated-slurry slope on both backends."""
    t_c = jnp.minimum(temperature_c, _ISOSTERIC_T_MAX_C_LICL - _ISOSTERIC_DT_K)

    t1, t2 = t_c - _ISOSTERIC_DT_K, t_c + _ISOSTERIC_DT_K
    # Clamp on the cold probe's saturation so both probes stay in one regime -- see the
    # note in physics.isosteric_h_des_j_per_kg; straddling it reads the dissolution step.
    xi_c = jnp.clip(xi, 1e-6, xi_sat_licl(t1))

    def _ln_p(t):
        aw = jnp.clip(vapor_pressure_ratio_licl(xi_c, t), 1e-300, None)
        return jnp.log(aw * saturation_vapor_pressure_pa(t))

    slope = (_ln_p(t2) - _ln_p(t1)) / (1.0 / (t2 + 273.15) - 1.0 / (t1 + 273.15))
    h = -GAS_CONSTANT_J_MOL_K * slope / WATER_MOLAR_MASS_KG_MOL
    return jnp.where(jnp.isfinite(h) & (h > 0.0), h, H_DES_J_PER_KG)


def parallel_plate_emissivity(eps_a, eps_b):
    # jnp.asarray first: as plain Python floats, "1.0 / 0.0" raises eagerly before
    # jnp.where can mask it; as arrays it is a lazy inf that jnp.where discards.
    eps_a = jnp.asarray(eps_a)
    eps_b = jnp.asarray(eps_b)
    return jnp.where((eps_a <= 0.0) | (eps_b <= 0.0), 0.0, 1.0 / (1.0 / eps_a + 1.0 / eps_b - 1.0))


def radiative_exchange_w_m2(t_hot_c, t_cold_c, emissivity):
    t_hot_k = t_hot_c + 273.15
    t_cold_k = t_cold_c + 273.15
    return emissivity * STEFAN_BOLTZMANN * (t_hot_k**4 - t_cold_k**4)


def _poly(coeffs, x):
    """Horner in ascending coefficient order; mirrors physics._poly."""
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def humid_air_props(t_film_c, x_v=0.0, p_atm_pa=P_ATM_SEA_LEVEL_PA):
    """(k, nu, alpha) of humid air at a film temperature; mirrors physics.humid_air_props.

    Unlike the CPU version this has no dry-air early return -- at x_v = 0 the mixing
    rules reduce to the dry-air values exactly (each vapor term is 0/positive), so one
    traced path serves both gaps and there is no branch to diverge.
    """
    t = jnp.clip(t_film_c, _AIR_PROPS_T_LO_C, _AIR_PROPS_T_HI_C)
    t_k = t + 273.15
    xv = jnp.clip(x_v, 0.0, 1.0)
    xa = 1.0 - xv

    mu_a = _poly(_MU_A_NS_M2, t_k) * 1e-6      # Tsilingiris Eq. (38)
    k_a = _poly(_K_A_W_M_K, t_k)               # Eq. (39) -- already W/m K
    cp_a = _poly(_CP_A_KJ_KG_K, t_k) * 1e3     # Eq. (40)
    mu_v = _poly(_MU_V_NS_M2, t) * 1e-7        # Eq. (41) -- 1e-7, not the printed 1e-6
    k_v = _poly(_K_V_W_M_K, t) * 1e-3          # Eq. (42)
    cp_v = _poly(_CP_V_KJ_KG_K, t) * 1e3       # Eq. (43)

    m_ratio = _M_AIR_KG_KMOL / _M_H2O_KG_KMOL
    root2_4 = jnp.sqrt(2.0) / 4.0
    phi_av = root2_4 / jnp.sqrt(1.0 + m_ratio) * (
        1.0 + jnp.sqrt(mu_a / mu_v) * (1.0 / m_ratio) ** 0.25
    ) ** 2
    phi_va = root2_4 / jnp.sqrt(1.0 + 1.0 / m_ratio) * (
        1.0 + jnp.sqrt(mu_v / mu_a) * m_ratio**0.25
    ) ** 2

    mu = xa * mu_a / (xa + xv * phi_av) + xv * mu_v / (xv + xa * phi_va)
    k = xa * k_a / (xa + xv * phi_av) + xv * k_v / (xv + xa * phi_va)
    cp = (cp_a * xa * _M_AIR_KG_KMOL + cp_v * xv * _M_H2O_KG_KMOL) / (
        _M_AIR_KG_KMOL * xa + _M_H2O_KG_KMOL * xv
    )
    rho = p_atm_pa * _M_AIR_KG_KMOL / (_R_J_KMOL_K * t_k) * (1.0 - xv * (1.0 - 1.0 / m_ratio))
    return k, mu / rho, k / (rho * cp)


def vapor_gap_water_mole_fraction(t_cond_c, p_atm_pa=P_ATM_SEA_LEVEL_PA):
    """x_v pinned by the condenser; mirrors physics.vapor_gap_water_mole_fraction.

    The `<= 0 C -> dry` guard mirrors the CPU path deliberately: there
    water_vapor_pressure_pa returns NaN over ice and the caller falls back to dry air,
    whereas this backend's saturation_vapor_pressure_pa clamps instead of failing. Without
    the explicit where, the two would disagree on sub-zero iterates.
    """
    p_v = saturation_vapor_pressure_pa(t_cond_c)
    return jnp.where(t_cond_c <= 0.0, 0.0, jnp.clip(p_v / p_atm_pa, 0.0, 1.0))


def d_h2o_air_m2_s(t_c, p_atm_pa=P_ATM_SEA_LEVEL_PA):
    """Marrero & Mason (1972) Table 13 air-H2O; mirrors physics.d_h2o_air_m2_s."""
    t_k = jnp.clip(t_c, _D_H2O_AIR_T_LO_C, _D_H2O_AIR_T_HI_C) + 273.15
    return _D_H2O_AIR_A_ATM_CM2 * t_k**_D_H2O_AIR_S / (p_atm_pa / _ATM_IN_PA) * 1e-4


def h_amb_density_factor(t_amb_c, p_atm_pa=P_ATM_SEA_LEVEL_PA, exponent=H_AMB_DENSITY_EXPONENT):
    """(rho_amb/rho_ref)^n; mirrors physics.h_amb_density_factor."""
    t_k = jnp.clip(t_amb_c + 273.15, 1.0, None)
    rho = jnp.clip(p_atm_pa, 0.0, None) * _M_AIR_KG_KMOL / (_R_J_KMOL_K * t_k)
    return (rho / _RHO_AIR_REF_KG_M3) ** exponent


def conduction_air_gap_w_m2(t_hot_c, t_cold_c, gap_m, p_atm_pa=P_ATM_SEA_LEVEL_PA):
    """Dry-air conduction across one sealed glazing gap, k at THAT gap's film temperature.

    Called separately for absorber-glass and pane-pane, which sit at genuinely different
    temperatures -- mirrors the _gap_conduction_w_m2 closure in physics._residuals.
    """
    k_air = humid_air_props(0.5 * (t_hot_c + t_cold_c), 0.0, p_atm_pa)[0]
    return jnp.where(gap_m <= 0.0, 0.0, k_air / jnp.clip(gap_m, 1e-30, None) * (t_hot_c - t_cold_c))


def _rayleigh_vapor_gap(gap_m, t_hot_c, t_cold_c, x_v=0.0, p_atm_pa=P_ATM_SEA_LEVEL_PA):
    """Note S1 Eq. S3 — exact Δρ buoyancy (ideal-gas air, ρ_air at film temperature);
    mirrors physics.rayleigh_vapor_gap. Not the Boussinesq β·ΔT linearization. ν and α are
    evaluated at the same film temperature, so one temperature serves the whole group."""
    t_hot_k = jnp.clip(t_hot_c + 273.15, 1.0, None)
    t_cold_k = jnp.clip(t_cold_c + 273.15, 1.0, None)
    t_film_k = 0.5 * (t_hot_k + t_cold_k)
    delta_t = jnp.clip(jnp.abs(t_hot_k - t_cold_k), 1e-6, None)
    d_rho_over_rho = t_film_k * delta_t / (t_hot_k * t_cold_k)
    _, nu_air, alpha_air = humid_air_props(t_film_k - 273.15, x_v, p_atm_pa)
    return GRAVITY_M_S2 * d_rho_over_rho * gap_m**3 / (nu_air * alpha_air)


def hollands_nu_eq_s3(ra, *, tilt_deg):
    cos_t = jnp.clip(jnp.cos(jnp.radians(tilt_deg)), 1e-6, None)
    ra_cos = ra * cos_t
    sin_18t_16 = jnp.sin(jnp.radians(1.8 * tilt_deg)) ** 1.6
    f1 = jnp.clip(1.0 - 1708.0 * sin_18t_16 / jnp.clip(ra_cos, 1e-30, None), 0.0, None)
    f2 = jnp.clip(1.0 - 1708.0 / jnp.clip(ra_cos, 1e-30, None), 0.0, None)
    f3 = jnp.clip((jnp.clip(ra_cos, 0.0, None) / 5830.0) ** (1.0 / 3.0) - 1.0, 0.0, None)
    nu = 1.0 + 1.44 * f1 * f2 + f3
    return jnp.where(ra_cos <= 0.0, 1.0, nu)


def iso15099_nu_vertical(ra, aspect_ratio):
    """ISO 15099 Eqs. 48-52 (vertical cavity); mirrors physics.iso15099_nu_vertical."""
    ra_s = jnp.clip(ra, 1e-30, None)
    nu1 = jnp.where(
        ra_s > 5e4,
        0.0673838 * ra_s ** (1.0 / 3.0),
        jnp.where(ra_s > 1e4, 0.028154 * ra_s**0.4134, 1.0 + 1.7596678e-10 * ra_s**2.2984755),
    )
    nu2 = 0.242 * (ra_s / jnp.clip(aspect_ratio, 1e-6, None)) ** 0.272
    return jnp.maximum(1.0, jnp.maximum(nu1, nu2))


def vapor_gap_h_conv_w_m2_k(gap_m, t_gel_c, t_cond_c, *, tilt_deg, p_atm_pa=P_ATM_SEA_LEVEL_PA):
    """Heated-from-above ISO 15099 sec. 5.3.3.5 gap; mirrors physics.vapor_gap_h_conv_w_m2_k.
    Hollands (heated-from-below) only for the inverted condenser-hotter case."""
    gap = jnp.clip(gap_m, 1e-30, None)
    # Humid, at this gap's own film temperature -- the condenser pins x_v. The glazing
    # gaps go through conduction_air_gap_w_m2 with their own (hotter, dry) film temps.
    x_v = vapor_gap_water_mole_fraction(t_cond_c, p_atm_pa)
    k_air = humid_air_props(0.5 * (t_gel_c + t_cond_c), x_v, p_atm_pa)[0]
    ra = _rayleigh_vapor_gap(gap, t_gel_c, t_cond_c, x_v, p_atm_pa)
    nu_v = iso15099_nu_vertical(ra, CAVITY_HEIGHT_M / gap)
    nu_above = 1.0 + (nu_v - 1.0) * jnp.sin(jnp.radians(180.0 - tilt_deg))
    nu = jnp.where(t_cond_c > t_gel_c, hollands_nu_eq_s3(ra, tilt_deg=tilt_deg), nu_above)
    return jnp.where(gap_m <= 0.0, 0.0, nu * k_air / gap)


def condenser_h_conv_w_m2_k(h_amb, *, fin_area_ratio, fin_thickness_m=None, fin_height_m=None):
    """Wilson's ideal fin (``A_r * h_amb``) unless complex mode (B3) supplies geometry,
    in which case the added fin area is derated by the straight-fin efficiency. Mirrors
    solar_lumped.physics.condenser_h_conv_w_m2_k."""
    if fin_thickness_m is None or fin_height_m is None:
        return fin_area_ratio * h_amb
    t = jnp.clip(fin_thickness_m, 1e-6, None)
    length = jnp.clip(fin_height_m, 1e-6, None)
    m = jnp.sqrt(2.0 * jnp.clip(h_amb, 0.0, None) / (K_AL_W_M_K * t))
    ml = jnp.clip(m * length, 1e-9, None)
    eta_f = jnp.tanh(ml) / ml
    # Only the finned area (A_r - 1) is derated; the exposed base plate is not.
    return h_amb * (1.0 + eta_f * jnp.clip(fin_area_ratio - 1.0, 0.0, None))


def blend_water_activity_from_c_w(
    c_w, *, c_s, h0_ref_m, formula_weight_g_mol, temperature_c, aw_table
):
    """B8 water activity by bilinear lookup on a host-built (T, f_b) -> a_w table.

    ``aw_table`` is ``(t_grid_c, fb_grid, table)`` from
    ``complex_model.zsr_inverse_table``, computed on the host because blend weights
    are constant for a whole instance. That is what lets the ZSR inversion -- a root
    solve on the CPU path -- run inside a jitted ODE right-hand side at all.
    """
    t_grid, fb_grid, table = aw_table
    f_b = licl_brine_salt_fraction_from_gel(
        c_w, c_s=c_s, h0_ref_m=h0_ref_m, formula_weight_g_mol=formula_weight_g_mol
    )
    # Fractional indices, edge-clamped: outside the tabulated range the nearest
    # column is the right answer (it is the saturated / most-dilute limit).
    fi = jnp.interp(temperature_c, t_grid, jnp.arange(t_grid.shape[0], dtype=t_grid.dtype))
    fj = jnp.interp(f_b, fb_grid, jnp.arange(fb_grid.shape[0], dtype=fb_grid.dtype))
    i0 = jnp.clip(jnp.floor(fi).astype(jnp.int32), 0, t_grid.shape[0] - 1)
    j0 = jnp.clip(jnp.floor(fj).astype(jnp.int32), 0, fb_grid.shape[0] - 1)
    i1 = jnp.minimum(i0 + 1, t_grid.shape[0] - 1)
    j1 = jnp.minimum(j0 + 1, fb_grid.shape[0] - 1)
    di, dj = fi - i0, fj - j0
    aw = (
        (1.0 - di) * (1.0 - dj) * table[i0, j0]
        + (1.0 - di) * dj * table[i0, j1]
        + di * (1.0 - dj) * table[i1, j0]
        + di * dj * table[i1, j1]
    )
    return jnp.where(c_w <= 0.0, 1.0, aw)


def mass_transfer_g_from_h_conv_m_s(h_conv, t_gel_c, t_cond_c, p_atm_pa=P_ATM_SEA_LEVEL_PA):
    """Note S1 Eq. S5 (Le ≈ 1); mirrors physics.mass_transfer_g_from_h_conv_m_s. D_air and
    k_air are both taken at the vapor gap's film temperature."""
    t_film_c = 0.5 * (t_gel_c + t_cond_c)
    x_v = vapor_gap_water_mole_fraction(t_cond_c, p_atm_pa)
    k_air = humid_air_props(t_film_c, x_v, p_atm_pa)[0]
    return jnp.where(h_conv <= 0.0, 0.0, h_conv * d_h2o_air_m2_s(t_film_c, p_atm_pa) / k_air)


def u_gel_w_m2_k(h_m):
    h = jnp.clip(h_m, H0_M * 0.25, None)
    resistance = RABS_M2_K_W + h / K_GEL_W_M_K
    return 1.0 / resistance


def concentration_ratio_desorption(t_gel_c, t_cond_c):
    p_g = saturation_vapor_pressure_pa(t_gel_c)
    p_c = saturation_vapor_pressure_pa(t_cond_c)
    t_g_k = t_gel_c + 273.15
    t_c_k = t_cond_c + 273.15
    ratio = (p_c / jnp.clip(p_g, 1e-30, None)) * (t_g_k / jnp.clip(t_c_k, 1e-30, None))
    return jnp.where((p_g <= 0.0) | (t_g_k <= 0.0) | (t_c_k <= 0.0), 0.0, ratio)


class MassParams:
    """Mirrors solar_lumped.physics.MassTransferParams for LiCl."""

    def __init__(
        self, *, h0_ref_m, vapor_gap_m, tilt_deg, c_s_mol_m3, formula_weight_g_mol,
        c_w_min_mol_m3, c_w_max_mol_m3, g_conv_m_s=0.015, aw_table=None,
        instant_equilibrium=False, p_atm_pa=P_ATM_SEA_LEVEL_PA,
    ):
        self.h0_ref_m = h0_ref_m
        self.vapor_gap_m = vapor_gap_m
        self.tilt_deg = tilt_deg
        self.c_s_mol_m3 = c_s_mol_m3
        self.formula_weight_g_mol = formula_weight_g_mol
        # Per-instance gel-water bounds from physics.hydrate_floor_c_w and
        # physics.dilution_ceiling_c_w -- salt- and blend-dependent, so they are traced
        # per-instance values, not module constants.
        self.c_w_min_mol_m3 = c_w_min_mol_m3
        self.c_w_max_mol_m3 = c_w_max_mol_m3
        self.g_conv_m_s = g_conv_m_s
        # Complex mode (B8): host-built (T, f_b) -> a_w table. None keeps the
        # closed-form LiCl isotherm, which is what the simple model uses.
        self.aw_table = aw_table
        # Ideal case: g -> infinity (static, uniform across the batch). Mirrors
        # SystemConfig.instant_equilibrium / MassTransferParams.instant_equilibrium.
        self.instant_equilibrium = instant_equilibrium
        # Site ambient pressure; mirrors MassTransferParams.p_atm_pa.
        self.p_atm_pa = p_atm_pa


def desorption_driving_force(c_w, *, t_gel_c, t_cond_c, h_m, mass: MassParams):
    """Eq. 5's desorption driving force c_r - a_w (dimensionless).

    Negative means the gel's surface vapour pressure exceeds the condenser's, i.e. it
    desorbs. Zero is the local-equilibrium constraint that instant_equilibrium imposes --
    see _joint_residuals, which uses THIS rather than a g-scaled rate as its fourth
    residual. Mirrors physics._mass_transfer_driving_force's desorption branch.
    """
    c_r = concentration_ratio_desorption(t_gel_c, t_cond_c)
    if mass.aw_table is None:
        aw = water_activity_licl_from_c_w(
            c_w, c_s=mass.c_s_mol_m3, h0_ref_m=mass.h0_ref_m,
            formula_weight_g_mol=mass.formula_weight_g_mol, temperature_c=t_gel_c,
        )
    else:
        aw = blend_water_activity_from_c_w(
            c_w, c_s=mass.c_s_mol_m3, h0_ref_m=mass.h0_ref_m,
            formula_weight_g_mol=mass.formula_weight_g_mol, temperature_c=t_gel_c,
            aw_table=mass.aw_table,
        )
    return jnp.where(jnp.isfinite(aw), c_r - aw, 0.0)


def dc_dh_desorption(c_w, *, t_gel_c, t_cond_c, h_m, mass: MassParams):
    """Eqs. 5-6 desorption branch (dc_w/dt, dH/dt), before the clip-to-<=0 in sorbent.py."""
    gap_m = jnp.clip(mass.vapor_gap_m - h_m, 0.0, None)
    if mass.instant_equilibrium:
        # g -> infinity. Static branch, same as the CPU path.
        g = _INSTANT_EQUILIBRIUM_G_SCALE * mass.g_conv_m_s
    else:
        h_conv = vapor_gap_h_conv_w_m2_k(
            gap_m, t_gel_c, t_cond_c, tilt_deg=mass.tilt_deg, p_atm_pa=mass.p_atm_pa
        )
        # No thermobuoyancy cutoff; see physics.mass_transfer_g_m_s for why.
        g = mass_transfer_g_from_h_conv_m_s(h_conv, t_gel_c, t_cond_c, mass.p_atm_pa)

    driving = desorption_driving_force(
        c_w, t_gel_c=t_gel_c, t_cond_c=t_cond_c, h_m=h_m, mass=mass
    )

    t_k = jnp.clip(t_gel_c + 273.15, 200.0, None)
    p_sat = saturation_vapor_pressure_pa(t_gel_c)
    pref = g / mass.h0_ref_m
    rate = pref * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving
    rate = jnp.where(jnp.isfinite(rate), rate, 0.0)
    dc = jnp.where((c_w >= mass.c_w_max_mol_m3) & (rate > 0.0), 0.0, rate)
    dc = jnp.where((c_w <= mass.c_w_min_mol_m3) & (dc < 0.0), 0.0, dc)

    dh = (
        g
        * WATER_MOLAR_MASS_KG_MOL
        / RHO_GEL_KG_M3
        * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k))
        * driving
    )
    dh = jnp.where(jnp.isfinite(dh), dh, 0.0)

    dc = jnp.minimum(dc, 0.0)
    dh = jnp.minimum(dh, 0.0)
    # Bounded at the hydrate floor, not at the as-cast H0 -- mirrors the CPU
    # evaluate_mass_rates so the two backends stop the gel shrinking together.
    dh = jnp.where(c_w <= mass.c_w_min_mol_m3, 0.0, dh)
    return dc, dh


def m_des_kg_s_m2_from_dc_w(dc_w_dt_val, *, h0_ref_m):
    return jnp.where(dc_w_dt_val >= 0.0, 0.0, -dc_w_dt_val * WATER_MOLAR_MASS_KG_MOL * h0_ref_m)


class ThermalParams:
    """Mirrors SystemThermalParams for the note_s1/has_glass=True Atacama baseline. The
    default eps_abs_ir/eps_glass_ir is Case 2 (selective surface); pass 1.0/1.0 for
    Case 1's original Wilson Eqs. 3/4 blackbody approximation, or 0.0/0.0 for Case 3."""

    def __init__(
        self, *, insulation_gap_m, vapor_gap_m, eps_abs, tau_glass, tilt_deg,
        eps_abs_ir=EPS_ABS_IR_CASE2, eps_glass_ir=EPS_GLASS_IR_CASE2,
        # Complex mode (B2). ``n_glazing_panes`` and ``evacuated_gap`` may be traced
        # per-instance arrays; only ``complex_mode`` (set on the builder) is static,
        # since it fixes how many unknowns the Newton solve carries.
        n_glazing_panes=1.0, evacuated_gap=0.0, complex_mode=False,
        # Static: mirrors SystemConfig.h_des_mode == "isosteric" on the CPU path.
        h_des_isosteric=False,
        p_atm_pa=P_ATM_SEA_LEVEL_PA,
    ):
        self.insulation_gap_m = insulation_gap_m
        self.vapor_gap_m = vapor_gap_m
        self.eps_abs = eps_abs
        self.tau_glass = tau_glass
        self.tilt_deg = tilt_deg
        self.eps_abs_ir = eps_abs_ir
        self.eps_glass_ir = eps_glass_ir
        self.n_glazing_panes = n_glazing_panes
        self.evacuated_gap = evacuated_gap
        self.complex_mode = complex_mode
        self.h_des_j_per_kg = H_DES_J_PER_KG
        self.h_des_isosteric = h_des_isosteric
        # Site ambient pressure; mirrors SystemThermalParams.p_atm_pa.
        self.p_atm_pa = p_atm_pa

    @property
    def n_thermal_unknowns(self) -> int:
        """4 in complex mode (the outer pane of a 2-pane stack), else Wilson's 3."""
        return 4 if self.complex_mode else 3


def _thermal_residuals(x, *, t_cond_c, t_amb_c, q_solar_w_m2, m_des, h_amb, params: ThermalParams, gap_eff_m, h_m, brine_salt_fraction=None):
    t_gel, t_abs, t_glass = x[0], x[1], x[2]
    u_gel = u_gel_w_m2_k(h_m)
    h_conv_g = vapor_gap_h_conv_w_m2_k(
        gap_eff_m, t_gel, t_cond_c, tilt_deg=params.tilt_deg, p_atm_pa=params.p_atm_pa
    )
    # Thinner air convects less; mirrors the same line in physics._residuals.
    h_amb = h_amb * h_amb_density_factor(t_amb_c, params.p_atm_pa)
    eps_gc = parallel_plate_emissivity(EPS_GEL, EPS_AL)
    q_rad_gc = radiative_exchange_w_m2(t_gel, t_cond_c, eps_gc)

    h_des = (
        params.h_des_j_per_kg
        if (brine_salt_fraction is None or not params.h_des_isosteric)
        else isosteric_h_des_j_per_kg(brine_salt_fraction, t_gel)
    )
    q_des = m_des * h_des
    r1 = u_gel * (t_abs - t_gel) - h_conv_g * (t_gel - t_cond_c) - q_des - q_rad_gc

    eps_ag = parallel_plate_emissivity(params.eps_abs_ir, params.eps_glass_ir)
    eps_ga = params.eps_glass_ir
    if not params.complex_mode:
        q_cond_ag = conduction_air_gap_w_m2(t_abs, t_glass, params.insulation_gap_m, params.p_atm_pa)
        q_rad_ag = radiative_exchange_w_m2(t_abs, t_glass, eps_ag)
        q_rad_ga = radiative_exchange_w_m2(t_glass, t_amb_c, eps_ga)
        r3 = q_cond_ag + q_rad_ag - h_amb * (t_glass - t_amb_c) - q_rad_ga
        r4 = (
            params.eps_abs * params.tau_glass * q_solar_w_m2
            - q_cond_ag - q_rad_ag - u_gel * (t_abs - t_gel)
        )
        return jnp.array([r1, r3, r4])

    # --- complex mode: outer pane is a real 4th unknown, switched on per instance ---
    t_glass_outer = x[3]
    two_pane = params.n_glazing_panes >= 1.5
    uncovered = params.n_glazing_panes < 0.5
    # An evacuated assembly kills gas conduction across every glazing gap but leaves
    # radiation untouched -- that asymmetry is the whole point of paying for it.
    cond_ag = conduction_air_gap_w_m2(t_abs, t_glass, params.insulation_gap_m, params.p_atm_pa)
    q_cond_ag = jnp.where(params.evacuated_gap >= 0.5, 0.0, cond_ag)
    q_rad_ag = radiative_exchange_w_m2(t_abs, t_glass, eps_ag)
    q_rad_ga = radiative_exchange_w_m2(t_glass, t_amb_c, eps_ga)

    cond_io = conduction_air_gap_w_m2(t_glass, t_glass_outer, params.insulation_gap_m, params.p_atm_pa)
    q_cond_io = jnp.where(params.evacuated_gap >= 0.5, 0.0, cond_io)
    eps_pane = parallel_plate_emissivity(eps_ga, eps_ga)
    q_rad_io = radiative_exchange_w_m2(t_glass, t_glass_outer, eps_pane)
    q_rad_oa = radiative_exchange_w_m2(t_glass_outer, t_amb_c, eps_ga)

    # Inner pane sheds to the outer pane when two-pane, else straight to ambient.
    r3 = jnp.where(
        two_pane,
        q_cond_ag + q_rad_ag - q_cond_io - q_rad_io,
        q_cond_ag + q_rad_ag - h_amb * (t_glass - t_amb_c) - q_rad_ga,
    )
    # Uncovered: no glass at all, so the absorber sees ambient directly.
    q_rad_abs_amb = radiative_exchange_w_m2(t_abs, t_amb_c, params.eps_abs)
    r4 = jnp.where(
        uncovered,
        params.eps_abs * q_solar_w_m2
        - h_amb * (t_abs - t_amb_c) - q_rad_abs_amb - u_gel * (t_abs - t_gel),
        params.eps_abs * params.tau_glass * q_solar_w_m2
        - q_cond_ag - q_rad_ag - u_gel * (t_abs - t_gel),
    )
    r3 = jnp.where(uncovered, t_glass - t_amb_c, r3)
    # Pin the unused outer pane to ambient so the Jacobian stays non-singular.
    r5 = jnp.where(
        two_pane,
        q_cond_io + q_rad_io - h_amb * (t_glass_outer - t_amb_c) - q_rad_oa,
        t_glass_outer - t_amb_c,
    )
    return jnp.array([r1, r3, r4, r5])


def solve_steady_thermal(
    *, t_cond_c, t_amb_c, q_solar_w_m2, m_des, h_amb, params: ThermalParams, h_m, gap_eff_m, x0,
    n_iter=8, brine_salt_fraction=None,
):
    """Fixed-iteration Newton solve of Eqs. 1/3/4, replacing scipy.optimize.root('hybr')."""

    def body(x, _):
        r = _thermal_residuals(
            x, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2,
            m_des=m_des, h_amb=h_amb, params=params, gap_eff_m=gap_eff_m, h_m=h_m,
            brine_salt_fraction=brine_salt_fraction,
        )
        jac = jax.jacfwd(
            lambda xx: _thermal_residuals(
                xx, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2,
                m_des=m_des, h_amb=h_amb, params=params, gap_eff_m=gap_eff_m, h_m=h_m,
                brine_salt_fraction=brine_salt_fraction,
            )
        )(x)
        step = jnp.linalg.solve(jac, r)
        x_new = clamp_temperature_c(x - step)
        return x_new, None

    x_final, _ = jax.lax.scan(body, x0, None, length=n_iter)
    h_conv_g = vapor_gap_h_conv_w_m2_k(
        gap_eff_m, x_final[0], t_cond_c, tilt_deg=params.tilt_deg, p_atm_pa=params.p_atm_pa
    )
    return x_final, h_conv_g


def _joint_residuals(z, *, c_w, h_m, t_cond_c, t_amb_c, q_solar_w_m2, h_amb, thermal, mass, gap_eff_m):
    """4x4 system [T_gel, T_abs, T_glass, m_des] (Eqs. 1/3/4 plus m_calc(m)-m=0) solved
    jointly: same fixed point as evaluate_coupled_rates at ~n_iter cost instead of
    n_bisect * n_iter. See gpu_sweep/FINDINGS.md."""
    n = thermal.n_thermal_unknowns
    x, m_des = z[:n], jnp.clip(z[n], 0.0, None)
    r_thermal = _thermal_residuals(
        x, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2,
        m_des=m_des, h_amb=h_amb, params=thermal, gap_eff_m=gap_eff_m, h_m=h_m,
        brine_salt_fraction=licl_brine_salt_fraction_from_gel(
            c_w, c_s=mass.c_s_mol_m3, h0_ref_m=mass.h0_ref_m,
            formula_weight_g_mol=mass.formula_weight_g_mol,
        ),
    )
    t_gel = x[0]
    if mass.instant_equilibrium:
        # g -> infinity as a constraint: the gel surface sits at equilibrium with the
        # condenser, and m_des is whatever the thermal balances need to hold it there
        # (energy-limited, which is what the ideal case means). Static branch.
        #
        # This is also why the reformulation matters numerically. The penalty residual
        # below carries a slope of order g, so the Newton system -- and the ODE it feeds
        # -- are stiff in proportion to the scale factor. The driving force is O(1) and
        # g-free, so neither is.
        r_m = desorption_driving_force(
            c_w, t_gel_c=t_gel, t_cond_c=t_cond_c, h_m=h_m, mass=mass
        )
    else:
        dc, _ = dc_dh_desorption(c_w, t_gel_c=t_gel, t_cond_c=t_cond_c, h_m=h_m, mass=mass)
        m_calc = m_des_kg_s_m2_from_dc_w(dc, h0_ref_m=mass.h0_ref_m)
        r_m = m_calc - z[n]
    return jnp.concatenate([r_thermal, jnp.array([r_m])])


def solve_desorption_state_joint(
    *, c_w, h_m, t_cond_c, t_amb_c, q_solar_w_m2, h_amb, thermal: ThermalParams, mass: MassParams, gap_eff_m,
    x0, n_iter=12,
):
    """Joint 4x4 Newton solve of (T_gel, T_abs, T_glass, m_des), replacing
    evaluate_coupled_rates's bisection-wraps-Newton."""
    n = thermal.n_thermal_unknowns
    z0 = jnp.concatenate([x0, jnp.array([0.0])])

    def body(z, _):
        r = _joint_residuals(
            z, c_w=c_w, h_m=h_m, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2,
            h_amb=h_amb, thermal=thermal, mass=mass, gap_eff_m=gap_eff_m,
        )
        jac = jax.jacfwd(
            lambda zz: _joint_residuals(
                zz, c_w=c_w, h_m=h_m, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2,
                h_amb=h_amb, thermal=thermal, mass=mass, gap_eff_m=gap_eff_m,
            )
        )(z)
        step = jnp.linalg.solve(jac, r)
        x_new = clamp_temperature_c(z[:n] - step[:n])
        m_new = jnp.clip(z[n] - step[n], 0.0, _M_DES_BRACKET_MAX)
        return jnp.concatenate([x_new, jnp.array([m_new])]), None

    z_final, _ = jax.lax.scan(body, z0, None, length=n_iter)
    x_star, m_star = z_final[:n], jnp.clip(z_final[n], 0.0, None)

    # No-desorption branch: a negative equilibrium m_des means m_des=0 with the thermal
    # state solved there (mirrors evaluate_coupled_rates's m_at_zero<=0 short-circuit).
    x_at_zero, _ = solve_steady_thermal(
        t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2, m_des=0.0,
        h_amb=h_amb, params=thermal, h_m=h_m, gap_eff_m=gap_eff_m, x0=x0,
    )
    # Carrying no latent load, can the gel desorb at all? If not, m_des = 0 with the
    # thermal state solved there.
    if mass.instant_equilibrium:
        # Off the driving force's sign, since the constraint route uses no rate law. Also
        # mask the hydrate floor explicitly, which is where this differs from the test
        # below: dc_dh_desorption clamps dc to 0 at the floor, so the penalty path reports
        # no-desorption there and keeps x_at_zero. Matching that exactly matters -- the
        # finite-g path is unchanged by this whole reformulation and has already produced
        # rows.
        driving0 = desorption_driving_force(
            c_w, t_gel_c=x_at_zero[0], t_cond_c=t_cond_c, h_m=h_m, mass=mass
        )
        no_desorption = (driving0 >= 0.0) | (c_w <= mass.c_w_min_mol_m3)
    else:
        dc0, _ = dc_dh_desorption(c_w, t_gel_c=x_at_zero[0], t_cond_c=t_cond_c, h_m=h_m, mass=mass)
        m_calc0 = m_des_kg_s_m2_from_dc_w(dc0, h0_ref_m=mass.h0_ref_m)
        no_desorption = m_calc0 <= 0.0

    x_final = jnp.where(no_desorption, x_at_zero, x_star)
    m_final = jnp.where(no_desorption, 0.0, m_star)
    if mass.instant_equilibrium:
        # Eq. 5's driving force is zero by construction here, so it cannot be what sets
        # the rate: invert the flux instead (CPU: dc_w_dt_from_m_des).
        dc_final = -m_final / (WATER_MOLAR_MASS_KG_MOL * mass.h0_ref_m)
        dc_final = jnp.where(c_w <= mass.c_w_min_mol_m3, 0.0, jnp.minimum(dc_final, 0.0))
    else:
        dc_final, _ = dc_dh_desorption(
            c_w, t_gel_c=x_final[0], t_cond_c=t_cond_c, h_m=h_m, mass=mass
        )
    return m_final, x_final, dc_final


def desorption_rhs(
    y,
    *,
    t_amb_c,
    q_solar_w_m2,
    h_amb,
    thermal: ThermalParams,
    mass: MassParams,
    h0_ref_m,
    h_floor_m,
    h_fg_j_per_kg,
    fin_area_ratio,
    x0_guess,
    h_amb_cond=None,
    fin_thickness_m=None,
    fin_height_m=None,
    condenser_tracks_ambient=False,
):
    """dy/dt for y = [c_w, H, T_cond]: the 3-state quasi_steady desorption ODE, matching
    evaluate_coupled_rates's desorption branch exactly (Eqs. 1-6 + Eq. 2).

    Complex mode adds B4's ``h_amb_cond`` (forced condenser air, decoupled from the
    absorber's ambient convection) and B3's fin geometry; both default to None,
    which is Wilson's shared h_amb and ideal fin. ``condenser_tracks_ambient``
    (static) is the idealized limit of infinite condenser cooling capacity: T_cond
    is pinned to T_amb every step instead of evolving via Eq. 2, and y[2]'s
    derivative is zeroed since that state is no longer read."""
    c_w, h_m_raw, t_cond = y[0], y[1], y[2]
    h_m = jnp.clip(h_m_raw, h_floor_m, None)
    t_cond_c = t_amb_c if condenser_tracks_ambient else clamp_temperature_c(t_cond)
    gap_eff_m = jnp.clip(thermal.vapor_gap_m - h_m, 0.0, None)
    q_sol = jnp.clip(q_solar_w_m2, 0.0, None)

    m_des, x_star, dc = solve_desorption_state_joint(
        c_w=c_w, h_m=h_m, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_sol,
        h_amb=h_amb, thermal=thermal, mass=mass, gap_eff_m=gap_eff_m, x0=x0_guess,
    )
    t_gel, t_abs, t_glass = x_star[0], x_star[1], x_star[2]

    if mass.instant_equilibrium:
        # dH/dt = dc_w/dt * MW * H0 / rho, the ratio dc_dh_desorption itself carries.
        # Taking it from the rate law would freeze the gel's thickness (zero driving
        # force) while its water drained. Mirrors CPU dh_dt_from_dc_w.
        dh = dc * WATER_MOLAR_MASS_KG_MOL * h0_ref_m / RHO_GEL_KG_M3
        dh = jnp.where(c_w <= mass.c_w_min_mol_m3, 0.0, jnp.minimum(dh, 0.0))
    else:
        _, dh = dc_dh_desorption(c_w, t_gel_c=t_gel, t_cond_c=t_cond_c, h_m=h_m, mass=mass)

    h_conv_g = vapor_gap_h_conv_w_m2_k(
        gap_eff_m, t_gel, t_cond_c, tilt_deg=thermal.tilt_deg, p_atm_pa=thermal.p_atm_pa
    )
    h_amb_for_cond = h_amb if h_amb_cond is None else h_amb_cond
    # Thinner air convects less, fans included; mirrors simulation.evaluate_coupled_rates.
    h_amb_for_cond = h_amb_for_cond * h_amb_density_factor(t_amb_c, thermal.p_atm_pa)
    h_conv_cond = condenser_h_conv_w_m2_k(
        h_amb_for_cond, fin_area_ratio=fin_area_ratio,
        fin_thickness_m=fin_thickness_m, fin_height_m=fin_height_m,
    )
    eps_gc = parallel_plate_emissivity(EPS_GEL, EPS_AL)
    q_rad = radiative_exchange_w_m2(t_gel, t_cond_c, eps_gc)
    tmass = jnp.clip(CONDENSER_THERMAL_MASS_J_M2_K, 1.0, None)
    dT_cond = (h_conv_g * (t_gel - t_cond_c) - h_conv_cond * (t_cond_c - t_amb_c) + m_des * h_fg_j_per_kg + q_rad) / tmass

    dh_masked = jnp.where(h_m > h_floor_m + 1e-12, dh, 0.0)
    dc_masked = jnp.minimum(dc, 0.0)
    if condenser_tracks_ambient:
        # Static flag, so this is a plain branch, not a traced select. y[2] stays in
        # the state vector (jit/vmap need a fixed shape) but is frozen and unread.
        dT_cond = jnp.zeros_like(dT_cond)

    return jnp.array([dc_masked, dh_masked, dT_cond]), (t_gel, t_abs, t_glass, m_des)


# ---- Absorption phase (Note S1, no thermal root-solve -- T_gel == T_amb) ----
def pam_licl_gravimetric_uptake_g_g(c_w, *, h0_ref_m, c_s_mol_m3, formula_weight_g_mol, salt_loading, salt_weight_factor=1.0):
    mw_eff = formula_weight_g_mol * salt_weight_factor
    mass_salt = jnp.clip(c_s_mol_m3, 0.0, None) * h0_ref_m * mw_eff / 1000.0
    mass_polymer = mass_salt / jnp.clip(salt_loading, 1e-9, None)
    m_dry = mass_salt + mass_polymer
    mass_water = jnp.clip(c_w, 0.0, None) * h0_ref_m * WATER_MOLAR_MASS_KG_MOL
    return jnp.where(m_dry <= 0.0, 0.0, mass_water / jnp.clip(m_dry, 1e-30, None))


def absorption_effective_water_activity(c_w, *, t_gel_c, mass: "MassParams", salt_loading):
    """Gel water activity during absorption: the salt-in-water activity, nothing else.

    Water in the gel has the chemical potential of pure water plus RT ln(x_w gamma_w), so
    the activity model IS the isotherm. This used to return max(brine, PAM-LiCl DVS cap);
    that cap was a measured composite curve which contradicted the activity model between
    RH 0.73 and 0.90 and then switched itself off above 0.90 where its data ran out.
    Mirrors solar_lumped.physics._absorption_effective_water_activity.
    """
    del salt_loading  # dry-mass bookkeeping only; not needed for the activity
    if mass.aw_table is not None:
        return blend_water_activity_from_c_w(
            c_w, c_s=mass.c_s_mol_m3, h0_ref_m=mass.h0_ref_m,
            formula_weight_g_mol=mass.formula_weight_g_mol, temperature_c=t_gel_c,
            aw_table=mass.aw_table,
        )
    return water_activity_licl_from_c_w(
        c_w, c_s=mass.c_s_mol_m3, h0_ref_m=mass.h0_ref_m,
        formula_weight_g_mol=mass.formula_weight_g_mol, temperature_c=t_gel_c,
    )


def dc_dh_absorption(c_w, *, t_gel_c, rh, h_m, mass: "MassParams", salt_loading):
    """Eqs. 5-6 absorption branch: g_chamber (constant), driven by rh - a_w,eff."""
    aw = absorption_effective_water_activity(c_w, t_gel_c=t_gel_c, mass=mass, salt_loading=salt_loading)
    driving = rh - aw

    t_k = jnp.clip(t_gel_c + 273.15, 200.0, None)
    p_sat = saturation_vapor_pressure_pa(t_gel_c)
    g = _INSTANT_EQUILIBRIUM_G_SCALE * mass.g_conv_m_s if mass.instant_equilibrium else mass.g_conv_m_s
    pref = g / mass.h0_ref_m
    rate = pref * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving
    rate = jnp.where(jnp.isfinite(rate), rate, 0.0)
    dc = jnp.where((c_w >= mass.c_w_max_mol_m3) & (rate > 0.0), 0.0, rate)
    dc = jnp.where((c_w <= mass.c_w_min_mol_m3) & (dc < 0.0), 0.0, dc)

    dh = g * WATER_MOLAR_MASS_KG_MOL / RHO_GEL_KG_M3 * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving
    dh = jnp.where(jnp.isfinite(dh), dh, 0.0)
    dh = jnp.where(c_w <= mass.c_w_min_mol_m3, jnp.maximum(dh, 0.0), dh)
    return dc, dh


def equilibrium_c_w_absorption(*, rh, t_gel_c, mass: "MassParams", salt_loading, n_iter=50):
    """Gel water concentration in equilibrium with ambient air (mol/m3).

    The g -> infinity limit for absorption: the gel tracks ambient RH exactly, so c_w
    solves a_w,eff(c_w) = c_r(rh). Mirrors physics.equilibrium_c_w_absorption, but by
    fixed-count bisection rather than brentq -- a data-dependent trip count cannot be
    vmapped, and 50 halvings of the [floor, ceiling] range land far inside float64.

    Monotone (a_w rises with water content), so bisection is unambiguous, and it converges
    onto whichever bound the equilibrium lies outside -- the same clamping the CPU does
    explicitly.
    """
    # physics.concentration_ratio_absorption is the identity on rh -- the ambient air's
    # own water activity -- which is why dc_dh_absorption drives on (rh - a_w) directly.
    c_r = jnp.asarray(rh, dtype=float)

    def body(bounds, _):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        aw = absorption_effective_water_activity(
            mid, t_gel_c=t_gel_c, mass=mass, salt_loading=salt_loading
        )
        too_dry = (c_r - aw) > 0.0        # ambient wetter than the gel: go up
        return (jnp.where(too_dry, mid, lo), jnp.where(too_dry, hi, mid)), None

    (lo, hi), _ = jax.lax.scan(
        body, (mass.c_w_min_mol_m3 * jnp.ones_like(c_r), mass.c_w_max_mol_m3 * jnp.ones_like(c_r)),
        None, length=n_iter,
    )
    return 0.5 * (lo + hi)


def absorption_rhs(y, *, t_amb_c, rh, h0_ref_m, h_floor_m, h_max_m, mass: "MassParams", salt_loading):
    """dy/dt for y = [c_w, H]: the 2-state absorption ODE. T_gel == T_amb during open
    absorption (Note S1 Eq. S1), so no thermal root-solve is needed."""
    c_w, h_m_raw = y[0], y[1]
    h_m = jnp.maximum(h_m_raw, h_floor_m)
    t_gel = t_amb_c
    dc, dh = dc_dh_absorption(c_w, t_gel_c=t_gel, rh=rh, h_m=h_m, mass=mass, salt_loading=salt_loading)
    dh = jnp.where(h_m > h_floor_m + 1e-12, dh, jnp.maximum(dh, 0.0))
    dh = jnp.where((h_m >= h_max_m) & (dh > 0.0), 0.0, dh)
    return jnp.array([dc, dh])
