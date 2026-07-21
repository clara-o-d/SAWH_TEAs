"""JAX port of the Wilson quasi-steady desorption RHS (Note S1 Eqs. 1-6 + Eq. 2).

Mirrors, for the ``physics_model="note_s1"`` / ``desorption_solver="quasi_steady"``
path only (the default -- see solar_lumped/scripts/grid_param_sweep.py):
  - solar_lumped.physics.device_balances.solve_steady_thermal (scipy.optimize.root
    "hybr" on Eqs. 1/3/4) -> replaced with a fixed-iteration Newton solve.
  - solar_lumped.simulation.coupled_dynamics._solve_m_des_coupled (scipy.brentq
    root of m_calc(m) - m = 0) -> replaced with fixed-iteration bisection.
  - solar_lumped.physics.mass_transfer.dc_w_dt / dH_dt (desorption branch only).
  - solar_lumped.physics.salt_properties.water_activity_from_c_w (LiCl branch only
    -- this prototype does not port the CaCl2/MgCl2/NaCl brine_equilibrium path).

All functions are pure, vmap-safe (no data-dependent Python control flow -- every
branch is a jnp.where), and operate on float64 (jax.config.update("jax_enable_x64",
True) is required by the caller -- c_w is O(1e4) mol/m3 and float32 loses the
absolute precision the original atol=1e-7 assumes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---- Table S3 / Note S1 constants (solar_lumped/src/solar_lumped/physics/table_s3.py) ----
H0_M = 0.004
L_G_M = 0.04
L_INS_M = 0.005
L_C_M = 0.005
L_GLASS_M = 0.125 * 0.0254
L_AL_STACK_M = L_C_M
L_SILICONE_M = 0.001
VAPOR_GAP_TRANSPORT_MIN_M = 0.007

RHO_SOL_KG_M3 = 1250.2
RHO_COMPOSITE_KG_M3 = 1250.2
H_DES_J_PER_KG = 2.32e6
H_FG_J_PER_KG = 2.256e6
K_AIR_W_M_K = 0.0286
K_AL_W_M_K = 167.0
K_SILICONE_W_M_K = 0.2
K_GEL_W_M_K = 0.6
K_GLASS_W_M_K = 1.2
RHO_AL_KG_M3 = 2700.0
CP_AL_J_KG_K = 900.0
RHO_GLASS_KG_M3 = 2230.0
CP_GLASS_J_KG_K = 830.0
CP_GEL_J_KG_K = 3500.0

EPS_GEL = 1.0
EPS_AL = 0.05
TILT_DEG_DEFAULT = 30.0

STEFAN_BOLTZMANN = 5.670374419e-8
D_AIR_M2_S = 2.62e-5
GRAVITY_M_S2 = 9.81
BETA_AIR_K = 1.0 / 300.0
NU_AIR_M2_S = 1.5e-5
RHO_AIR_KG_M3 = 1.2
CP_AIR_J_KG_K = 1005.0
ALPHA_AIR_M2_S = K_AIR_W_M_K / (RHO_AIR_KG_M3 * CP_AIR_J_KG_K)

WATER_MOLAR_MASS_KG_MOL = 0.018015
GAS_CONSTANT_J_MOL_K = 8.314462618
C_W_MAX_MOL_M3 = 400000.0
C_W_MIN_MOL_M3 = 100.0

CONDENSER_THERMAL_MASS_J_M2_K = RHO_AL_KG_M3 * CP_AL_J_KG_K * L_C_M
GLASS_THERMAL_MASS_J_M2_K = RHO_GLASS_KG_M3 * CP_GLASS_J_KG_K * L_GLASS_M
ABSORBER_THERMAL_MASS_J_M2_K = RHO_AL_KG_M3 * CP_AL_J_KG_K * L_AL_STACK_M

# Conde (2004) Table 3 LiCl vapour-pressure correlation parameters.
_PI0, _PI1, _PI2, _PI3, _PI4 = 0.28, 4.30, 0.60, 0.21, 5.10
_PI5, _PI6, _PI7, _PI8, _PI9 = 0.49, 0.362, -4.75, -0.40, 0.03
_XI_MAX_LICL = 0.56
T_CRIT_H2O_K = 647.096
P_CRIT_H2O_PA = 22.064e6
_SAUL_WAGNER_A = jnp.array([-7.858230, 1.839910, -11.781100, 22.670500, -15.939300, 1.775160])
_SAUL_WAGNER_EXP = jnp.array([1.0, 1.5, 3.0, 3.5, 4.0, 7.5])

TEMP_CLAMP_LO_C = -40.0
TEMP_CLAMP_HI_C = 120.0


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


def vapor_pressure_ratio_licl(xi, temperature_c):
    """pi = p_sol / p_H2O = brine water activity a_w (Conde 2004 Table 3, LiCl)."""
    xi = jnp.clip(xi, 1e-12, _XI_MAX_LICL)
    theta = (temperature_c + 273.15) / T_CRIT_H2O_K
    pi25 = 1.0 - (1.0 + (xi / _PI6) ** _PI7) ** _PI8 - _PI9 * jnp.exp(-((xi - 0.1) ** 2) / 0.005)
    a_term = 2.0 - (1.0 + (xi / _PI0) ** _PI1) ** _PI2
    b_term = (1.0 + (xi / _PI3) ** _PI4) ** _PI5 - 1.0
    f = a_term + b_term * theta
    return jnp.clip(pi25 * f, 0.0, 1.0)


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
    aw = vapor_pressure_ratio_licl(f_b, jnp.minimum(temperature_c, 150.0))
    return jnp.where(c_w <= 0.0, 1.0, aw)


def parallel_plate_emissivity(eps_a, eps_b):
    # jnp.asarray first: if eps_a/eps_b arrive as plain Python floats (e.g. a
    # closure constant from the serial, non-batched daily-cycle path) rather
    # than JAX arrays, "1.0 / 0.0" is eager Python division and raises
    # ZeroDivisionError immediately -- jnp.where can't mask a branch that
    # never got the chance to be lazy. As real arrays, "1.0 / 0.0" is a safe,
    # lazy `inf` that jnp.where then correctly discards.
    eps_a = jnp.asarray(eps_a)
    eps_b = jnp.asarray(eps_b)
    return jnp.where((eps_a <= 0.0) | (eps_b <= 0.0), 0.0, 1.0 / (1.0 / eps_a + 1.0 / eps_b - 1.0))


def radiative_exchange_w_m2(t_hot_c, t_cold_c, emissivity):
    t_hot_k = t_hot_c + 273.15
    t_cold_k = t_cold_c + 273.15
    return emissivity * STEFAN_BOLTZMANN * (t_hot_k**4 - t_cold_k**4)


def conduction_air_gap_w_m2(t_hot_c, t_cold_c, gap_m):
    return jnp.where(gap_m <= 0.0, 0.0, K_AIR_W_M_K / jnp.clip(gap_m, 1e-30, None) * (t_hot_c - t_cold_c))


def _rayleigh_vapor_gap(gap_m, t_hot_c, t_cold_c):
    delta_t = jnp.clip(jnp.abs(t_hot_c - t_cold_c), 1e-6, None)
    return GRAVITY_M_S2 * BETA_AIR_K * delta_t * gap_m**3 / (NU_AIR_M2_S * ALPHA_AIR_M2_S)


def hollands_nu_eq_s3(ra, *, tilt_deg):
    cos_t = jnp.clip(jnp.cos(jnp.radians(tilt_deg)), 1e-6, None)
    ra_cos = ra * cos_t
    sin_18t_16 = jnp.sin(jnp.radians(1.8 * tilt_deg)) ** 1.6
    f1 = jnp.clip(1.0 - 1708.0 * sin_18t_16 / jnp.clip(ra_cos, 1e-30, None), 0.0, None)
    f2 = jnp.clip(1.0 - 1708.0 / jnp.clip(ra_cos, 1e-30, None), 0.0, None)
    f3 = jnp.clip((jnp.clip(ra_cos, 0.0, None) / 5830.0) ** (1.0 / 3.0) - 1.0, 0.0, None)
    nu = 1.0 + 1.44 * f1 * f2 + f3
    return jnp.where(ra_cos <= 0.0, 1.0, nu)


def hollands_vapor_gap_h_conv_w_m2_k(gap_m, t_hot_c, t_cold_c, *, tilt_deg):
    ra = _rayleigh_vapor_gap(jnp.clip(gap_m, 1e-30, None), t_hot_c, t_cold_c)
    nu = hollands_nu_eq_s3(ra, tilt_deg=tilt_deg)
    h = nu * K_AIR_W_M_K / jnp.clip(gap_m, 1e-30, None)
    return jnp.where(gap_m <= 0.0, 0.0, h)


def condenser_h_conv_w_m2_k(h_amb, *, fin_area_ratio):
    return fin_area_ratio * h_amb


def mass_transfer_g_from_h_conv_m_s(h_conv):
    return jnp.where(h_conv <= 0.0, 0.0, h_conv * D_AIR_M2_S / K_AIR_W_M_K)


def u_gel_w_m2_k(h_m):
    h = jnp.clip(h_m, H0_M * 0.25, None)
    resistance = L_AL_STACK_M / K_AL_W_M_K + L_SILICONE_M / K_SILICONE_W_M_K + h / K_GEL_W_M_K
    return 1.0 / resistance


def gel_thermal_mass_j_m2_k(h_m):
    return RHO_COMPOSITE_KG_M3 * CP_GEL_J_KG_K * jnp.clip(h_m, H0_M * 0.25, None)


def concentration_ratio_desorption(t_gel_c, t_cond_c):
    p_g = saturation_vapor_pressure_pa(t_gel_c)
    p_c = saturation_vapor_pressure_pa(t_cond_c)
    t_g_k = t_gel_c + 273.15
    t_c_k = t_cond_c + 273.15
    ratio = (p_c / jnp.clip(p_g, 1e-30, None)) * (t_g_k / jnp.clip(t_c_k, 1e-30, None))
    return jnp.where((p_g <= 0.0) | (t_g_k <= 0.0) | (t_c_k <= 0.0), 0.0, ratio)


class MassParams:
    """Mirrors solar_lumped.physics.mass_transfer.MassTransferParams for LiCl."""

    def __init__(self, *, h0_ref_m, vapor_gap_m, tilt_deg, c_s_mol_m3, formula_weight_g_mol, g_conv_m_s=0.0085):
        self.h0_ref_m = h0_ref_m
        self.vapor_gap_m = vapor_gap_m
        self.tilt_deg = tilt_deg
        self.c_s_mol_m3 = c_s_mol_m3
        self.formula_weight_g_mol = formula_weight_g_mol
        self.g_conv_m_s = g_conv_m_s


def dc_dh_desorption(c_w, *, t_gel_c, t_cond_c, h_m, mass: MassParams):
    """Eqs. 5-6 desorption branch (dc_w/dt, dH/dt), before the clip-to-<=0 in sorbent.py."""
    gap_m = jnp.clip(mass.vapor_gap_m - h_m, 0.0, None)
    h_conv = hollands_vapor_gap_h_conv_w_m2_k(gap_m, t_gel_c, t_cond_c, tilt_deg=mass.tilt_deg)
    g = mass_transfer_g_from_h_conv_m_s(h_conv)
    g = jnp.where(gap_m < VAPOR_GAP_TRANSPORT_MIN_M, 0.0, g)

    c_r = concentration_ratio_desorption(t_gel_c, t_cond_c)
    aw = water_activity_licl_from_c_w(
        c_w,
        c_s=mass.c_s_mol_m3,
        h0_ref_m=mass.h0_ref_m,
        formula_weight_g_mol=mass.formula_weight_g_mol,
        temperature_c=t_gel_c,
    )
    driving = jnp.where(jnp.isfinite(aw), c_r - aw, 0.0)

    t_k = jnp.clip(t_gel_c + 273.15, 200.0, None)
    p_sat = saturation_vapor_pressure_pa(t_gel_c)
    pref = g / mass.h0_ref_m
    rate = pref * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving
    rate = jnp.where(jnp.isfinite(rate), rate, 0.0)
    dc = jnp.where((c_w >= C_W_MAX_MOL_M3) & (rate > 0.0), 0.0, rate)
    dc = jnp.where((c_w <= C_W_MIN_MOL_M3) & (dc < 0.0), 0.0, dc)

    dh = (
        g
        * WATER_MOLAR_MASS_KG_MOL
        / RHO_SOL_KG_M3
        * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k))
        * driving
    )
    dh = jnp.where(jnp.isfinite(dh), dh, 0.0)

    dc = jnp.minimum(dc, 0.0)
    dh = jnp.minimum(dh, 0.0)
    dh = jnp.where(h_m <= mass.h0_ref_m + 1e-12, 0.0, dh)
    return dc, dh


def m_des_kg_s_m2_from_dc_w(dc_w_dt_val, *, h0_ref_m):
    return jnp.where(dc_w_dt_val >= 0.0, 0.0, -dc_w_dt_val * WATER_MOLAR_MASS_KG_MOL * h0_ref_m)


class ThermalParams:
    """Mirrors DeviceThermalParams for the note_s1/has_glass=True Atacama-baseline path.

    eps_abs_ir/eps_glass_ir default to 1.0 (blackbody), which makes
    parallel_plate_emissivity(1.0, 1.0) == 1.0 -- i.e. the default reproduces
    the original Wilson Eqs. 3/4 cavity/blackbody approximation exactly (Case
    1). Pass real IR emissivities to activate the modified physics (Case 2/3,
    see device_balances.py's _residuals for the CPU-side mirror of this).
    """

    def __init__(self, *, insulation_gap_m, vapor_gap_m, eps_abs, tau_glass, tilt_deg, eps_abs_ir=1.0, eps_glass_ir=1.0):
        self.insulation_gap_m = insulation_gap_m
        self.vapor_gap_m = vapor_gap_m
        self.eps_abs = eps_abs
        self.tau_glass = tau_glass
        self.tilt_deg = tilt_deg
        self.eps_abs_ir = eps_abs_ir
        self.eps_glass_ir = eps_glass_ir
        self.h_des_j_per_kg = H_DES_J_PER_KG


def _thermal_residuals(x, *, t_cond_c, t_amb_c, q_solar_w_m2, m_des, h_amb, params: ThermalParams, gap_eff_m, h_m):
    t_gel, t_abs, t_glass = x[0], x[1], x[2]
    u_gel = u_gel_w_m2_k(h_m)
    h_conv_g = hollands_vapor_gap_h_conv_w_m2_k(gap_eff_m, t_gel, t_cond_c, tilt_deg=params.tilt_deg)
    eps_gc = parallel_plate_emissivity(EPS_GEL, EPS_AL)
    q_rad_gc = radiative_exchange_w_m2(t_gel, t_cond_c, eps_gc)

    q_des = m_des * params.h_des_j_per_kg
    r1 = u_gel * (t_abs - t_gel) - h_conv_g * (t_gel - t_cond_c) - q_des - q_rad_gc

    eps_ag = parallel_plate_emissivity(params.eps_abs_ir, params.eps_glass_ir)
    eps_ga = params.eps_glass_ir
    q_cond_ag = conduction_air_gap_w_m2(t_abs, t_glass, params.insulation_gap_m)
    q_rad_ag = radiative_exchange_w_m2(t_abs, t_glass, eps_ag)
    q_rad_ga = radiative_exchange_w_m2(t_glass, t_amb_c, eps_ga)
    r3 = q_cond_ag + q_rad_ag - h_amb * (t_glass - t_amb_c) - q_rad_ga
    r4 = (
        params.eps_abs * params.tau_glass * q_solar_w_m2
        - q_cond_ag
        - q_rad_ag
        - u_gel * (t_abs - t_gel)
    )
    return jnp.array([r1, r3, r4])


def solve_steady_thermal(
    *, t_cond_c, t_amb_c, q_solar_w_m2, m_des, h_amb, params: ThermalParams, h_m, gap_eff_m, x0, n_iter=8
):
    """Fixed-iteration Newton solve of Eqs. 1/3/4, replacing scipy.optimize.root('hybr')."""

    def body(x, _):
        r = _thermal_residuals(
            x, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2,
            m_des=m_des, h_amb=h_amb, params=params, gap_eff_m=gap_eff_m, h_m=h_m,
        )
        jac = jax.jacfwd(
            lambda xx: _thermal_residuals(
                xx, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2,
                m_des=m_des, h_amb=h_amb, params=params, gap_eff_m=gap_eff_m, h_m=h_m,
            )
        )(x)
        step = jnp.linalg.solve(jac, r)
        x_new = clamp_temperature_c(x - step)
        return x_new, None

    x_final, _ = jax.lax.scan(body, x0, None, length=n_iter)
    h_conv_g = hollands_vapor_gap_h_conv_w_m2_k(gap_eff_m, x_final[0], t_cond_c, tilt_deg=params.tilt_deg)
    return x_final, h_conv_g


def _m_calc_residual(m_des, *, c_w, h_m, t_cond_c, t_amb_c, q_solar_w_m2, h_amb, thermal, mass, gap_eff_m, x0):
    x_star, _ = solve_steady_thermal(
        t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2, m_des=jnp.clip(m_des, 0.0, None),
        h_amb=h_amb, params=thermal, h_m=h_m, gap_eff_m=gap_eff_m, x0=x0,
    )
    t_gel = x_star[0]
    dc, _ = dc_dh_desorption(c_w, t_gel_c=t_gel, t_cond_c=t_cond_c, h_m=h_m, mass=mass)
    m_calc = m_des_kg_s_m2_from_dc_w(dc, h0_ref_m=mass.h0_ref_m)
    return m_calc - m_des, x_star, dc, m_calc


_M_DES_BRACKET_MAX = 0.01


def solve_m_des_and_thermal(
    *, c_w, h_m, t_cond_c, t_amb_c, q_solar_w_m2, h_amb, thermal: ThermalParams, mass: MassParams, gap_eff_m, x0,
    n_bisect=22,
):
    """Fixed-iteration bisection root of m_calc(m) - m = 0, replacing scipy.brentq.

    Brackets [0, _M_DES_BRACKET_MAX] always (the CPU code's adaptive doubling is a
    performance optimization, not a correctness requirement -- the residual is
    monotonically decreasing in m over the physical range, per the original comment
    "avoids fixed-point cycling").
    """
    r0, x0_star, dc0, _ = _m_calc_residual(
        0.0, c_w=c_w, h_m=h_m, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2,
        h_amb=h_amb, thermal=thermal, mass=mass, gap_eff_m=gap_eff_m, x0=x0,
    )
    no_desorption = r0 <= 0.0

    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        r_mid, _, _, _ = _m_calc_residual(
            mid, c_w=c_w, h_m=h_m, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2,
            h_amb=h_amb, thermal=thermal, mass=mass, gap_eff_m=gap_eff_m, x0=x0,
        )
        lo_new = jnp.where(r_mid > 0.0, mid, lo)
        hi_new = jnp.where(r_mid > 0.0, hi, mid)
        return (lo_new, hi_new), None

    (lo_f, hi_f), _ = jax.lax.scan(body, (0.0, _M_DES_BRACKET_MAX), None, length=n_bisect)
    m_star = 0.5 * (lo_f + hi_f)
    m_star = jnp.where(no_desorption, 0.0, m_star)

    _, x_star, dc_star, m_calc_star = _m_calc_residual(
        m_star, c_w=c_w, h_m=h_m, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2,
        h_amb=h_amb, thermal=thermal, mass=mass, gap_eff_m=gap_eff_m, x0=x0,
    )
    x_star = jnp.where(no_desorption, x0_star, x_star)
    dc_star = jnp.where(no_desorption, dc0, dc_star)
    m_final = jnp.where(no_desorption, 0.0, m_star)
    return m_final, x_star, dc_star


def _joint_residuals(z, *, c_w, h_m, t_cond_c, t_amb_c, q_solar_w_m2, h_amb, thermal, mass, gap_eff_m):
    """4x4 system [T_gel, T_abs, T_glass, m_des]: Eqs. 1/3/4 plus m_calc(m)-m=0,
    solved jointly instead of an outer scalar root wrapping an inner 3x3 root.
    Same fixed point as solve_m_des_and_thermal, ~n_iter cost instead of
    n_bisect * n_iter (this is the fix the bisection-nesting-Newton architecture
    needs for practical per-step cost -- see gpu_sweep/FINDINGS.md).
    """
    x, m_des = z[:3], jnp.clip(z[3], 0.0, None)
    r_thermal = _thermal_residuals(
        x, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2,
        m_des=m_des, h_amb=h_amb, params=thermal, gap_eff_m=gap_eff_m, h_m=h_m,
    )
    t_gel = x[0]
    dc, _ = dc_dh_desorption(c_w, t_gel_c=t_gel, t_cond_c=t_cond_c, h_m=h_m, mass=mass)
    m_calc = m_des_kg_s_m2_from_dc_w(dc, h0_ref_m=mass.h0_ref_m)
    r_m = m_calc - z[3]
    return jnp.concatenate([r_thermal, jnp.array([r_m])])


def solve_desorption_state_joint(
    *, c_w, h_m, t_cond_c, t_amb_c, q_solar_w_m2, h_amb, thermal: ThermalParams, mass: MassParams, gap_eff_m,
    x0, n_iter=12,
):
    """Joint Newton solve of (T_gel, T_abs, T_glass, m_des); replaces
    solve_m_des_and_thermal's bisection-wraps-Newton with a single 4x4 Newton.
    """
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
        x_new = clamp_temperature_c(z[:3] - step[:3])
        m_new = jnp.clip(z[3] - step[3], 0.0, _M_DES_BRACKET_MAX)
        return jnp.concatenate([x_new, jnp.array([m_new])]), None

    z_final, _ = jax.lax.scan(body, z0, None, length=n_iter)
    x_star, m_star = z_final[:3], jnp.clip(z_final[3], 0.0, None)

    # No-desorption branch: if the equilibrium m_des would be negative, the
    # physical answer is m_des=0 with thermal state solved at m_des=0 (mirrors
    # solve_m_des_and_thermal's m_at_zero<=0 short-circuit).
    x_at_zero, _ = solve_steady_thermal(
        t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_solar_w_m2, m_des=0.0,
        h_amb=h_amb, params=thermal, h_m=h_m, gap_eff_m=gap_eff_m, x0=x0,
    )
    dc0, _ = dc_dh_desorption(c_w, t_gel_c=x_at_zero[0], t_cond_c=t_cond_c, h_m=h_m, mass=mass)
    m_calc0 = m_des_kg_s_m2_from_dc_w(dc0, h0_ref_m=mass.h0_ref_m)
    no_desorption = m_calc0 <= 0.0

    x_final = jnp.where(no_desorption, x_at_zero, x_star)
    m_final = jnp.where(no_desorption, 0.0, m_star)
    dc_final, _ = dc_dh_desorption(c_w, t_gel_c=x_final[0], t_cond_c=t_cond_c, h_m=h_m, mass=mass)
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
    h_fg_j_per_kg,
    fin_area_ratio,
    x0_guess,
):
    """dy/dt for y = [c_w, H, T_cond] -- the 3-state quasi_steady desorption ODE
    integrated by scipy.integrate.solve_ivp(method='Radau') in
    ode_system.py::_integrate_desorption. Matches evaluate_coupled_rates's
    desorption branch exactly (Eqs. 1-6 + Eq. 2).
    """
    c_w, h_m_raw, t_cond = y[0], y[1], y[2]
    h_m = jnp.clip(h_m_raw, h0_ref_m, None)
    t_cond_c = clamp_temperature_c(t_cond)
    gap_eff_m = jnp.clip(thermal.vapor_gap_m - h_m, 0.0, None)
    q_sol = jnp.clip(q_solar_w_m2, 0.0, None)

    m_des, x_star, dc = solve_desorption_state_joint(
        c_w=c_w, h_m=h_m, t_cond_c=t_cond_c, t_amb_c=t_amb_c, q_solar_w_m2=q_sol,
        h_amb=h_amb, thermal=thermal, mass=mass, gap_eff_m=gap_eff_m, x0=x0_guess,
    )
    t_gel, t_abs, t_glass = x_star[0], x_star[1], x_star[2]

    _, dh = dc_dh_desorption(c_w, t_gel_c=t_gel, t_cond_c=t_cond_c, h_m=h_m, mass=mass)

    h_conv_g = hollands_vapor_gap_h_conv_w_m2_k(gap_eff_m, t_gel, t_cond_c, tilt_deg=thermal.tilt_deg)
    h_conv_cond = condenser_h_conv_w_m2_k(h_amb, fin_area_ratio=fin_area_ratio)
    eps_gc = parallel_plate_emissivity(EPS_GEL, EPS_AL)
    q_rad = radiative_exchange_w_m2(t_gel, t_cond_c, eps_gc)
    tmass = jnp.clip(CONDENSER_THERMAL_MASS_J_M2_K, 1.0, None)
    dT_cond = (h_conv_g * (t_gel - t_cond_c) - h_conv_cond * (t_cond_c - t_amb_c) + m_des * h_fg_j_per_kg + q_rad) / tmass

    dh_masked = jnp.where(h_m > h0_ref_m + 1e-12, dh, 0.0)
    dc_masked = jnp.minimum(dc, 0.0)

    return jnp.array([dc_masked, dh_masked, dT_cond]), (t_gel, t_abs, t_glass, m_des)


# ---- Absorption phase (Note S1, no thermal root-solve -- T_gel == T_amb) ----
# Wilson Note S2 PAM-LiCl DVS isotherm (RH %, gravimetric uptake g/g), sorted by RH
# ascending, verbatim from data/materials/PAM-LiCL_isotherm.csv. Interpolation for
# the inverse (uptake -> RH) lookup replicates salt_properties.py's np.interp
# exactly, including its non-monotonic first two uptake points -- not "fixed" here.
_DVS_RH_PCT = jnp.array([
    0.0, 4.912669270074408, 9.915874958714616, 14.900817936313652,
    19.908297876474123, 24.81513862174817, 29.92286918847506, 34.83048707038915,
    39.8375784422296, 44.845058382390086, 49.85176118591051, 54.95716034271725,
    59.86205824639115, 64.96551456159779, 69.96522313535777, 74.7624876143848,
    79.9560917798372, 84.8419498358299, 90.01806842688116,
])
_DVS_UPTAKE_G_G = jnp.array([
    0.011627906976745095, 0.0, 0.27906976744186096, 1.1046511627906987,
    1.2558139534883725, 1.4186046511627906, 1.5697674418604652, 1.7093023255813957,
    1.8720930232558146, 2.0232558139534884, 2.1976744186046515, 2.4186046511627914,
    2.6395348837209305, 2.9186046511627914, 3.3023255813953494, 3.744186046511628,
    4.325581395348838, 5.116279069767442, 6.220930232558139,
])


def pam_licl_water_activity_from_uptake_g_g(uptake_g_g):
    u = jnp.clip(uptake_g_g, _DVS_UPTAKE_G_G[0], _DVS_UPTAKE_G_G[-1])
    aw_pct = jnp.interp(u, _DVS_UPTAKE_G_G, _DVS_RH_PCT)
    return jnp.clip(aw_pct / 100.0, 0.0, 1.0)


def pam_licl_gravimetric_uptake_g_g(c_w, *, h0_ref_m, c_s_mol_m3, formula_weight_g_mol, salt_to_polymer_ratio, salt_weight_factor=1.0):
    mw_eff = formula_weight_g_mol * salt_weight_factor
    mass_salt = jnp.clip(c_s_mol_m3, 0.0, None) * h0_ref_m * mw_eff / 1000.0
    mass_polymer = mass_salt / jnp.clip(salt_to_polymer_ratio, 1e-9, None)
    m_dry = mass_salt + mass_polymer
    mass_water = jnp.clip(c_w, 0.0, None) * h0_ref_m * WATER_MOLAR_MASS_KG_MOL
    return jnp.where(m_dry <= 0.0, 0.0, mass_water / jnp.clip(m_dry, 1e-30, None))


def absorption_effective_water_activity(c_w, *, t_gel_c, mass: "MassParams", salt_to_polymer_ratio):
    """max(brine activity, PAM-LiCl DVS cap) -- salt_properties.py's LiCl branch."""
    aw_brine = water_activity_licl_from_c_w(
        c_w, c_s=mass.c_s_mol_m3, h0_ref_m=mass.h0_ref_m,
        formula_weight_g_mol=mass.formula_weight_g_mol, temperature_c=t_gel_c,
    )
    u = pam_licl_gravimetric_uptake_g_g(
        c_w, h0_ref_m=mass.h0_ref_m, c_s_mol_m3=mass.c_s_mol_m3,
        formula_weight_g_mol=mass.formula_weight_g_mol, salt_to_polymer_ratio=salt_to_polymer_ratio,
    )
    aw_dvs = pam_licl_water_activity_from_uptake_g_g(u)
    return jnp.maximum(aw_brine, aw_dvs)


def dc_dh_absorption(c_w, *, t_gel_c, rh, h_m, mass: "MassParams", salt_to_polymer_ratio):
    """Eqs. 5-6 absorption branch: g_chamber (constant), driven by rh - a_w,eff."""
    aw = absorption_effective_water_activity(c_w, t_gel_c=t_gel_c, mass=mass, salt_to_polymer_ratio=salt_to_polymer_ratio)
    driving = rh - aw

    t_k = jnp.clip(t_gel_c + 273.15, 200.0, None)
    p_sat = saturation_vapor_pressure_pa(t_gel_c)
    g = mass.g_conv_m_s
    pref = g / mass.h0_ref_m
    rate = pref * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving
    rate = jnp.where(jnp.isfinite(rate), rate, 0.0)
    dc = jnp.where((c_w >= C_W_MAX_MOL_M3) & (rate > 0.0), 0.0, rate)
    dc = jnp.where((c_w <= C_W_MIN_MOL_M3) & (dc < 0.0), 0.0, dc)

    dh = g * WATER_MOLAR_MASS_KG_MOL / RHO_SOL_KG_M3 * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving
    dh = jnp.where(jnp.isfinite(dh), dh, 0.0)
    dh = jnp.where(h_m <= mass.h0_ref_m + 1e-12, jnp.maximum(dh, 0.0), dh)
    return dc, dh


def absorption_rhs(y, *, t_amb_c, rh, h0_ref_m, h_max_m, mass: "MassParams", salt_to_polymer_ratio):
    """dy/dt for y = [c_w, H] -- the 2-state absorption ODE (ode_system.py's
    _integrate_absorption). T_gel == T_amb during open absorption (fast gel
    thermal storage, Note S1 Eq. S1) -- no thermal root-solve needed here.
    """
    c_w, h_m_raw = y[0], y[1]
    h_m = jnp.maximum(h_m_raw, h0_ref_m)
    t_gel = t_amb_c
    dc, dh = dc_dh_absorption(c_w, t_gel_c=t_gel, rh=rh, h_m=h_m, mass=mass, salt_to_polymer_ratio=salt_to_polymer_ratio)
    dh = jnp.where(h_m > h0_ref_m + 1e-12, dh, jnp.maximum(dh, 0.0))
    dh = jnp.where((h_m >= h_max_m) & (dh > 0.0), 0.0, dh)
    return jnp.array([dc, dh])
