"""System physics: geometry/material constants, brine/salt thermodynamics, heat-transfer
correlations, thermal balances, mass transfer, and the PAM-salt hydrogel sorbent model."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import root

from solar_lumped._parameters_xlsx import SALTS, physics_value as _pv
from solar_lumped.utils import find_root_bracketed, load_two_column_csv

if TYPE_CHECKING:
    from solar_lumped.simulation import SystemConfig


# --- Table S3 / Note S1 system parameters (Wilson & Díaz-Marín *Device* 2025) ---
# All values load from docs/parameters.xlsx (Physics sheet).

# Geometry
H0_M: float = _pv("Hydrogel reference thickness (H0)", mm_to_m=True)  # hydrogel reference thickness H₀ (m)
L_G_M: float = _pv("Vapor gap (L_g)", mm_to_m=True)  # vapor gap L_g (m)
L_INS_M: float = _pv("Insulation gap (L_ins)", mm_to_m=True)  # insulation gap L_ins (m)
L_C_M: float = _pv("Condenser aluminum plate thickness (L_c)", mm_to_m=True)  # condenser aluminum plate thickness L_c (m)

# Wilson §2.2 / Note S1: thermobuoyancy and mass transport inhibited below ~7 mm gap
VAPOR_GAP_TRANSPORT_MIN_M: float = _pv("Vapor-gap transport floor", mm_to_m=True)

# Baseline ambient convection coefficient, Wilson et al. (2025) Methods (Cambridge test).
H_AMB_W_M2_K: float = _pv("Ambient convection coefficient (h_amb)")

# Salt loading default (g salt / g polymer); dual role, also economics.py sorbent cost.
SALT_LOADING_DEFAULT: float = _pv("Salt loading (SL)")

# Materials / transport
G_CHAMBER_M_S: float = _pv("Chamber convection coefficient, absorption (g_chamber)")
RHO_GEL_KG_M3: float = _pv("Composite (hydrogel) density at 20% RH (rho_gel)")  # ρ_gel composite/hydrogel density (kg/m³)
RHO_COMPOSITE_KG_M3: float = RHO_GEL_KG_M3  # alias -- composite density at fabrication (25 °C, 20% RH)
H_DES_J_PER_KG: float = _pv("Desorption enthalpy, LiCl (h_des)")  # h_des (J/kg)
H_FG_J_PER_KG: float = _pv("Condensation enthalpy (h_fg)")  # h_fg condensation (J/kg)
K_GEL_W_M_K: float = _pv("Hydrogel thermal conductivity (k_gel)")  # k_w hydrogel (W/m·K) — Table S3
RABS_M2_K_W: float = _pv("Absorber-to-gel constant resistance (Rabs)")  # R_aluminum + R_silicone, lumped and rounded
RHO_AL_KG_M3: float = _pv("Aluminum density (rho_Al)")
CP_AL_J_KG_K: float = _pv("Aluminum specific heat (cp_Al)")

# Optical / radiative
EPS_GEL: float = _pv("Gel emissivity (eps_gel)")
EPS_AL: float = _pv("Condenser (Al) emissivity (eps_Al)")
EPS_ABS: float = _pv("Absorber emissivity (eps_abs)")
TAU_GLASS: float = _pv("Glass transmittance (tau_glass)")
# This package's Case 2 ("selective surface") base-case IR emissivities -- see
# SystemConfig.thermal_params() in simulation.py, which is where these are applied.
EPS_ABS_IR_CASE2: float = _pv("Absorber IR emissivity (eps_abs_ir)")
EPS_GLASS_IR_CASE2: float = _pv("Glass IR emissivity (eps_glass_ir)")

# System orientation / condenser fins
TILT_DEG: float = _pv("Tilt angle (theta)")
FIN_AREA_RATIO: float = _pv("Condenser fin area ratio (A_r)")  # A_r


def u_gel_w_m2_k(h_m: float) -> float:
    """Note S1 gel–absorber conductance in series:
    1/U_gel = Rabs + H(t)/k_hydrogel, where Rabs lumps the (fixed) aluminum
    and silicone resistances into one rounded constant."""
    h = max(float(h_m), H0_M * 0.25)
    resistance = RABS_M2_K_W + h / K_GEL_W_M_K
    return 1.0 / resistance


# Reference value at fabrication thickness H₀ (for tests / docs)
U_GEL_W_M2_K: float = u_gel_w_m2_k(H0_M)

# Condenser thermal mass per footprint area (ρ_al c_p L_c)
CONDENSER_THERMAL_MASS_J_M2_K: float = RHO_AL_KG_M3 * CP_AL_J_KG_K * L_C_M


# --- Conde (2004) aqueous LiCl and CaCl2 solution properties ---
# Table 3: π ≡ p_sol(ξ,T)/p_H2O(T) = π_25(ξ)·f(ξ,θ), ξ = brine salt mass fraction,
# θ = T/T_c,H2O. π is the interface water activity a_w; p_H2O from Saul–Wagner.

# Conde (2004): θ ≡ T / T_c,H2O
T_CRIT_H2O_K: float = _pv("Water critical temperature (T_crit,H2O)")
P_CRIT_H2O_PA: float = _pv("Water critical pressure (P_crit,H2O)")
GAS_CONSTANT_J_MOL_K: float = _pv("Universal gas constant (R)")  # also the BET c below
WATER_MOLAR_MASS_KG_MOL: float = _pv("Water molar mass (MW_w)")

# Saturation (solubility) brine salt mass fraction: the most concentrated the liquid can
# get. Past it the excess salt is precipitated solid, so a_w is pinned here rather than
# following the correlation into its supersaturated tail. Derived from solubility in grams
# per litre of water, so xi = s / (s + 1000): LiCl 845 g/L -> 45.8 wt%, CaCl2 745 g/L ->
# 42.7 wt%. Both sit inside Conde's stated validity range (§ Density: LiCl 0.56, CaCl2
# 0.60), so the correlation is never evaluated outside its own domain.
# Read only for salts with no crystallization line -- i.e. NaCl and MgCl2. The LiCl,
# CaCl2 and LiBr entries are inert: those three resolve saturation against temperature
# instead, and their sheet values are kept only as the 25 C reference.
SOLUBILITY_G_PER_L: dict[str, float] = {
    name: float(row["solubility_g_per_l"]) for name, row in SALTS.items()
}


# Conde (2004) Tables 1-2, crystallization line: theta = A0 + A1*xi + A2*xi^2, with
# theta = T / T_c,H2O. One branch per solid phase, each valid over the temperature range
# where that phase is the one that crystallizes. Conde reproduces Fig. 1/2 rather than
# tabulating the transition points, so the branch is chosen as the *lowest* xi any branch
# admits at this temperature -- the solution saturates as soon as some solid can form.
# Validated against literature solubility: LiCl within 0.002 mass fraction over 0-100 C,
# CaCl2 within 0.004 except near 40 C, in the alpha-tetrahydrate range Conde itself flags
# as partly metastable. Kept inline with the other Conde fits (_SAUL_WAGNER_A, the pi
# parameters) rather than in the workbook: a multi-branch published fit, not a knob.
# Range over which some branch admits a root in (0, 1) for EVERY salt here, measured by
# sweeping the lines; evaluation temperature is clamped into it (see below). The upper
# bound is set by whichever salt runs out first, so adding a salt can lower it: LiBr's
# line reaches xi = 1 at 360.7 C, against 937 C for CaCl2 and 1080 C for LiCl, and this
# was 440 C until LiBr arrived. Above the bound no candidate root is in (0, 1) and the
# "unreachable" raise below becomes reachable, which is why the clamp tracks the
# minimum rather than any one salt's limit. Nothing physical lives up here -- the gel
# tops out near 160 C -- this only has to stay total for wild Newton iterates.
_CRYSTALLIZATION_T_LO_C: float = -200.0
_CRYSTALLIZATION_T_HI_C: float = 360.0

_CRYSTALLIZATION_LINE: dict[str, tuple[tuple[float, float, float], ...]] = {
    "LiCl": (
        (-0.005340, 2.015890, -3.114590),  # LiCl.5H2O
        (-0.560360, 4.723080, -5.811050),  # LiCl.3H2O
        (-0.315220, 2.882480, -2.624330),  # LiCl.2H2O
        (-1.312310, 6.177670, -5.034790),  # LiCl.H2O
        (-1.356800, 3.448540, 0.0),        # anhydrous LiCl
    ),
    "CaCl2": (
        (-0.378950, 3.456900, -3.531310),  # CaCl2.6H2O
        (-0.519970, 3.400970, -2.851290),  # CaCl2.4H2O alpha
        (-1.149044, 5.509111, -4.642544),  # CaCl2.4H2O beta
        (-2.385836, 8.084829, -5.303476),  # CaCl2.2H2O
        (-2.807560, 4.678250, 0.0),        # CaCl2.H2O
    ),
    # LiBr is in neither Conde nor Patek -- Patek & Klomfar's LiBr formulation states
    # validity "from 273 K or from the crystallization line" but does not supply that
    # line (it is in their companion Fluid Phase Equilibria paper). So this is derived
    # here, from Greenspan (1977) J. Res. NBS 81A 89-96: the measured equilibrium RH
    # over *saturated* LiBr solution, which IS the deliquescence point, fitted there
    # over 0-100 C from 21 points (sigma 0.22%):
    #     RH% = 7.75437 - 0.0654994 t + 0.420737e-3 t^2      (t in C)
    # Inverting patek_libr_water_activity at that RH gives the saturated composition,
    # xi_sat 0.585 (0 C) to 0.694 (100 C). Against literature solubility (~0.578 at
    # 0 C, 0.588 at 20 C, 0.638 at 60 C, 0.718 at 100 C) that is good to ~0.02-0.03 --
    # the cost of deriving saturation from DRH plus an isotherm instead of measuring
    # it. Boryta (1970) or the Patek FPE paper would decouple the two.
    # Refit by fit_libr_bet.py, which owns the whole LiBr derivation.
    #
    # One linear branch, deliberately, for three reasons. Greenspan's smooth DRH fit
    # does not resolve LiBr's hydrate transitions, so nothing justifies separate
    # branches. A quadratic fits xi better (0.0032 vs 0.0075) but opens upward with its
    # vertex at xi = 0.427, putting a spurious second root INSIDE (0, 1) and *below*
    # the real one -- and the branch selection below takes the minimum, so it would
    # return the wrong root at every temperature. And linear stays total across the
    # whole clamp range, where the quadratic would not.
    #
    # Fitted over Greenspan's full 0-100 C span even though a 0-80 C window fits xi
    # nearly 3x better (0.0028 vs 0.0075). What matters downstream is deliquescence
    # a_w, not xi, and on that the full-span fit wins: 0.0043 against 0.0069 over
    # 10-100 C. Greenspan's quadratic turns back upward above 77.8 C, which is a
    # fitting artifact and is what costs the xi fit its accuracy at the top end.
    "LiBr": (
        (-0.370950, 1.350549, 0.0),        # single branch, hydrates unresolved
    ),
}


@lru_cache(maxsize=4096)
def saturation_brine_salt_fraction(
    salt_name: str, temperature_c: float = 25.0
) -> float:
    """Saturation brine salt mass fraction at this temperature.

    LiCl and CaCl2 invert Conde's crystallization line, so saturation follows
    temperature (LiCl 0.458 at 25 C -> 0.528 at 80 C). That matters twice over: it is
    where regime 1 ends, and it is the composition a_w pins to once salt precipitates.
    Freezing it at 25 C -- which is what the single tabulated solubility does -- put the
    saturation boundary ~24% too wet at desorption temperature.

    LiBr has no Conde line but gets an equivalent one derived from Greenspan's measured
    deliquescence humidities (see _CRYSTALLIZATION_LINE), so it follows temperature too:
    0.585 at 0 C -> 0.694 at 100 C. Pinning it at the 20 C solubility instead did not
    merely lose accuracy, it inverted the trend -- deliquescence RH came out *rising*
    with temperature (0.029 at 0 C to 0.147 at 100 C) where the measurement falls
    (0.078 to 0.054), overstating the hot plateau 2.3x at 80 C.

    NaCl and MgCl2 still have no line, so they fall back to the tabulated solubility and
    stay temperature-independent. Their isotherms have no temperature dependence either,
    so there is nothing better to be had without replacing those fits first.
    """
    branches = _CRYSTALLIZATION_LINE.get(salt_name)
    if branches is None:
        try:
            s = SOLUBILITY_G_PER_L[salt_name]
        except KeyError:
            raise KeyError(
                f"No solubility for {salt_name!r}; add it to the Salts sheet "
                "(its deliquescence RH is derived from it)."
            ) from None
        return s / (s + 1000.0)

    # Clamped, because this is evaluated inside the thermal Newton solve and a wild
    # iterate (seen: -3185 C) would otherwise fall off the correlation entirely.
    # Clamping keeps the function total so a transient bad iterate cannot abort the
    # solve; see _CRYSTALLIZATION_T_HI_C for how the bound is chosen.
    t_c = min(max(float(temperature_c), _CRYSTALLIZATION_T_LO_C), _CRYSTALLIZATION_T_HI_C)
    theta = (t_c + 273.15) / T_CRIT_H2O_K
    best = math.inf
    for a0, a1, a2 in branches:
        if a2 == 0.0:
            candidates = ((theta - a0) / a1,) if a1 != 0.0 else ()
        else:
            disc = a1 * a1 - 4.0 * a2 * (a0 - theta)
            if disc < 0.0:
                continue
            root = math.sqrt(disc)
            candidates = ((-a1 + root) / (2.0 * a2), (-a1 - root) / (2.0 * a2))
        for xi in candidates:
            if 0.0 < xi < 1.0:
                best = min(best, xi)
    if not math.isfinite(best):  # unreachable given the clamp above
        raise ValueError(f"No crystallization branch for {salt_name!r} at {t_c} C")
    return best


# 25 C references, for callers that want a single number (and the Conde pi correlations'
# nominal validity limits). The temperature-resolved value is the function above.
XI_SAT_LICL: float = saturation_brine_salt_fraction("LiCl")
XI_SAT_CACL2: float = saturation_brine_salt_fraction("CaCl2")

_BRACKET_LO: float = _pv("Brine mass-fraction bracket lower")
_BRACKET_HI: float = _pv("Brine mass-fraction bracket upper")


# Upper temperature bound on the Conde (2004) Table 3 brine correlations, for BOTH salts.
# Conde states no range in the text; the bound comes from the data his temperature
# correction f(xi, theta) was fitted to -- Gibbard [26] for LiCl (MIT PhD thesis, 1966,
# static vapour pressure method) and Baker & Waite [27] for CaCl2 -- neither of which
# extends past ~100 C. LiCl was previously capped at 150 C here, which was unsupported.
#
# The gel exceeds this (~157 C in the evacuated two-pane case). LiCl no longer clamps
# there: it carries the Zeng & Zhou BET isotherm below as a temperature extension (see
# _bet_temperature_factor). CaCl2 has no such partner and still clamps.
CONDE_T_MAX_C: float = 100.0

# --- Zeng & Zhou (2006) BET isotherm: LiCl's temperature slope above Conde's cap ---
# J. Chem. Eng. Data 51(2), 315-321, eq 3:
#
#     a_w * m / (55.51 * (1 - a_w)) = 1/(c*r) + (c - 1) * a_w / (c*r)
#
# with m the molality (mol salt / kg water), r the number of hydration sites and c the
# BET energy parameter. Both are linear in T: r = r0 + r1*T, dE = e0 + e1*T, c =
# exp(-dE/RT). Eq 3 is a quadratic in a_w and linear in m, so BOTH directions are closed
# form -- no root solve, unlike the Conde inversion.
#
# Sign note: the paper prints "dE = R T ln c", which would give c < 1. c = exp(-dE/RT)
# is what reproduces their own published numbers (r = 3.278, c = 9.329 at 373.15 K; a_w
# = 0.130 at m = 18.542, 298.15 K -> this code gives 3.322, 9.100, 0.1290), so the
# printed sign is a transcription slip. test_bet_isotherm pins all three.
BET_T_MAX_C: float = 155.5  # Zeng & Zhou fitted 273.15-428.65 K


@dataclass(frozen=True, slots=True)
class BetParams:
    r0: float
    r1: float
    e0: float
    e1: float
    formula_weight_g_mol: float


# Zeng & Zhou Figs. 4-5, fitted 0-155.5 C. Used only as a temperature *ratio* above
# CONDE_T_MAX_C, never as the absolute isotherm: below 40 wt% the BET runs 0.04-0.06 dry
# against Conde (0.63 vs 0.69 at xi = 0.20, 25 C), because Zeng & Zhou fit eq 3's linear
# region a_w < 0.4 and extrapolate outward. Conde is the better absolute isotherm there;
# what it lacks, and this supplies, is a data-backed dT slope past 100 C.
LICL_BET = BetParams(r0=4.7323, r1=-0.00378, e0=-8166.6, e1=3.526, formula_weight_g_mol=42.394)

_BET_MOLES_WATER_PER_KG: float = 1000.0 / 18.015  # 55.51, Zeng & Zhou's constant
BET_T_MIN_C: float = 0.0  # Zeng & Zhou's lower fit bound, 273.15 K


def _bet_r_c(temperature_c: float, p: BetParams) -> tuple[float, float]:
    # Clamped to the fit range for the same reason saturation_brine_salt_fraction is:
    # this runs inside the thermal Newton solve, and a wild iterate overflows exp() in
    # c and aborts the solve. Outside [0, 155.5] C both ends are extrapolation anyway.
    t_k = min(max(float(temperature_c), BET_T_MIN_C), BET_T_MAX_C) + 273.15
    return p.r0 + p.r1 * t_k, math.exp(-(p.e0 + p.e1 * t_k) / (GAS_CONSTANT_J_MOL_K * t_k))


def bet_water_activity(
    salt_mass_fraction: float, temperature_c: float, p: BetParams
) -> float:
    """a_w from Zeng & Zhou eq 3, solved as a quadratic in a_w."""
    xi = float(salt_mass_fraction)
    if not math.isfinite(xi) or xi <= 0.0:
        return 1.0 if xi == 0.0 else float("nan")
    if xi >= 1.0:
        return float("nan")
    m = 1000.0 * xi / (p.formula_weight_g_mol * (1.0 - xi))
    r, c = _bet_r_c(temperature_c, p)
    n = _BET_MOLES_WATER_PER_KG
    a, b, k = n * (c - 1.0), m * c * r - n * (c - 2.0), -n
    if abs(a) < 1e-12:
        return max(0.0, min(1.0, -k / b))
    disc = b * b - 4.0 * a * k
    if disc < 0.0:
        return float("nan")
    return max(0.0, min(1.0, (-b + math.sqrt(disc)) / (2.0 * a)))


def bet_brine_salt_fraction(
    water_activity: float, temperature_c: float, p: BetParams
) -> float:
    """Inverse of :func:`bet_water_activity`; eq 3 is linear in m, so this is explicit."""
    aw = float(water_activity)
    if not (0.0 < aw < 1.0):
        return float("nan")
    r, c = _bet_r_c(temperature_c, p)
    m = _BET_MOLES_WATER_PER_KG * (1.0 - aw) * (1.0 + (c - 1.0) * aw) / (c * r * aw)
    ms = m * p.formula_weight_g_mol
    return ms / (1000.0 + ms)


def _bet_temperature_factor(xi: float, temperature_c: float, p: BetParams) -> float:
    """a_w(xi, T) / a_w(xi, 100 C) from the BET -- the dT slope Conde does not have.

    Anchoring on the ratio rather than substituting the BET wholesale keeps Conde's
    (better) absolute isotherm and is continuous at 100 C by construction, while landing
    within 0.02 of the pure BET at 155 C. Beyond BET_T_MAX_C the factor is frozen: past
    that both correlations are extrapolating and the honest answer is to stop.
    """
    t = min(float(temperature_c), BET_T_MAX_C)
    ref = bet_water_activity(xi, CONDE_T_MAX_C, p)
    if not (math.isfinite(ref) and ref > 0.0):
        return 1.0
    hot = bet_water_activity(xi, t, p)
    return hot / ref if math.isfinite(hot) else 1.0


@dataclass(frozen=True, slots=True)
class VaporPressureParams:
    """Table 3 parameters for the Conde vapour-pressure equation."""

    pi0: float
    pi1: float
    pi2: float
    pi3: float
    pi4: float
    pi5: float
    pi6: float
    pi7: float
    pi8: float
    pi9: float
    # 25 C reference only. The clip in vapor_pressure_ratio resolves saturation at the
    # evaluation temperature via saturation_brine_salt_fraction(salt_name, T).
    xi_sat: float
    salt_name: str
    # BET partner supplying the temperature slope above CONDE_T_MAX_C, or None to clamp
    # there (CaCl2: Zeng & Zhou fitted LiCl only).
    bet: BetParams | None = None
    # Highest temperature this salt's isotherm is defined at, extension included.
    t_max_c: float = 100.0


LICL_VAPOR_PRESSURE = VaporPressureParams(
    pi0=0.28,
    pi1=4.30,
    pi2=0.60,
    pi3=0.21,
    pi4=5.10,
    pi5=0.49,
    pi6=0.362,
    pi7=-4.75,
    pi8=-0.40,
    pi9=0.03,
    xi_sat=XI_SAT_LICL,
    salt_name="LiCl",
    bet=LICL_BET,
    t_max_c=BET_T_MAX_C,
)

CACL2_VAPOR_PRESSURE = VaporPressureParams(
    pi0=0.31,
    pi1=3.698,
    pi2=0.60,
    pi3=0.231,
    pi4=4.584,
    pi5=0.49,
    pi6=0.478,
    pi7=-5.20,
    pi8=-0.40,
    pi9=0.018,
    xi_sat=XI_SAT_CACL2,
    salt_name="CaCl2",
)

# Saul–Wagner (Appendix A, Table 12)
_SAUL_WAGNER_A: tuple[float, ...] = (
    -7.858230,
    1.839910,
    -11.781100,
    22.670500,
    -15.939300,
    1.775160,
)
_SAUL_WAGNER_EXP: tuple[float, ...] = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)


def vapor_pressure_ratio(
    salt_mass_fraction: float,
    temperature_c: float,
    params: VaporPressureParams,
) -> float:
    """π = p_sol / p_H2O; equals brine water activity a_w."""
    xi = float(salt_mass_fraction)
    if not math.isfinite(xi):
        return float("nan")
    if xi <= 0.0:
        return 1.0
    if xi >= 1.0:
        return float("nan")
    # Past saturation the excess salt is precipitated, not dissolved: the liquid stays at
    # its saturated composition, so a_w is pinned there rather than undefined. Returning
    # nan instead made dc_w_dt clamp to 0, freezing a dry gel exactly when deliquescence
    # should drive the strongest uptake. Mirrored by jax_physics.vapor_pressure_ratio_licl
    # (_XI_SAT_LICL) -- keep the two in step or CPU and GPU diverge above saturation.
    xi = min(xi, saturation_brine_salt_fraction(params.salt_name, float(temperature_c)))
    # Conde's f(xi, theta) was fitted to <=100 C data, so evaluate it at its own ceiling
    # and carry the rest of the way on the BET slope. Salts with no BET partner clamp,
    # which is what this did for both before.
    t_conde = min(float(temperature_c), CONDE_T_MAX_C)
    theta = (t_conde + 273.15) / T_CRIT_H2O_K  # θ = T / T_c,H2O
    pi_25 = (
        1.0
        - (1.0 + (xi / params.pi6) ** params.pi7) ** params.pi8
        - params.pi9 * math.exp(-((xi - 0.1) ** 2) / 0.005)
    )
    a_term = 2.0 - (1.0 + (xi / params.pi0) ** params.pi1) ** params.pi2
    b_term = (1.0 + (xi / params.pi3) ** params.pi4) ** params.pi5 - 1.0
    pi = pi_25 * (a_term + b_term * theta)
    if params.bet is not None and float(temperature_c) > CONDE_T_MAX_C:
        pi *= _bet_temperature_factor(xi, float(temperature_c), params.bet)
    return max(0.0, min(1.0, float(pi)))


def water_activity_licl(
    salt_mass_fraction: float,
    temperature_c: float = 25.0,
) -> float:
    """LiCl–H2O brine water activity (Conde 2004 Table 3)."""
    return vapor_pressure_ratio(salt_mass_fraction, temperature_c, LICL_VAPOR_PRESSURE)


def equilibrium_salt_mass_fraction(
    relative_humidity: float,
    params: VaporPressureParams,
    *,
    temperature_c: float = 25.0,
    temperature_max_c: float | None = None,
) -> float:
    """Invert a_w(ξ) = RH for brine salt mass fraction ξ."""
    rh = float(relative_humidity)
    if rh <= 0.0:
        return 1.0
    if temperature_c > (params.t_max_c if temperature_max_c is None else temperature_max_c):
        return float("nan")
    # Bracket exhaustion, not a dilution cap: above the activity of the most dilute
    # bracketed brine the root lies below _BRACKET_LO and find_root_bracketed has
    # nothing to bisect, so _BRACKET_LO is the closest representable answer. Derived
    # per salt and temperature rather than approximated by a flat 0.99 (the true
    # threshold is 0.9930 for LiCl, 0.9961 for CaCl2).
    if rh >= vapor_pressure_ratio(_BRACKET_LO, temperature_c, params):
        return _BRACKET_LO

    def residual(xi: float) -> float:
        return rh - vapor_pressure_ratio(xi, temperature_c, params)

    hi = min(_BRACKET_HI, saturation_brine_salt_fraction(params.salt_name, temperature_c))
    return find_root_bracketed(residual, _BRACKET_LO, hi)


def equilibrium_salt_mass_fraction_licl(
    relative_humidity: float,
    temperature_c: float = 25.0,
) -> float:
    """Equilibrium LiCl brine salt mass fraction at RH and T."""
    return equilibrium_salt_mass_fraction(
        relative_humidity,
        LICL_VAPOR_PRESSURE,
        temperature_c=temperature_c,
    )


def equilibrium_salt_mass_fraction_cacl2(
    relative_humidity: float,
    temperature_c: float = 25.0,
) -> float:
    """Equilibrium CaCl2 brine salt mass fraction at RH and T."""
    return equilibrium_salt_mass_fraction(
        relative_humidity,
        CACL2_VAPOR_PRESSURE,
        temperature_c=temperature_c,
    )


def _saul_wagner_pa(t_k: float) -> float:
    """Saul-Wagner (Conde 2004 Appendix A) with no lower domain guard.

    Split out for the Patek Duhring path, which evaluates p_s at an effective
    temperature theta well below the triple point for concentrated brine (-20 C at
    65 wt% / 25 C). Extrapolating there is safe and deliberate: against tabulated
    SUPERCOOLED liquid water this reads +0.14% at 0 C, +0.19% at -20 C and +0.50% at
    -30 C, an order of magnitude inside Patek's own 2.1% RMS. The public wrapper below
    keeps the guard, so nothing outside that path changes.
    """
    if t_k >= T_CRIT_H2O_K or t_k <= 0.0:
        return float("nan")
    tau = 1.0 - t_k / T_CRIT_H2O_K
    numer = sum(a * tau**exp for a, exp in zip(_SAUL_WAGNER_A, _SAUL_WAGNER_EXP, strict=True))
    return float(P_CRIT_H2O_PA * math.exp(numer / (1.0 - tau)))


def water_vapor_pressure_pa(temperature_c: float) -> float:
    """Saul–Wagner vapour pressure of pure liquid water (Conde 2004 Appendix A)."""
    t_k = float(temperature_c) + 273.15
    if t_k <= 273.15:
        return float("nan")
    return _saul_wagner_pa(t_k)


# --- Patek & Klomfar (2006) LiBr-H2O isotherm ---
# Int. J. Refrig. 29(4), 566-578, Eq. (1). A Duhring form: the solution's vapour
# pressure at T equals pure water's saturation pressure at a lower effective
# temperature theta, so the water activity is just a ratio of p_s evaluations.
#
#     p(T,x) = p_s(theta),  theta = T - sum_i a_i x^m_i (0.4-x)^n_i (T/T_c)^t_i
#     a_w    = p(T,x) / p_s(T) = p_s(theta) / p_s(T)
#
# x is the MOLE fraction of (undissociated) LiBr, not the mass fraction the rest of
# this module carries; Eq. (7) converts. Coefficients are Table 4, RMS 2.1%.
#
# This replaced a BET fitted here to the two CSVs in data/materials. Patek is strictly
# better on all three counts that matter:
#   * 155 C is measured, not extrapolated. Eq. (1) is fitted to 1449 primary points
#     from 30 works, including Lenard et al. 1992 (398-483 K, 44-65 wt%, RMS 1.31%),
#     Feuerecker et al. 1993 (318-462 K, 40-75 wt%) and Iyoki & Uemura 1989
#     (362-452 K, 39-70 wt%) -- dense coverage exactly where the gel desorbs.
#   * It rejects Patil & Tripathi 1990. Patek collected all 40 points of it and used
#     ZERO (RMS 4.76%, against 1.3-2.0% for the sets he kept). That is the same PT90
#     that is half the local CSV data, and its disagreement with the other CSV was the
#     bulk of the retired BET's fit residual.
#   * The BET was 2-3x too dry above 60 wt% at 25 C (0.038 vs 0.079 at 60 wt%),
#     because it extrapolated past its 60.4 wt% data ceiling -- which is precisely
#     where saturation sits.
_PATEK_LIBR: tuple[tuple[int, int, int, float], ...] = (  # (m_i, n_i, t_i, a_i)
    (3, 0, 0, -2.41303e2),
    (4, 5, 0, 1.91750e7),
    (4, 6, 0, -1.75521e8),
    (8, 3, 0, 3.25430e7),
    (1, 0, 1, 3.92571e2),
    (1, 2, 1, -2.12626e3),
    (4, 6, 1, 1.85127e8),
    (6, 0, 1, 1.91216e3),
)
LIBR_FORMULA_WEIGHT_G_MOL: float = float(SALTS["LiBr"]["formula_weight_g_mol"])
# Patek fitted p-T-x data below 483 K and demonstrates the equation extrapolates with
# defined uncertainty to 623 K; 483 K is where the *data* stops, so that is the cap.
LIBR_T_MAX_C: float = 210.0
LIBR_T_MIN_C: float = 0.0  # Patek's stated lower bound, 273 K
# Composition ceiling, Patek's stated 75 wt%. This binds above ~142 C, where the
# crystallization line would otherwise push saturation past it: real LiBr solubility
# keeps climbing but the correlation has no support, so xi is clipped here instead.
LIBR_W_MAX: float = 0.75


def patek_libr_water_activity(salt_mass_fraction: float, temperature_c: float) -> float:
    """LiBr-H2O brine water activity, Patek & Klomfar (2006) Eq. (1)."""
    w = float(salt_mass_fraction)
    if not math.isfinite(w) or w < 0.0 or w >= 1.0:
        return float("nan")
    if w <= 0.0:
        return 1.0
    w = min(w, LIBR_W_MAX)
    # Clamped like the BET and the crystallization line: this is evaluated inside the
    # thermal Newton solve, where one wild iterate must not poison the whole solve.
    t_k = min(max(float(temperature_c), LIBR_T_MIN_C), LIBR_T_MAX_C) + 273.15
    x = (w / LIBR_FORMULA_WEIGHT_G_MOL) / (
        w / LIBR_FORMULA_WEIGHT_G_MOL + (1.0 - w) / (WATER_MOLAR_MASS_KG_MOL * 1000.0)
    )
    theta = t_k - sum(
        a * x**m * (0.4 - x) ** n * (t_k / T_CRIT_H2O_K) ** t for m, n, t, a in _PATEK_LIBR
    )
    p_sol, p_pure = _saul_wagner_pa(theta), _saul_wagner_pa(t_k)
    if not (math.isfinite(p_sol) and math.isfinite(p_pure) and p_pure > 0.0):
        return float("nan")
    return max(0.0, min(1.0, p_sol / p_pure))


# --- Heat-transfer correlations (Hollands et al. 1976; Wilson Note S1) ---

STEFAN_BOLTZMANN_W_M2_K4: float = _pv("Stefan-Boltzmann constant (sigma)")
GRAVITY_M_S2: float = _pv("Gravitational acceleration (g)")
# Sea-level reference, and the default for every property function below. Real sites
# override it: SystemConfig.site_elevation_m -> pressure_from_elevation_m -> the p_atm_pa
# carried on SystemThermalParams/MassTransferParams. It is a default, not "the" pressure.
P_ATM_SEA_LEVEL_PA: float = _pv("Atmospheric pressure (P0)")


def pressure_from_elevation_m(elevation_m: float) -> float:
    """ISA barometric pressure (Pa) for a site elevation.

    The troposphere branch, p = p0·(1 − 2.25577e-5·h)^5.25588, valid to 11 km and so over
    every inhabited elevation. Standard-atmosphere temperature, not the site's own -- a
    real column is warmer or colder than ISA, which shifts pressure by a percent or two,
    far inside what the transport correlations themselves are worth (D_air is ±5-10%).
    Prefer measured surface pressure where the weather feed carries it.
    """
    return P_ATM_SEA_LEVEL_PA * (1.0 - 2.25577e-5 * max(float(elevation_m), 0.0)) ** 5.25588


# --- Air transport properties: Tsilingiris (2008) humid air, Marrero & Mason (1972) D ---
#
# These used to be five fixed workbook constants near ambient. The device does not run
# there: the vapor gap sits at a 45-90 C film temperature and the absorber-glass gap
# hotter still, and across that span nu rises ~30%, alpha ~15% and D_air ~25%. The
# retired constants were not even mutually consistent -- k_air = 0.0286 is the 63 C
# value while nu_air = 1.5e-5 and rho_air = 1.2 are ~20 C values -- so the old Ra mixed
# two temperatures inside one dimensionless group.
#
# Tsilingiris, Energy Convers. Manage. 49 (2008) 1098, transcribed in
# data/materials/tsilingiris2008.tex; equation numbers below are that paper's. Humid air
# as a binary dry-air/water-vapor mixture over 0-100 C, i.e. exactly this device's band.
#
# Two simplifications, both sanctioned by the paper itself, and together they remove
# every exponential from the hot path and leave plain polynomial evaluation:
#   - enhancement factor f = 1 (drops Eqs. 6-8): under 0.5% on x_v;
#   - compressibility z = 1 (drops Eqs. 13-16): 0.4-1.5% on density.
# Its Eq. (9) saturation-pressure polynomial is deliberately NOT used: it is 27% low at
# 20 C and biased low across the whole range (a defect in the source, not the
# transcription). water_vapor_pressure_pa -- Saul-Wagner, already here for the isotherms
# -- is both correct and what the rest of this module reads, so the two agree by
# construction.
#
# Two unit captions in the source are wrong; the .tex flags both and this follows the
# corrected reading. Eq. (39) is captioned "W/m K x 10^-3" but already returns W/m K
# (it gives 0.02403 at 273.15 K). Eq. (41) is captioned "Ns/m2 x 10^-6" but needs
# 10^-7: with 10^-6 the mixture viscosity comes out +905% at 100 C, with 10^-7 it
# reproduces the paper's own Table 4 saturated-mixture fit to 0.17%.
#
# Table 4's fitted polynomials are NOT used either. Its k/alpha/Pr rows cannot be
# reconciled with Eq. (29) above ~65 C by any single-coefficient repair (the best SK2 is
# -3.86e-7, still 5.8% off), and 65 C is mid-range here. Eqs. (12)/(21)/(29)/(34)/(35)
# reproduce Table 4's density, viscosity and cp to under 0.7% over 0-100 C, so the
# first-principles route is the internally consistent one.

# Eqs. (38)-(40), dry air, argument T in KELVIN. Eqs. (41)-(43), water vapor, argument
# t in CELSIUS. The mismatch is the paper's, kept so the coefficients stay greppable
# against it.
_MU_A_NS_M2: tuple[float, ...] = (
    -9.8601e-1, 9.080125e-2, -1.17635575e-4, 1.2349703e-7, -5.7971299e-11,
)
_K_A_W_M_K: tuple[float, ...] = (
    -2.276501e-3, 1.2598485e-4, -1.4815235e-7, 1.73550646e-10, -1.066657e-13,
    2.47663035e-17,
)
_CP_A_KJ_KG_K: tuple[float, ...] = (
    0.103409e1, -0.284887e-3, 0.7816818e-6, -0.4970786e-9, 0.1077024e-12,
)
_MU_V_NS_M2: tuple[float, ...] = (8.058131868e1, 4.000549451e-1)
_K_V_W_M_K: tuple[float, ...] = (1.761758242e1, 5.558941059e-2, -1.663336663e-4)
_CP_V_KJ_KG_K: tuple[float, ...] = (1.86910989, -2.578421578e-4, 1.941058941e-5)

# Eq. (37): standard dry-air composition. Water's molar mass and R come from the
# workbook (per mole there, per kmol in Tsilingiris's equations).
_M_AIR_KG_KMOL: float = 28.9635
_M_H2O_KG_KMOL: float = WATER_MOLAR_MASS_KG_MOL * 1e3
_R_J_KMOL_K: float = GAS_CONSTANT_J_MOL_K * 1e3

# The water-vapor fits (Eqs. 41-43) are stated for 0-120 C, the narrowest of the set.
# Property evaluation is clamped into it. This is for solver iterates only -- the same
# reason _CRYSTALLIZATION_T_LO_C/_HI_C exist above. solve_steady_thermal probes wildly
# inverted (T_gel, T_cond) pairs on ~16% of calls, and nothing physical lives out there;
# converged states are inside by a wide margin.
_AIR_PROPS_T_LO_C: float = 0.0
_AIR_PROPS_T_HI_C: float = 120.0


def _poly(coeffs: tuple[float, ...], x: float) -> float:
    """Horner, ascending coefficient order (as the papers print them)."""
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


@lru_cache(maxsize=4096)
def humid_air_props(
    t_film_c: float, *, x_v: float = 0.0, p_atm_pa: float = P_ATM_SEA_LEVEL_PA
) -> tuple[float, float, float]:
    """(k, nu, alpha) in SI for humid air at one film temperature.

    Cached: pure in its arguments, and the root solve re-probes identical iterates often
    enough to hit ~72%. Together with the cache on vapor_gap_water_mole_fraction this
    brings the cost of replacing five constants with correlations down from 1.52x to
    1.06x on a daily cycle. No approximation -- exact same values, just not recomputed.

    Tsilingiris Eq. (21) viscosity, (29) conductivity, (34) specific heat, (12) density,
    (35) alpha. The Wilke/Wassiljewa interaction parameters of Eqs. (22)-(23) serve both
    the viscosity and the conductivity mixing rules, because Reid's epsilon = 1 collapses
    Theta_ij onto Phi_ij (Eq. 27) -- so they are computed once.

    ``x_v`` is the water-vapor mole fraction; the 0.0 default is dry air, which is what
    the sealed glazing gaps are. Pass vapor_gap_water_mole_fraction(T_cond) for the
    vapor gap.
    """
    t = min(max(float(t_film_c), _AIR_PROPS_T_LO_C), _AIR_PROPS_T_HI_C)
    t_k = t + 273.15
    xv = min(max(float(x_v), 0.0), 1.0)
    xa = 1.0 - xv

    mu_a = _poly(_MU_A_NS_M2, t_k) * 1e-6      # Eq. (38)
    k_a = _poly(_K_A_W_M_K, t_k)               # Eq. (39) -- already W/m K, see above
    cp_a = _poly(_CP_A_KJ_KG_K, t_k) * 1e3     # Eq. (40)
    mu_v = _poly(_MU_V_NS_M2, t) * 1e-7        # Eq. (41) -- 1e-7, not the printed 1e-6
    k_v = _poly(_K_V_W_M_K, t) * 1e-3          # Eq. (42)
    cp_v = _poly(_CP_V_KJ_KG_K, t) * 1e3       # Eq. (43)

    # Dry air only: skip the mixing rules entirely rather than multiply zeros through
    # them. This is the glazing-gap path, called twice per residual evaluation.
    if xv <= 0.0:
        rho = p_atm_pa * _M_AIR_KG_KMOL / (_R_J_KMOL_K * t_k)
        return k_a, mu_a / rho, k_a / (rho * cp_a)

    m_ratio = _M_AIR_KG_KMOL / _M_H2O_KG_KMOL
    root2_4 = math.sqrt(2.0) / 4.0
    phi_av = root2_4 / math.sqrt(1.0 + m_ratio) * (
        1.0 + math.sqrt(mu_a / mu_v) * (1.0 / m_ratio) ** 0.25
    ) ** 2
    phi_va = root2_4 / math.sqrt(1.0 + 1.0 / m_ratio) * (
        1.0 + math.sqrt(mu_v / mu_a) * m_ratio**0.25
    ) ** 2

    mu = xa * mu_a / (xa + xv * phi_av) + xv * mu_v / (xv + xa * phi_va)
    k = xa * k_a / (xa + xv * phi_av) + xv * k_v / (xv + xa * phi_va)
    cp = (cp_a * xa * _M_AIR_KG_KMOL + cp_v * xv * _M_H2O_KG_KMOL) / (
        _M_AIR_KG_KMOL * xa + _M_H2O_KG_KMOL * xv
    )
    rho = p_atm_pa * _M_AIR_KG_KMOL / (_R_J_KMOL_K * t_k) * (1.0 - xv * (1.0 - 1.0 / m_ratio))
    return k, mu / rho, k / (rho * cp)


@lru_cache(maxsize=256)
def vapor_gap_water_mole_fraction(
    t_cond_c: float, *, p_atm_pa: float = P_ATM_SEA_LEVEL_PA
) -> float:
    """x_v in the vapor gap, set by the condenser.

    Cached, and this one is nearly free: T_cond is an ODE state held fixed for the whole
    m_des root solve, so the Saul-Wagner evaluation behind it gets asked the same question
    ~760k times per daily cycle against ~2.8k distinct answers (99.6% hit rate).

    The condenser is the condensing surface, so it -- not the gel, and not saturation at
    the film temperature -- pins the gap's vapor partial pressure at p_sat(T_cond).
    Getting this wrong is the easy mistake and it is not small: assuming saturation at a
    65 C film temperature gives x_v = 0.25 and a 12% error in alpha, where the correct
    x_v over a 30 C condenser is 0.04 and costs 2%.
    """
    p_v = water_vapor_pressure_pa(t_cond_c)
    if not math.isfinite(p_v):  # Saul-Wagner is NaN at/below 0 C; a frozen gap is dry
        return 0.0
    return min(max(p_v / p_atm_pa, 0.0), 1.0)


# Marrero & Mason (1972) J. Phys. Chem. Ref. Data 1, 3 -- Table 13 ("Correlation
# Parameters of Eq. 4.3-2"), row air-H2O, low-temperature branch: 10^5 A = 0.187,
# s = 2.072, S = none, 282-450 K, uncertainty +/-5 to 10% (their Table 11). With the
# Sutherland term absent, Eq. (4.3-2) ln(p D) = ln A + s ln T - S/T reduces to a plain
# power law. A = 1.87e-6 atm cm^2/(s K^s), so in SI D = 1.87e-10 T^2.072 / P[atm].
#
# Their 450-1070 K second branch is unreachable here, so this stays single-branch. Air
# rows in that table are trace diffusion through excess air, which is the vapor gap's
# composition. Uniquely among the paper's air systems this fit is not from Blanc's law:
# it synthesizes O'Connell (1969) H2O-N2 with Walker & Westenberg (1960) H2O-O2, which
# is why its uncertainty is wider than the +/-4% of H2O-N2 alone.
_D_H2O_AIR_A_ATM_CM2: float = 1.87e-6
_D_H2O_AIR_S: float = 2.072
# 282-450 K in Celsius. The paper warns against downward extrapolation specifically
# (Eqs. 4.3-1/2 are unusable where London dispersion dominates), hence the clamp.
_D_H2O_AIR_T_LO_C: float = 8.85
_D_H2O_AIR_T_HI_C: float = 176.85
_ATM_IN_PA: float = 101325.0  # definition of the atmosphere, not the site pressure


def d_h2o_air_m2_s(t_c: float, *, p_atm_pa: float = P_ATM_SEA_LEVEL_PA) -> float:
    """Binary diffusivity of water vapor in air (m²/s), Marrero & Mason Table 13."""
    t_k = min(max(float(t_c), _D_H2O_AIR_T_LO_C), _D_H2O_AIR_T_HI_C) + 273.15
    d_cm2_s = _D_H2O_AIR_A_ATM_CM2 * t_k**_D_H2O_AIR_S / (p_atm_pa / _ATM_IN_PA)
    return d_cm2_s * 1e-4


# --- Ambient convection: density dependence of h_amb ---
#
# h_amb is an empirical Wilson calibration (~10 W/m2 K at 0.5 m/s, Methods) and stays
# that. What it was missing is that external forced convection thins out with the air:
# h = Nu*k/L with Nu ~ Re^n Pr^(1/3) and Re = u*L/nu, and nu ~ 1/rho, so h ~ rho^n. Air at
# 2400 m is 25% less dense than at sea level, which is a 14% cut in convective cooling --
# a real penalty on a device whose condenser rejects heat to ambient, and the offset that
# keeps the elevation gain in D_air from being a free lunch.
#
# n = 0.5 (laminar flat plate) is the default: at 0.5 m/s over the ~1 m aperture Re ~ 3e4,
# well below the 5e5 transition. n = 0.8 is the turbulent value, and the workbook carries
# it as the sweep upper bound.
#
# Honest caveat: the wind law this multiplies is linear in u, which is not itself a
# flat-plate form (laminar would be u^0.5), so a theoretically-derived pressure exponent
# is being grafted onto an empirical wind fit. That is why n is a workbook knob rather
# than a hardcoded 0.5 -- the exponent is the least defensible number in this block.
H_AMB_DENSITY_EXPONENT: float = _pv("Ambient convection density exponent")
# Wilson's calibration point: dry air, sea level, 25 C.
_H_AMB_REF_T_C: float = 25.0
_RHO_AIR_REF_KG_M3: float = (
    P_ATM_SEA_LEVEL_PA * _M_AIR_KG_KMOL / (_R_J_KMOL_K * (_H_AMB_REF_T_C + 273.15))
)


def h_amb_density_factor(
    t_amb_c: float,
    *,
    p_atm_pa: float = P_ATM_SEA_LEVEL_PA,
    exponent: float = H_AMB_DENSITY_EXPONENT,
) -> float:
    """(ρ_amb/ρ_ref)^n, the factor multiplying every h_amb at the point of use.

    Density, not bare pressure, so a hot day gets the same treatment as a high one:
    ρ = p·M/(R·T) falls with either. Dry air -- ambient humidity lowers ρ by under 1% over
    this range, which is not worth threading RH through for.

    Returns 1.0 exactly at sea level and 25 C, so a sea-level site is bit-for-bit
    unchanged by this whole mechanism.
    """
    t_k = max(float(t_amb_c) + 273.15, 1.0)
    rho = max(float(p_atm_pa), 0.0) * _M_AIR_KG_KMOL / (_R_J_KMOL_K * t_k)
    return (rho / _RHO_AIR_REF_KG_M3) ** exponent


def parallel_plate_emissivity(eps_a: float, eps_b: float) -> float:
    """Note S1 Eq. S2 — infinite parallel plates."""
    if eps_a <= 0.0 or eps_b <= 0.0:
        return 0.0
    return 1.0 / (1.0 / eps_a + 1.0 / eps_b - 1.0)


def mass_transfer_g_from_h_conv_m_s(
    h_conv_w_m2_k: float,
    *,
    t_gel_c: float,
    t_cond_c: float,
    p_atm_pa: float = P_ATM_SEA_LEVEL_PA,
) -> float:
    """Note S1 Eq. S5 (Le ≈ 1): g = h_conv · D_air / k_air.

    Both properties are evaluated at the vapor gap's own film temperature, so the ratio
    stays a ratio of the same air. k_air cancels analytically against h_conv = Nu·k_air/L
    (leaving g = Nu·D_air/L), and keeping the cancelling form is deliberate: it is the
    published Eq. S5, and the Sh == Nu identity it implies is what the tests assert.
    """
    if h_conv_w_m2_k <= 0.0:
        return 0.0
    t_film_c = 0.5 * (t_gel_c + t_cond_c)
    x_v = vapor_gap_water_mole_fraction(t_cond_c, p_atm_pa=p_atm_pa)
    k_air = humid_air_props(t_film_c, x_v=x_v, p_atm_pa=p_atm_pa)[0]
    return h_conv_w_m2_k * d_h2o_air_m2_s(t_film_c, p_atm_pa=p_atm_pa) / k_air


def radiative_exchange_w_m2(t_hot_c: float, t_cold_c: float, *, emissivity: float = 0.9) -> float:
    t_hot_k = t_hot_c + 273.15
    t_cold_k = t_cold_c + 273.15
    return emissivity * STEFAN_BOLTZMANN_W_M2_K4 * (t_hot_k**4 - t_cold_k**4)


def rayleigh_vapor_gap(
    gap_m: float,
    t_hot_c: float,
    t_cold_c: float,
    *,
    x_v: float = 0.0,
    p_atm_pa: float = P_ATM_SEA_LEVEL_PA,
) -> float:
    """Note S1 Eq. S3: Ra = g·Δρ(T_gel,T_cond)·(L−H)³/(ρ_air·ν_air·α_air).

    Air is an ideal gas at constant pressure (ρ ∝ 1/T), so with ρ_air evaluated at the
    film temperature T_film = ½(T_gel + T_cond) the reference density cancels:
    Δρ/ρ_air = T_film·|T_hot − T_cold|/(T_hot·T_cold). This is the exact density-difference
    buoyancy, not the Boussinesq β·ΔT linearization with a fixed 300 K reference that this
    used to apply (which overstates peak Ra by ~7% at the baseline ΔT≈47 K desorption peak).

    ν and α are now evaluated at that same T_film rather than at fixed near-ambient
    constants, so every term in the group refers to one temperature. ``x_v`` is passed in
    rather than derived here because this signature cannot tell which surface is the
    condenser -- the caller can (see vapor_gap_h_conv_w_m2_k).
    """
    # Solver iterates can transiently pass unclamped temperatures; floor above absolute zero.
    t_hot_k = max(t_hot_c + 273.15, 1.0)
    t_cold_k = max(t_cold_c + 273.15, 1.0)
    t_film_k = 0.5 * (t_hot_k + t_cold_k)
    delta_t = max(abs(t_hot_k - t_cold_k), 1e-6)
    d_rho_over_rho = t_film_k * delta_t / (t_hot_k * t_cold_k)
    _, nu_air, alpha_air = humid_air_props(t_film_k - 273.15, x_v=x_v, p_atm_pa=p_atm_pa)
    return GRAVITY_M_S2 * d_rho_over_rho * gap_m**3 / (nu_air * alpha_air)


# Cavity height setting the ISO 15099 aspect ratio A_g,i = height/gap (Eq. 52). The
# device is modelled per m² of absorber, hence 1 m. A_g,i only enters Nu₂, which Nu₁
# dominates for A_g,i ≳ 7 at this device's Ra, so the exact value does not bind.
CAVITY_HEIGHT_M: float = _pv("Cavity height (H_cav)")


def hollands_nu(ra: float, *, tilt_deg: float) -> float:
    """Hollands et al. (1976) tilted-cavity Nu — heated from BELOW only.

    The 1708 is the Rayleigh–Bénard onset threshold, so this applies solely when the
    lower surface is the hot one. Validated to Ra·cos(tilt) ≈ 1e5.
    """
    cos_t = max(math.cos(math.radians(tilt_deg)), 1e-6)
    ra_cos = ra * cos_t
    if ra_cos <= 0.0:
        return 1.0
    sin_18t_16 = math.sin(math.radians(1.8 * tilt_deg)) ** 1.6
    f1 = max(0.0, 1.0 - 1708.0 * sin_18t_16 / ra_cos)
    f2 = max(0.0, 1.0 - 1708.0 / ra_cos)
    f3 = max(0.0, (ra_cos / 5830.0) ** (1.0 / 3.0) - 1.0)
    return 1.0 + 1.44 * f1 * f2 + f3


def iso15099_nu_vertical(ra: float, aspect_ratio: float) -> float:
    """ISO 15099 §5.3.3.4 Eqs. 48–52: Nu = max(Nu₁, Nu₂) for a vertical cavity.

    Eq. 49 is anchored at Ra < 1e6; above that its Ra^(1/3) form is the turbulent
    asymptote, so holding it is a mild extrapolation rather than a regime error.
    """
    if ra <= 0.0:
        return 1.0
    if ra > 5e4:
        nu1 = 0.0673838 * ra ** (1.0 / 3.0)  # Eq. 49
    elif ra > 1e4:
        nu1 = 0.028154 * ra**0.4134  # Eq. 50
    else:
        nu1 = 1.0 + 1.7596678e-10 * ra**2.2984755  # Eq. 51
    nu2 = 0.242 * (ra / max(aspect_ratio, 1e-6)) ** 0.272  # Eq. 52
    return max(1.0, nu1, nu2)


def vapor_gap_h_conv_w_m2_k(
    gap_m: float,
    t_gel_c: float,
    t_cond_c: float,
    *,
    tilt_deg: float = TILT_DEG,
    p_atm_pa: float = P_ATM_SEA_LEVEL_PA,
) -> float:
    """Note S1 Eq. S4: h_conv,g = Nu·k_air/(L_g − H), with Nu chosen by heating direction.

    Wilson's stack is glass/absorber/hydrogel on top with a downward-facing finned
    condenser beneath (Note S1 "top hydrogel stage"; "downward oriented fin arrays"),
    so during desorption the gel is the UPPER surface and the cavity is heated from
    above — stably stratified, with no Rayleigh–Bénard onset. That is ISO 15099
    §5.3.3.5 (cavity inclined 90°–180°, "downward facing"):

        Nu = 1 + [Nu_v − 1]·sin θ,   θ = 180° − tilt,   Nu_v from Eqs. 48–52.

    Note S1 cites Hollands et al. (1976) here, but that is a heated-from-BELOW
    correlation — its 1708 onset threshold does not exist with the hot plate on top,
    so it overstates Nu (4.79 vs 2.65 at the baseline desorption peak).

    Hollands survives only for the inverted iterate (condenser hotter than gel), and
    that is a solver artefact, not an operating state: at converged points the gel is
    always the hotter surface during desorption, while absorption is isothermal
    (T_gel := T_amb, condenser not tracked at all — the gel stage is physically
    detached from the device). solve_m_des_and_thermal nevertheless probes inverted
    (T_gel, T_cond) pairs on ~16% of calls (median 69 K inverted, max ~346 K), so the
    branch must return something; a truly bottom-heated layer IS destabilizing, and
    handing the root finder the stably-stratified form there would invert the physics.
    """
    if gap_m <= 0.0:
        return 0.0
    # This gap's own film temperature and composition: humid, because the condenser is
    # evaporating/condensing into it. The glazing gaps above run hotter and dry, and are
    # evaluated separately in _residuals -- one global k_air cannot serve both.
    x_v = vapor_gap_water_mole_fraction(t_cond_c, p_atm_pa=p_atm_pa)
    k_air = humid_air_props(0.5 * (t_gel_c + t_cond_c), x_v=x_v, p_atm_pa=p_atm_pa)[0]
    ra = rayleigh_vapor_gap(gap_m, t_gel_c, t_cond_c, x_v=x_v, p_atm_pa=p_atm_pa)
    if t_cond_c > t_gel_c:
        nu = hollands_nu(ra, tilt_deg=tilt_deg)  # hot surface below: destabilizing
    else:
        nu_v = iso15099_nu_vertical(ra, CAVITY_HEIGHT_M / gap_m)
        nu = 1.0 + (nu_v - 1.0) * math.sin(math.radians(180.0 - tilt_deg))
    return nu * k_air / gap_m


def wind_to_h_amb_w_m2_k(wind_speed_m_s: float, *, base: float = H_AMB_W_M2_K) -> float:
    """Map 10 m wind speed to external convection coefficient (paper: ~10 at 0.5 m/s)."""
    w = max(0.0, float(wind_speed_m_s))
    return base * (0.5 + w) / 1.0


def condenser_h_conv_w_m2_k(
    h_amb: float,
    *,
    fin_area_ratio: float = FIN_AREA_RATIO,
    fin_thickness_m: float | None = None,
    fin_height_m: float | None = None,
) -> float:
    """Condenser-side convection coefficient referenced to the base plate area.

    Wilson assumes an ideally efficient fin, i.e. ``A_r * h_amb``, which is what
    both fin geometry arguments being ``None`` (the default) reproduces. Complex
    mode (B3) supplies the geometry and derates the added area by the straight-fin
    efficiency, so fin area stops paying off linearly forever.
    """
    if fin_thickness_m is None or fin_height_m is None:
        return fin_area_ratio * h_amb
    from solar_lumped.complex_model import fin_efficiency

    eta_f = fin_efficiency(
        h_amb, fin_thickness_m=fin_thickness_m, fin_height_m=fin_height_m
    )
    # Only the finned area (A_r - 1) is derated; the exposed base plate is not.
    return h_amb * (1.0 + eta_f * max(fin_area_ratio - 1.0, 0.0))


# --- Salt catalog and PAM-LiCl water-activity models for Wilson Eq. 5 ---

C_W_MAX_MOL_M3: float = _pv("Gel water concentration numerical backstop, upper")
C_W_MIN_MOL_M3: float = _pv("Gel water concentration numerical backstop, lower")

# The one dilution cap. a_w -> 1 is infinite dilution: c_w diverges and the a_w -> f_b
# inversion goes ill-conditioned (da_w/df_b -> 0), so an upper bound is structural, not
# cosmetic. It used to be expressed three inconsistent ways -- per-salt rh_max, an
# rh>=0.99 short-circuit, and C_W_MAX -- of which the tightest (C_W_MAX, a_w~0.948) was
# the one that actually bound. Now one number gates the isotherm, clamps the equilibrium
# inversion, and sets each salt's c_w ceiling through dilution_ceiling_c_w.
DILUTION_CAP_RH: float = _pv("Dilution cap (maximum water activity)")


@dataclass(frozen=True, slots=True)
class SaltProperties:
    name: str
    formula_weight_g_mol: float
    ions_per_formula: int
    price_usd_per_kg: float
    h_des_j_per_kg: float
    rho_solution_kg_m3: float
    default_sl: float
    rh_min: float
    rh_max: float
    # Moles of water per mole of salt in the lowest hydrate that is stable at this
    # model's desorption temperatures (gel runs ~60-120 C). This is the physical floor
    # on gel water: below it the remaining water is crystal-bound, and removing it
    # costs far more than h_fg, which Eq. 5 does not model. See hydrate_floor_c_w.
    hydrate_h2o_per_formula: float


@lru_cache(maxsize=1)
def _load_salt_catalog() -> dict[str, SaltProperties]:
    """Per-salt properties from the Salts sheet of docs/parameters.xlsx.

    Formerly two CSVs under data/materials; salt_heat_of_desorption.csv duplicated the
    catalog's h_des exactly (verified equal before the collapse) and contributed only
    its provenance notes, which now live in the sheet's Source column."""
    out: dict[str, SaltProperties] = {}
    for name, row in SALTS.items():
        out[name] = SaltProperties(
            name=name,
            formula_weight_g_mol=float(row["formula_weight_g_mol"]),
            ions_per_formula=int(row["ions_per_formula"]),
            price_usd_per_kg=float(row["price_usd_per_kg"]),
            h_des_j_per_kg=float(row["h_des_j_per_kg"]),
            rho_solution_kg_m3=float(row["rho_solution_kg_m3"]),
            default_sl=float(row["default_sl"]),
            # Both bounds are derived, not tabulated per salt: rh_min from solubility
            # (deliquescence_rh), rh_max from the one dilution cap shared by every salt.
            rh_min=deliquescence_rh(name),
            rh_max=DILUTION_CAP_RH,
            hydrate_h2o_per_formula=float(row["hydrate_h2o_per_formula"]),
        )
    return out


def get_salt(name: str) -> SaltProperties:
    catalog = _load_salt_catalog()
    if name not in catalog:
        raise KeyError(f"Unknown salt {name!r}; available: {sorted(catalog)}")
    return catalog[name]


def get_salt_price_usd_per_kg(name: str) -> float:
    return get_salt(name).price_usd_per_kg


# Shared with the JAX backend, which reads the same two rows -- these used to be a
# hardcoded tuple here and separate constants there, i.e. one clamp with two sources.
TEMPERATURE_CLAMP_C: tuple[float, float] = (
    _pv("Temperature clamp lower bound"),
    _pv("Temperature clamp upper bound"),
)


def clamp_temperature_c(temperature_c: float) -> float:
    lo, hi = TEMPERATURE_CLAMP_C
    if not math.isfinite(temperature_c):
        return 25.0
    return max(lo, min(hi, float(temperature_c)))


def saturation_vapor_pressure_pa(temperature_c: float) -> float:
    """Pure-water vapour pressure (Pa): Conde (2004) Saul–Wagner, Tetens fallback."""
    t = clamp_temperature_c(temperature_c)
    p = water_vapor_pressure_pa(t)
    if math.isfinite(p):
        return p
    return 1000.0 * 0.61078 * math.exp(17.27 * t / (t + 237.3))


def salt_molarity_from_composite(
    salt_loading: float,
    hydrogel_density_kg_m3: float,
    formula_weight_g_mol: float,
) -> float:
    """Fixed salt molar concentration c_s (mol/m³ gel) in desorbed composite."""
    f_salt = salt_loading / (1.0 + salt_loading)
    mass_salt_kg_m3 = hydrogel_density_kg_m3 * f_salt
    return mass_salt_kg_m3 / (formula_weight_g_mol / 1000.0)


# Díaz-Marín Methods — one-pot batch in 50 mL, poured into 60 mm petri dishes.
_CHAMBER_DISH_DIAMETER_M: float = _pv("Chamber dish diameter")
_SYNTHESIS_BATCH_ML: float = _pv("Synthesis batch volume")
_PAM_LICL_STANDARD_POUR_ML: float = _pv("PAM-LiCl standard pour volume")
_PAM_LICL_2GG_CHAMBER_POUR_ML: float = _pv("PAM-LiCl 2 g/g chamber pour volume")
_LICL_BATCH_G_BY_SL: dict[int, float] = {
    1: 4.18,
    2: 8.36,
    4: 16.72,
    8: 33.44,
}
# Table S3 reference for anchoring synthesis c_s to the 4 g/g DVS dry-basis density.
_CHAMBER_CS_CALIB_SL: float = _pv("Chamber c_s calibration salt loading")
_CHAMBER_CS_CALIB_H0_MM: float = _pv("Chamber c_s calibration hydrogel thickness")
_CHAMBER_CS_CALIB_POUR_ML: float = _PAM_LICL_STANDARD_POUR_ML


def chamber_pour_volume_ml(
    salt_loading: float,
    *,
    pam_licl_chamber: bool = True,
) -> float:
    """Solution pour volume (mL) for environmental-chamber kinetics samples."""
    if pam_licl_chamber and int(round(salt_loading)) == 2:
        return _PAM_LICL_2GG_CHAMBER_POUR_ML
    return _PAM_LICL_STANDARD_POUR_ML


def _chamber_c_s_from_pour_inventory(
    salt_loading: float,
    h0_mm: float,
    *,
    pour_ml: float,
    formula_weight_g_mol: float,
) -> float:
    """c_s [mol/m³ gel] from LiCl mass in pour / gel volume at measured H₀."""
    sl_key = int(round(salt_loading))
    if sl_key not in _LICL_BATCH_G_BY_SL:
        raise ValueError(f"unsupported PAM-LiCl salt loading for synthesis c_s: {sl_key}")
    salt_in_pour_kg = _LICL_BATCH_G_BY_SL[sl_key] * (pour_ml / _SYNTHESIS_BATCH_ML) / 1000.0
    moles = salt_in_pour_kg / (formula_weight_g_mol / 1000.0)
    area_m2 = math.pi * (_CHAMBER_DISH_DIAMETER_M / 2.0) ** 2
    vol_m3 = area_m2 * max(h0_mm * 1e-3, 1e-6)
    return moles / vol_m3


def chamber_c_s_from_synthesis(
    salt_loading: float,
    h0_mm: float,
    *,
    formula_weight_g_mol: float = 42.394,
    pour_ml: float | None = None,
    calibrate_to_dvs: bool = True,
) -> float:
    """Fixed c_s for Eq. 8: poured LiCl moles spread over footprint × H₀ (SI Note S9),
    with the 4 g/g reference scaled to DRY_COMPOSITE_DENSITY for DVS consistency."""
    pour = pour_ml if pour_ml is not None else chamber_pour_volume_ml(salt_loading)
    cs_synth = _chamber_c_s_from_pour_inventory(
        salt_loading,
        h0_mm,
        pour_ml=pour,
        formula_weight_g_mol=formula_weight_g_mol,
    )
    if not calibrate_to_dvs:
        return cs_synth

    cs_dvs_ref = salt_molarity_from_composite(
        _CHAMBER_CS_CALIB_SL,
        DRY_COMPOSITE_DENSITY_KG_M3,
        formula_weight_g_mol,
    )
    cs_synth_ref = _chamber_c_s_from_pour_inventory(
        _CHAMBER_CS_CALIB_SL,
        _CHAMBER_CS_CALIB_H0_MM,
        pour_ml=_CHAMBER_CS_CALIB_POUR_ML,
        formula_weight_g_mol=formula_weight_g_mol,
    )
    return cs_dvs_ref * (cs_synth / cs_synth_ref)


def chamber_c_s_with_constant_density(
    salt_loading: float,
    h0_mm: float,
    *,
    formula_weight_g_mol: float = 42.394,
    pour_ml: float | None = None,
) -> float:
    """``c_s`` for Eq. 8 at SI Note S7 constant 20 % RH solution density. c_s stays on the
    4 g/g calibration thickness while g/H₀ uses measured H₀ -- matches panel 5d."""
    cs = chamber_c_s_from_synthesis(
        salt_loading,
        h0_mm,
        formula_weight_g_mol=formula_weight_g_mol,
        pour_ml=pour_ml,
    )
    if int(round(salt_loading)) != 2:
        return cs
    if abs(h0_mm - _CHAMBER_CS_CALIB_H0_MM) < 1e-6:
        return cs
    cs_ref = chamber_c_s_from_synthesis(
        _CHAMBER_CS_CALIB_SL,
        _CHAMBER_CS_CALIB_H0_MM,
        formula_weight_g_mol=formula_weight_g_mol,
    )
    return cs_ref * (_CHAMBER_CS_CALIB_H0_MM / h0_mm)


@lru_cache(maxsize=1)
def _load_pam_licl_dvs_isotherm() -> tuple[np.ndarray, np.ndarray]:
    """Note S2 DVS isotherm: RH (%), gravimetric uptake (g water / g dry composite)."""
    path = Path(__file__).resolve().parent / "data" / "materials" / "PAM-LiCL_isotherm.csv"
    return load_two_column_csv(path)


def pam_licl_uptake_g_g_at_rh(rh_fraction: float) -> float:
    """Forward DVS isotherm: equilibrium uptake (g/g) at relative humidity."""
    rh_pct, uptake = _load_pam_licl_dvs_isotherm()
    r = max(0.0, min(100.0, float(rh_fraction) * 100.0))
    return float(np.interp(r, rh_pct, uptake))


# Dry-basis composite density = rho_composite(20% RH) / (1 + uptake(20% RH)); the wet
# density would over-count dry sorbent mass by (1 + u20) ≈ 2.26x.
DRY_COMPOSITE_DENSITY_KG_M3: float = RHO_COMPOSITE_KG_M3 / (
    1.0 + pam_licl_uptake_g_g_at_rh(0.20)
)


def pam_licl_dry_mass_kg_m2(
    h0_ref_m: float,
    *,
    dry_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3,
) -> float:
    """Dry PAM-LiCl composite mass per m² at reference thickness H₀ (DVS basis)."""
    return dry_density_kg_m3 * h0_ref_m


def pam_licl_gravimetric_uptake_g_g(
    c_w: float,
    h_m: float,
    *,
    h0_ref_m: float,
    dry_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3,
    c_s_mol_m3: float | None = None,
    formula_weight_g_mol: float | None = None,
    salt_loading: float | None = None,
    salt_weight_factor: float = 1.0,
) -> float:
    """Gravimetric moisture content m_w/m_dry (g/g) per footprint, referenced to the fixed
    fabrication thickness H₀ (Note S1), not swollen H(t) -- ``h_m`` is unused. With composite
    state, dry mass uses ``formula_weight_g_mol * salt_weight_factor``; else the DVS density."""
    del h_m
    if (
        c_s_mol_m3 is not None
        and formula_weight_g_mol is not None
        and salt_loading is not None
    ):
        mw_eff = formula_weight_g_mol * salt_weight_factor
        mass_salt = max(0.0, c_s_mol_m3) * h0_ref_m * mw_eff / 1000.0
        mass_polymer = mass_salt / max(salt_loading, 1e-9)
        m_dry = mass_salt + mass_polymer
    else:
        m_dry = pam_licl_dry_mass_kg_m2(h0_ref_m, dry_density_kg_m3=dry_density_kg_m3)
    if m_dry <= 0.0:
        return 0.0
    mass_water = max(0.0, c_w) * h0_ref_m * WATER_MOLAR_MASS_KG_MOL
    return mass_water / m_dry


def licl_water_activity_at_brine_fraction(
    brine_salt_fraction: float,
    temperature_c: float,
) -> float:
    """LiCl brine water activity — Conde (2004) Table 3, Zeng & Zhou BET above 100 C."""
    t_corr = min(float(temperature_c), BET_T_MAX_C)
    return water_activity_licl(brine_salt_fraction, t_corr)


def licl_equilibrium_brine_salt_fraction(
    relative_humidity: float,
    temperature_c: float = 25.0,
) -> float:
    """Invert the LiCl isotherm: brine salt fraction at equilibrium with RH."""
    t_corr = min(float(temperature_c), BET_T_MAX_C)
    return equilibrium_salt_mass_fraction_licl(relative_humidity, t_corr)


def water_activity_from_c_w(
    c_w: float,
    *,
    c_s: float,
    ions_per_formula: int,
    temperature_c: float = 25.0,
    salt_name: str = "LiCl",
    formula_weight_g_mol: float = 42.394,
    salt_loading: float = SALT_LOADING_DEFAULT,
    h_m: float | None = None,
    h0_ref_m: float | None = None,
    salt_weight_factor: float = 1.0,
    blend_weights: tuple[float, ...] | None = None,
) -> float:
    """Brine a_w,s in Eq. 5 (Wilson Device); activity of water in the salt solution.

    With ``blend_weights`` (complex mode, B8) the brine is a ZSR mixture and a_w
    comes from inverting the mixing rule at the current salt mass fraction,
    instead of a single salt's closed-form isotherm.
    """
    del ions_per_formula, salt_loading, h_m  # h_m unused; inventory is on H₀ basis
    if c_w <= 0.0 or c_s <= 0.0:
        return 1.0
    h_ref = h0_ref_m if h0_ref_m is not None else H0_M
    mw_eff = formula_weight_g_mol * salt_weight_factor

    if blend_weights is not None:
        from solar_lumped.complex_model import zsr_water_activity_at_brine_fraction

        mass_water = max(0.0, c_w) * h_ref * WATER_MOLAR_MASS_KG_MOL
        mass_salt = c_s * h_ref * mw_eff / 1000.0
        total = mass_salt + mass_water
        if total <= 0.0:
            return float("nan")
        aw = zsr_water_activity_at_brine_fraction(blend_weights, mass_salt / total, temperature_c)
        return aw if math.isfinite(aw) else float("nan")

    if salt_name == "LiCl":
        # Brine salt mass fraction m_s / (m_s + m_w) -- LiCl solution a_w,s (Eq. 5).
        salt_mol_m2 = c_s * h_ref
        mass_salt = salt_mol_m2 * mw_eff / 1000.0
        mass_water = max(0.0, c_w) * h_ref * WATER_MOLAR_MASS_KG_MOL
        total = mass_salt + mass_water
        f_b = 1.0 if total <= 0.0 else mass_salt / total
        aw = licl_water_activity_at_brine_fraction(f_b, temperature_c)
        return aw if math.isfinite(aw) else float("nan")

    # Brine salt mass fraction from gel water/salt molarities (mol/m³ gel).
    if not all(map(math.isfinite, (c_w, c_s, mw_eff))) or c_w < 0.0 or c_s < 0.0:
        return float("nan")
    mass_water = c_w * 18.015 / 1000.0
    mass_salt = c_s * mw_eff / 1000.0
    total = mass_water + mass_salt
    if total <= 0.0:
        return float("nan")
    f_b = float(mass_salt / total)
    aw = water_activity_at_brine_fraction(salt_name, f_b, temperature_c)
    return aw if math.isfinite(aw) else float("nan")


def equilibrium_c_w_at_rh(
    rh: float,
    *,
    c_s: float,
    ions_per_formula: int,
    temperature_c: float = 25.0,
    salt_name: str = "LiCl",
    formula_weight_g_mol: float = 42.394,
    salt_loading: float = SALT_LOADING_DEFAULT,
    h_m: float | None = None,
    h0_ref_m: float | None = None,
    blend_weights: tuple[float, ...] | None = None,
) -> float:
    """Invert a_w(RH) to c_w at reference hydrogel thickness H₀."""
    del ions_per_formula
    if rh <= 0.0:
        return 0.0
    # Clamp to the dilution cap rather than short-circuiting to a constant: above the
    # cap the isotherm has no finite answer, and the capped equilibrium *is* the ceiling
    # (that is exactly what dilution_ceiling_c_w asks for), so returning it keeps the
    # bound self-consistent instead of pinning to an unrelated C_W_MAX.
    rh = min(float(rh), DILUTION_CAP_RH)

    h_ref = h0_ref_m if h0_ref_m is not None else H0_M
    del h_m  # inventory referenced to H₀ (see pam_licl_gravimetric_uptake_g_g)

    if blend_weights is not None:
        # ZSR is posed a_w -> composition, so the forward direction is the direct
        # evaluation here -- no root solve needed on this path.
        from solar_lumped.complex_model import zsr_brine_state

        f_b, _ions, _mw = zsr_brine_state(blend_weights, rh, temperature_c)
        if not math.isfinite(f_b):
            return C_W_MIN_MOL_M3
    elif salt_name == "LiCl":
        f_b = licl_equilibrium_brine_salt_fraction(rh, temperature_c)
    else:
        f_b = equilibrate_salt_mf(salt_name, rh, temperature_c)
        if not math.isfinite(f_b):
            return C_W_MIN_MOL_M3

    salt_mol_m2 = c_s * h_ref
    mass_salt = salt_mol_m2 * formula_weight_g_mol / 1000.0
    if f_b <= 0.0:
        return C_W_MAX_MOL_M3
    mass_water = mass_salt * (1.0 - f_b) / f_b
    if mass_water <= 0.0:
        return C_W_MIN_MOL_M3
    # No C_W_MAX clamp: rh is capped above, so c_w cannot exceed the dilution ceiling.
    return mass_water / (h_ref * WATER_MOLAR_MASS_KG_MOL)


# Methods: hydrogel cast at equilibrium with ~20% RH ambient.
FABRICATION_EQUILIBRIUM_RH: float = _pv("Fabrication equilibrium RH")


def equilibrium_c_w_from_dvs_at_rh(
    rh: float,
    *,
    h_m: float,
    h0_ref_m: float,
    dry_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3,
) -> float:
    """Paper Note S2: DVS isotherm sets sorbent equilibrium uptake at ambient RH."""
    del h_m  # inventory referenced to H₀ (see pam_licl_gravimetric_uptake_g_g)
    if rh <= 0.0:
        return C_W_MIN_MOL_M3
    u = pam_licl_uptake_g_g_at_rh(rh)
    m_dry = dry_density_kg_m3 * h0_ref_m
    mass_water_kg_m2 = u * m_dry
    c_w = mass_water_kg_m2 / (h0_ref_m * WATER_MOLAR_MASS_KG_MOL)
    return max(C_W_MIN_MOL_M3, min(C_W_MAX_MOL_M3, c_w))


def desorption_water_activity(
    condenser_temperature_c: float,
    gel_temperature_c: float,
) -> float:
    """Effective desorption water activity at sealed condenser / sun-heated gel equilibrium."""
    p_sat_cond = saturation_vapor_pressure_pa(condenser_temperature_c)
    p_sat_gel = saturation_vapor_pressure_pa(gel_temperature_c)
    if p_sat_gel <= 0.0 or not math.isfinite(p_sat_gel) or not math.isfinite(p_sat_cond):
        return float("nan")
    t_cond_k = condenser_temperature_c + 273.15
    t_gel_k = gel_temperature_c + 273.15
    if t_cond_k <= 0.0 or t_gel_k <= 0.0:
        return float("nan")
    return p_sat_cond * t_gel_k / (p_sat_gel * t_cond_k)


def fabrication_c_w_initial(
    *,
    salt_name: str,
    salt_loading: float,
    hydrogel_thickness_m: float,
    hydrogel_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3,
    formula_weight_g_mol: float | None = None,
    blend_weights: tuple[float, ...] | None = None,
    use_dvs_cap: bool = True,
) -> float:
    """Initial gel water state after fabrication at ~20% RH ambient.

    A ZSR blend (complex mode, B8) never takes the PAM-LiCl DVS branch: that
    isotherm was measured on the LiCl composite and says nothing about a mixed
    brine. It also casts at the blend's own clamped fabrication RH, since CaCl2 and
    MgCl2 have no brine at 20% RH and would otherwise start the gel bone dry --
    which silently zeroed a blend's whole year.
    """
    h0 = hydrogel_thickness_m
    if blend_weights is not None:
        from solar_lumped.complex_model import (
            clamp_reference_rh,
            zsr_effective_formula_weight_g_mol,
        )

        fw = (
            formula_weight_g_mol
            if formula_weight_g_mol is not None
            else zsr_effective_formula_weight_g_mol(
                blend_weights, reference_rh=FABRICATION_EQUILIBRIUM_RH
            )
        )
        return equilibrium_c_w_at_rh(
            clamp_reference_rh(blend_weights, FABRICATION_EQUILIBRIUM_RH),
            c_s=salt_molarity_from_composite(salt_loading, hydrogel_density_kg_m3, fw),
            ions_per_formula=2,  # unused on this path; ZSR carries effective ions
            salt_name=salt_name,
            formula_weight_g_mol=fw,
            salt_loading=salt_loading,
            h_m=h0,
            h0_ref_m=h0,
            blend_weights=blend_weights,
        )
    if salt_name == "LiCl" and use_dvs_cap:
        return equilibrium_c_w_from_dvs_at_rh(
            FABRICATION_EQUILIBRIUM_RH,
            h_m=h0,
            h0_ref_m=h0,
            dry_density_kg_m3=hydrogel_density_kg_m3,
        )
    s = get_salt(salt_name)
    fw = formula_weight_g_mol if formula_weight_g_mol is not None else s.formula_weight_g_mol
    c_s = salt_molarity_from_composite(
        salt_loading,
        hydrogel_density_kg_m3,
        fw,
    )
    # Same clamp as the blend branch, for the same reason: CaCl2 (DRH 0.35) and
    # MgCl2 (0.33) have no brine at Wilson's 20% RH casting condition, and leaving
    # it unclamped starts the gel dry and silently zeroes the year.
    return equilibrium_c_w_at_rh(
        min(max(FABRICATION_EQUILIBRIUM_RH, s.rh_min), s.rh_max),
        c_s=c_s,
        ions_per_formula=s.ions_per_formula,
        salt_name=salt_name,
        formula_weight_g_mol=fw,
        salt_loading=salt_loading,
        h_m=h0,
        h0_ref_m=h0,
    )


# --- Equilibrium brine isotherms for NaCl, LiCl, CaCl2, and MgCl2 ---


def _aw_polynomial(salt_fraction: float, coeffs: tuple[float, ...]) -> float:
    """a_w(ξ) = Σ coeffs[k]·ξ^k (coeffs[0] first, i.e. increasing powers)."""
    if not (0.0 <= salt_fraction < 1.0) or not math.isfinite(salt_fraction):
        return float("nan")
    a_w = 0.0
    for k, coeff in enumerate(coeffs):
        a_w += coeff * (salt_fraction**k)
    return float(a_w)


# a_w(ξ) polynomial coefficients (increasing powers of ξ): forward isotherm, and inverted
# via find_root_bracketed for the equilibrium brine fraction.
_NACL_AW_COEFFS: tuple[float, ...] = (0.9998, -0.5597, -0.332, -5.545, 5.863)
_MGCL2_AW_COEFFS: tuple[float, ...] = (1.16231287, -4.86704441, 38.21982328, -153.67496570, 186.32487108)


def mf_NaCl(relative_humidity: float) -> float:
    """Equilibrium brine salt fraction for NaCl at 25°C."""
    if not (0.0 < relative_humidity < 1.0):
        return float("nan")
    return find_root_bracketed(
        lambda xi: relative_humidity - _aw_polynomial(xi, _NACL_AW_COEFFS),
        0.0116,
        0.264,
    )


def mf_LiCl(relative_humidity: float, temperature_c: float = 25.0) -> float:
    """Equilibrium brine salt fraction for LiCl (Conde 2004 + Zeng & Zhou above 100 C)."""
    if not (0.0 < relative_humidity < 1.0) or temperature_c > BET_T_MAX_C:
        return float("nan")
    return equilibrium_salt_mass_fraction_licl(relative_humidity, temperature_c)


def mf_LiBr(relative_humidity: float, temperature_c: float = 25.0) -> float:
    """Equilibrium brine salt fraction for LiBr (Patek & Klomfar 2006).

    Bracketed rather than closed form -- unlike the BET this replaced, Eq. (1) has no
    analytic inverse. a_w is strictly decreasing in composition at every temperature
    across 0-210 C (checked on a 200-point sweep), so the bracket is always valid.
    """
    if not (0.0 < relative_humidity < 1.0) or temperature_c > LIBR_T_MAX_C:
        return float("nan")
    hi = min(LIBR_W_MAX, saturation_brine_salt_fraction("LiBr", temperature_c))
    if relative_humidity >= patek_libr_water_activity(_BRACKET_LO, temperature_c):
        return _BRACKET_LO
    return find_root_bracketed(
        lambda xi: relative_humidity - patek_libr_water_activity(xi, temperature_c),
        _BRACKET_LO,
        hi,
    )


def mf_CaCl2(relative_humidity: float, temperature_c: float = 25.0) -> float:
    """Equilibrium brine salt fraction for CaCl2 (Conde 2004)."""
    if not (0.0 < relative_humidity < 1.0) or temperature_c > CONDE_T_MAX_C:
        return float("nan")
    return equilibrium_salt_mass_fraction_cacl2(relative_humidity, temperature_c)


def mf_MgCl2(relative_humidity: float) -> float:
    """Equilibrium brine salt fraction for MgCl2 (polynomial fit)."""
    if not (0.0 < relative_humidity < 1.0):
        return float("nan")
    return find_root_bracketed(
        lambda xi: relative_humidity - _aw_polynomial(xi, _MGCL2_AW_COEFFS),
        0.01,
        0.75,
        scan=True,
        n_intervals=19,
    )


_isotherm_by_salt: dict[str, Callable[[float, float], float]] = {
    "NaCl": lambda rh, t: mf_NaCl(rh),
    "LiCl": lambda rh, t: mf_LiCl(rh, t),
    "LiBr": lambda rh, t: mf_LiBr(rh, t),
    "CaCl2": lambda rh, t: mf_CaCl2(rh, t),
    "MgCl2": lambda rh, t: mf_MgCl2(rh),
}


def equilibrate_salt_mf(
    salt_name: str,
    relative_humidity: float,
    temperature_c: float = 25.0,
) -> float:
    """Return equilibrium brine salt mass fraction, or nan if outside the salt's RH range."""
    rec = get_salt(salt_name)
    if rec.name not in _isotherm_by_salt:
        return float("nan")
    # Gate on the deliquescence point *at this temperature*, not the 25 C catalog value.
    # Saturation concentration rises with temperature faster than activity does, so DRH
    # falls (LiCl 0.106 at 25 C -> 0.096 at 80 C): a fixed 25 C gate would reject the
    # driest states a hot gel legitimately reaches.
    if not (deliquescence_rh(rec.name, temperature_c) <= relative_humidity <= rec.rh_max):
        return float("nan")
    return float(_isotherm_by_salt[rec.name](relative_humidity, temperature_c))


def _water_activity_at_brine_fraction_by_name(
    salt_name: str,
    brine_salt_fraction: float,
    temperature_c: float,
) -> float:
    """Forward isotherm dispatch keyed on the salt name only.

    Deliberately does not call ``get_salt``: the catalog loader derives each salt's
    deliquescence RH through here, and looking the record up would re-enter the loader.
    """
    f = float(brine_salt_fraction)
    if not (0.0 <= f < 1.0) or not math.isfinite(f):
        return float("nan")
    if salt_name == "NaCl":
        return _aw_polynomial(f, _NACL_AW_COEFFS)
    if salt_name == "MgCl2":
        return _aw_polynomial(f, _MGCL2_AW_COEFFS)
    if salt_name == "LiCl":
        if temperature_c > BET_T_MAX_C:
            return float("nan")
        return water_activity_licl(f, temperature_c)
    if salt_name == "LiBr":
        if temperature_c > LIBR_T_MAX_C:
            return float("nan")
        return patek_libr_water_activity(
            min(f, saturation_brine_salt_fraction("LiBr", temperature_c)), temperature_c
        )
    if salt_name == "CaCl2":
        if temperature_c > 100.0:
            return float("nan")
        return vapor_pressure_ratio(f, temperature_c, CACL2_VAPOR_PRESSURE)
    return float("nan")


def isotherm_t_max_c(salt_name: str) -> float:
    """Highest temperature this salt's forward isotherm is defined at.

    One source of truth for every caller that has to clamp before evaluating a_w --
    the ZSR bucketing in complex_model, the isosteric slope below, and the NaN gates in
    _water_activity_at_brine_fraction_by_name. It used to be a single module-level
    CONDE_T_MAX_C everywhere, which capped LiCl at its weakest sibling's limit.
    """
    if salt_name == "LiCl":
        return BET_T_MAX_C
    if salt_name == "LiBr":
        return LIBR_T_MAX_C
    if salt_name == "CaCl2":
        return CONDE_T_MAX_C
    # NaCl and MgCl2 fit a_w(xi) with temperature-independent polynomials, so they have
    # no ceiling to hit -- and no temperature dependence to be right about either.
    return math.inf


@lru_cache(maxsize=1024)
def deliquescence_rh(salt_name: str, temperature_c: float = 25.0) -> float:
    """Deliquescence RH: the brine water activity at saturation, at this temperature.

    Derived, not tabulated. Below it no brine exists at equilibrium, so the salt stays
    solid and cannot take up water -- which also makes it the floor a_w pins to once
    the excess salt has precipitated.

    Temperature matters: solubility rises with temperature, so the saturation *mass
    fraction* climbs faster than the activity at fixed composition does, and DRH falls
    (LiCl 0.150 at 0 C -> 0.106 at 25 C -> 0.096 at 80 C -> 0.091 at 100 C). A gel
    desorbing at 80 C therefore pins ~10% *lower* than the 25 C value, not higher.
    ``SaltProperties.rh_min`` keeps the 25 C default as the catalog reference.

    NaCl and MgCl2 use temperature-independent polynomial isotherms, so their
    deliquescence point does not move -- a limit of those correlations, not of this.
    """
    return _water_activity_at_brine_fraction_by_name(
        salt_name,
        saturation_brine_salt_fraction(salt_name, float(temperature_c)),
        float(temperature_c),
    )


def water_activity_at_brine_fraction(
    salt_name: str,
    brine_salt_fraction: float,
    temperature_c: float = 25.0,
) -> float:
    """Forward isotherm: brine water activity at salt mass fraction and temperature."""
    return _water_activity_at_brine_fraction_by_name(
        get_salt(salt_name).name, brine_salt_fraction, temperature_c
    )


# --- Wilson et al. 2025 Eqs. 1, 3, 4 — steady absorber, glass, and gel temperatures ---


@dataclass(frozen=True, slots=True)
class ThermalState:
    t_gel_c: float
    t_abs_c: float
    t_glass_c: float
    h_conv_g: float
    m_des_kg_s_m2: float
    # Complex mode (B2) only: outer pane temperature of a 2-pane stack. None for
    # the single-pane / uncovered system, where t_glass_c is the whole story.
    t_glass_outer_c: float | None = None


@dataclass(frozen=True, slots=True)
class SystemThermalParams:
    insulation_gap_m: float = L_INS_M
    vapor_gap_m: float = L_G_M
    eps_abs: float = EPS_ABS
    tau_glass: float = TAU_GLASS
    eps_gel: float = EPS_GEL
    eps_al: float = EPS_AL
    # Real absorber/glass IR emissivities for the modified Eqs. 3/4 radiative terms.
    # Case 2 (selective surface) is the default; pass both as 1.0 (or None) for
    # Case 1's original Wilson blackbody/cavity approximation -- see _residuals.
    eps_abs_ir: float | None = EPS_ABS_IR_CASE2
    eps_glass_ir: float | None = EPS_GLASS_IR_CASE2
    tilt_deg: float = TILT_DEG
    h_des_j_per_kg: float = H_DES_J_PER_KG
    has_glass: bool = True
    # Complex mode (B2). n_glazing_panes=1 (default) is Wilson's single cover and
    # leaves _residuals a 3-unknown solve; 2 adds an outer pane as a real 4th
    # unknown; 0 is the uncovered system (equivalent to has_glass=False).
    n_glazing_panes: int = 1
    # An evacuated inter-pane gap suppresses conduction but not radiation.
    evacuated_gap: bool = False
    # Salt whose isotherm supplies an isosteric h_des(xi, T); None keeps the constant
    # h_des_j_per_kg above. Only ISOSTERIC_H_DES_SALTS qualify.
    h_des_salt_name: str | None = None
    # Site ambient pressure, from SystemConfig.site_elevation_m. Sets every air property
    # in the gaps and the h_amb density factor; the sea-level default reproduces the
    # previous behaviour exactly.
    p_atm_pa: float = P_ATM_SEA_LEVEL_PA


# --- Isosteric (Clausius-Clapeyron) heat of sorption ---

# Only the Conde-backed salts qualify: the isosteric slope needs a_w with real
# temperature dependence, and NaCl/MgCl2 use temperature-independent polynomial
# isotherms, so d(ln a_w)/d(1/T) is identically zero there and this would collapse to
# plain h_fg -- strictly worse than their tabulated constants.
ISOSTERIC_H_DES_SALTS: frozenset[str] = frozenset({"LiCl", "LiBr", "CaCl2"})

# Each salt caps where its own isotherm stops. LiCl and LiBr now carry the Zeng & Zhou
# BET to 155.5 C, so their slope is defined across the whole gel range and no longer
# collapses onto a clamped-a_w value (which read as pure h_fg above 100 C). CaCl2 has
# only Conde, so it still stops at 100 C.
_ISOSTERIC_T_MAX_C: dict[str, float] = {
    name: isotherm_t_max_c(name) for name in ISOSTERIC_H_DES_SALTS
}
_ISOSTERIC_DT_K: float = _pv("Isosteric h_des finite-difference step")


def brine_salt_fraction_from_c_w(
    c_w: float,
    *,
    c_s_mol_m3: float,
    h0_ref_m: float,
    formula_weight_g_mol: float,
    salt_weight_factor: float = 1.0,
) -> float:
    """Brine salt mass fraction xi = m_s/(m_s + m_w), fixed-H0 footprint basis."""
    mw_eff = formula_weight_g_mol * salt_weight_factor
    mass_salt = max(0.0, c_s_mol_m3) * h0_ref_m * mw_eff / 1000.0
    mass_water = max(0.0, c_w) * h0_ref_m * WATER_MOLAR_MASS_KG_MOL
    total = mass_salt + mass_water
    return 1.0 if total <= 0.0 else mass_salt / total


def isosteric_h_des_j_per_kg(
    salt_name: str, brine_salt_fraction: float, temperature_c: float
) -> float:
    """Heat of sorption from Clausius-Clapeyron at fixed composition (J per kg water).

        P(xi, T) = a_w(xi, T) * p_sat(T),      h = -R * d(ln P)/d(1/T) at fixed xi

    This is Díaz-Marín Eq. 7 (h_sorp = h_fg + h_mix) evaluated numerically rather than
    split analytically: the p_sat slope contributes h_fg and the a_w slope the salt
    binding term. Validated against their PAM-LiCl measurements -- 2834 kJ/kg at
    saturation/25 C against a reported 2800-2900, and it converges onto h_fg at high
    RH where the brine is dilute, which is the limit they describe.

    Returns NaN when the isotherm is out of domain at either probe point. Callers must
    fall back to the tabulated constant rather than propagate it: a NaN reaching the
    energy balance gets swallowed downstream as a zero driving force.
    """
    if salt_name not in ISOSTERIC_H_DES_SALTS:
        return float("nan")
    t_c = min(float(temperature_c), _ISOSTERIC_T_MAX_C[salt_name] - _ISOSTERIC_DT_K)
    # ponytail: past saturation the excess salt is solid and a_w pins, so the slope
    # pins with it. The real answer there adds a hydrate-formation enthalpy -- the
    # LiCl/LiBr gap Díaz-Marín attributes to exactly that -- which needs per-salt
    # hydrate data the catalog does not carry beyond hydrate_h2o_per_formula.
    # Clamp against the COLD probe's saturation, not the centre's. xi_sat rises with
    # temperature, and the isotherm re-clips internally at each probe point, so a gel
    # sitting exactly at saturation would otherwise be clipped at t-dt but not at t+dt
    # -- straddling the phase boundary and reading the dissolution step as if it were
    # the brine slope (2626 vs 2834 kJ/kg at 25 C). Both probes must sit in the same
    # regime for the two-point derivative to mean anything.
    xi_max = saturation_brine_salt_fraction(salt_name, t_c - _ISOSTERIC_DT_K)
    xi = min(max(float(brine_salt_fraction), 1e-6), xi_max)
    pts = []
    for t in (t_c - _ISOSTERIC_DT_K, t_c + _ISOSTERIC_DT_K):
        aw = water_activity_at_brine_fraction(salt_name, xi, t)
        p_sat = water_vapor_pressure_pa(t)
        if not (math.isfinite(aw) and aw > 0.0 and math.isfinite(p_sat) and p_sat > 0.0):
            return float("nan")
        pts.append((1.0 / (t + 273.15), math.log(aw * p_sat)))
    (x1, y1), (x2, y2) = pts
    h = -GAS_CONSTANT_J_MOL_K * (y2 - y1) / (x2 - x1) / WATER_MOLAR_MASS_KG_MOL
    return h if math.isfinite(h) and h > 0.0 else float("nan")


def effective_h_des_j_per_kg(
    params: SystemThermalParams, t_gel_c: float, brine_salt_fraction: float | None
) -> float:
    """Isosteric h_des where the isotherm supports it, else the tabulated constant."""
    if brine_salt_fraction is None or params.h_des_salt_name is None:
        return params.h_des_j_per_kg
    h = isosteric_h_des_j_per_kg(params.h_des_salt_name, brine_salt_fraction, t_gel_c)
    return h if math.isfinite(h) else params.h_des_j_per_kg


def _residuals(
    x: np.ndarray,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    m_des_kg_s_m2: float,
    h_amb: float,
    params: SystemThermalParams,
    vapor_gap_effective_m: float,
    h_m: float,
    brine_salt_fraction: float | None = None,
) -> np.ndarray:
    t_gel, t_abs, t_glass = float(x[0]), float(x[1]), float(x[2])
    # Complex mode (B2): a second pane carries its own temperature as a 4th unknown.
    two_pane = params.has_glass and params.n_glazing_panes >= 2
    t_glass_outer = float(x[3]) if two_pane else t_glass
    u_gel = u_gel_w_m2_k(h_m)
    h_conv_g = vapor_gap_h_conv_w_m2_k(
        vapor_gap_effective_m, t_gel, t_cond_c, tilt_deg=params.tilt_deg, p_atm_pa=params.p_atm_pa
    )
    # Every h_amb below is the thinner-air one. Applied here, at the point of use, rather
    # than baked into the weather profile: this is where both the site pressure and the
    # ambient temperature that set air density are known, and doing it once covers the
    # absorber, both panes and (via simulation's condenser ODE) the fins.
    h_amb = h_amb * h_amb_density_factor(t_amb_c, p_atm_pa=params.p_atm_pa)
    eps_gc = parallel_plate_emissivity(params.eps_gel, params.eps_al)
    q_rad_gc = radiative_exchange_w_m2(t_gel, t_cond_c, emissivity=eps_gc)

    # Eq 1 gel. h_des is resolved at this iterate's t_gel, not at the warm start, so
    # the isosteric value is self-consistent with the temperature it is solving for.
    q_des = m_des_kg_s_m2 * effective_h_des_j_per_kg(params, t_gel, brine_salt_fraction)
    r1 = (
        u_gel * (t_abs - t_gel)
        - h_conv_g * (t_gel - t_cond_c)
        - q_des
        - q_rad_gc
    )

    if params.eps_abs_ir is not None and params.eps_glass_ir is not None:
        # Modified Eqs. 3/4: eps_ag from the parallel-plate formula (as eps_gel/eps_al),
        # eps_ga is the glass IR emissivity directly (it radiates to ambient, not a cavity).
        eps_ag = parallel_plate_emissivity(params.eps_abs_ir, params.eps_glass_ir)
        eps_ga = params.eps_glass_ir
    else:
        # Absorber→glass: Wilson Eq. 4 writes σ(T_abs⁴ − T_glass⁴) without an
        # explicit emissivity factor (cavity / blackbody approximation).
        eps_ag = 1.0
        # Glass→surroundings: Wilson Eq. 3 writes σ(T_glass⁴ − T_amb⁴) with no emissivity
        # factor (blackbody in the IR).
        eps_ga = 1.0

    if not params.has_glass:
        q_rad_abs_amb = radiative_exchange_w_m2(t_abs, t_amb_c, emissivity=params.eps_abs)
        r4 = (
            params.eps_abs * q_solar_w_m2
            - h_amb * (t_abs - t_amb_c)
            - q_rad_abs_amb
            - u_gel * (t_abs - t_gel)
        )
        r3 = t_glass - t_amb_c
    else:
        gap_m = params.insulation_gap_m

        def _gap_conduction_w_m2(t_hot: float, t_cold: float) -> float:
            """k_air/L·ΔT across one sealed glazing gap, k at THAT gap's film temperature.

            The two cavities are not at the same temperature: absorber-glass spans roughly
            90-110 C down to 40 C, while the pane-pane cavity above it sits far cooler, and
            k_air moves ~11% over the device's range. Dry air (x_v = 0) -- these cavities
            are sealed with no water path, unlike the vapor gap below the gel.

            An evacuated cover assembly (B2) kills gas conduction across every glazing gap
            but leaves radiation untouched -- that asymmetry is exactly why it buys
            anything, and why it only pays off with a low-eps absorber.
            """
            if gap_m <= 0.0 or params.evacuated_gap:
                return 0.0
            k_air = humid_air_props(0.5 * (t_hot + t_cold), p_atm_pa=params.p_atm_pa)[0]
            return k_air / gap_m * (t_hot - t_cold)

        q_cond_ag = _gap_conduction_w_m2(t_abs, t_glass)
        q_rad_ag = radiative_exchange_w_m2(t_abs, t_glass, emissivity=eps_ag)
        r4 = (
            params.eps_abs * params.tau_glass * q_solar_w_m2
            - q_cond_ag
            - q_rad_ag
            - u_gel * (t_abs - t_gel)
        )
        if two_pane:
            # Inner pane exchanges with the outer pane, which alone sees ambient.
            eps_pane_pane = parallel_plate_emissivity(eps_ga, eps_ga)
            q_cond_io = _gap_conduction_w_m2(t_glass, t_glass_outer)
            q_rad_io = radiative_exchange_w_m2(t_glass, t_glass_outer, emissivity=eps_pane_pane)
            q_rad_oa = radiative_exchange_w_m2(t_glass_outer, t_amb_c, emissivity=eps_ga)
            r3 = q_cond_ag + q_rad_ag - q_cond_io - q_rad_io
            r5 = q_cond_io + q_rad_io - h_amb * (t_glass_outer - t_amb_c) - q_rad_oa
            return np.array([r1, r3, r4, r5], dtype=float)
        q_rad_ga = radiative_exchange_w_m2(t_glass, t_amb_c, emissivity=eps_ga)
        r3 = q_cond_ag + q_rad_ag - h_amb * (t_glass - t_amb_c) - q_rad_ga

    return np.array([r1, r3, r4], dtype=float)


def solve_steady_thermal(
    *,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    m_des_kg_s_m2: float,
    h_amb: float,
    params: SystemThermalParams,
    h_m: float,
    t_guess: tuple[float, float, float] | None = None,
    vapor_gap_m: float | None = None,
    brine_salt_fraction: float | None = None,
) -> ThermalState:
    """Solve Eqs. 1, 3, 4 for (T_gel, T_abs, T_glass)."""
    if vapor_gap_m is None:
        gap_m = max(params.vapor_gap_m - h_m, 0.0)
    else:
        gap_m = vapor_gap_m
    if t_guess is None:
        t_gel0 = clamp_temperature_c(max(t_amb_c + 5.0, t_cond_c + 5.0))
        t_abs0 = clamp_temperature_c(t_gel0 + min(30.0, max(5.0, q_solar_w_m2 / 40.0)))
        t_glass0 = clamp_temperature_c(t_amb_c + 2.0)
    else:
        t_gel0, t_abs0, t_glass0 = (
            clamp_temperature_c(t_guess[0]),
            clamp_temperature_c(t_guess[1]),
            clamp_temperature_c(t_guess[2]),
        )

    # A 2-pane stack (B2) adds the outer pane as a 4th unknown; it starts a little
    # cooler than the inner pane, which is the physically correct ordering.
    two_pane = params.has_glass and params.n_glazing_panes >= 2
    x0 = [t_gel0, t_abs0, t_glass0]
    if two_pane:
        x0.append(clamp_temperature_c(0.5 * (t_glass0 + t_amb_c)))

    sol = root(
        _residuals,
        x0=np.array(x0),
        args=(
            t_cond_c, t_amb_c, q_solar_w_m2, m_des_kg_s_m2, h_amb, params, gap_m, h_m,
            brine_salt_fraction,
        ),
        method="hybr",
        tol=1e-8,
    )
    if not sol.success:
        t_gel, t_abs, t_glass = t_gel0, t_abs0, t_glass0
        t_glass_outer = x0[3] if two_pane else None
    else:
        t_gel = clamp_temperature_c(float(sol.x[0]))
        t_abs = clamp_temperature_c(float(sol.x[1]))
        t_glass = clamp_temperature_c(float(sol.x[2]))
        t_glass_outer = clamp_temperature_c(float(sol.x[3])) if two_pane else None

    h_conv_g = vapor_gap_h_conv_w_m2_k(
        gap_m, t_gel, t_cond_c, tilt_deg=params.tilt_deg, p_atm_pa=params.p_atm_pa
    )
    return ThermalState(
        t_gel_c=t_gel,
        t_abs_c=t_abs,
        t_glass_c=t_glass,
        h_conv_g=h_conv_g,
        m_des_kg_s_m2=m_des_kg_s_m2,
        t_glass_outer_c=t_glass_outer,
    )


def thermal_residual_norm(
    *,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    m_des_kg_s_m2: float,
    h_amb: float,
    params: SystemThermalParams,
    h_m: float = H0_M,
) -> float:
    gap_m = max(params.vapor_gap_m - h_m, 0.0)
    state = solve_steady_thermal(
        t_cond_c=t_cond_c,
        t_amb_c=t_amb_c,
        q_solar_w_m2=q_solar_w_m2,
        m_des_kg_s_m2=m_des_kg_s_m2,
        h_amb=h_amb,
        params=params,
        h_m=h_m,
        vapor_gap_m=gap_m,
    )
    x = [state.t_gel_c, state.t_abs_c, state.t_glass_c]
    if state.t_glass_outer_c is not None:
        x.append(state.t_glass_outer_c)
    r = _residuals(
        np.array(x),
        t_cond_c,
        t_amb_c,
        q_solar_w_m2,
        m_des_kg_s_m2,
        h_amb,
        params,
        gap_m,
        h_m,
    )
    return float(np.linalg.norm(r))


# --- Wilson et al. 2025 Eqs. 5–6 — convection-limited mass transfer ---

MassTransferPhase = Literal["absorption", "desorption"]


def _absorption_effective_water_activity(
    c_w: float,
    *,
    t_gel_c: float,
    params: MassTransferParams,
    h_m: float,
) -> float:
    """Composite gel a_w for Eq. 5 during open absorption: LiCl uses brine activity plus
    the PAM-LiCl DVS cap (Note S2); other salts use brine only."""
    aw_brine = water_activity_from_c_w(
        c_w,
        c_s=params.c_s_mol_m3,
        ions_per_formula=params.ions_per_formula,
        temperature_c=t_gel_c,
        salt_name=params.salt_name,
        formula_weight_g_mol=params.formula_weight_g_mol,
        salt_loading=params.salt_loading,
        h_m=h_m,
        h0_ref_m=params.h0_ref_m,
        salt_weight_factor=params.salt_weight_factor,
        blend_weights=params.blend_weights,
    )
    # The DVS cap is a measurement of *PAM-LiCl* specifically (Note S2): it applies
    # only to the pure-LiCl composite, never to a ZSR blend (whose polymer-bound
    # uptake was never characterized), and complex mode disables it outright via
    # use_dvs_cap so the whole simplex rests on one brine model.
    if params.salt_name != "LiCl" or params.blend_weights is not None or not params.use_dvs_cap:
        return aw_brine
    u = pam_licl_gravimetric_uptake_g_g(
        c_w,
        h_m,
        h0_ref_m=params.h0_ref_m,
        c_s_mol_m3=params.c_s_mol_m3,
        formula_weight_g_mol=params.formula_weight_g_mol,
        salt_loading=params.salt_loading,
        salt_weight_factor=params.salt_weight_factor,
    )
    rh_pct, uptake = _load_pam_licl_dvs_isotherm()
    if u <= float(uptake[0]):
        aw_dvs = max(0.0, float(rh_pct[0]) / 100.0)
    elif u >= float(uptake[-1]):
        aw_dvs = min(1.0, float(rh_pct[-1]) / 100.0)
    else:
        aw_dvs = max(0.0, min(1.0, float(np.interp(u, uptake, rh_pct)) / 100.0))
    return max(aw_brine, aw_dvs)


def _mass_transfer_driving_force(
    c_w: float,
    *,
    t_gel_c: float,
    c_r: float,
    params: MassTransferParams,
    h_m: float,
    phase: MassTransferPhase,
) -> float:
    if phase == "absorption":
        aw = _absorption_effective_water_activity(
            c_w,
            t_gel_c=t_gel_c,
            params=params,
            h_m=h_m,
        )
        if not math.isfinite(aw):
            return 0.0
        return c_r - aw

    aw = water_activity_from_c_w(
        c_w,
        c_s=params.c_s_mol_m3,
        ions_per_formula=params.ions_per_formula,
        temperature_c=t_gel_c,
        salt_name=params.salt_name,
        formula_weight_g_mol=params.formula_weight_g_mol,
        salt_loading=params.salt_loading,
        h_m=h_m,
        h0_ref_m=params.h0_ref_m,
        salt_weight_factor=params.salt_weight_factor,
        blend_weights=params.blend_weights,
    )
    if not math.isfinite(aw):
        return 0.0
    return c_r - aw


# Big enough that the residual driving force is negligible against _ODE_RTOL, small
# enough that g*driving stays far from float overflow at any reachable p_sat.
_INSTANT_EQUILIBRIUM_G_SCALE: float = _pv("Instant-equilibrium g scale factor")


@dataclass(frozen=True, slots=True)
class MassTransferParams:
    g_conv_m_s: float  # g_chamber (Table S3) for open absorption
    h0_ref_m: float
    vapor_gap_m: float
    tilt_deg: float
    c_s_mol_m3: float
    ions_per_formula: int
    rho_solution_kg_m3: float
    # Gel-water bounds (mol/m³), both salt- and blend-dependent so neither can be a
    # module constant: hydrate_floor_c_w below, dilution_ceiling_c_w above.
    c_w_min_mol_m3: float
    c_w_max_mol_m3: float
    salt_name: str = "LiCl"
    formula_weight_g_mol: float = 42.394
    salt_loading: float = SALT_LOADING_DEFAULT
    salt_weight_factor: float = 1.0
    # Complex mode (B8): ZSR molality weights over complex_model.ZSR_SALTS. None
    # (default) keeps the single-salt path, so gpu_sweep and every existing caller
    # are untouched.
    blend_weights: tuple[float, ...] | None = None
    # Ideal-case switch: the g -> infinity limit, where the gel sits on its
    # equilibrium isotherm at every instant (a_w == c_r, zero driving force).
    # See mass_transfer_g_m_s and SystemConfig.instant_equilibrium.
    instant_equilibrium: bool = False
    # The PAM-LiCl DVS cap (Note S2) is a measurement of the LiCl composite only.
    # Complex mode switches it off so the brine model is self-consistent across the
    # whole blend simplex -- otherwise the pure-LiCl corner sits on a cliff that a
    # GP would chase for a modeling reason rather than a physical one.
    use_dvs_cap: bool = True
    # Site ambient pressure (see SystemThermalParams.p_atm_pa). Desorption's g runs on
    # D_air/k_air, both pressure-dependent, so this has to reach the mass side too.
    p_atm_pa: float = P_ATM_SEA_LEVEL_PA


def mass_transfer_g_m_s(
    *,
    phase: MassTransferPhase,
    params: MassTransferParams,
    h_m: float,
    t_gel_c: float,
    t_cond_c: float | None = None,
) -> float:
    """Note S1: g_chamber in absorption; heat–mass analogy in desorption (Eq. S5)."""
    if params.instant_equilibrium:
        # g -> infinity. Not literally infinite: scaling g is enough because nothing
        # downstream depends on g except through the product g*(c_r - a_w), so the
        # desorption root solve in simulation.evaluate_coupled_rates lands on
        # driving ~ 1e-6 of its finite-g value, i.e. numerically on the isotherm.
        # Desorption is then energy-limited (m_des set by Eqs. 1-4), not
        # transport-limited, which is the point of the ideal case.
        #
        # This deliberately bypasses the 7 mm thermobuoyancy cutoff below: that
        # cutoff *is* a mass-transfer limit, so it has no place in the g -> inf bound.
        return _INSTANT_EQUILIBRIUM_G_SCALE * params.g_conv_m_s
    if phase == "absorption":
        return params.g_conv_m_s
    if t_cond_c is None:
        raise ValueError("t_cond_c required for desorption mass transfer")
    gap_m = max(params.vapor_gap_m - h_m, 0.0)
    if gap_m < VAPOR_GAP_TRANSPORT_MIN_M:  # Wilson's ~7 mm thermobuoyancy / transport limit
        return 0.0
    h_conv = vapor_gap_h_conv_w_m2_k(
        gap_m, t_gel_c, t_cond_c, tilt_deg=params.tilt_deg, p_atm_pa=params.p_atm_pa
    )
    return mass_transfer_g_from_h_conv_m_s(
        h_conv, t_gel_c=t_gel_c, t_cond_c=t_cond_c, p_atm_pa=params.p_atm_pa
    )


def concentration_ratio_absorption(rh: float) -> float:
    return float(rh)


def concentration_ratio_desorption(t_gel_c: float, t_cond_c: float) -> float:
    p_g = saturation_vapor_pressure_pa(t_gel_c)
    p_c = saturation_vapor_pressure_pa(t_cond_c)
    t_g_k = t_gel_c + 273.15
    t_c_k = t_cond_c + 273.15
    if p_g <= 0.0 or t_g_k <= 0.0 or t_c_k <= 0.0:
        return 0.0
    return (p_c / p_g) * (t_g_k / t_c_k)


def equilibrium_t_gel_desorption_c(
    c_w: float,
    *,
    t_cond_c: float,
    params: MassTransferParams,
    h_m: float,
) -> float:
    """Gel temperature at which desorption's driving force vanishes (deg C).

    The local-equilibrium (g -> infinity) closure. Eq. 5's desorption driving force is
    ``c_r(T_gel, T_cond) - a_w(c_w, T_gel)``; the ideal-kinetics limit is the constraint
    that it is exactly zero, i.e. the gel's surface vapour pressure equals the
    condenser's. Solving that for T_gel replaces the old approach of scaling g by a large
    factor until the residual was negligible -- same limit, but as an algebraic condition
    rather than a stiff relaxation, so the ODE that consumes it is not stiff.

    Monotone, hence a unique root: a_w varies weakly with temperature while c_r falls as
    p_sat(T_gel) grows. At T_gel = T_cond, c_r is exactly 1 and a_w <= 1, so the residual
    starts non-positive; it turns positive once the gel is hot enough. Returns nan if no
    bracket exists (e.g. no water left to have an activity), which callers read as "this
    state cannot desorb".
    """

    def residual(t_gel_c: float) -> float:
        # a_w - c_r, i.e. the negated Eq. 5 driving force: positive means desorbing.
        return -_mass_transfer_driving_force(
            c_w,
            t_gel_c=t_gel_c,
            c_r=concentration_ratio_desorption(t_gel_c, t_cond_c),
            params=params,
            h_m=h_m,
            phase="desorption",
        )

    lo = clamp_temperature_c(t_cond_c)
    hi = clamp_temperature_c(t_cond_c + 200.0)
    return find_root_bracketed(residual, lo, hi)


def dc_w_dt_from_m_des(m_des_kg_s_m2: float, *, h0_ref_m: float) -> float:
    """Inverse of ``m_des_kg_s_m2_from_dc_w``: the loading rate a given desorption flux
    implies. Under local equilibrium the flux comes from the energy balance rather than
    from Eq. 5's rate law, so the causality runs this way round."""
    if m_des_kg_s_m2 <= 0.0:
        return 0.0
    return -m_des_kg_s_m2 / (WATER_MOLAR_MASS_KG_MOL * h0_ref_m)


def dh_dt_from_dc_w(dc_w_dt_val: float, *, rho_solution_kg_m3: float, h0_ref_m: float) -> float:
    """Eq. 6 written as a ratio to Eq. 5 rather than as its own rate law.

    dH/dt = dc_w/dt * (MW * H0 / rho_sol) identically in ``dH_dt``, so H and c_w stay on
    the same trajectory whether the rate came from the mass law or from the energy
    balance. Deriving it here keeps the equilibrium path from needing a second driving
    force it is not allowed to have (its driving force is zero by construction).
    """
    return dc_w_dt_val * WATER_MOLAR_MASS_KG_MOL * h0_ref_m / rho_solution_kg_m3


def _mass_transfer_rate_terms(
    c_w: float,
    *,
    t_gel_c: float,
    c_r: float,
    params: MassTransferParams,
    h_m: float,
    phase: MassTransferPhase,
    t_cond_c: float | None,
) -> tuple[float, float, float, float]:
    """Shared (T_k, p_sat, g, driving) terms behind Eqs. 5-6 (dc_w/dt, dH/dt)."""
    t_k = max(t_gel_c + 273.15, 200.0)
    p_sat = saturation_vapor_pressure_pa(t_gel_c)
    g = mass_transfer_g_m_s(phase=phase, params=params, h_m=h_m, t_gel_c=t_gel_c, t_cond_c=t_cond_c)
    driving = _mass_transfer_driving_force(
        c_w,
        t_gel_c=t_gel_c,
        c_r=c_r,
        params=params,
        h_m=h_m,
        phase=phase,
    )
    return t_k, p_sat, g, driving


def dc_w_dt(
    c_w: float,
    *,
    t_gel_c: float,
    c_r: float,
    params: MassTransferParams,
    h_m: float,
    phase: MassTransferPhase = "absorption",
    t_cond_c: float | None = None,
) -> float:
    """Eq. 5: dc_w/dt (mol/m³/s); g_chamber/H₀ (abs) or heat–mass analogy (des)."""
    t_k, p_sat, g, driving = _mass_transfer_rate_terms(
        c_w, t_gel_c=t_gel_c, c_r=c_r, params=params, h_m=h_m, phase=phase, t_cond_c=t_cond_c
    )
    if not math.isfinite(driving):
        return 0.0
    rate = (g / params.h0_ref_m) * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving
    if not math.isfinite(rate):
        return 0.0
    if c_w >= params.c_w_max_mol_m3 and rate > 0.0:
        return 0.0
    if c_w <= params.c_w_min_mol_m3 and rate < 0.0:
        return 0.0
    return rate


def dH_dt(
    c_w: float,
    *,
    t_gel_c: float,
    c_r: float,
    params: MassTransferParams,
    h_m: float,
    phase: MassTransferPhase = "absorption",
    t_cond_c: float | None = None,
) -> float:
    """Eq. 6 hydrogel thickness rate dH/dt = g·(MW/ρ_sol)·(p_sat/RT)·driving (m/s), i.e.
    dc_w/dt·(MW·H₀/ρ_sol), so H and c_w evolve on the same timescale."""
    t_k, p_sat, g, driving = _mass_transfer_rate_terms(
        c_w, t_gel_c=t_gel_c, c_r=c_r, params=params, h_m=h_m, phase=phase, t_cond_c=t_cond_c
    )
    return g * WATER_MOLAR_MASS_KG_MOL / params.rho_solution_kg_m3 * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k)) * driving


def hydrate_floor_c_w(
    *,
    c_s_mol_m3: float,
    salt_name: str,
    blend_weights: tuple[float, ...] | None = None,
) -> float:
    """Lowest physically reachable gel water concentration (mol/m³), on the H₀ basis.

    Eq. 5's rate depends on c_w only through a_w, and a_w is floored at the salt's
    deliquescence RH -- correct for the brine (below DRH the salt precipitates and a
    saturated solution's activity stops falling), but it leaves the rate with no
    knowledge of how much water is actually left. Desorption therefore never
    self-terminates for a salt whose DRH sits above the condenser's concentration
    ratio, and the ODE will drive c_w through zero. The bound that is missing is
    chemical, not numerical: once every remaining water molecule is crystal-bound as
    ``salt·nH₂O``, removing more is a dehydration reaction costing far more than
    h_fg, which this model does not represent.

    So the floor is n·c_s, with n blend-weighted over ZSR_SALTS. That replaces the
    single global C_W_MIN_MOL_M3, which corresponds to 0.01-0.03 water per salt --
    roughly 100x drier than any hydrate, i.e. no constraint at all.
    """
    if blend_weights is None:
        n_eff = get_salt(salt_name).hydrate_h2o_per_formula
    else:
        from solar_lumped.complex_model import ZSR_SALTS, normalized_blend_weights

        w = normalized_blend_weights(blend_weights)
        n_eff = float(
            sum(wi * get_salt(name).hydrate_h2o_per_formula for name, wi in zip(ZSR_SALTS, w))
        )
    return max(0.0, n_eff * float(c_s_mol_m3))


def drh_floor_c_w(
    *,
    c_s_mol_m3: float,
    salt_name: str,
    formula_weight_g_mol: float,
    blend_weights: tuple[float, ...] | None = None,
    temperature_c: float = 25.0,
) -> float:
    """Alternative lower bound to :func:`hydrate_floor_c_w`: stop at brine saturation.

    The equilibrium c_w at the deliquescence RH -- the water still held when the brine
    reaches its solubility limit and any further drying precipitates solid salt. That is
    where the Conde/ZSR activity model stops being a description of a real solution, so
    it is the conservative place to stop: strictly wetter than the hydrate floor (a
    saturated LiCl brine holds ~2.8 H2O per LiCl vs. 1 in the monohydrate), ending
    desorption earlier and yielding less. Selected by ``SystemConfig(c_w_floor_mode="drh")``.

    Taken as the saturation *composition* rather than by inverting a_w at the DRH: the
    DRH is by construction the activity at saturation, so the inversion is being asked
    for a root sitting exactly on its own bracket edge and returns NaN. The saturated
    brine fraction is the same answer, directly.

    Composition-only, so temperature enters solely through the blend window; a single
    salt's solubility is not temperature-resolved in this model.
    """
    if blend_weights is None:
        f_b = saturation_brine_salt_fraction(salt_name)
    else:
        from solar_lumped.complex_model import blend_water_activity_window, zsr_brine_state

        aw_lo, _hi = blend_water_activity_window(blend_weights, temperature_c)
        f_b, _ions, _mw = zsr_brine_state(blend_weights, aw_lo, temperature_c)
    if not (0.0 < f_b <= 1.0):
        return float("nan")
    # Same H₀-basis algebra as equilibrium_c_w_at_rh, with H₀ cancelling out.
    mass_salt_per_h = c_s_mol_m3 * formula_weight_g_mol / 1000.0
    return mass_salt_per_h * (1.0 - f_b) / f_b / WATER_MOLAR_MASS_KG_MOL


def dilution_ceiling_c_w(
    *,
    c_s_mol_m3: float,
    salt_name: str,
    formula_weight_g_mol: float,
    blend_weights: tuple[float, ...] | None = None,
    temperature_c: float = 25.0,
) -> float:
    """Highest physically modelled gel water concentration (mol/m³), on the H₀ basis.

    The upper mirror of :func:`hydrate_floor_c_w`: the equilibrium c_w at
    ``DILUTION_CAP_RH``. Beyond the cap the brine is so dilute that a_w -> 1, c_w
    diverges, and the a_w -> f_b inversion loses conditioning, so the model has
    nothing meaningful to say. Per salt and blend, because the same a_w corresponds
    to a different water inventory for each.
    """
    return equilibrium_c_w_at_rh(
        DILUTION_CAP_RH,
        c_s=c_s_mol_m3,
        ions_per_formula=0,
        temperature_c=temperature_c,
        salt_name=salt_name,
        formula_weight_g_mol=formula_weight_g_mol,
        blend_weights=blend_weights,
    )


def m_des_kg_s_m2_from_state(
    c_w: float,
    h_m: float,
    dc_w_dt_val: float,
    dH_dt_val: float,
) -> float:
    """Desorption flux (kg/m²/s) from the Eqs. 5–6 inventory N = c_w·H:
    ṁ = -MW·(dc_w/dt·H + c_w·dH/dt); Note S1's -dc_w/dt·MW·H₀ is its dH/dt ≈ 0 limit."""
    flux = -WATER_MOLAR_MASS_KG_MOL * (dc_w_dt_val * h_m + c_w * dH_dt_val)
    return max(0.0, flux)


def m_des_kg_s_m2_from_dc_w(
    dc_w_dt_val: float,
    *,
    h0_ref_m: float,
) -> float:
    """Note S1 limit: ṁ_des = -dc_w/dt · MW · H₀ (reference thickness, negligible dH/dt)."""
    if dc_w_dt_val >= 0.0:
        return 0.0
    return -dc_w_dt_val * WATER_MOLAR_MASS_KG_MOL * h0_ref_m


# --- Sorbent interface: PAM-salt hydrogel; PhaseResult.c_w stores mol/m³. ---


def evaluate_mass_rates(
    *,
    loading: float,
    h_m: float,
    t_gel_c: float,
    t_cond_c: float | None,
    rh: float,
    phase: str,
    mass: MassTransferParams,
    config: SystemConfig,
    vapor_gap_m: float,
) -> tuple[float, float, float]:
    """Return (dloading/dt, dH/dt, m_des_kg_s_m2)."""
    if phase == "absorption":
        c_r = concentration_ratio_absorption(rh)
        dc = dc_w_dt(
            loading,
            t_gel_c=t_gel_c,
            c_r=c_r,
            params=mass,
            h_m=h_m,
            phase="absorption",
        )
        dh = dH_dt(
            loading,
            t_gel_c=t_gel_c,
            c_r=c_r,
            params=mass,
            h_m=h_m,
            phase="absorption",
        )
        # Bounded in the same coordinate as dc_w_dt (which clamps at c_w_min/c_w_max),
        # not at H₀: the gel's volume follows its water content, so the thinnest it gets
        # is set by the hydrate floor, not by its as-cast thickness. See
        # SystemConfig.hydrogel_floor_thickness_m.
        if loading <= mass.c_w_min_mol_m3:
            dh = max(0.0, dh)
        return dc, dh, 0.0

    assert t_cond_c is not None
    c_r = concentration_ratio_desorption(t_gel_c, t_cond_c)
    dc = dc_w_dt(
        loading,
        t_gel_c=t_gel_c,
        c_r=c_r,
        params=mass,
        h_m=h_m,
        phase="desorption",
        t_cond_c=t_cond_c,
    )
    dh = dH_dt(
        loading,
        t_gel_c=t_gel_c,
        c_r=c_r,
        params=mass,
        h_m=h_m,
        phase="desorption",
        t_cond_c=t_cond_c,
    )
    if loading <= mass.c_w_min_mol_m3:
        dh = 0.0
    if dc > 0.0:
        dc = 0.0
    if dh > 0.0:
        dh = 0.0
    m_des = m_des_kg_s_m2_from_dc_w(dc, h0_ref_m=mass.h0_ref_m)
    return dc, dh, m_des


def inventory_label(config: SystemConfig) -> str:
    return "gel"


def inventory_ylabel(config: SystemConfig) -> str:
    return "Water in gel (L/m²)"


def inventory_prefix(config: SystemConfig) -> str:
    return "water_in_gel"


def initial_loading(config: SystemConfig) -> float:
    # Route the resolved blend (None for single-salt, including complex mode's
    # single-salt corners) so fabrication state and transport agree on the brine.
    mass = config.mass_params()
    return fabrication_c_w_initial(
        salt_name=mass.salt_name,
        salt_loading=config.salt_loading,
        hydrogel_thickness_m=config.hydrogel_thickness_m,
        hydrogel_density_kg_m3=config.hydrogel_density_kg_m3,
        blend_weights=mass.blend_weights,
        use_dvs_cap=mass.use_dvs_cap,
    )


def water_in_gel_l_m2(
    c_w: float,
    h_m: float,
    *,
    h0_ref_m: float = H0_M,
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


def water_in_sorbent_l_m2(loading: float, h_m: float, *, config: SystemConfig) -> float:
    return water_in_gel_l_m2(loading, h_m, h0_ref_m=config.hydrogel_thickness_m)

