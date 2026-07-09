"""Conde (2004) aqueous LiCl and CaCl2 solution properties.

Vapour-pressure correlation (Table 3, ``conde2004.tex``):

    π ≡ p_sol(ξ, T) / p_H2O(T) = π_25(ξ) · f(ξ, θ)

where ξ is the salt mass fraction in the brine and θ = T / T_c,H2O.
π equals the water activity a_w at the solution interface (Díaz-Marín Eq. 5 / 8).

Pure-water vapour pressure uses Saul–Wagner (Appendix A, same reference).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from solar_lumped.utils.numerics import find_root_bracketed

# Conde (2004): θ ≡ T / T_c,H2O
T_CRIT_H2O_K: float = 647.096
P_CRIT_H2O_PA: float = 22.064e6

# Valid brine salt mass-fraction ranges (Conde § Density)
XI_MAX_LICL: float = 0.56
XI_MAX_CACL2: float = 0.60

_BRACKET_LO: float = 0.01
_BRACKET_HI: float = 0.75


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
    xi_max: float


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
    xi_max=XI_MAX_LICL,
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
    xi_max=XI_MAX_CACL2,
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


def reduced_temperature(temperature_c: float) -> float:
    """θ = T / T_c,H2O."""
    return (float(temperature_c) + 273.15) / T_CRIT_H2O_K


def _pi_25(xi: float, params: VaporPressureParams) -> float:
    return (
        1.0
        - (1.0 + (xi / params.pi6) ** params.pi7) ** params.pi8
        - params.pi9 * math.exp(-((xi - 0.1) ** 2) / 0.005)
    )


def _f_xi_theta(xi: float, theta: float, params: VaporPressureParams) -> float:
    a_term = 2.0 - (1.0 + (xi / params.pi0) ** params.pi1) ** params.pi2
    b_term = (1.0 + (xi / params.pi3) ** params.pi4) ** params.pi5 - 1.0
    return a_term + b_term * theta


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
    if xi > params.xi_max:
        return float("nan")
    theta = reduced_temperature(temperature_c)
    pi = _pi_25(xi, params) * _f_xi_theta(xi, theta, params)
    return max(0.0, min(1.0, float(pi)))


def water_activity_licl(
    salt_mass_fraction: float,
    temperature_c: float = 25.0,
) -> float:
    """LiCl–H2O brine water activity (Conde 2004 Table 3)."""
    return vapor_pressure_ratio(salt_mass_fraction, temperature_c, LICL_VAPOR_PRESSURE)


def water_activity_cacl2(
    salt_mass_fraction: float,
    temperature_c: float = 25.0,
) -> float:
    """CaCl2–H2O brine water activity (Conde 2004 Table 3)."""
    return vapor_pressure_ratio(salt_mass_fraction, temperature_c, CACL2_VAPOR_PRESSURE)


def equilibrium_salt_mass_fraction(
    relative_humidity: float,
    params: VaporPressureParams,
    *,
    temperature_c: float = 25.0,
    temperature_max_c: float = 150.0,
) -> float:
    """Invert a_w(ξ) = RH for brine salt mass fraction ξ."""
    rh = float(relative_humidity)
    if rh <= 0.0:
        return 1.0
    if rh >= 0.99:
        return _BRACKET_LO
    if temperature_c > temperature_max_c:
        return float("nan")

    def residual(xi: float) -> float:
        return rh - vapor_pressure_ratio(xi, temperature_c, params)

    hi = min(_BRACKET_HI, params.xi_max)
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


def water_vapor_pressure_pa(temperature_c: float) -> float:
    """Saul–Wagner vapour pressure of pure liquid water (Conde 2004 Appendix A)."""
    t_k = float(temperature_c) + 273.15
    if t_k <= 273.15 or t_k >= T_CRIT_H2O_K:
        return float("nan")
    tau = 1.0 - t_k / T_CRIT_H2O_K
    numer = sum(a * tau**exp for a, exp in zip(_SAUL_WAGNER_A, _SAUL_WAGNER_EXP, strict=True))
    ln_p_pc = numer / (1.0 - tau)
    return float(P_CRIT_H2O_PA * math.exp(ln_p_pc))
