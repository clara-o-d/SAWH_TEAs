"""Equilibrium brine isotherms for NaCl, LiCl, CaCl2, and MgCl2."""

from __future__ import annotations

import math
from collections.abc import Callable

from solar_lumped.physics.conde2004 import (
    equilibrium_salt_mass_fraction_cacl2,
    equilibrium_salt_mass_fraction_licl,
    water_activity_cacl2,
    water_activity_licl,
)
from solar_lumped.physics.salt_properties import get_salt
from solar_lumped.utils.numerics import find_root_bracketed

_BRACKET_LO = 0.01
_BRACKET_HI = 0.75


def mf_NaCl(relative_humidity: float) -> float:
    """Equilibrium brine salt fraction for NaCl at 25°C."""
    if not (0.0 < relative_humidity < 1.0):
        return float("nan")
    a4, a3, a2, a1, a0 = 5.863, -5.545, -0.332, -0.5597, 0.9998

    def residual(salt_fraction: float) -> float:
        return (
            relative_humidity
            - a0
            - a1 * salt_fraction
            - a2 * salt_fraction**2
            - a3 * salt_fraction**3
            - a4 * salt_fraction**4
        )

    return find_root_bracketed(residual, 0.0116, 0.264)


def mf_LiCl(relative_humidity: float, temperature_c: float = 25.0) -> float:
    """Equilibrium brine salt fraction for LiCl (Conde 2004)."""
    if not (0.0 < relative_humidity < 1.0) or temperature_c > 150.0:
        return float("nan")
    return equilibrium_salt_mass_fraction_licl(relative_humidity, temperature_c)


def mf_CaCl2(relative_humidity: float, temperature_c: float = 25.0) -> float:
    """Equilibrium brine salt fraction for CaCl2 (Conde 2004)."""
    if not (0.0 < relative_humidity < 1.0) or temperature_c > 100.0:
        return float("nan")
    return equilibrium_salt_mass_fraction_cacl2(relative_humidity, temperature_c)


def mf_MgCl2(relative_humidity: float) -> float:
    """Equilibrium brine salt fraction for MgCl2 (polynomial fit)."""
    if not (0.0 < relative_humidity < 1.0):
        return float("nan")
    a4, a3, a2, a1, a0 = 186.32487108, -153.67496570, 38.21982328, -4.86704441, 1.16231287

    def residual(salt_fraction: float) -> float:
        return (
            relative_humidity
            - a0
            - a1 * salt_fraction
            - a2 * salt_fraction**2
            - a3 * salt_fraction**3
            - a4 * salt_fraction**4
        )

    return find_root_bracketed(residual, 0.01, 0.75, scan=True, n_intervals=19)


_isotherm_by_salt: dict[str, Callable[[float, float], float]] = {
    "NaCl": lambda rh, t: mf_NaCl(rh),
    "LiCl": lambda rh, t: mf_LiCl(rh, t),
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
    if not (rec.rh_min <= relative_humidity <= rec.rh_max):
        return float("nan")
    return float(_isotherm_by_salt[rec.name](relative_humidity, temperature_c))


def _aw_polynomial(salt_fraction: float, coeffs: tuple[float, ...]) -> float:
    if not (0.0 <= salt_fraction < 1.0) or not math.isfinite(salt_fraction):
        return float("nan")
    a_w = 0.0
    for k, coeff in enumerate(coeffs):
        a_w += coeff * (salt_fraction**k)
    return float(a_w)


def water_activity_at_brine_fraction(
    salt_name: str,
    brine_salt_fraction: float,
    temperature_c: float = 25.0,
) -> float:
    """Forward isotherm: brine water activity at salt mass fraction and temperature."""
    rec = get_salt(salt_name)
    f = float(brine_salt_fraction)
    if not (0.0 <= f < 1.0) or not math.isfinite(f):
        return float("nan")
    if rec.name == "NaCl":
        return _aw_polynomial(f, (0.9998, -0.5597, -0.332, -5.545, 5.863))
    if rec.name == "MgCl2":
        return _aw_polynomial(
            f, (1.16231287, -4.86704441, 38.21982328, -153.67496570, 186.32487108)
        )
    if rec.name == "LiCl":
        if temperature_c > 150.0:
            return float("nan")
        return water_activity_licl(f, min(temperature_c, 150.0))
    if rec.name == "CaCl2":
        if temperature_c > 100.0:
            return float("nan")
        return water_activity_cacl2(f, temperature_c)
    return float("nan")


def brine_salt_fraction_from_c_w(
    c_w_mol_m3: float,
    c_s_mol_m3: float,
    effective_formula_weight_g_per_mol: float,
) -> float:
    """Brine salt mass fraction from gel water/salt molarities (mol/m³ gel)."""
    if not all(map(math.isfinite, (c_w_mol_m3, c_s_mol_m3, effective_formula_weight_g_per_mol))):
        return float("nan")
    if c_w_mol_m3 < 0.0 or c_s_mol_m3 < 0.0:
        return float("nan")
    mass_water = c_w_mol_m3 * 18.015 / 1000.0
    mass_salt = c_s_mol_m3 * effective_formula_weight_g_per_mol / 1000.0
    total = mass_water + mass_salt
    if total <= 0.0:
        return float("nan")
    return float(mass_salt / total)
