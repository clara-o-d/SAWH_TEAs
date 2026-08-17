"""Provenance for LiBr: refits ``physics._CRYSTALLIZATION_LINE["LiBr"]``, and checks
``physics.patek_libr_water_activity`` against the two local CSVs.

    python -m solar_lumped.data.materials.fit_libr_saturation

The isotherm itself is no longer fitted here -- it is Patek & Klomfar (2006) Eq. (1),
transcribed into physics.py from patek2006.tex Table 4 and validated against that
paper's own Table 9. What is still derived locally is saturation, because Patek states
validity "from 273 K or from the crystallization line" without supplying that line.

Saturation source:
  Greenspan (1977) J. Res. NBS 81A, 89-96, Table 1 -- measured equilibrium RH over
  SATURATED LiBr solution, i.e. the deliquescence point, as a polynomial in t over
  0-100 C. Inverting the isotherm at that RH gives the saturated composition.

Validation CSVs (digitised; these used to be the fit data for a BET that Patek
replaced):
  libr_temperature_vle.csv       Patil & Tripathi (1990) osmotic coefficients phi,
                                 30-70 C, 14.8-58.1 wt%; a_w = exp(-phi*nu*m*Mw/1000).
  libr_water_vle_isothermal.csv  isothermal P-x VLE at 65/75/85/95 C to 60.4 wt%;
                                 a_w = P / P0(T), P0 read off the x1 = 0 row.

Expect PT90 to agree noticeably worse than the VLE set. That is not a bug in the
implementation -- Patek collected all 40 PT90 points and used none of them (his Table 1
gives it RMS 4.76% against 1.3-2.0% for the sets he kept), so this run reproduces his
verdict from local data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from solar_lumped.physics import (
    LIBR_FORMULA_WEIGHT_G_MOL,
    T_CRIT_H2O_K,
    _CRYSTALLIZATION_LINE,
    patek_libr_water_activity,
)

_DIR = Path(__file__).parent
_NU = 2  # LiBr dissociates to Li+ + Br-
_MW_WATER_G = 18.015

# Greenspan Table 1: RH% = sum(A_i * t^i), t in C. 21 points over 0-100 C, residual
# sigma 0.22%, sources Acheson and Richardson & Malthers (1955). Checks out
# independently: this gives 6.38% at 25 C against the widely quoted 6.37%.
GREENSPAN_LIBR_DRH_PCT: tuple[float, float, float] = (7.75437, -0.0654994, 0.420737e-3)
GREENSPAN_T_RANGE_C: tuple[float, float] = (0.0, 100.0)


def greenspan_deliquescence_rh(t_c: float) -> float:
    a0, a1, a2 = GREENSPAN_LIBR_DRH_PCT
    return (a0 + a1 * t_c + a2 * t_c**2) / 100.0


def fit_crystallization_line() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conde-form theta = A0 + A1*xi over Greenspan's span; returns (coeffs, xi, resid).

    Linear, not quadratic: the quadratic fits xi better but opens upward with its vertex
    inside the composition range, which puts a spurious root below the real one -- and
    physics picks the minimum root. See the comment on the committed coefficients.

    Fitted over the full 0-100 C span rather than stopping at Greenspan's 77.8 C
    turning point, because deliquescence a_w is what matters downstream and the full
    span is better on it (0.0043 vs 0.0069) even though it is worse on xi.
    """
    t = np.arange(GREENSPAN_T_RANGE_C[0], GREENSPAN_T_RANGE_C[1] + 0.01, 2.5)
    xi = np.array([
        brentq(lambda w: patek_libr_water_activity(w, x) - greenspan_deliquescence_rh(x), 0.35, 0.74)
        for x in t
    ])
    theta = (t + 273.15) / T_CRIT_H2O_K
    a1, a0 = np.polyfit(xi, theta, 1)
    return np.array([a0, a1, 0.0]), xi, (a0 + a1 * xi - theta) / a1


def libr_activity_points() -> pd.DataFrame:
    """Both CSVs reduced to (T_K, molality, a_w, source)."""
    pt = pd.read_csv(_DIR / "libr_temperature_vle.csv")
    osm = pt[pt.quantity == "osmotic"]
    a = pd.DataFrame({
        "T_K": osm.temperature_C + 273.15,
        "m": osm.molality_mol_per_kg,
        "aw": np.exp(-osm.value * _NU * osm.molality_mol_per_kg * _MW_WATER_G / 1000.0),
        "source": "PT90 (rejected by Patek)",
    })

    vle = pd.read_csv(_DIR / "libr_water_vle_isothermal.csv")
    p0 = vle[vle.x1_LiBr == 0].set_index("T_K").P_kPa
    wet = vle[vle.x1_LiBr > 0]
    b = pd.DataFrame({
        "T_K": wet.T_K,
        "m": wet.m_LiBr_mol_per_kg,
        "aw": wet.P_kPa / wet.T_K.map(p0),
        "source": "VLE",
    })
    return pd.concat([a, b], ignore_index=True)


if __name__ == "__main__":
    coeffs, xi, resid = fit_crystallization_line()
    print(f"crystallization line, Greenspan {GREENSPAN_T_RANGE_C[0]:.0f}-"
          f"{GREENSPAN_T_RANGE_C[1]:.0f} C, xi_sat {xi.min():.3f}-{xi.max():.3f}")
    print(f"  refit     A0={coeffs[0]:.6f} A1={coeffs[1]:.6f} A2={coeffs[2]:.6f}")
    print("  committed A0={:.6f} A1={:.6f} A2={:.6f}".format(*_CRYSTALLIZATION_LINE["LiBr"][0]))
    print(f"  residual  RMS {np.sqrt((resid**2).mean()):.5f}  max {abs(resid).max():.5f} (in xi)")
    drh = [(t, greenspan_deliquescence_rh(t)) for t in (0, 25, 50, 80, 100)]
    print("  deliquescence a_w error vs Greenspan: " + ", ".join(
        f"{t:.0f}C {patek_libr_water_activity(((t + 273.15) / T_CRIT_H2O_K - coeffs[0]) / coeffs[1], t) - g:+.4f}"
        for t, g in drh))

    pts = libr_activity_points()
    mw = LIBR_FORMULA_WEIGHT_G_MOL
    pts["w"] = pts.m * mw / (1000.0 + pts.m * mw)
    pred = np.array([patek_libr_water_activity(w, T - 273.15)
                     for w, T in zip(pts.w, pts.T_K)])
    pts["e"] = pred - pts.aw
    # Percent, because that is the metric Patek's Table 1 reports (RMS on PRESSURE, and
    # a_w is pressure over a common p_s(T), so the ratio is the same number). Absolute
    # a_w error is not comparable to it: 4.76% of a_w = 0.9 is 0.043, of a_w = 0.1 is
    # 0.005, so a dilute-heavy set looks bad in percent and good in absolute.
    pts["pct"] = 100.0 * pts.e / pts.aw
    print(f"\nPatek Eq. (1) against the local CSVs ({len(pts)} points, "
          f"{pts.T_K.min() - 273.15:.0f}-{pts.T_K.max() - 273.15:.0f} C):")
    for src, g in pts.groupby("source"):
        print(f"  {src:26s} n={len(g):3d}  RMS {np.sqrt((g.pct**2).mean()):5.2f}%  "
              f"max {g.pct.abs().max():5.2f}%  mean {g.pct.mean():+5.2f}%   "
              f"(abs a_w: RMS {np.sqrt((g.e**2).mean()):.4f})")
    print("  Patek Table 1 reports RMS 4.76% for PT90 (0 of 40 points used) against "
          "1.3-2.0% for the sets he kept.")
