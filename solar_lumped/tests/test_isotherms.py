"""High-temperature brine isotherms: Zeng & Zhou BET for LiCl above Conde's cap, and
Patek & Klomfar (2006) for LiBr.

Each correlation is pinned against its own paper's published numbers, because both were
transcribed by hand and both have a way to be silently wrong: the BET's sign convention
on c, and Patek's mole-vs-mass fraction basis.
"""

from __future__ import annotations

import math

import pytest

from solar_lumped.physics import (
    BET_T_MAX_C,
    CONDE_T_MAX_C,
    LIBR_T_MAX_C,
    LIBR_W_MAX,
    LICL_BET,
    _saul_wagner_pa,
    bet_brine_salt_fraction,
    bet_water_activity,
    deliquescence_rh,
    equilibrate_salt_mf,
    isosteric_h_des_j_per_kg,
    isotherm_t_max_c,
    patek_libr_water_activity,
    saturation_brine_salt_fraction,
    water_activity_at_brine_fraction,
    water_activity_licl,
)


def _xi(m: float, mw: float) -> float:
    return m * mw / (1000.0 + m * mw)


# --- Zeng & Zhou (2006), LiCl above 100 C ---------------------------------------


def test_reproduces_zeng_zhou_published_values():
    """r and c at 373.15 K (paper: 3.278, 9.329) and a_w at m=18.542/298.15 K (0.130).

    Pins the sign convention: the paper prints dE = R T ln c, which gives c < 1 and a
    wrong isotherm. Only c = exp(-dE/RT) reproduces its own tabulated numbers.
    """
    t_k = 373.15
    assert LICL_BET.r0 + LICL_BET.r1 * t_k == pytest.approx(3.278, abs=0.05)
    c = math.exp(-(LICL_BET.e0 + LICL_BET.e1 * t_k) / (8.314 * t_k))
    assert c == pytest.approx(9.329, abs=0.3)
    assert bet_water_activity(_xi(18.542, 42.394), 25.0, LICL_BET) == pytest.approx(0.130, abs=0.002)


@pytest.mark.parametrize("aw", [0.05, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("t_c", [25.0, 100.0, 155.0])
def test_bet_forward_and_inverse_round_trip(aw, t_c):
    xi = bet_brine_salt_fraction(aw, t_c, LICL_BET)
    assert 0.0 < xi < 1.0
    assert bet_water_activity(xi, t_c, LICL_BET) == pytest.approx(aw, rel=1e-9)


def test_licl_extension_is_continuous_at_condes_cap():
    """The BET enters as a ratio anchored at 100 C, so it must not step there."""
    for xi in (0.30, 0.45, 0.55):
        below = water_activity_licl(xi, CONDE_T_MAX_C - 1e-6)
        above = water_activity_licl(xi, CONDE_T_MAX_C + 1e-6)
        assert above == pytest.approx(below, rel=1e-6)


def test_licl_activity_rises_past_condes_cap_instead_of_clamping():
    """The bug this fixes: a_w frozen at 100 C understated p_gel by ~40% at 155 C."""
    for xi in (0.40, 0.50, 0.55):
        aw = [water_activity_licl(xi, t) for t in (100.0, 120.0, 140.0, 155.0)]
        assert all(b > a for a, b in zip(aw, aw[1:])), f"not monotonic in T at xi={xi}: {aw}"
        assert aw[-1] / aw[0] > 1.15


@pytest.mark.parametrize("t_c", [-3185.0, -300.0, 1e4])
def test_wild_temperature_iterate_stays_total(t_c):
    """The thermal Newton solve throws iterates like -3185 C at this; exp() in c
    overflowed and aborted the whole solve rather than returning a clamped answer."""
    assert 0.0 <= bet_water_activity(0.4, t_c, LICL_BET) <= 1.0
    assert 0.0 <= patek_libr_water_activity(0.4, t_c) <= 1.0


# --- Patek & Klomfar (2006), LiBr ------------------------------------------------


def _table9_mass_fraction(x: float) -> float:
    mw_libr, mw_water = 86.845, 18.015
    return x * mw_libr / (x * mw_libr + (1 - x) * mw_water)


# Table 9's sixth row (x = 0.4) is omitted: it sits at 76.3 wt%, past Patek's own
# stated 0-75 wt% validity, so LIBR_W_MAX clips it and the comparison is meaningless.
# The clip is the conservative reading of his stated range, kept even though his
# validation table steps outside it. test_patek_composition_ceiling_clips covers it.
@pytest.mark.parametrize(
    "x,t_k,p_ref",
    [(0.05, 300, 3025.1805), (0.05, 450, 835097.47), (0.1, 300, 2286.4858),
     (0.1, 450, 647702.12), (0.3, 350, 2237.3986)],
)
def test_reproduces_patek_table9(x, t_k, p_ref):
    """Their Table 9 exists to validate an implementation; x there is the MOLE fraction,
    so this also pins the mole-vs-mass basis, which is the other way to be silently
    wrong here.

    Tolerance is 0.15%, which is the Saul-Wagner vs IAPWS-95 gap in p_s -- the residual
    is systematic in temperature (+0.09% at 300 K, -0.02% at 450 K), not scatter.
    """
    w = _table9_mass_fraction(x)
    assert w <= LIBR_W_MAX
    p = patek_libr_water_activity(w, t_k - 273.15) * _saul_wagner_pa(t_k)
    assert p == pytest.approx(p_ref, rel=0.0015)


def test_patek_covers_the_whole_gel_range_by_measurement():
    """Fitted to p-T-x data below 483 K, so the ceiling is data-backed, not extrapolated.

    This is the point of replacing the BET, which stopped at 95 C of LiBr data.
    """
    assert isotherm_t_max_c("LiBr") == LIBR_T_MAX_C > BET_T_MAX_C
    for t in (25.0, 100.0, 155.0, LIBR_T_MAX_C):
        aw = [patek_libr_water_activity(w, t) for w in (0.40, 0.50, 0.60, 0.70)]
        assert all(0.0 < a < 1.0 for a in aw), (t, aw)
        assert all(b < a for a, b in zip(aw, aw[1:])), f"not monotone in w at {t} C: {aw}"


def test_patek_composition_ceiling_clips_rather_than_extrapolating():
    """75 wt% is Patek's stated limit; above ~142 C the crystallization line exceeds it."""
    assert patek_libr_water_activity(0.80, 155.0) == patek_libr_water_activity(LIBR_W_MAX, 155.0)
    assert saturation_brine_salt_fraction("LiBr", 155.0) > LIBR_W_MAX
    # Table 9's x = 0.4 row is itself past the limit, so we clip where Patek did not.
    assert _table9_mass_fraction(0.4) > LIBR_W_MAX


@pytest.mark.parametrize("salt", ["LiCl", "LiBr"])
def test_isotherm_and_h_des_finite_to_the_gel_ceiling(salt):
    """~157 C is reachable in the evacuated two-pane case; nothing may go NaN there."""
    for t in (25.0, 100.0, 120.0, isotherm_t_max_c(salt)):
        xi = 0.9 * saturation_brine_salt_fraction(salt, t)
        aw = water_activity_at_brine_fraction(salt, xi, t)
        assert 0.0 < aw < 1.0, f"{salt} a_w={aw} at {t} C"
        h = isosteric_h_des_j_per_kg(salt, xi, t)
        assert 2.0e6 < h < 3.5e6, f"{salt} h_des={h} at {t} C"


@pytest.mark.parametrize("t_c", [25.0, 80.0, 155.0])
@pytest.mark.parametrize("rh", [0.2, 0.5, 0.8])
def test_libr_forward_and_inverse_round_trip(rh, t_c):
    """mf_LiBr is a bracketed solve, not closed form like the BET it replaced."""
    xi = equilibrate_salt_mf("LiBr", rh, t_c)
    assert 0.0 < xi < 1.0
    assert water_activity_at_brine_fraction("LiBr", xi, t_c) == pytest.approx(rh, abs=1e-6)


def test_saul_wagner_extrapolates_below_the_triple_point():
    """The Duhring path evaluates p_s at theta ~ -20 C for concentrated brine at 25 C.
    The public wrapper still guards; only the unguarded split does this."""
    for t_c, ref_pa in ((0.0, 611.2), (-10.0, 286.6), (-20.0, 125.6)):
        assert _saul_wagner_pa(t_c + 273.15) == pytest.approx(ref_pa, rel=0.005)
    assert math.isnan(__import__("solar_lumped.physics", fromlist=["x"])
                      .water_vapor_pressure_pa(-10.0))


# --- Saturation and deliquescence (Greenspan-derived crystallization line) ---------


def _greenspan_libr_drh(t_c: float) -> float:
    """Greenspan (1977) Table 1, LiBr: RH% over saturated solution, 0-100 C."""
    return (7.75437 - 0.0654994 * t_c + 0.420737e-3 * t_c**2) / 100.0


@pytest.mark.parametrize("t_c", [0.0, 10.0, 25.0, 50.0, 80.0, 100.0])
def test_libr_deliquescence_matches_greenspan_across_its_whole_range(t_c):
    """Not just at 25 C. A tabulated 25 C solubility passed the point check and still
    got the temperature TREND backwards -- DRH rose 0.029 -> 0.147 over 0-100 C where
    Greenspan falls 0.078 -> 0.054, overstating the hot plateau 2.3x at 80 C."""
    assert deliquescence_rh("LiBr", t_c) == pytest.approx(_greenspan_libr_drh(t_c), abs=0.005)


def test_libr_deliquescence_falls_with_temperature():
    """The sign of the trend, stated separately so it cannot regress silently."""
    drh = [deliquescence_rh("LiBr", t) for t in (0.0, 25.0, 50.0, 80.0, 100.0)]
    assert all(b < a for a, b in zip(drh, drh[1:])), drh
    # ...and saturation concentrates as it heats, which is what drives that.
    xi = [saturation_brine_salt_fraction("LiBr", t) for t in (0.0, 25.0, 50.0, 80.0, 155.0)]
    assert all(b > a for a, b in zip(xi, xi[1:])), xi


def test_libr_crystallization_line_stays_total_across_the_clamp():
    """A quadratic fit would put a spurious root below the real one inside (0, 1), and
    the branch selection takes the minimum -- so linear, and total across the clamp."""
    for t in (-3185.0, -200.0, 0.0, 155.0, 440.0, 1e4):
        xi = saturation_brine_salt_fraction("LiBr", t)
        assert 0.0 < xi < 1.0, f"xi_sat={xi} at {t} C"


def test_libr_is_the_weaker_desiccant_per_unit_salt_mass():
    """Sanity on the new salt: LiBr holds roughly half LiCl's water at equal RH."""
    for rh in (0.2, 0.4, 0.6):
        g_per_g = {
            s: (1.0 - x) / x
            for s in ("LiCl", "LiBr")
            for x in (equilibrate_salt_mf(s, rh, 25.0),)
        }
        ratio = g_per_g["LiBr"] / g_per_g["LiCl"]
        assert 0.3 < ratio < 0.7, f"RH={rh}: LiBr/LiCl uptake ratio {ratio}"
