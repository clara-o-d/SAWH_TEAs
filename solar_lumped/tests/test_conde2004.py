"""Tests for Conde (2004) LiCl/CaCl2 brine isotherms."""

from __future__ import annotations

import math

from solar_lumped.physics.conde2004 import (
    LICL_VAPOR_PRESSURE,
    equilibrium_salt_mass_fraction_licl,
    vapor_pressure_ratio,
    water_activity_licl,
    water_vapor_pressure_pa,
)
from solar_lumped.physics.salt_properties import (
    licl_equilibrium_brine_salt_fraction,
    licl_water_activity_at_brine_fraction,
    saturation_vapor_pressure_pa,
)


def test_licl_table3_parameters_match_tex():
    p = LICL_VAPOR_PRESSURE
    assert p.pi0 == 0.28
    assert p.pi1 == 4.30
    assert p.pi6 == 0.362
    assert p.pi9 == 0.03


def test_pure_water_activity_at_zero_salt_fraction():
    aw = water_activity_licl(0.0, 25.0)
    assert math.isfinite(aw)
    assert aw == 1.0


def test_licl_activity_decreases_with_salt_fraction():
    aw_dilute = water_activity_licl(0.05, 25.0)
    aw_brine = water_activity_licl(0.30, 25.0)
    assert aw_dilute > aw_brine


def test_licl_equilibrium_inverts_forward_isotherm():
    for rh in (0.20, 0.30, 0.50, 0.70):
        xi = equilibrium_salt_mass_fraction_licl(rh, 25.0)
        aw = water_activity_licl(xi, 25.0)
        assert math.isclose(aw, rh, rel_tol=0.0, abs_tol=1e-4)


def test_salt_properties_delegates_to_conde2004():
    xi = 0.25
    assert licl_water_activity_at_brine_fraction(xi, 25.0) == water_activity_licl(xi, 25.0)
    assert licl_equilibrium_brine_salt_fraction(0.50, 25.0) == equilibrium_salt_mass_fraction_licl(
        0.50, 25.0
    )


def test_saul_wagner_near_tetens_at_25c():
    p_conde = water_vapor_pressure_pa(25.0)
    p_api = saturation_vapor_pressure_pa(25.0)
    assert math.isfinite(p_conde)
    assert math.isclose(p_conde, p_api, rel_tol=0.02)


def test_fig5_relative_uptake_via_chamber_model():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    import run_solar_sim as rss

    params = rss.build_hydrogel_chamber_params(
        salt="LiCl", salt_loading=4.0, h0_mm=2.34, g_conv_m_s=0.0095
    )
    u20 = rss._dry_uptake_g_g(rss.chamber_equilibrium_c_w(params, 0.20), params)
    for rh, target in ((0.30, 0.12), (0.70, 0.91)):
        cw = rss.chamber_equilibrium_c_w(params, rh)
        rel = rss.chamber_relative_uptake(cw, params, u_baseline=u20)
        assert math.isclose(rel, target, abs_tol=0.03)
