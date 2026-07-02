"""Tests for ported brine equilibrium isotherms."""

from __future__ import annotations

import math

import pytest

from solar_lumped.physics.brine_equilibrium import (
    equilibrate_salt_mf,
    mf_NaCl,
    water_activity_at_brine_fraction,
)
from solar_lumped.physics.salt_properties import get_salt


def test_mf_nacl_finite_at_high_rh():
    mf = mf_NaCl(0.80)
    assert math.isfinite(mf) and 0.0 < mf < 1.0


def test_equilibrate_nacl_below_drh_is_nan():
    assert math.isnan(equilibrate_salt_mf("NaCl", 0.70))


def test_nacl_aw_decreases_with_salt_fraction():
    aw_dilute = water_activity_at_brine_fraction("NaCl", 0.05)
    aw_brine = water_activity_at_brine_fraction("NaCl", 0.20)
    assert math.isfinite(aw_dilute) and math.isfinite(aw_brine)
    assert aw_dilute > aw_brine


def test_catalog_drh_matches_electrolyte():
    nacl = get_salt("NaCl")
    assert nacl.rh_min == 0.757
    assert nacl.h_des_j_per_kg == 2_500_000
    assert nacl.price_usd_per_kg == pytest.approx(0.045)
