"""Tests for site feasibility and salt LCOW simulation."""

from __future__ import annotations

import math

from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.simulation.site_feasibility import (
    FAIL_LCO,
    salt_climate_feasible,
    simulate_salt_lcow,
)
from solar_lumped.physics.salt_properties import get_salt
from solar_lumped.weather.profiles import baseline_profile


def test_drh_rejects_nacl_at_low_rh():
    salt = get_salt("NaCl")
    ok, reason = salt_climate_feasible(salt, rh_abs=0.50, t_cond_c=25.0, t_gel_c=45.0)
    assert not ok
    assert "absorption RH" in reason


def test_simulate_licl_baseline_finite():
    config = DeviceConfig.baseline(salt_name="LiCl")
    result = simulate_salt_lcow(
        baseline_profile(),
        config,
        skip_feasibility=True,
    )
    assert result.feasible
    assert result.lcow < FAIL_LCO
    assert result.yield_kg_m2 > 0.0


def test_simulate_nacl_baseline_low_rh_infeasible():
    config = DeviceConfig.baseline(salt_name="NaCl")
    result = simulate_salt_lcow(baseline_profile(relative_humidity=0.5), config)
    assert not result.feasible
    assert result.lcow >= 0.99 * FAIL_LCO
