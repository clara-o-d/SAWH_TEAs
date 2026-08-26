"""The gel/condenser ceiling and its detector.

Regression cover for the change that replaced Wilson's 7 mm transport floor with
GEL_CONDENSER_CLEARANCE_M as the swelling cap. The old value was ~70% of a narrow vapor
gap, so it pinned the gel every cycle and set night-time uptake by fiat; the new one is a
numerical clearance and leaves near-contact to be punished by the k/L heat leak.
"""

from __future__ import annotations

import numpy as np
import pytest

from solar_lumped import simulation
from solar_lumped.physics import GEL_CONDENSER_CLEARANCE_M, SWELLING_CAP_TOL_M
from solar_lumped.simulation import SystemConfig, run_daily_cycle
from solar_lumped.weather import baseline_profile


# A narrow-gap design: 9.891 mm gap against a 3.529 mm reference thickness. Under the old
# 7 mm cap its ceiling was 2.891 mm -- below the dry floor's own reference -- and
# absorption terminated on it every single day.
NARROW = dict(hydrogel_thickness_m=0.0035289, vapor_gap_m=0.009891)


def _cap(config: SystemConfig, clearance: float) -> float:
    return max(config.vapor_gap_m - clearance, config.hydrogel_floor_thickness_m() + 1e-6)


def _absorption_end_h(config: SystemConfig) -> float:
    _y, _eta, abs_res, _des = run_daily_cycle(baseline_profile(), config, cyclic_initial=True)
    return float(np.asarray(abs_res.H)[-1])


@pytest.fixture
def narrow_config() -> SystemConfig:
    return SystemConfig(**NARROW)


def test_clearance_is_numerical_not_a_physical_setback():
    """Sized as a solver guard, not a margin. If this grows toward millimetres it is
    governing uptake again -- which is the bug this replaced."""
    assert 0.0 < GEL_CONDENSER_CLEARANCE_M <= 5e-4
    assert SWELLING_CAP_TOL_M < GEL_CONDENSER_CLEARANCE_M / 100.0


def test_ceiling_leaves_room_to_swell(narrow_config: SystemConfig):
    """With the real clearance the gel swells freely: absorption ends strictly below the
    ceiling, so the isotherm sets uptake rather than the constant."""
    h_max = _cap(narrow_config, GEL_CONDENSER_CLEARANCE_M)
    assert h_max > narrow_config.hydrogel_thickness_m  # ceiling above the reference
    assert _absorption_end_h(narrow_config) < h_max - SWELLING_CAP_TOL_M


def test_detector_fires_when_the_ceiling_actually_binds(monkeypatch, narrow_config):
    """Positive control for SiteResult.swelling_cap_bound's test. Restoring the old 7 mm
    setback pins absorption on the ceiling, and the >= h_max - tol check must catch it --
    otherwise the flag is unfalsifiable and would stay False through the failure it
    exists to report."""
    old = 0.007
    monkeypatch.setattr(simulation, "GEL_CONDENSER_CLEARANCE_M", old)
    h_max = _cap(narrow_config, old)
    assert h_max < narrow_config.hydrogel_thickness_m  # ceiling below the reference
    assert _absorption_end_h(narrow_config) >= h_max - SWELLING_CAP_TOL_M
