"""Tests for Díaz-Marín Fig. 5 RH schedule loading."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_DIAZ_SCRIPTS = Path(__file__).resolve().parents[1] / "diaz-marin-et-al._re-creation" / "scripts"
if str(_DIAZ_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_DIAZ_SCRIPTS))

from chamber_rh_schedule import (  # noqa: E402
    detect_high_to_20_switch_from_reference,
    load_chamber_rh_schedules,
    load_chamber_rh_schedules_from_reference,
)


def test_reference_switch_5c_30_matches_plateau_end():
    ref_dir = _DIAZ_SCRIPTS.parent / "reference" / "figure5"
    t, u = np.loadtxt(ref_dir / "5c_30.csv", delimiter=",").T
    switch = detect_high_to_20_switch_from_reference(t, u)
    assert switch == pytest.approx(2449.2, abs=1.0)


def test_reference_switch_filters_outlier_uptake():
    t = np.array([0.0, 1000.0, 3500.0, 4000.0, 5000.0])
    u = np.array([0.0, 0.10, 0.12, 0.04, 0.01])
    u_with_outlier = u.copy()
    u_with_outlier[1] = 1.8
    assert detect_high_to_20_switch_from_reference(t, u) == pytest.approx(3500.0, abs=1.0)
    assert detect_high_to_20_switch_from_reference(t, u_with_outlier) == pytest.approx(
        3500.0, abs=1.0
    )


def test_load_reference_covers_all_figure5_panels():
    schedules = load_chamber_rh_schedules_from_reference()
    assert len(schedules) == 12
    assert ("5c", 70) in schedules
    assert schedules[("5c", 70)].t_high_to_20_min == pytest.approx(3220.3, abs=1.0)


def test_reference_5i_30_end_time_ignores_outlier_row():
    schedules = load_chamber_rh_schedules_from_reference()
    assert schedules[("5i", 30)].t_end_min == pytest.approx(4963.7, abs=1.0)


def test_default_loader_uses_reference_not_esm():
    ref = load_chamber_rh_schedules(source="reference")
    esm = load_chamber_rh_schedules(source="esm")
    assert ref[("5c", 30)].t_high_to_20_min > esm[("5c", 30)].t_high_to_20_min
    assert ref[("5c", 30)].t_end_min < esm[("5c", 30)].t_end_min
