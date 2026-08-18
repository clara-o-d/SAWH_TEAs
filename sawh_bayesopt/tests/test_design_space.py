from __future__ import annotations

import numpy as np
import pytest

from sawh_bayesopt.design_space import (
    DesignBounds,
    FULL_VAR_ORDER,
    SIMPLE_FIXED,
    VAR_ORDER,
    VAPOR_GAP_TRANSPORT_MIN_M,
    from_unit_cube,
    is_gap_degenerate,
    latin_hypercube_design,
    to_profile_kwargs,
    to_system_config_kwargs,
    to_unit_cube,
)


def test_var_order_matches_bounds_fields():
    bounds = DesignBounds()
    for name in VAR_ORDER:
        assert hasattr(bounds, name)


def test_to_system_config_kwargs_maps_by_name_and_pins_the_rest():
    x = np.arange(len(VAR_ORDER), dtype=float)
    kwargs = to_system_config_kwargs(x)
    # Every case (including the default, "case2") now gets an explicit
    # "thermal" override -- see design_space.py::CASE_EPS_IR. The condenser mode is
    # always passed explicitly rather than left to SystemConfig's default, so a
    # sweep cannot silently fall back to the ODE condenser -- same for the sorption
    # kinetics limit.
    assert list(kwargs.keys()) == [
        "hydrogel_thickness_m", "vapor_gap_m", "tilt_deg",
        *SIMPLE_FIXED, "condenser_tracks_ambient", "instant_equilibrium", "thermal",
    ]
    assert kwargs["condenser_tracks_ambient"] is False
    assert kwargs["instant_equilibrium"] is False
    for i, name in enumerate(VAR_ORDER):
        if name in kwargs:
            assert kwargs[name] == float(i)
    # The dims simple mode no longer optimizes are pinned, not dropped, and the two
    # schedule offsets stay out of SystemConfig entirely -- they reshape the profile.
    for name, value in SIMPLE_FIXED.items():
        assert kwargs[name] == value
    assert kwargs["thermal"].insulation_gap_m == SIMPLE_FIXED["insulation_gap_m"]
    assert "seal_offset_h" not in kwargs and "open_offset_h" not in kwargs


def test_to_profile_kwargs_carries_the_schedule_and_transposes_onto_the_design_tilt():
    x = np.arange(len(VAR_ORDER), dtype=float)
    profile = to_profile_kwargs(x)
    assert profile["seal_offset_h"] == float(VAR_ORDER.index("seal_offset_h"))
    assert profile["open_offset_h"] == float(VAR_ORDER.index("open_offset_h"))
    # POA on in simple mode too, at this design's own tilt -- not
    # weather.POA_DEFAULT_TILT_DEG, which would transpose every design onto the same
    # aperture and leave tilt a pure gap-convection knob.
    assert profile["poa_tilt_deg"] == float(VAR_ORDER.index("tilt_deg"))
    # B4's fan stays complex-only; simple mode does not price it.
    assert profile["condenser_air_speed_m_s"] == 0.0

    xc = np.arange(len(FULL_VAR_ORDER), dtype=float)
    profile_complex = to_profile_kwargs(xc, complex_mode=True)
    assert profile_complex["poa_tilt_deg"] == float(FULL_VAR_ORDER.index("tilt_deg"))
    assert profile_complex["seal_offset_h"] == float(FULL_VAR_ORDER.index("seal_offset_h"))


def test_unit_cube_round_trip():
    bounds = DesignBounds()
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = rng.uniform(0.0, 1.0, size=len(VAR_ORDER))
        x = from_unit_cube(u, bounds)
        u_back = to_unit_cube(x, bounds)
        assert np.allclose(u, u_back, atol=1e-10)


def test_unit_cube_bounds_map_to_endpoints():
    bounds = DesignBounds()
    lo = from_unit_cube(np.zeros(len(VAR_ORDER)), bounds)
    hi = from_unit_cube(np.ones(len(VAR_ORDER)), bounds)
    for name, lo_v, hi_v in zip(VAR_ORDER, lo, hi):
        expected_lo, expected_hi = getattr(bounds, name)
        assert lo_v == pytest.approx(expected_lo)
        assert hi_v == pytest.approx(expected_hi)


def test_is_gap_degenerate_true_when_gap_too_small():
    x = to_system_config_kwargs_to_array(
        {
            "hydrogel_thickness_m": 0.005,
            "vapor_gap_m": 0.005 + VAPOR_GAP_TRANSPORT_MIN_M - 0.001,  # < margin
            "tilt_deg": 30.0,
            "seal_offset_h": 0.0,
            "open_offset_h": 0.0,
        }
    )
    assert is_gap_degenerate(x)


def test_is_gap_degenerate_false_when_gap_ample():
    x = to_system_config_kwargs_to_array(
        {
            "hydrogel_thickness_m": 0.004,
            "vapor_gap_m": 0.040,
            "tilt_deg": 30.0,
            "seal_offset_h": 0.0,
            "open_offset_h": 0.0,
        }
    )
    assert not is_gap_degenerate(x)


def to_system_config_kwargs_to_array(kwargs: dict[str, float]) -> np.ndarray:
    return np.array([kwargs[name] for name in VAR_ORDER], dtype=float)


def test_latin_hypercube_design_within_bounds_and_deterministic():
    bounds = DesignBounds()
    x1 = latin_hypercube_design(30, bounds, seed=42)
    x2 = latin_hypercube_design(30, bounds, seed=42)
    assert x1.shape == (30, len(VAR_ORDER))
    assert np.array_equal(x1, x2)

    lo, hi = bounds.as_array()[:, 0], bounds.as_array()[:, 1]
    assert np.all(x1 >= lo - 1e-12)
    assert np.all(x1 <= hi + 1e-12)


def test_latin_hypercube_design_rejects_gap_degenerate_rows():
    bounds = DesignBounds()
    x = latin_hypercube_design(50, bounds, seed=7, reject_gap_degenerate=True)
    assert not any(is_gap_degenerate(row) for row in x)
