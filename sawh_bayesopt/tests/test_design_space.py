from __future__ import annotations

import numpy as np
import pytest

from sawh_bayesopt.design_space import (
    DesignBounds,
    VAR_ORDER,
    VAPOR_GAP_TRANSPORT_MIN_M,
    from_unit_cube,
    is_gap_degenerate,
    latin_hypercube_design,
    to_device_config_kwargs,
    to_unit_cube,
)


def test_var_order_matches_bounds_fields():
    bounds = DesignBounds()
    for name in VAR_ORDER:
        assert hasattr(bounds, name)


def test_to_device_config_kwargs_maps_in_order():
    x = np.arange(len(VAR_ORDER), dtype=float)
    kwargs = to_device_config_kwargs(x)
    assert list(kwargs.keys()) == list(VAR_ORDER)
    assert kwargs[VAR_ORDER[0]] == 0.0
    assert kwargs[VAR_ORDER[-1]] == len(VAR_ORDER) - 1


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
    x = to_device_config_kwargs_to_array(
        {
            "hydrogel_thickness_m": 0.005,
            "vapor_gap_m": 0.005 + VAPOR_GAP_TRANSPORT_MIN_M - 0.001,  # < margin
            "insulation_gap_m": 0.005,
            "fin_area_ratio": 7.0,
            "tilt_deg": 30.0,
            "salt_to_polymer_ratio": 4.0,
        }
    )
    assert is_gap_degenerate(x)


def test_is_gap_degenerate_false_when_gap_ample():
    x = to_device_config_kwargs_to_array(
        {
            "hydrogel_thickness_m": 0.004,
            "vapor_gap_m": 0.040,
            "insulation_gap_m": 0.005,
            "fin_area_ratio": 7.0,
            "tilt_deg": 30.0,
            "salt_to_polymer_ratio": 4.0,
        }
    )
    assert not is_gap_degenerate(x)


def to_device_config_kwargs_to_array(kwargs: dict[str, float]) -> np.ndarray:
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
