from __future__ import annotations

import numpy as np
import pytest

from sawh_bayesopt.design_space import (
    CASE_EPS_IR,
    CASE_SOLAR_OPTICS,
    DesignBounds,
    FULL_VAR_ORDER,
    SIMPLE_FIXED,
    VAR_GRID,
    VAR_ORDER,
    from_unit_cube,
    latin_hypercube_design,
    to_profile_kwargs,
    to_system_config_kwargs,
    to_unit_cube,
)
from solar_lumped.simulation import SystemConfig


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


def test_unit_cube_round_trip_is_exact_off_the_ordinal_grid():
    """Round-tripping is the identity on continuous dims only. VAR_GRID dims snap on the
    way out, so u -> x -> u lands on the nearest lattice point instead of coming back."""
    bounds = DesignBounds()
    rng = np.random.default_rng(0)
    continuous = [i for i, n in enumerate(VAR_ORDER) if n not in VAR_GRID]
    ordinal = [i for i, n in enumerate(VAR_ORDER) if n in VAR_GRID]
    assert ordinal, "expected seal/open_offset_h to be gridded in simple mode"
    for _ in range(20):
        u = rng.uniform(0.0, 1.0, size=len(VAR_ORDER))
        x = from_unit_cube(u, bounds)
        u_back = to_unit_cube(x, bounds)
        assert np.allclose(u[continuous], u_back[continuous], atol=1e-10)
        for i in ordinal:
            step = VAR_GRID[VAR_ORDER[i]]
            assert x[i] == pytest.approx(round(x[i] / step) * step)


def test_gridded_dims_have_no_sub_grid_resolution():
    """The staircase this snapping exists for: seal_offset_h only moves the desorption
    window when it crosses a 15-minute weather sample, so two proposals inside one 0.25 h
    block must collapse to one design vector -- and hence one cache key. The 33 levels are
    the count measured against _shifted_desorption_mask directly on Atacama weather."""
    bounds = DesignBounds()
    i = VAR_ORDER.index("seal_offset_h")
    lo, hi = bounds.seal_offset_h

    def snapped(raw_h: float) -> float:
        u = np.full(len(VAR_ORDER), 0.5)
        u[i] = (raw_h - lo) / (hi - lo)
        return float(from_unit_cube(u, bounds)[i])

    assert snapped(0.13) == snapped(0.24) == pytest.approx(0.25)
    assert snapped(0.12) == pytest.approx(0.0)

    u = np.full((400, len(VAR_ORDER)), 0.5)
    u[:, i] = np.linspace(0.0, 1.0, 400)
    levels = np.unique(from_unit_cube(u, bounds)[:, i])
    assert len(levels) == 33
    assert np.allclose(levels, np.arange(lo, hi + 1e-9, 0.25))


def test_unit_cube_bounds_map_to_endpoints():
    bounds = DesignBounds()
    lo = from_unit_cube(np.zeros(len(VAR_ORDER)), bounds)
    hi = from_unit_cube(np.ones(len(VAR_ORDER)), bounds)
    for name, lo_v, hi_v in zip(VAR_ORDER, lo, hi):
        expected_lo, expected_hi = getattr(bounds, name)
        assert lo_v == pytest.approx(expected_lo)
        assert hi_v == pytest.approx(expected_hi)


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


def test_latin_hypercube_design_cannot_start_gel_in_condenser():
    """Replaces the old is_gap_degenerate rejection test. No design in the bounds can
    start in contact, which is why the a-priori predicate was removed: the dry thickness
    is 0.6034 x hydrogel_thickness_m, so the worst corner is 6.03 mm against a 7 mm gap.
    Swelling into the ceiling is a runtime state -- SiteResult.swelling_cap_bound."""
    bounds = DesignBounds()
    x = latin_hypercube_design(200, bounds, seed=7)
    for row in x:
        cfg = SystemConfig(**to_system_config_kwargs(row, case="case2", complex_mode=False))
        assert cfg.hydrogel_floor_thickness_m() < cfg.vapor_gap_m


def test_case_optics_match_site_sweep_scenarios():
    """CASE_EPS_IR/CASE_SOLAR_OPTICS duplicate site_sweep's three optics bases so this
    module needn't import weather (and so cartopy). This is the guard on that copy: a case
    must reproduce its scenario's optics EXACTLY, all four numbers. Before
    CASE_SOLAR_OPTICS existed, case3 matched _LIMITS on the IR pair but silently kept a
    real absorber behind real glass, so the BayesOpt path could not express any of the four
    optical_limits scenarios."""
    from solar_lumped.site_sweep import SCENARIOS

    for case, scenario in (("case1", "wilson"), ("case2", "improved"), ("case3", "optical_limits")):
        sc = SCENARIOS[scenario]
        assert CASE_EPS_IR[case] == (sc.eps_abs_ir, sc.eps_glass_ir), case
        assert CASE_SOLAR_OPTICS[case] == (sc.eps_abs, sc.tau_glass), case


def test_every_scenario_is_reachable_through_the_case_flag():
    """All 8 SCENARIOS must be expressible as (case, instant_equilibrium,
    condenser_ambient) -- that mapping is what sbatch_bayesopt_global_12deg.sh derives its
    per-task flags from, so an unreachable scenario there means a silently wrong campaign."""
    from solar_lumped.site_sweep import SCENARIOS

    for name, sc in SCENARIOS.items():
        matches = [
            c for c in CASE_EPS_IR
            if CASE_EPS_IR[c] == (sc.eps_abs_ir, sc.eps_glass_ir)
            and CASE_SOLAR_OPTICS[c] == (sc.eps_abs, sc.tau_glass)
        ]
        assert len(matches) >= 1, f"{name} has no case with matching optics"


def test_perturbed_neighbors_land_on_the_grid():
    """The first campaign's actual defect: verification promoted a perturbed neighbor at
    31% of sites, and those neighbors skipped the snapping that proposals got, so 131 of
    435 reported optima carried an off-lattice schedule offset. Both birth sites for a
    design vector must snap."""
    from sawh_bayesopt.verification import _perturbed_neighbors

    bounds = DesignBounds()
    i, j = VAR_ORDER.index("seal_offset_h"), VAR_ORDER.index("open_offset_h")
    x_best = from_unit_cube(np.full(len(VAR_ORDER), 0.5), bounds)
    neighbors = _perturbed_neighbors(x_best, bounds, n=25, frac=0.10, seed=0)
    assert len(neighbors) == 25
    for x in neighbors:
        for k in (i, j):
            assert x[k] == pytest.approx(round(x[k] / 0.25) * 0.25), x[k]
    # ...and the perturbation still moves: snapping must not collapse every neighbor onto
    # x_best, which would make verification a no-op.
    assert len({(round(x[i], 6), round(x[j], 6)) for x in neighbors}) > 1
