from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from sawh_bayesopt.acquisition import expected_improvement, propose_batch, propose_next
from sawh_bayesopt.design_space import DesignBounds, VAR_ORDER, from_unit_cube, to_unit_cube
from sawh_bayesopt.surrogate import SurrogateState, append_observations, build_gp, fit, fit_feasibility


def test_expected_improvement_zero_when_certain():
    mu = np.array([1.0, 2.0])
    sigma = np.array([0.0, 0.0])
    ei = expected_improvement(mu, sigma, y_best=0.5)
    assert np.allclose(ei, 0.0)


def test_expected_improvement_positive_when_uncertain_near_best():
    mu = np.array([0.4])
    sigma = np.array([0.2])
    ei = expected_improvement(mu, sigma, y_best=0.5)
    assert ei[0] > 0.0


def _quadratic_bowl(u: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.sum((u - center) ** 2, axis=-1)


def _fitted_state(seed: int, n: int, center: np.ndarray) -> SurrogateState:
    bounds = DesignBounds()
    rng = np.random.default_rng(seed)
    Xu = rng.uniform(0.0, 1.0, size=(n, len(VAR_ORDER)))
    y = _quadratic_bowl(Xu, center)
    X_raw = from_unit_cube(Xu, bounds)
    state = SurrogateState(gp=build_gp(seed=seed), bounds=bounds)
    return fit(append_observations(state, X_raw, y))


def test_propose_next_within_bounds():
    state = _fitted_state(seed=0, n=15, center=np.full(len(VAR_ORDER), 0.5))
    x_next = propose_next(state, seed=0, maxiter=30, popsize=8)

    bounds_arr = state.bounds.as_array()
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
    assert np.all(x_next >= lo - 1e-9)
    assert np.all(x_next <= hi + 1e-9)


def test_propose_batch_returns_distinct_points():
    state = _fitted_state(seed=1, n=15, center=np.full(len(VAR_ORDER), 0.5))
    batch = propose_batch(state, batch_size=3, seed=1, maxiter=30, popsize=8)

    assert len(batch) == 3
    us = np.array([to_unit_cube(x, state.bounds) for x in batch])
    dists = pdist(us)
    assert np.all(dists > 1e-6)


def test_neg_ei_unit_cube_suppressed_by_low_feasibility_probability():
    from sawh_bayesopt.acquisition import _neg_ei_unit_cube

    state = _fitted_state(seed=0, n=15, center=np.full(len(VAR_ORDER), 0.5))
    u = np.full(len(VAR_ORDER), 0.5)

    class _AlwaysFeasible:
        classes_ = [False, True]

        def predict_proba(self, u):
            return np.tile([0.0, 1.0], (len(u), 1))

    class _AlwaysInfeasible:
        classes_ = [False, True]

        def predict_proba(self, u):
            return np.tile([1.0, 0.0], (len(u), 1))

    neg_ei_unconstrained = _neg_ei_unit_cube(u, state.gp, state.y_best, 0.01, clf=None)
    neg_ei_feasible = _neg_ei_unit_cube(u, state.gp, state.y_best, 0.01, clf=_AlwaysFeasible())
    neg_ei_infeasible = _neg_ei_unit_cube(u, state.gp, state.y_best, 0.01, clf=_AlwaysInfeasible())

    assert neg_ei_feasible == pytest.approx(neg_ei_unconstrained)
    assert neg_ei_infeasible == pytest.approx(0.0)  # EI * P(feasible)=0 -> exactly 0


def test_propose_next_avoids_region_a_classifier_flags_infeasible():
    """A classifier that flags the whole upper half of one dimension as
    infeasible should keep DE from proposing there, even though the
    unconstrained EI optimum (a flat/uninformative GP here) has no preference."""
    bounds = DesignBounds()
    rng = np.random.default_rng(0)
    Xu = rng.uniform(0.0, 1.0, size=(15, len(VAR_ORDER)))
    y = np.sum((Xu - 0.5) ** 2, axis=-1)
    X_raw = from_unit_cube(Xu, bounds)
    feasible = Xu[:, 0] < 0.5

    state = SurrogateState(gp=build_gp(seed=0), bounds=bounds)
    state = fit(append_observations(state, X_raw, y, feasible))
    state = fit_feasibility(state)
    assert state.clf is not None

    x_next = propose_next(state, seed=0, maxiter=60, popsize=16)
    u_next = to_unit_cube(x_next, bounds)
    assert u_next[0] < 0.5
