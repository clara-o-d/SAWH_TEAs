from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from sawh_bayesopt.acquisition import expected_improvement, propose_batch, propose_next
from sawh_bayesopt.design_space import DesignBounds, VAR_ORDER, from_unit_cube, to_unit_cube
from sawh_bayesopt.surrogate import SurrogateState, append_observations, build_gp, fit


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
