from __future__ import annotations

import numpy as np

from sawh_bayesopt.design_space import DesignBounds, VAR_ORDER, from_unit_cube
from sawh_bayesopt.surrogate import (
    SurrogateState,
    append_observations,
    build_gp,
    fit,
    predict,
    predict_batch,
)


def _quadratic_bowl(u: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.sum((u - center) ** 2, axis=-1)


def _fit_on_quadratic(seed: int, n: int, center: np.ndarray) -> tuple[SurrogateState, np.ndarray, np.ndarray]:
    bounds = DesignBounds()
    rng = np.random.default_rng(seed)
    Xu = rng.uniform(0.0, 1.0, size=(n, len(VAR_ORDER)))
    y = _quadratic_bowl(Xu, center)
    X_raw = from_unit_cube(Xu, bounds)
    state = SurrogateState(gp=build_gp(seed=seed), bounds=bounds)
    state = fit(append_observations(state, X_raw, y))
    return state, X_raw, y


def test_gp_predictions_correlate_with_true_function_on_training_points():
    center = np.full(len(VAR_ORDER), 0.4)
    state, X_raw, y = _fit_on_quadratic(seed=0, n=25, center=center)

    mu_train, _ = predict_batch(state, X_raw)
    corr = np.corrcoef(mu_train, y)[0, 1]
    assert corr > 0.9


def test_gp_predicts_lower_near_bowl_center_than_far_corner():
    center = np.full(len(VAR_ORDER), 0.4)
    state, _, _ = _fit_on_quadratic(seed=0, n=25, center=center)
    bounds = state.bounds

    x_center = from_unit_cube(center, bounds)
    x_far = from_unit_cube(np.zeros(len(VAR_ORDER)), bounds)

    mu_center, _ = predict(state, x_center)
    mu_far, _ = predict(state, x_far)
    assert mu_center < mu_far


def test_surrogate_state_y_best_and_x_best():
    center = np.full(len(VAR_ORDER), 0.5)
    state, X_raw, y = _fit_on_quadratic(seed=2, n=10, center=center)
    assert state.y_best == float(np.min(y))
    assert np.array_equal(state.x_best, X_raw[int(np.argmin(y))])
