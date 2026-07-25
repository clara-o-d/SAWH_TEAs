from __future__ import annotations

import numpy as np
import pytest

from sawh_bayesopt.design_space import DesignBounds, VAR_ORDER, from_unit_cube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from sawh_bayesopt.surrogate import (
    SurrogateState,
    append_observations,
    build_gp,
    check_hyperparameter_convergence,
    fit,
    fit_feasibility,
    predict,
    predict_batch,
    predict_feasibility_batch,
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


def test_fit_raises_with_fewer_than_2_feasible_points_even_if_many_total():
    bounds = DesignBounds()
    rng = np.random.default_rng(0)
    Xu = rng.uniform(0.0, 1.0, size=(10, len(VAR_ORDER)))
    y = _quadratic_bowl(Xu, np.full(len(VAR_ORDER), 0.5))
    X_raw = from_unit_cube(Xu, bounds)
    feasible = np.zeros(10, dtype=bool)
    feasible[0] = True  # only 1 feasible point among 10 evaluated designs

    state = SurrogateState(gp=build_gp(seed=0), bounds=bounds)
    state = append_observations(state, X_raw, y, feasible)
    with pytest.raises(ValueError, match="feasible"):
        fit(state)


def test_fit_uses_only_feasible_points_not_contaminated_by_infeasible_outliers():
    """A couple of extreme 'infeasible' y values (mimicking evaluator.py's
    penalty sentinel) must not affect the fitted GP at all -- fit() should
    produce the exact same model whether or not contaminated rows are present,
    since they're excluded before fitting either way."""
    center = np.full(len(VAR_ORDER), 0.4)
    bounds = DesignBounds()
    rng = np.random.default_rng(0)
    Xu = rng.uniform(0.0, 1.0, size=(20, len(VAR_ORDER)))
    y = _quadratic_bowl(Xu, center)
    X_raw = from_unit_cube(Xu, bounds)
    feasible = np.ones(20, dtype=bool)

    state_clean = fit(append_observations(
        SurrogateState(gp=build_gp(seed=0), bounds=bounds), X_raw, y, feasible,
    ))

    # Same 20 feasible points, plus 2 extreme "infeasible" outliers appended.
    Xu_outliers = rng.uniform(0.0, 1.0, size=(2, len(VAR_ORDER)))
    X_outliers = from_unit_cube(Xu_outliers, bounds)
    y_outliers = np.array([1.0e4, 5026.2])
    X_all = np.vstack([X_raw, X_outliers])
    y_all = np.concatenate([y, y_outliers])
    feasible_all = np.concatenate([feasible, np.array([False, False])])

    state_contaminated = fit(append_observations(
        SurrogateState(gp=build_gp(seed=0), bounds=bounds), X_all, y_all, feasible_all,
    ))

    x_probe = from_unit_cube(np.full(len(VAR_ORDER), 0.4), bounds)
    mu_clean, sigma_clean = predict(state_clean, x_probe)
    mu_contaminated, sigma_contaminated = predict(state_contaminated, x_probe)
    assert mu_clean == pytest.approx(mu_contaminated, rel=1e-9)
    assert sigma_clean == pytest.approx(sigma_contaminated, rel=1e-9)


def test_y_best_and_x_best_ignore_infeasible_points_even_when_numerically_smaller():
    """Guards against a mask bug: y_best/x_best must ignore infeasible rows
    by the feasibility mask, not merely because penalty values are usually
    numerically large -- construct an infeasible point with a *smaller* y
    than the true best feasible point and confirm it's still ignored."""
    bounds = DesignBounds()
    X_raw = from_unit_cube(np.array([[0.5] * len(VAR_ORDER), [0.9] * len(VAR_ORDER)]), bounds)
    y = np.array([2.0, -100.0])  # index 1 is numerically smaller but infeasible
    feasible = np.array([True, False])
    state = append_observations(SurrogateState(gp=build_gp(seed=0), bounds=bounds), X_raw, y, feasible)
    assert state.y_best == 2.0
    assert np.array_equal(state.x_best, X_raw[0])


def test_fit_feasibility_returns_none_clf_with_single_class():
    bounds = DesignBounds()
    X_raw = from_unit_cube(np.random.default_rng(0).uniform(0.0, 1.0, size=(5, len(VAR_ORDER))), bounds)
    y = np.zeros(5)
    feasible = np.ones(5, dtype=bool)  # only one class (all feasible) observed so far
    state = append_observations(SurrogateState(gp=build_gp(seed=0), bounds=bounds), X_raw, y, feasible)
    state = fit_feasibility(state)
    assert state.clf is None
    assert np.array_equal(predict_feasibility_batch(state, X_raw), np.ones(5))


def test_check_hyperparameter_convergence_flags_values_pinned_at_bounds():
    kernel = ConstantKernel(1.0, (0.5, 2.0)) * RBF(1.0, (0.5, 2.0)) + WhiteKernel(1e-5, (1e-8, 1e-2))
    kernel.theta = np.log([2.0, 2.0, 1e-8])  # constant & length_scale at upper bound, noise at lower bound
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.kernel_ = kernel  # what fit() would set; skip an actual .fit() call

    warnings = check_hyperparameter_convergence(gp)
    assert len(warnings) == 3
    assert any("k1__k1__constant_value" in w and "upper bound" in w for w in warnings)
    assert any("k1__k2__length_scale" in w and "upper bound" in w for w in warnings)
    assert any("k2__noise_level" in w and "lower bound" in w for w in warnings)


def test_check_hyperparameter_convergence_silent_when_well_interior():
    kernel = ConstantKernel(1.0, (0.5, 2.0)) * RBF(1.0, (0.5, 2.0)) + WhiteKernel(1e-5, (1e-8, 1e-2))
    kernel.theta = np.log([1.0, 1.0, 1e-5])  # comfortably interior for all three
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.kernel_ = kernel

    assert check_hyperparameter_convergence(gp) == []


def test_fit_feasibility_learns_a_real_boundary_with_both_classes_present():
    bounds = DesignBounds()
    rng = np.random.default_rng(0)
    Xu = rng.uniform(0.0, 1.0, size=(40, len(VAR_ORDER)))
    # Feasible iff the first coordinate is below 0.5 -- a clean, learnable boundary.
    feasible = Xu[:, 0] < 0.5
    assert feasible.any() and (~feasible).any()
    X_raw = from_unit_cube(Xu, bounds)
    y = np.where(feasible, _quadratic_bowl(Xu, np.full(len(VAR_ORDER), 0.25)), 1.0e4)

    state = append_observations(SurrogateState(gp=build_gp(seed=0), bounds=bounds), X_raw, y, feasible)
    state = fit_feasibility(state)
    assert state.clf is not None

    x_feasible_side = from_unit_cube(np.array([0.1] * len(VAR_ORDER)), bounds)
    x_infeasible_side = from_unit_cube(np.array([0.9] * len(VAR_ORDER)), bounds)
    p_feasible = predict_feasibility_batch(state, x_feasible_side.reshape(1, -1))[0]
    p_infeasible = predict_feasibility_batch(state, x_infeasible_side.reshape(1, -1))[0]
    assert p_feasible > 0.5
    assert p_infeasible < 0.5
