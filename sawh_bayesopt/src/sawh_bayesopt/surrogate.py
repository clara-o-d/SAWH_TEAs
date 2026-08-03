"""Gaussian-process surrogate for combined_lcow(design_vector) plus a separate
feasibility classifier.

scikit-learn over BoTorch/GPyTorch: 6-D with a ~50-80 point budget and no GPU need.
Feasibility is modelled separately because penalized/partially-penalized LCOWs aren't real
measurements and corrupt the regression's calibration -- the GP fits only fully-feasible
designs, a GaussianProcessClassifier learns P(feasible | x) from all of them, and
acquisition.py weights EI by it ("constrained EI").

# ponytail: sklearn GP, ~50-80 point budget; move to BoTorch for batch qEI or multi-output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from sawh_bayesopt.design_space import DesignBounds, VAR_ORDER, to_unit_cube


def build_gp(
    *,
    n_dims: int = len(VAR_ORDER),
    n_restarts_optimizer: int = 10,
    seed: int = 0,
) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * n_dims,
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(1e-3, (1e-8, 1e-1))
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=seed,
    )


def build_gp_classifier(
    *,
    n_dims: int = len(VAR_ORDER),
    n_restarts_optimizer: int = 10,
    seed: int = 0,
) -> GaussianProcessClassifier:
    """P(feasible | x). No WhiteKernel: GaussianProcessClassifier's Laplace approximation
    handles noise itself, unlike the regressor above."""
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * n_dims,
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    )
    return GaussianProcessClassifier(
        kernel=kernel,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=seed,
    )


@dataclass
class SurrogateState:
    gp: GaussianProcessRegressor
    bounds: DesignBounds
    # Width is the run's design dimensionality (6 simple / 13 complex); the empty
    # default is reshaped to match on the first append_observations call.
    X_raw: np.ndarray = field(default_factory=lambda: np.zeros((0, len(VAR_ORDER))))
    y: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    # True iff y is a real LCOW measurement (all sites feasible). gp fits on
    # X_raw[feasible]/y[feasible]; clf fits on all X_raw against this array.
    feasible: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    clf: GaussianProcessClassifier | None = None

    @property
    def y_best(self) -> float:
        """Best feasible observation, or the unmasked min if nothing feasible has been seen
        yet (keeps bootstrap steps usable; rare, since LHS init usually finds several)."""
        if self.feasible.any():
            return float(np.min(self.y[self.feasible]))
        if self.y.size == 0:
            return float("inf")
        return float(np.min(self.y))

    @property
    def x_best(self) -> np.ndarray:
        if self.feasible.any():
            idx = np.flatnonzero(self.feasible)
            return self.X_raw[idx[int(np.argmin(self.y[idx]))]]
        return self.X_raw[int(np.argmin(self.y))]

    @property
    def n_feasible(self) -> int:
        return int(self.feasible.sum())


def append_observations(
    state: SurrogateState,
    X_new: np.ndarray,
    y_new: np.ndarray,
    feasible_new: np.ndarray | None = None,
) -> SurrogateState:
    """feasible_new=None treats every new point as feasible -- right for synthetic objectives
    and Kriging-Believer fantasy points; bayesopt.py must pass real flags."""
    X_new = np.asarray(X_new, dtype=float).reshape(-1, len(state.bounds.names()))
    y_new = np.asarray(y_new, dtype=float).reshape(-1)
    feasible_new = (
        np.ones(y_new.shape[0], dtype=bool) if feasible_new is None
        else np.asarray(feasible_new, dtype=bool).reshape(-1)
    )
    X_all = np.vstack([state.X_raw, X_new]) if state.X_raw.size else X_new
    y_all = np.concatenate([state.y, y_new]) if state.y.size else y_new
    feasible_all = np.concatenate([state.feasible, feasible_new]) if state.feasible.size else feasible_new
    return SurrogateState(gp=state.gp, bounds=state.bounds, X_raw=X_all, y=y_all, feasible=feasible_all, clf=state.clf)


def fit(state: SurrogateState) -> SurrogateState:
    """Fit the LCOW regression on feasible points only. Raises below 2 feasible points --
    callers should check state.n_feasible >= 2 and explore via LHS instead."""
    n_feasible = int(state.feasible.sum())
    if n_feasible < 2:
        raise ValueError(
            f"Need at least 2 feasible observations to fit the LCOW GP, have {n_feasible} "
            f"(of {state.X_raw.shape[0]} total evaluated). Fall back to exploration until this "
            "many feasible points have been observed."
        )
    mask = state.feasible
    u = to_unit_cube(state.X_raw[mask], state.bounds)
    state.gp.fit(u, state.y[mask])
    return state


def fit_feasibility(state: SurrogateState, *, seed: int = 0) -> SurrogateState:
    """Fit state.clf on every evaluated point against state.feasible. No-op (clf stays None)
    with only one class observed -- downstream that reads as P(feasible)=1, not an error."""
    if state.X_raw.shape[0] < 2 or len(np.unique(state.feasible)) < 2:
        state.clf = None
        return state
    u = to_unit_cube(state.X_raw, state.bounds)
    clf = build_gp_classifier(n_dims=u.shape[1], seed=seed)
    clf.fit(u, state.feasible)
    state.clf = clf
    return state


def check_hyperparameter_convergence(gp: GaussianProcessRegressor, *, edge_tol: float = 0.01) -> list[str]:
    """Flag fitted hyperparameters within edge_tol (relative, log space) of their bounds --
    sklearn's ConvergenceWarning made machine-checkable. A pinned hyperparameter usually
    means the bound is too tight, not that more restarts are needed."""
    # theta/bounds are the log-space fitted hyperparameters, in lockstep with the non-fixed
    # kernel_.hyperparameters (hp.n_elements slots each); sklearn exposes no .value.
    warnings_out = []
    theta, log_bounds = gp.kernel_.theta, gp.kernel_.bounds
    pos = 0
    for hp in gp.kernel_.hyperparameters:
        if hp.fixed:
            continue
        for i in range(hp.n_elements):
            v, (lo, hi) = theta[pos], log_bounds[pos]
            pos += 1
            span = hi - lo
            if span <= 0:
                continue
            dist_from_lo = (v - lo) / span
            dist_from_hi = (hi - v) / span
            suffix = "" if hp.n_elements == 1 else f"[{i}]"
            if dist_from_lo < edge_tol:
                warnings_out.append(
                    f"{hp.name}{suffix}: fitted value {np.exp(v):.4g} is within "
                    f"{dist_from_lo * 100:.2g}% of its lower bound {np.exp(lo):.4g} -- "
                    "the optimizer likely wants to go lower; consider widening the bound."
                )
            elif dist_from_hi < edge_tol:
                warnings_out.append(
                    f"{hp.name}{suffix}: fitted value {np.exp(v):.4g} is within "
                    f"{dist_from_hi * 100:.2g}% of its upper bound {np.exp(hi):.4g} -- "
                    "the optimizer likely wants to go higher; consider widening the bound."
                )
    return warnings_out


def predict(state: SurrogateState, x: np.ndarray) -> tuple[float, float]:
    """(mu, sigma) of combined_lcow at raw (un-normalized) design vector x."""
    u = to_unit_cube(np.asarray(x, dtype=float).reshape(1, -1), state.bounds)
    mu, sigma = state.gp.predict(u, return_std=True)
    return float(mu[0]), float(sigma[0])


def predict_batch(state: SurrogateState, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = to_unit_cube(np.asarray(X, dtype=float).reshape(-1, len(state.bounds.names())), state.bounds)
    mu, sigma = state.gp.predict(u, return_std=True)
    return mu, sigma


def predict_feasibility_batch(state: SurrogateState, X: np.ndarray) -> np.ndarray:
    """P(feasible) at raw design vectors X, shape (n,); all-ones when state.clf is None."""
    X = np.asarray(X, dtype=float).reshape(-1, len(state.bounds.names()))
    if state.clf is None:
        return np.ones(X.shape[0])
    u = to_unit_cube(X, state.bounds)
    proba = state.clf.predict_proba(u)
    feasible_col = list(state.clf.classes_).index(True)
    return proba[:, feasible_col]


def save_state(state: SurrogateState, path: str | Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"gp": state.gp, "clf": state.clf}, path)
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "bounds": {name: list(getattr(state.bounds, name)) for name in state.bounds.names()},
                "X_raw": state.X_raw.tolist(),
                "y": state.y.tolist(),
                "feasible": state.feasible.tolist(),
                "kernel": str(state.gp.kernel_) if hasattr(state.gp, "kernel_") else str(state.gp.kernel),
                "clf_kernel": (
                    str(state.clf.kernel_) if state.clf is not None and hasattr(state.clf, "kernel_") else None
                ),
            },
            indent=2,
        )
    )

