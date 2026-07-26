"""Gaussian-process surrogate for combined_lcow(design_vector), plus a
separate feasibility classifier.

scikit-learn's GaussianProcessRegressor, not BoTorch/GPyTorch: the problem is
6-D with a ~50-80 point evaluation budget, no GPU/multi-fidelity need, and
scikit-learn+scipy are already shared, lightweight dependencies across every
sibling solar_lumped / waste-heat/lumped package. BoTorch is the natural
upgrade for principled batch qEI or a joint multi-output (both-sites) model
if a future version needs it.

Feasibility is modeled separately from LCOW, not folded into a single
regression: infeasible designs get a finite sentinel LCOW
(evaluator.PENALTY_LCOW_USD_PER_M3), and with combine_rule="mean" a
partially-infeasible design (one site ok, one site penalized) lands
somewhere between a real value and the sentinel -- neither is a real LCOW
measurement, and feeding either into the LCOW regression corrupts its
calibration (see gpu_sweep/../sawh_bayesopt diagnostics: a couple of such
points out of 39 pushed standardized_residual_std to ~86 when they should
have been excluded). The LCOW GP here fits *only* on fully-feasible designs;
a separate GaussianProcessClassifier learns P(feasible | x) from every
evaluated design (feasible and infeasible both -- that's exactly the signal
a classifier needs), and acquisition.py weights EI by that probability
("constrained EI") so proposals near known-infeasible regions are naturally
discouraged without ever training the regression on contaminated values.
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
    """P(feasible | x). No WhiteKernel term: GaussianProcessClassifier fits a
    latent GP via Laplace approximation with its own noise handling, unlike
    the regressor above which needs an explicit observation-noise term.
    """
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
    X_raw: np.ndarray = field(default_factory=lambda: np.zeros((0, len(VAR_ORDER))))
    y: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    # True iff the corresponding y is a real, uncontaminated LCOW measurement
    # (all sites feasible for that design) -- see module docstring. gp is fit
    # on X_raw[feasible]/y[feasible] only; clf is fit on all of X_raw against
    # this same array.
    feasible: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    clf: GaussianProcessClassifier | None = None

    @property
    def y_best(self) -> float:
        """Best *feasible* observation. Falls back to the unmasked min only
        if literally nothing feasible has been observed yet (keeps this
        finite/usable during the earliest bootstrap steps rather than
        crashing), which should be rare -- LHS init over a reasonable design
        space normally finds several feasible points immediately.
        """
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
    """feasible_new=None means "treat every new point as feasible" -- the
    right default for synthetic/test objectives with no feasibility concept,
    and for acquisition.py's Kriging-Believer fantasized points (which stand
    in for a believed-good real observation). Real evaluator.py-backed
    callers (bayesopt.py) should always pass real feasibility flags.
    """
    X_new = np.asarray(X_new, dtype=float).reshape(-1, len(VAR_ORDER))
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
    """Fits the LCOW regression on feasible points only. Raises if fewer than
    2 feasible points exist yet -- callers (bayesopt.py's main loop) should
    check state.n_feasible >= 2 before calling this, and fall back to pure
    exploration (Latin hypercube) otherwise rather than treating this as a
    real error.
    """
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
    """Fits (or refits) state.clf on every evaluated point (feasible and
    infeasible both) against state.feasible. A no-op (clf left as None) if
    only one class has been observed so far -- GaussianProcessClassifier
    can't fit with a single class, and "no evidence of infeasibility
    anywhere yet" is correctly represented downstream as P(feasible)=1
    everywhere (see predict_feasibility_batch), not an error.
    """
    if state.X_raw.shape[0] < 2 or len(np.unique(state.feasible)) < 2:
        state.clf = None
        return state
    u = to_unit_cube(state.X_raw, state.bounds)
    clf = build_gp_classifier(seed=seed)
    clf.fit(u, state.feasible)
    state.clf = clf
    return state


def check_hyperparameter_convergence(gp: GaussianProcessRegressor, *, edge_tol: float = 0.01) -> list[str]:
    """Flags fitted hyperparameters that landed within edge_tol (relative,
    in log space) of their optimization bounds.

    n_restarts_optimizer=10 (see build_gp above) restarts sklearn's internal
    L-BFGS-B from 10 random points and keeps the best log marginal likelihood
    found, but that's still a local, gradient-based search that can silently
    converge to a boundary rather than a true interior optimum (sklearn's own
    ConvergenceWarning fires for exactly this case -- this is the same
    signal, made an explicit, machine-checkable diagnostic instead of
    something you only notice by reading warnings scroll by). A
    hyperparameter pinned at its bound usually means the search wants a wider
    bound, not just more restarts -- see length_scale_bounds/noise_level
    bounds in build_gp/build_gp_classifier.
    """
    # gp.kernel_.theta/.bounds are the fitted hyperparameters and their bounds
    # in log space, in lockstep order with the non-fixed entries of
    # gp.kernel_.hyperparameters (each contributing hp.n_elements slots, e.g.
    # a per-dimension length_scale contributes n_dims slots) -- reading
    # .value directly off each Hyperparameter isn't available in sklearn's
    # namedtuple, so theta/bounds is the supported way to get fitted values.
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
    u = to_unit_cube(np.asarray(X, dtype=float).reshape(-1, len(VAR_ORDER)), state.bounds)
    mu, sigma = state.gp.predict(u, return_std=True)
    return mu, sigma


def predict_feasibility_batch(state: SurrogateState, X: np.ndarray) -> np.ndarray:
    """P(feasible) at raw design vectors X, shape (n,). All-ones if state.clf
    is None (no evidence of infeasibility observed yet -- see fit_feasibility).
    """
    X = np.asarray(X, dtype=float).reshape(-1, len(VAR_ORDER))
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
                "bounds": {name: list(getattr(state.bounds, name)) for name in VAR_ORDER},
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


def load_state(path: str | Path) -> SurrogateState:
    import joblib

    path = Path(path)
    models = joblib.load(path)
    sidecar = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    bounds = DesignBounds(**{name: tuple(v) for name, v in sidecar["bounds"].items()})
    n = len(sidecar["y"])
    feasible = np.array(sidecar.get("feasible", [True] * n), dtype=bool)  # old saves predate feasibility tracking
    return SurrogateState(
        gp=models["gp"],
        bounds=bounds,
        X_raw=np.array(sidecar["X_raw"], dtype=float),
        y=np.array(sidecar["y"], dtype=float),
        feasible=feasible,
        clf=models.get("clf"),
    )
