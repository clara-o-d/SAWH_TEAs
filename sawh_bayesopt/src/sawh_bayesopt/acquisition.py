"""Expected Improvement acquisition + batch proposal via Kriging-Believer. EI's landscape
is often flat/multi-modal early, so the inner maximization uses gradient-free
scipy.optimize.differential_evolution."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

from sawh_bayesopt.design_space import from_unit_cube
from sawh_bayesopt.surrogate import SurrogateState, append_observations, fit, predict


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, *, xi: float = 0.01) -> np.ndarray:
    """Minimization EI (lower combined_lcow is better)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    z = (y_best - mu - xi) / sigma_safe
    ei = (y_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(sigma > 1e-12, np.maximum(ei, 0.0), 0.0)


def _neg_ei_unit_cube(u: np.ndarray, gp, y_best: float, xi: float, clf=None) -> float:
    """Negative constrained EI: EI(x)·P(feasible(x)) when a classifier exists, plain EI
    otherwise. Without the P(feasible) term, DE would re-propose infeasible designs the
    LCOW GP is blind to (it only ever fits feasible points)."""
    u = np.asarray(u, dtype=float).reshape(1, -1)
    mu, sigma = gp.predict(u, return_std=True)
    ei = expected_improvement(mu, sigma, y_best, xi=xi)
    if clf is not None:
        p_feasible = clf.predict_proba(u)[:, list(clf.classes_).index(True)]
        ei = ei * p_feasible
    return -float(ei[0])


def propose_next(
    state: SurrogateState,
    *,
    xi: float = 0.01,
    seed: int = 0,
    maxiter: int = 1000,
    popsize: int = 40,
    record: list[dict] | None = None,
) -> np.ndarray:
    """Raw (denormalized) design vector maximizing constrained EI over the unit cube.
    ``record`` collects one success/nit/maxiter/-EI dict per DE call: an unconverged DE run
    is only an approximate EI maximizer, so apparent under-exploration may be the optimizer,
    not the landscape (surfaced by ../analysis/performance/optimization/diagnostics_bo/gp_diagnostics.py).

    The maxiter/popsize/tol of 1000/40/1e-6 replaced 200/20/1e-8 after hp_sweep_1 showed
    33-67% of DE calls hitting maxiter regardless of ei_xi -- tol=1e-8 demanded 10,000x
    tighter population uniformity than scipy's default on a flat EI landscape. Cheap, since
    a DE generation costs one GP predict() vs. the ~160s/round of JAX physics."""
    bounds_unit = [(0.0, 1.0)] * len(state.bounds.names())
    y_best = state.y_best
    result = differential_evolution(
        _neg_ei_unit_cube,
        bounds_unit,
        args=(state.gp, y_best, xi, state.clf),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=True,
        tol=1e-6,
    )
    if record is not None:
        record.append({
            "success": bool(result.success),
            "nit": int(result.nit),
            "maxiter": maxiter,
            "hit_maxiter": bool(result.nit >= maxiter),
            "message": str(result.message),
            "neg_ei": float(result.fun),
        })
    return from_unit_cube(result.x, state.bounds)


def propose_batch(
    state: SurrogateState,
    *,
    batch_size: int,
    seed: int = 0,
    xi: float = 0.01,
    maxiter: int = 1000,
    popsize: int = 40,
    record: list[dict] | None = None,
) -> list[np.ndarray]:
    """Kriging-Believer: propose_next, fantasize y=mu(x), refit a scratch GP, repeat -- a
    cheap way to get *batch_size* diverse candidates from a non-batch GP/EI without qEI.
    Fantasized points count as feasible, and the classifier is carried read-only (one more
    assumed-feasible point wouldn't move its boundary)."""
    scratch = SurrogateState(
        gp=deepcopy(state.gp),
        bounds=state.bounds,
        X_raw=state.X_raw.copy(),
        y=state.y.copy(),
        feasible=state.feasible.copy(),
        clf=state.clf,
    )
    proposals: list[np.ndarray] = []
    for i in range(batch_size):
        x_next = propose_next(scratch, xi=xi, seed=seed + i, maxiter=maxiter, popsize=popsize, record=record)
        mu, _ = predict(scratch, x_next)
        scratch = append_observations(
            scratch, x_next.reshape(1, len(scratch.bounds.names())), np.array([mu])
        )
        fit(scratch)
        proposals.append(x_next)
    return proposals
