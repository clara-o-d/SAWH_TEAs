"""Expected Improvement acquisition + batch proposal via Kriging-Believer.

EI's landscape over the GP is often flat/multi-modal (especially early with
few observations), so the inner maximization uses
scipy.optimize.differential_evolution (gradient-free, already a shared
dependency) rather than a gradient method.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

from sawh_bayesopt.design_space import VAR_ORDER, from_unit_cube
from sawh_bayesopt.surrogate import SurrogateState, append_observations, fit, predict


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, *, xi: float = 0.01) -> np.ndarray:
    """Minimization EI (lower combined_lcow is better)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    z = (y_best - mu - xi) / sigma_safe
    ei = (y_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(sigma > 1e-12, np.maximum(ei, 0.0), 0.0)


def _neg_ei_unit_cube(u: np.ndarray, gp, y_best: float, xi: float) -> float:
    u = np.asarray(u, dtype=float).reshape(1, -1)
    mu, sigma = gp.predict(u, return_std=True)
    ei = expected_improvement(mu, sigma, y_best, xi=xi)
    return -float(ei[0])


def propose_next(
    state: SurrogateState,
    *,
    xi: float = 0.01,
    seed: int = 0,
    maxiter: int = 200,
    popsize: int = 20,
) -> np.ndarray:
    """Raw (denormalized) design vector maximizing EI over the unit cube."""
    bounds_unit = [(0.0, 1.0)] * len(VAR_ORDER)
    y_best = state.y_best
    result = differential_evolution(
        _neg_ei_unit_cube,
        bounds_unit,
        args=(state.gp, y_best, xi),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=True,
        tol=1e-8,
    )
    return from_unit_cube(result.x, state.bounds)


def propose_batch(
    state: SurrogateState,
    *,
    batch_size: int,
    seed: int = 0,
    xi: float = 0.01,
    maxiter: int = 200,
    popsize: int = 20,
) -> list[np.ndarray]:
    """Kriging-Believer: propose_next, fantasize y=mu(x) there, refit a scratch
    GP, repeat -- cheap way to get *batch_size* diverse parallel candidates
    from a vanilla (non-batch) GP/EI without a qEI dependency."""
    scratch = SurrogateState(
        gp=deepcopy(state.gp),
        bounds=state.bounds,
        X_raw=state.X_raw.copy(),
        y=state.y.copy(),
    )
    proposals: list[np.ndarray] = []
    for i in range(batch_size):
        x_next = propose_next(scratch, xi=xi, seed=seed + i, maxiter=maxiter, popsize=popsize)
        mu, _ = predict(scratch, x_next)
        scratch = append_observations(scratch, x_next.reshape(1, len(VAR_ORDER)), np.array([mu]))
        fit(scratch)
        proposals.append(x_next)
    return proposals
