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


def _neg_ei_unit_cube(u: np.ndarray, gp, y_best: float, xi: float, clf=None) -> float:
    """Negative *constrained* EI: EI(x) * P(feasible(x)) when a feasibility
    classifier is available, plain EI otherwise (clf=None -- no infeasible
    observations yet, see surrogate.py::fit_feasibility). This is what keeps
    DE from proposing designs deep in a known-infeasible region: EI alone has
    no way to know a region is bad if the LCOW GP was never fit on points
    there (it's only ever fit on feasible points -- see surrogate.py::fit),
    so without this term the optimizer could repeatedly re-propose infeasible
    designs the LCOW surrogate is silently blind to.
    """
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

    ``record``, if given, gets one dict appended per call summarizing the
    inner differential_evolution optimization itself (success, nit vs.
    maxiter, final -EI). DE is gradient-free and stochastic-population-based,
    so it can silently exhaust ``maxiter`` without its internal convergence
    tolerance being satisfied; a proposal from an unconverged DE run is only
    an approximate EI maximizer, which matters when interpreting apparent
    under-exploration -- it could be the acquisition landscape genuinely
    favors exploitation, or it could be DE not finding the true (possibly
    farther-out) maximizer in time. See scripts/diagnostics/gp_diagnostics.py
    for where this is surfaced.

    maxiter/popsize/tol were previously 200/20/1e-8 -- outputs/runs/hp_sweep_1
    (scripts/hp_sweep.py) showed 33-67% of DE calls hitting maxiter without
    declaring success *regardless of ei_xi*, which is what actually explained
    that sweep's ei_xi values producing near-identical proposals: an
    under-converged inner optimizer, not real acquisition-function
    insensitivity. tol=1e-8 in particular was 10,000x tighter than
    differential_evolution's own default (0.01) -- DE declares "success" when
    the population's fitness spread relative to its mean falls under
    ``tol``, and demanding that much uniformity from a stochastic
    population-based optimizer on a possibly-flat EI landscape rarely
    happens no matter how many generations run. Raised maxiter 200->1000,
    popsize 20->40 (actual population is popsize*n_dims, so 120->240), and
    loosened tol 1e-8->1e-6 (still 100x tighter than scipy's default, not
    fully reverting to it) -- cheap to do since each DE generation only costs
    a GP predict() (milliseconds), nowhere near the ~160s/round the JAX
    physics evaluate_batch() calls cost.
    """
    bounds_unit = [(0.0, 1.0)] * len(VAR_ORDER)
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
    """Kriging-Believer: propose_next, fantasize y=mu(x) there, refit a scratch
    GP, repeat -- cheap way to get *batch_size* diverse parallel candidates
    from a vanilla (non-batch) GP/EI without a qEI dependency.

    Fantasized points are always treated as feasible (a believed-good real
    LCOW estimate, not a real evaluation -- see append_observations). The
    feasibility classifier itself is carried over read-only, not refit each
    iteration: it's only ever consulted (predict_proba) inside the DE
    objective, and one more assumed-feasible point wouldn't meaningfully move
    its decision boundary anyway.
    """
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
        scratch = append_observations(scratch, x_next.reshape(1, len(VAR_ORDER)), np.array([mu]))
        fit(scratch)
        proposals.append(x_next)
    return proposals
