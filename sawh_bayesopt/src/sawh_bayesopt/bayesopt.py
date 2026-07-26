"""EGO driver: Latin-hypercube init -> batched Expected-Improvement infill,
run directly against solar_lumped/gpu_sweep's JAX fast path."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from sawh_bayesopt.acquisition import propose_batch
from sawh_bayesopt.design_space import DesignBounds, latin_hypercube_design
from sawh_bayesopt.evaluator import DesignEvalResult, EvalCache, evaluate_batch
from sawh_bayesopt.sites import DEFAULT_SITES, SiteSpec, fetch_monthly_profiles
from sawh_bayesopt.surrogate import (
    SurrogateState,
    append_observations,
    build_gp,
    check_hyperparameter_convergence,
    fit,
    fit_feasibility,
)

logger = logging.getLogger(__name__)

StoppedReason = Literal["budget", "stalled"]

# Below this many feasible observations, the LCOW GP can't be fit at all
# (surrogate.py::fit requires >=2) -- fall back to pure Latin-hypercube
# exploration instead of EI-based proposal until enough accumulate. Should be
# rare (LHS init over a reasonable design space normally finds several
# feasible points immediately), but a design space where most of the box is
# infeasible shouldn't crash the optimizer, it should just explore longer.
MIN_FEASIBLE_TO_FIT = 2


@dataclass
class BayesOptConfig:
    bounds: DesignBounds = DesignBounds()
    sites: tuple[SiteSpec, ...] = DEFAULT_SITES
    combine_rule: str = "mean"
    n_init: int = 24
    n_total: int = 50
    batch_size: int = 3
    seed: int = 0
    ei_xi: float = 0.01
    stall_rel_tol: float = 0.005
    stall_rounds: int = 3
    resolution: str = "monthly"
    weather_cache_dir: str = ".weather_cache"
    # Absorber/glass IR emissivity variant (design_space.CASE_EPS_IR) -- see
    # to_device_config_kwargs. "case2" (default) matches solar_lumped's own
    # base-case physics; "case1" reproduces Wilson's original blackbody/cavity
    # approximation instead.
    case: str = "case2"
    # Inner differential_evolution search budget for each EI proposal (see
    # acquisition.propose_next) -- defaults match acquisition.py's own
    # (1000/40), overridable here so a cheap synthetic-objective test doesn't
    # have to pay real-problem DE cost just to exercise the loop's control
    # flow (see tests/test_bayesopt_loop_synthetic.py).
    de_maxiter: int = 1000
    de_popsize: int = 40


@dataclass
class BayesOptResult:
    history: list[DesignEvalResult]
    best: DesignEvalResult
    surrogate: SurrogateState
    stopped_reason: StoppedReason
    # One dict per differential_evolution call made during EI-based proposal
    # (see acquisition.propose_next's `record` param) -- success/nit/maxiter
    # for each, tagged with the round it was proposed in. Empty for rounds
    # that fell back to pure LHS exploration (no DE involved).
    de_diagnostics: list[dict]


def _to_xyf(results: list[DesignEvalResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([r.design_vector for r in results], dtype=float)
    y = np.array([r.combined_lcow for r in results], dtype=float)
    feasible = np.array([r.is_feasible for r in results], dtype=bool)
    return X, y, feasible


def _try_fit(state: SurrogateState, *, seed: int) -> tuple[SurrogateState, bool]:
    """Fit the LCOW GP if there are enough feasible points yet, and the
    feasibility classifier whenever both classes have been observed (a
    no-op, clf left as None, otherwise -- see fit_feasibility). Returns
    (state, fitted) so the caller knows whether EI-based proposal is usable
    this round or it should still fall back to exploration.
    """
    state = fit_feasibility(state, seed=seed)
    if state.n_feasible < MIN_FEASIBLE_TO_FIT:
        return state, False
    state = fit(state)
    for w in check_hyperparameter_convergence(state.gp):
        logger.warning("LCOW GP hyperparameter near optimization bound: %s", w)
    return state, True


def run_bayesopt(cfg: BayesOptConfig, run_dir: str | Path) -> BayesOptResult:
    from solar_lumped.economics import LCOEconomicParams

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    econ = LCOEconomicParams()

    site_profiles = {
        s.name: fetch_monthly_profiles(s, cache_dir=cfg.weather_cache_dir) for s in cfg.sites
    }
    cache = EvalCache(run_dir / "cache.jsonl")

    def _evaluate(xs: list[np.ndarray]) -> list[DesignEvalResult]:
        return evaluate_batch(
            xs,
            cache=cache,
            sites=cfg.sites,
            site_profiles=site_profiles,
            econ=econ,
            combine_rule=cfg.combine_rule,
            resolution=cfg.resolution,
            case=cfg.case,
        )

    X0 = latin_hypercube_design(cfg.n_init, cfg.bounds, seed=cfg.seed, reject_gap_degenerate=True)
    history = _evaluate(list(X0))

    state = SurrogateState(gp=build_gp(seed=cfg.seed), bounds=cfg.bounds)
    X_all, y_all, feasible_all = _to_xyf(history)
    state = append_observations(state, X_all, y_all, feasible_all)
    state, fitted = _try_fit(state, seed=cfg.seed)

    best_so_far = state.y_best
    stall_count = 0
    stopped_reason: StoppedReason = "budget"
    de_diagnostics: list[dict] = []
    round_idx = 0

    while len(history) < cfg.n_total:
        remaining = cfg.n_total - len(history)
        batch_n = min(cfg.batch_size, remaining)

        if fitted:
            round_record: list[dict] = []
            batch = propose_batch(
                state, batch_size=batch_n, seed=cfg.seed + len(history), xi=cfg.ei_xi, record=round_record,
                maxiter=cfg.de_maxiter, popsize=cfg.de_popsize,
            )
            for d in round_record:
                d["round"] = round_idx
            de_diagnostics.extend(round_record)
        else:
            # Not enough feasible observations yet to fit the LCOW GP --
            # keep exploring blindly (this design space region's feasibility
            # is still unknown) rather than proposing via a surrogate that
            # can't exist yet.
            batch = list(
                latin_hypercube_design(batch_n, cfg.bounds, seed=cfg.seed + len(history), reject_gap_degenerate=True)
            )

        new_results = _evaluate(batch)
        history.extend(new_results)

        X_new, y_new, feasible_new = _to_xyf(new_results)
        state = append_observations(state, X_new, y_new, feasible_new)
        state, fitted = _try_fit(state, seed=cfg.seed)

        new_best = state.y_best
        if fitted and math.isfinite(best_so_far) and best_so_far != 0.0:
            rel_improve = (best_so_far - new_best) / abs(best_so_far)
        else:
            # Still bootstrapping (or just became fittable this round) --
            # don't let the stall counter fire off an undefined/meaningless
            # comparison against the pre-fit y_best.
            rel_improve = 1.0
        stall_count = 0 if rel_improve >= cfg.stall_rel_tol else stall_count + 1
        best_so_far = new_best

        if fitted and stall_count >= cfg.stall_rounds:
            stopped_reason = "stalled"
            break

        round_idx += 1

    best_idx = int(np.argmin([r.combined_lcow for r in history]))
    return BayesOptResult(
        history=history,
        best=history[best_idx],
        surrogate=state,
        de_diagnostics=de_diagnostics,
        stopped_reason=stopped_reason,
    )
