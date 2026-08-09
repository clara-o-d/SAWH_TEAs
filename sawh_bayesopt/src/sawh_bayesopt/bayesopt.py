"""EGO driver: Latin-hypercube init -> batched Expected-Improvement infill,
run directly against solar_lumped/gpu_sweep's JAX fast path."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from sawh_bayesopt.acquisition import propose_batch
from sawh_bayesopt.design_space import DesignBounds, latin_hypercube_design
from sawh_bayesopt.evaluator import DesignEvalResult, EvalCache, evaluate_batch
from sawh_bayesopt.sites import ATACAMA, SiteSpec, fetch_daily_profiles, fetch_site_frame
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

# Below this many feasible observations the LCOW GP can't be fit, so proposal falls back
# to pure LHS exploration -- a mostly-infeasible box should explore longer, not crash.
MIN_FEASIBLE_TO_FIT = 2


@dataclass
class BayesOptConfig:
    bounds: DesignBounds = DesignBounds()
    # Exactly one site: see evaluator.site_lcow_or_penalty for why optimization is
    # single-site. A 1-tuple rather than a bare SiteSpec so per-site plumbing
    # (profiles, frames, SiteResult) keeps its existing shape.
    sites: tuple[SiteSpec, ...] = (ATACAMA,)
    n_init: int = 24
    n_total: int = 50
    batch_size: int = 3
    seed: int = 0
    ei_xi: float = 0.01
    stall_rel_tol: float = 0.005
    stall_rounds: int = 3
    weather_cache_dir: str = ".weather_cache"
    # IR emissivity variant (design_space.CASE_EPS_IR): "case2" matches solar_lumped's
    # base case, "case1" Wilson's original blackbody/cavity approximation.
    case: str = "case2"
    # Complex fidelity (solar_lumped.complex_model): 13 design dims evaluated on the
    # CPU ODE path. Must agree with ``bounds.complex_mode``; the JAX fast path is
    # LiCl-hardcoded and cannot represent glazing stacks or ZSR blends.
    complex_mode: bool = False
    # False (default): solar_lumped's Eq. 2 condenser ODE. True: T_cond == T_amb, the
    # infinite-cooling-capacity limit, in either fidelity mode.
    condenser_tracks_ambient: bool = False
    # "jax" batches every (design, site) into one vmapped call -- the backend for
    # global sweeps. "cpu" is sequential and needs no GPU stack, which is what a
    # single-site study wants. Both support simple and complex fidelity.
    backend: str = "jax"
    # Inner differential_evolution budget per EI proposal; matches acquisition.py's
    # defaults (1000/40) but overridable so synthetic tests don't pay real DE cost.
    de_maxiter: int = 1000
    de_popsize: int = 40


@dataclass
class BayesOptResult:
    history: list[DesignEvalResult]
    best: DesignEvalResult
    surrogate: SurrogateState
    stopped_reason: StoppedReason
    # One success/nit/maxiter dict per differential_evolution call, tagged with its round.
    # Empty for rounds that fell back to pure LHS exploration.
    de_diagnostics: list[dict]


def _to_xyf(results: list[DesignEvalResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([r.design_vector for r in results], dtype=float)
    y = np.array([r.combined_lcow for r in results], dtype=float)
    feasible = np.array([r.is_feasible for r in results], dtype=bool)
    return X, y, feasible


def _try_fit(state: SurrogateState, *, seed: int) -> tuple[SurrogateState, bool]:
    """Fit the LCOW GP if enough feasible points exist, and the classifier once both classes
    are observed. Returns (state, fitted) so the caller knows if EI proposal is usable."""
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

    # Complex mode rebuilds profiles per design point (A1/B4/POA are design
    # variables that live in the profile), so it needs the raw frames instead.
    if cfg.complex_mode:
        site_profiles = {}
        site_frames = {
            s.name: fetch_site_frame(s, cache_dir=cfg.weather_cache_dir) for s in cfg.sites
        }
    else:
        site_profiles = {
            s.name: fetch_daily_profiles(s, cache_dir=cfg.weather_cache_dir) for s in cfg.sites
        }
        site_frames = None
    cache = EvalCache(run_dir / "cache.jsonl")

    def _evaluate(xs: list[np.ndarray]) -> list[DesignEvalResult]:
        return evaluate_batch(
            xs,
            cache=cache,
            sites=cfg.sites,
            site_profiles=site_profiles,
            econ=econ,
            case=cfg.case,
            complex_mode=cfg.complex_mode,
            condenser_tracks_ambient=cfg.condenser_tracks_ambient,
            site_frames=site_frames,
            backend=cfg.backend,
        )

    # A complex-mode run is hours long and otherwise prints nothing until it finishes,
    # so each round reports progress and a rate-based ETA. cache.jsonl is the ground
    # truth (one fsync'd line per completed design) -- this just saves tailing it.
    t_start = time.perf_counter()

    def _progress(label: str, n_done: int, best: float) -> None:
        elapsed = time.perf_counter() - t_start
        eta = elapsed / n_done * (cfg.n_total - n_done) if n_done else 0.0
        best_str = f"{best:.4f}" if math.isfinite(best) else "n/a"
        print(
            f"[{n_done:4d}/{cfg.n_total}] {label:<9s} best {best_str:>10s} USD/m3  "
            f"elapsed {elapsed / 60:.1f}m  eta {eta / 60:.0f}m",
            flush=True,
        )

    X0 = latin_hypercube_design(cfg.n_init, cfg.bounds, seed=cfg.seed, reject_gap_degenerate=True)
    history = _evaluate(list(X0))
    _progress("lhs-init", len(history), min((r.combined_lcow for r in history), default=float("inf")))

    # The Matern kernel is anisotropic (one length scale per dimension), so it must
    # be built at this run's width -- 6 simple, 13 complex.
    n_dims = len(cfg.bounds.names())
    state = SurrogateState(gp=build_gp(n_dims=n_dims, seed=cfg.seed), bounds=cfg.bounds)
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
            # Too few feasible observations to fit the LCOW GP -- keep exploring blindly
            # rather than proposing from a surrogate that can't exist yet.
            batch = list(
                latin_hypercube_design(batch_n, cfg.bounds, seed=cfg.seed + len(history), reject_gap_degenerate=True)
            )

        new_results = _evaluate(batch)
        history.extend(new_results)
        _progress(f"round {round_idx}", len(history), min(r.combined_lcow for r in history))

        X_new, y_new, feasible_new = _to_xyf(new_results)
        state = append_observations(state, X_new, y_new, feasible_new)
        state, fitted = _try_fit(state, seed=cfg.seed)

        new_best = state.y_best
        if fitted and math.isfinite(best_so_far) and best_so_far != 0.0:
            rel_improve = (best_so_far - new_best) / abs(best_so_far)
        else:
            # Still bootstrapping: don't let the stall counter compare against a
            # meaningless pre-fit y_best.
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
