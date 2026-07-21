"""EGO driver: Latin-hypercube init -> batched Expected-Improvement infill,
run directly against solar_lumped/gpu_sweep's JAX fast path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from sawh_bayesopt.acquisition import propose_batch
from sawh_bayesopt.design_space import DesignBounds, latin_hypercube_design
from sawh_bayesopt.evaluator import DesignEvalResult, EvalCache, evaluate_batch
from sawh_bayesopt.sites import DEFAULT_SITES, SiteSpec, fetch_monthly_profiles
from sawh_bayesopt.surrogate import SurrogateState, append_observations, build_gp, fit

StoppedReason = Literal["budget", "stalled"]


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


@dataclass
class BayesOptResult:
    history: list[DesignEvalResult]
    best: DesignEvalResult
    surrogate: SurrogateState
    stopped_reason: StoppedReason


def _to_xy(results: list[DesignEvalResult]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([r.design_vector for r in results], dtype=float)
    y = np.array([r.combined_lcow for r in results], dtype=float)
    return X, y


def run_bayesopt(cfg: BayesOptConfig, run_dir: str | Path) -> BayesOptResult:
    from solar_lumped.economics.params import LCOEconomicParams

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
        )

    X0 = latin_hypercube_design(cfg.n_init, cfg.bounds, seed=cfg.seed, reject_gap_degenerate=True)
    history = _evaluate(list(X0))

    state = SurrogateState(gp=build_gp(seed=cfg.seed), bounds=cfg.bounds)
    X_all, y_all = _to_xy(history)
    state = fit(append_observations(state, X_all, y_all))

    best_so_far = state.y_best
    stall_count = 0
    stopped_reason: StoppedReason = "budget"

    while len(history) < cfg.n_total:
        remaining = cfg.n_total - len(history)
        batch_n = min(cfg.batch_size, remaining)
        batch = propose_batch(state, batch_size=batch_n, seed=cfg.seed + len(history), xi=cfg.ei_xi)
        new_results = _evaluate(batch)
        history.extend(new_results)

        X_new, y_new = _to_xy(new_results)
        state = fit(append_observations(state, X_new, y_new))

        new_best = state.y_best
        if math.isfinite(best_so_far) and best_so_far != 0.0:
            rel_improve = (best_so_far - new_best) / abs(best_so_far)
        else:
            rel_improve = 1.0
        stall_count = 0 if rel_improve >= cfg.stall_rel_tol else stall_count + 1
        best_so_far = new_best

        if stall_count >= cfg.stall_rounds:
            stopped_reason = "stalled"
            break

    best_idx = int(np.argmin([r.combined_lcow for r in history]))
    return BayesOptResult(
        history=history,
        best=history[best_idx],
        surrogate=state,
        stopped_reason=stopped_reason,
    )
