"""Re-evaluate a BayesOpt result's reported optimum directly on the true
model, plus a few perturbed neighbors, to flag surrogate artifacts rather
than trusting the GP's optimum blindly -- in the same empirical-sanity-check
spirit as the ZSR track's sanity_check.py (verify a claimed optimum by direct
re-simulation, don't just trust the model)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sawh_bayesopt.bayesopt import BayesOptConfig, BayesOptResult
from sawh_bayesopt.design_space import VAR_ORDER
from sawh_bayesopt.evaluator import DesignEvalResult, EvalCache, evaluate_batch
from sawh_bayesopt.sites import fetch_monthly_profiles
from sawh_bayesopt.surrogate import predict


@dataclass
class VerificationReport:
    best_design_vector: tuple[float, ...]
    best_true_combined_lcow: float
    best_surrogate_mu: float
    best_surrogate_sigma: float
    neighbor_results: list[DesignEvalResult]
    neighbor_combined_lcows: list[float]
    max_neighbor_improvement_frac: float
    flagged_as_surrogate_artifact: bool


def _perturbed_neighbors(
    x_best: np.ndarray,
    bounds,
    *,
    n: int,
    frac: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    bounds_arr = bounds.as_array()
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
    span = hi - lo
    out = []
    for _ in range(n):
        delta = rng.uniform(-frac, frac, size=len(VAR_ORDER)) * span
        out.append(np.clip(x_best + delta, lo, hi))
    return out


def verify_optimum(
    result: BayesOptResult,
    cfg: BayesOptConfig,
    run_dir: str | Path,
    *,
    n_neighbors: int = 5,
    perturbation_frac: float = 0.10,
    seed: int = 0,
    artifact_tolerance: float = 0.02,
) -> VerificationReport:
    from solar_lumped.economics.params import LCOEconomicParams

    run_dir = Path(run_dir)
    econ = LCOEconomicParams()
    site_profiles = {
        s.name: fetch_monthly_profiles(s, cache_dir=cfg.weather_cache_dir) for s in cfg.sites
    }
    cache = EvalCache(run_dir / "cache.jsonl")

    x_best = np.array(result.best.design_vector, dtype=float)
    neighbors = _perturbed_neighbors(
        x_best, cfg.bounds, n=n_neighbors, frac=perturbation_frac, seed=seed
    )
    xs = [x_best, *neighbors]
    results = evaluate_batch(
        xs,
        cache=cache,
        sites=cfg.sites,
        site_profiles=site_profiles,
        econ=econ,
        combine_rule=cfg.combine_rule,
        resolution=cfg.resolution,
    )

    best_true = results[0].combined_lcow
    neighbor_results = results[1:]
    neighbor_lcows = [r.combined_lcow for r in neighbor_results]

    improvements = [
        (best_true - v) / best_true
        for v in neighbor_lcows
        if math.isfinite(v) and math.isfinite(best_true) and best_true != 0.0
    ]
    max_improvement = max(improvements) if improvements else 0.0

    mu, sigma = predict(result.surrogate, x_best)

    return VerificationReport(
        best_design_vector=tuple(float(v) for v in x_best),
        best_true_combined_lcow=best_true,
        best_surrogate_mu=mu,
        best_surrogate_sigma=sigma,
        neighbor_results=neighbor_results,
        neighbor_combined_lcows=neighbor_lcows,
        max_neighbor_improvement_frac=max_improvement,
        flagged_as_surrogate_artifact=max_improvement > artifact_tolerance,
    )
