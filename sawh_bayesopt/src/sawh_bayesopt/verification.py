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
from sawh_bayesopt.design_space import snap_to_grid
from sawh_bayesopt.evaluator import DesignEvalResult, EvalCache, evaluate_for_config
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
        delta = rng.uniform(-frac, frac, size=span.shape[0]) * span
        # Snapped for the same reason proposals are: an off-lattice seal/open offset
        # quantizes to a different window when simulated, so a promoted neighbor would
        # report a schedule that was never the one evaluated.
        out.append(snap_to_grid(np.clip(x_best + delta, lo, hi), bounds))
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
    """One site's verification. Wrapper over :func:`verify_optima` so the single-site CLI
    and the global sweep verify identically."""
    if len(cfg.sites) != 1:
        raise ValueError(f"verify_optimum is single-site, got {len(cfg.sites)}; use verify_optima")
    name = cfg.sites[0].name
    return verify_optima(
        {name: result}, cfg, {name: Path(run_dir)},
        n_neighbors=n_neighbors, perturbation_frac=perturbation_frac,
        seed=seed, artifact_tolerance=artifact_tolerance,
    )[name]


def verify_optima(
    results: dict[str, BayesOptResult],
    cfg: BayesOptConfig,
    run_dirs: dict[str, Path],
    *,
    n_neighbors: int = 5,
    perturbation_frac: float = 0.10,
    seed: int = 0,
    artifact_tolerance: float = 0.02,
    site_inputs: tuple[dict, dict[str, float]] | None = None,
) -> dict[str, VerificationReport]:
    """Verify every site's reported optimum in ONE batched evaluation.

    Every site's (best + neighbours) go into a single request list, because an evaluation
    call costs ~the same at any batch width -- verifying site-by-site would pay a full
    ~1 h call per site for what fits in one (see ``evaluator.evaluate_requests``).
    """
    from solar_lumped.economics import LCOEconomicParams

    econ = LCOEconomicParams()
    caches = {
        name: EvalCache(Path(run_dirs[name]) / "cache.jsonl") for name in results
    }
    specs = {spec.name: spec for spec in cfg.sites}

    requests: list[tuple] = []
    spans: dict[str, tuple[int, int]] = {}
    for name, result in results.items():
        x_best = np.array(result.best.design_vector, dtype=float)
        neighbors = _perturbed_neighbors(
            x_best, cfg.bounds, n=n_neighbors, frac=perturbation_frac, seed=seed
        )
        start = len(requests)
        requests.extend((specs[name], x) for x in (x_best, *neighbors))
        spans[name] = (start, len(requests))

    evaluated = evaluate_for_config(
        requests, cfg=cfg, caches=caches, econ=econ, site_inputs=site_inputs
    )

    reports: dict[str, VerificationReport] = {}
    for name, result in results.items():
        lo, hi = spans[name]
        site_evaluated = evaluated[lo:hi]
        best_true = site_evaluated[0].combined_lcow
        neighbor_results = site_evaluated[1:]
        neighbor_lcows = [r.combined_lcow for r in neighbor_results]

        improvements = [
            (best_true - v) / best_true
            for v in neighbor_lcows
            if math.isfinite(v) and math.isfinite(best_true) and best_true != 0.0
        ]
        max_improvement = max(improvements) if improvements else 0.0
        x_best = np.array(result.best.design_vector, dtype=float)
        mu, sigma = predict(result.surrogate, x_best)

        reports[name] = VerificationReport(
            best_design_vector=tuple(float(v) for v in x_best),
            best_true_combined_lcow=best_true,
            best_surrogate_mu=mu,
            best_surrogate_sigma=sigma,
            neighbor_results=neighbor_results,
            neighbor_combined_lcows=neighbor_lcows,
            max_neighbor_improvement_frac=max_improvement,
            flagged_as_surrogate_artifact=max_improvement > artifact_tolerance,
        )
    return reports
