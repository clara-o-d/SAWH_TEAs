"""EGO driver: Latin-hypercube init -> batched Expected-Improvement infill,
run directly against solar_lumped/gpu_sweep's JAX fast path."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from sawh_bayesopt.acquisition import propose_batch
from sawh_bayesopt.design_space import DesignBounds, latin_hypercube_design
from sawh_bayesopt.evaluator import (
    DesignEvalResult,
    EvalCache,
    evaluate_requests,
    fetch_site_inputs,
)
from sawh_bayesopt.sites import ATACAMA, SiteSpec
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
    # Many sites here means many independent single-site optimizations advanced in
    # lockstep (run_bayesopt_sites), not one design scored against several climates.
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
    # False (default): finite mass transfer (Eq. 5's g). True: the g -> infinity limit --
    # instantaneous sorption on the equilibrium isotherm, in either fidelity mode.
    instant_equilibrium: bool = False
    # "jax" batches every (design, site) into one vmapped call -- the backend for
    # global sweeps. "cpu" is sequential, needs no GPU stack, and is single-site only:
    # one simulation at a time in one location. Both support simple and complex fidelity.
    backend: str = "jax"
    # 1 (default) evaluates every calendar day. >1 keeps every Nth, which is the only
    # lever that shortens an evaluation: the year is ~366 *sequential* day-steps and is
    # ~100% of a call's cost (compile is ~20s of it). Stride 5 is ~5x cheaper but a
    # different objective -- the mean is over sampled days and the sorbent state chain
    # advances in 5-day jumps -- so it joins the cache key (evaluator.design_vector_hash)
    # and must not be mixed with stride-1 results in one comparison.
    day_stride: int = 1
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
    """One site's EGO loop. Thin wrapper over :func:`run_bayesopt_sites`, so the
    single-site CLI and the global sweep run the identical loop rather than two
    implementations that drift."""
    if len(cfg.sites) != 1:
        raise ValueError(f"run_bayesopt is single-site, got {len(cfg.sites)}; use run_bayesopt_sites")
    site = cfg.sites[0]
    return run_bayesopt_sites(cfg, {site.name: Path(run_dir)})[site.name]


@dataclass
class _SiteLoop:
    """One site's independent optimization state, advanced one round at a time.

    Sites share nothing but the batched evaluation call: separate GP, separate history,
    separate stall counter, separate cache.jsonl. That independence is the point -- a
    design is scored where it will be built (evaluator.site_lcow_or_penalty), so there is
    no shared objective to pool.
    """
    spec: SiteSpec
    cache: EvalCache
    state: SurrogateState
    history: list[DesignEvalResult]
    fitted: bool = False
    best_so_far: float = float("inf")
    stall_count: int = 0
    stopped_reason: StoppedReason = "budget"
    done: bool = False
    de_diagnostics: list[dict] = field(default_factory=list)


def run_bayesopt_sites(
    cfg: BayesOptConfig,
    run_dirs: dict[str, Path],
    *,
    site_inputs: tuple[dict, dict[str, float]] | None = None,
) -> dict[str, BayesOptResult]:
    """Optimize every site in ``cfg.sites`` independently, advanced in **lockstep** so
    each round's designs across all sites go in ONE batched evaluation.

    Why lockstep rather than a site-at-a-time loop: a year is ~366 sequential day-steps
    whatever the batch width, so an evaluation call costs ~the same for 8 instances as for
    400 (measured on an A100: 60.1 min for 1 design, 68.2 min for 8). Cost is therefore
    ``number of rounds``, not number of designs -- and a site-at-a-time sweep pays every
    site's rounds separately. N sites in lockstep pay them once.

    Each site keeps its own GP, history, stall counter and ``run_dirs[name]/cache.jsonl``;
    only the evaluation is shared. A site that stalls or exhausts its budget drops out of
    later rounds while the others continue.

    The cost of sharing a call is a shared blast radius: the vmapped physics call is fused,
    so a raise inside it marks every request in that round failed (evaluate_requests'
    ``batch_error``), across sites rather than just across one site's designs. Real bugs
    still propagate -- see evaluator._BUG_EXCEPTIONS.
    """
    from solar_lumped.economics import LCOEconomicParams

    econ = LCOEconomicParams()
    for d in run_dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)

    # Both fidelities rebuild profiles per design point (the schedule offsets and POA tilt
    # are optimized dims that live in the profile), so both need the raw frames.
    # Elevations come off the same frames -- a site property, not a design variable.
    site_frames, site_elevations = (
        fetch_site_inputs(cfg) if site_inputs is None else site_inputs
    )

    n_dims = len(cfg.bounds.names())
    loops = [
        _SiteLoop(
            spec=spec,
            cache=EvalCache(Path(run_dirs[spec.name]) / "cache.jsonl"),
            state=SurrogateState(gp=build_gp(n_dims=n_dims, seed=cfg.seed), bounds=cfg.bounds),
            history=[],
        )
        for spec in cfg.sites
    ]
    caches = {loop.spec.name: loop.cache for loop in loops}

    def _evaluate(work: list[tuple[_SiteLoop, list[np.ndarray]]]) -> None:
        """Evaluate every (loop, designs) pair in one call and append to each history."""
        requests = [(loop.spec, x) for loop, xs in work for x in xs]
        if not requests:
            return
        results = evaluate_requests(
            requests,
            caches=caches,
            econ=econ,
            case=cfg.case,
            complex_mode=cfg.complex_mode,
            condenser_tracks_ambient=cfg.condenser_tracks_ambient,
            instant_equilibrium=cfg.instant_equilibrium,
            site_frames=site_frames,
            site_elevations=site_elevations,
            backend=cfg.backend,
            day_stride=cfg.day_stride,
        )
        at = 0
        for loop, xs in work:
            loop.history.extend(results[at : at + len(xs)])
            at += len(xs)

    t_start = time.perf_counter()

    def _progress(label: str) -> None:
        """One line per round covering every site: a 400-instance round is one call, so
        per-site progress would be 400 identical timestamps."""
        n_done = sum(len(loop.history) for loop in loops)
        n_target = cfg.n_total * len(loops)
        elapsed = time.perf_counter() - t_start
        eta = elapsed / n_done * (n_target - n_done) if n_done else 0.0
        best = min((r.combined_lcow for loop in loops for r in loop.history), default=float("inf"))
        active = sum(1 for loop in loops if not loop.done)
        print(
            f"[{n_done:5d}/{n_target}] {label:<9s} {active:3d}/{len(loops)} site(s) active  "
            f"best {best:10.4f} USD/m3  elapsed {elapsed / 60:.1f}m  eta {eta / 60:.0f}m",
            flush=True,
        )

    def _refit(loop: _SiteLoop) -> None:
        loop.state, loop.fitted = _try_fit(loop.state, seed=cfg.seed)
        if loop.fitted:
            for w in check_hyperparameter_convergence(loop.state.gp):
                logger.warning("%s: LCOW GP hyperparameter near optimization bound: %s", loop.spec.name, w)

    # Same seed at every site on purpose: an identical initial design set makes per-site
    # optima comparable, and the cache keys already differ by site name.
    X0 = latin_hypercube_design(cfg.n_init, cfg.bounds, seed=cfg.seed, reject_gap_degenerate=True)
    _evaluate([(loop, list(X0)) for loop in loops])
    for loop in loops:
        X_all, y_all, feasible_all = _to_xyf(loop.history)
        loop.state = append_observations(loop.state, X_all, y_all, feasible_all)
        _refit(loop)
        loop.best_so_far = loop.state.y_best
    _progress("lhs-init")

    round_idx = 0
    while any(not loop.done for loop in loops):
        work: list[tuple[_SiteLoop, list[np.ndarray]]] = []
        for loop in loops:
            if loop.done:
                continue
            remaining = cfg.n_total - len(loop.history)
            if remaining <= 0:
                loop.done = True
                continue
            batch_n = min(cfg.batch_size, remaining)
            if loop.fitted:
                # ponytail: the EI proposals are sequential across sites and DE-bound on
                # CPU (~batch_size differential_evolution runs per site per round). Fine
                # against a ~1h GPU call for tens of sites; parallelize over sites with
                # joblib if a task ever carries hundreds.
                record: list[dict] = []
                batch = propose_batch(
                    loop.state, batch_size=batch_n, seed=cfg.seed + len(loop.history),
                    xi=cfg.ei_xi, record=record,
                    maxiter=cfg.de_maxiter, popsize=cfg.de_popsize,
                )
                for d in record:
                    d["round"] = round_idx
                loop.de_diagnostics.extend(record)
            else:
                # Too few feasible observations to fit the LCOW GP -- keep exploring
                # blindly rather than proposing from a surrogate that can't exist yet.
                batch = list(
                    latin_hypercube_design(
                        batch_n, cfg.bounds, seed=cfg.seed + len(loop.history),
                        reject_gap_degenerate=True,
                    )
                )
            work.append((loop, batch))

        if not work:
            break
        _evaluate(work)

        for loop, batch in work:
            X_new, y_new, feasible_new = _to_xyf(loop.history[-len(batch):])
            loop.state = append_observations(loop.state, X_new, y_new, feasible_new)
            _refit(loop)
            new_best = loop.state.y_best
            if loop.fitted and math.isfinite(loop.best_so_far) and loop.best_so_far != 0.0:
                rel_improve = (loop.best_so_far - new_best) / abs(loop.best_so_far)
            else:
                # Still bootstrapping: don't let the stall counter compare against a
                # meaningless pre-fit y_best.
                rel_improve = 1.0
            loop.stall_count = 0 if rel_improve >= cfg.stall_rel_tol else loop.stall_count + 1
            loop.best_so_far = new_best
            if loop.fitted and loop.stall_count >= cfg.stall_rounds:
                loop.stopped_reason = "stalled"
                loop.done = True
            elif len(loop.history) >= cfg.n_total:
                loop.done = True

        _progress(f"round {round_idx}")
        round_idx += 1

    out: dict[str, BayesOptResult] = {}
    for loop in loops:
        best_idx = int(np.argmin([r.combined_lcow for r in loop.history]))
        out[loop.spec.name] = BayesOptResult(
            history=loop.history,
            best=loop.history[best_idx],
            surrogate=loop.state,
            de_diagnostics=loop.de_diagnostics,
            stopped_reason=loop.stopped_reason,
        )
    return out
