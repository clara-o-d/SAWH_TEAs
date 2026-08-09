"""Exercise the EGO loop end-to-end against a cheap synthetic objective --
solar_lumped's real physics and network weather fetch are monkeypatched out
so this stays fast and hermetic."""

from __future__ import annotations

import numpy as np

from sawh_bayesopt.bayesopt import BayesOptConfig, run_bayesopt
from sawh_bayesopt.design_space import DesignBounds
from sawh_bayesopt.evaluator import DesignEvalResult, SiteResult

_BOUNDS = DesignBounds()
_TARGET_RAW = _BOUNDS.as_array().mean(axis=1)


def _fake_evaluate_batch(
    xs,
    *,
    cache=None,
    sites,
    site_profiles=None,
    econ=None,
    combine_rule="mean",
    resolution="annual",
    case="case1",
    # Accepted and ignored: these are real-evaluator plumbing (complex fidelity, its
    # per-design weather frames, and the condenser mode) that the synthetic bowl has
    # no use for.
    complex_mode=False,
    condenser_tracks_ambient=False,
    site_frames=None,
    backend="jax",
):
    results = []
    for x in xs:
        y = float(np.sum((np.asarray(x, dtype=float) - _TARGET_RAW) ** 2))
        site_results = tuple(SiteResult(s.name, y, True, "", 1.0, 0.5) for s in sites)
        results.append(
            DesignEvalResult(
                design_vector=tuple(float(v) for v in x),
                site_results=site_results,
                combined_lcow=y,
                wall_time_s=0.0,
            )
        )
    return results


def _patch(monkeypatch):
    monkeypatch.setattr("sawh_bayesopt.bayesopt.fetch_daily_profiles", lambda site, cache_dir: [])
    monkeypatch.setattr("sawh_bayesopt.bayesopt.evaluate_batch", _fake_evaluate_batch)


def test_run_bayesopt_best_so_far_is_monotone_and_respects_budget(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = BayesOptConfig(
        n_init=8,
        n_total=16,
        batch_size=2,
        seed=0,
        stall_rel_tol=0.5,
        stall_rounds=2,
        de_maxiter=30,
        de_popsize=8,
    )
    result = run_bayesopt(cfg, tmp_path / "run1")

    lcows = [r.combined_lcow for r in result.history]
    best_so_far = np.minimum.accumulate(lcows)
    assert np.all(np.diff(best_so_far) <= 1e-12)
    assert result.best.combined_lcow == min(lcows)
    assert len(result.history) <= cfg.n_total
    assert result.stopped_reason in ("budget", "stalled")


def test_run_bayesopt_stops_early_when_stalled(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = BayesOptConfig(
        n_init=8,
        n_total=100,
        batch_size=2,
        seed=0,
        stall_rel_tol=0.99,
        stall_rounds=1,
        de_maxiter=30,
        de_popsize=8,
    )
    result = run_bayesopt(cfg, tmp_path / "run2")

    assert result.stopped_reason == "stalled"
    assert len(result.history) < cfg.n_total


def _fake_evaluate_batch_with_infeasible_region(
    xs,
    *,
    cache=None,
    sites,
    site_profiles=None,
    econ=None,
    combine_rule="mean",
    resolution="annual",
    case="case1",
    # Accepted and ignored: these are real-evaluator plumbing (complex fidelity, its
    # per-design weather frames, and the condenser mode) that the synthetic bowl has
    # no use for.
    complex_mode=False,
    condenser_tracks_ambient=False,
    site_frames=None,
    backend="jax",
):
    """Same quadratic-bowl objective, but the whole region where the first
    design variable exceeds its midpoint is infeasible (penalty y) -- exactly
    the shape evaluator.py produces for a real infeasible design."""
    lo, hi = _BOUNDS.hydrogel_thickness_m
    midpoint = (lo + hi) / 2.0
    results = []
    for x in xs:
        x = np.asarray(x, dtype=float)
        infeasible = x[0] > midpoint
        y = 1.0e4 if infeasible else float(np.sum((x - _TARGET_RAW) ** 2))
        site_results = tuple(
            SiteResult(s.name, y, not infeasible, "" if not infeasible else "synthetic failure", 1.0, 0.5)
            for s in sites
        )
        results.append(
            DesignEvalResult(
                design_vector=tuple(float(v) for v in x),
                site_results=site_results,
                combined_lcow=y,
                wall_time_s=0.0,
            )
        )
    return results


def test_run_bayesopt_handles_an_infeasible_region_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr("sawh_bayesopt.bayesopt.fetch_daily_profiles", lambda site, cache_dir: [])
    monkeypatch.setattr(
        "sawh_bayesopt.bayesopt.evaluate_batch", _fake_evaluate_batch_with_infeasible_region
    )
    cfg = BayesOptConfig(
        n_init=8,
        n_total=20,
        batch_size=2,
        seed=0,
        stall_rel_tol=0.5,
        stall_rounds=3,
        de_maxiter=30,
        de_popsize=8,
    )
    result = run_bayesopt(cfg, tmp_path / "run_infeasible")

    assert result.best.is_feasible
    assert result.surrogate.n_feasible >= 2
    # The surrogate must never have been fit on a contaminated (infeasible)
    # observation -- every X_raw row marked infeasible should be excluded,
    # which state.feasible existing at all (and n_feasible < len(history) if
    # any infeasible points were sampled) is a reasonable proxy check for.
    assert result.surrogate.feasible.shape[0] == len(result.history)
    assert any(not r.is_feasible for r in result.history), "test should actually sample the infeasible region"
