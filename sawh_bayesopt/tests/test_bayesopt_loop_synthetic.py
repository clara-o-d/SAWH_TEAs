"""Exercise the EGO loop end-to-end against a cheap synthetic objective --
solar_lumped's real physics and network weather fetch are monkeypatched out
so this stays fast and hermetic."""

from __future__ import annotations

import numpy as np

from sawh_bayesopt.bayesopt import BayesOptConfig, run_bayesopt, run_bayesopt_sites
from sawh_bayesopt.design_space import DesignBounds
from sawh_bayesopt.evaluator import DesignEvalResult, SiteResult
from sawh_bayesopt.sites import SiteSpec

_BOUNDS = DesignBounds()
_TARGET_RAW = _BOUNDS.as_array().mean(axis=1)


def _fake_evaluate_requests(
    requests,
    *,
    caches=None,
    econ=None,
    case="case1",
    # Accepted and ignored: these are real-evaluator plumbing (complex fidelity, its
    # per-design weather frames, the condenser mode, the sorption-kinetics limit) that
    # the synthetic bowl has no use for.
    complex_mode=False,
    condenser_tracks_ambient=False,
    site_elevations=None,
    instant_equilibrium=False,
    site_frames=None,
    backend="jax",
    day_stride=1,
):
    results = []
    for spec, x in requests:
        y = float(np.sum((np.asarray(x, dtype=float) - _TARGET_RAW) ** 2))
        # Per-site offset, so a multi-site lockstep run cannot pass by scoring every
        # site identically.
        y += abs(spec.lat)
        results.append(
            DesignEvalResult(
                design_vector=tuple(float(v) for v in x),
                site_results=(SiteResult(spec.name, y, True, "", 1.0, 0.5),),
                combined_lcow=y,
                wall_time_s=0.0,
            )
        )
    return results


def _patch(monkeypatch, evaluate=_fake_evaluate_requests):
    monkeypatch.setattr("sawh_bayesopt.bayesopt.fetch_site_inputs", lambda cfg: ({}, {}))
    monkeypatch.setattr("sawh_bayesopt.bayesopt.evaluate_requests", evaluate)


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


def _fake_evaluate_requests_with_infeasible_region(
    requests,
    *,
    caches=None,
    econ=None,
    case="case1",
    # Accepted and ignored: these are real-evaluator plumbing (complex fidelity, its
    # per-design weather frames, the condenser mode, the sorption-kinetics limit) that
    # the synthetic bowl has no use for.
    complex_mode=False,
    condenser_tracks_ambient=False,
    site_elevations=None,
    instant_equilibrium=False,
    site_frames=None,
    backend="jax",
    day_stride=1,
):
    """Same quadratic-bowl objective, but the whole region where the first
    design variable exceeds its midpoint is infeasible (penalty y) -- exactly
    the shape evaluator.py produces for a real infeasible design."""
    lo, hi = _BOUNDS.hydrogel_thickness_m
    midpoint = (lo + hi) / 2.0
    results = []
    for spec, x in requests:
        x = np.asarray(x, dtype=float)
        infeasible = x[0] > midpoint
        y = 1.0e4 if infeasible else float(np.sum((x - _TARGET_RAW) ** 2))
        results.append(
            DesignEvalResult(
                design_vector=tuple(float(v) for v in x),
                site_results=(
                    SiteResult(
                        spec.name, y, not infeasible,
                        "" if not infeasible else "synthetic failure", 1.0, 0.5,
                    ),
                ),
                combined_lcow=y,
                wall_time_s=0.0,
            )
        )
    return results


def test_run_bayesopt_handles_an_infeasible_region_without_crashing(tmp_path, monkeypatch):
    _patch(monkeypatch, _fake_evaluate_requests_with_infeasible_region)
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


def test_lockstep_sites_share_one_call_per_round_but_not_their_histories(tmp_path, monkeypatch):
    """The whole point of run_bayesopt_sites: N sites cost N *designs*, not N rounds.

    A year is ~366 sequential day-steps whatever the batch width, so the number of
    evaluation calls is the cost. This asserts the call count is per round, not per
    (round, site), and that the sites stay independent optimizations while sharing them.
    """
    calls: list[list[str]] = []

    def counting_evaluate(requests, **kwargs):
        calls.append([spec.name for spec, _x in requests])
        return _fake_evaluate_requests(requests, **kwargs)

    _patch(monkeypatch, counting_evaluate)
    sites = (SiteSpec("s_hot", -23.0, -70.0), SiteSpec("s_cold", 60.0, 20.0), SiteSpec("s_mid", 10.0, 5.0))
    cfg = BayesOptConfig(
        sites=sites, n_init=6, n_total=10, batch_size=2, seed=0,
        stall_rel_tol=0.0, stall_rounds=99, de_maxiter=30, de_popsize=8,
    )
    results = run_bayesopt_sites(cfg, {s.name: tmp_path / s.name for s in sites})

    assert set(results) == {s.name for s in sites}
    # 6 init + 2 rounds x 2 = 10 evaluations per site, and every call carries all 3 sites.
    for names in calls:
        counts = {n: names.count(n) for n in set(names)}
        assert sorted(counts) == sorted(s.name for s in sites)
        assert len(set(counts.values())) == 1, f"uneven designs per site in one call: {counts}"
    assert len(calls) == 3, f"expected 1 init + 2 infill calls for all sites, got {len(calls)}"

    for spec in sites:
        r = results[spec.name]
        assert len(r.history) == cfg.n_total
        # Each site scored only itself -- no cross-site pooling.
        assert {sr.site_name for res in r.history for sr in res.site_results} == {spec.name}
        assert r.best.combined_lcow == min(res.combined_lcow for res in r.history)
        assert (tmp_path / spec.name / "cache.jsonl").parent.is_dir()

    # Independent GPs: different sites see different objectives (the per-site offset in
    # the fake), so their proposed designs must diverge after the shared LHS init.
    tails = {
        spec.name: tuple(tuple(res.design_vector) for res in results[spec.name].history[cfg.n_init:])
        for spec in sites
    }
    assert len(set(tails.values())) > 1, "sites proposed identical infill -- GPs are not independent"
