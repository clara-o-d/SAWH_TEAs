from __future__ import annotations

import numpy as np

from sawh_bayesopt import evaluator
from sawh_bayesopt.design_space import DesignBounds, latin_hypercube_design
from sawh_bayesopt.evaluator import (
    DesignEvalResult,
    EvalCache,
    SiteResult,
    design_vector_hash,
    evaluate_batch,
)
from sawh_bayesopt.sites import SiteSpec


def _sites(n: int = 1) -> tuple[SiteSpec, ...]:
    return tuple(SiteSpec(f"dummy{i}", 0.0, 0.0) for i in range(n))


def _fake_result(x: np.ndarray, *, site_names, sentinel: float) -> DesignEvalResult:
    return DesignEvalResult(
        design_vector=tuple(float(v) for v in x),
        site_results=tuple(
            SiteResult(name, sentinel, True, "", 1.0, 0.5) for name in site_names
        ),
        combined_lcow=sentinel,
        wall_time_s=0.0,
    )


def test_design_vector_hash_stable_under_jitter_but_distinguishes_real_diffs():
    bounds = DesignBounds()
    x = latin_hypercube_design(1, bounds, seed=3)[0]
    x_jittered = x + 1e-10
    x_different = latin_hypercube_design(1, bounds, seed=99)[0]

    k = design_vector_hash(x, sites=("cambridge", "atacama"))
    k_jittered = design_vector_hash(x_jittered, sites=("cambridge", "atacama"))
    k_different = design_vector_hash(x_different, sites=("cambridge", "atacama"))

    assert k == k_jittered
    assert k != k_different


def test_design_vector_hash_distinguishes_site_set():
    bounds = DesignBounds()
    x = latin_hypercube_design(1, bounds, seed=1)[0]
    assert design_vector_hash(x, sites=("cambridge",)) != design_vector_hash(
        x, sites=("cambridge", "atacama")
    )


def test_eval_cache_round_trips_and_resumes(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = EvalCache(path)
    assert len(cache) == 0

    r1 = _fake_result(np.array([1.0] * 6), site_names=("cambridge",), sentinel=12.3)
    r2 = _fake_result(np.array([2.0] * 6), site_names=("cambridge",), sentinel=45.6)
    cache.put("key1", r1)
    cache.put("key2", r2)
    assert len(cache) == 2

    # Simulate a crash: fresh EvalCache instance on the same path replays both.
    resumed = EvalCache(path)
    assert len(resumed) == 2
    assert resumed.get_or_none("key1").combined_lcow == 12.3
    assert resumed.get_or_none("key2").combined_lcow == 45.6
    assert resumed.get_or_none("missing") is None


def test_evaluate_batch_skips_cached_points(tmp_path):
    bounds = DesignBounds()
    xs = list(latin_hypercube_design(2, bounds, seed=5))
    sites = _sites(1)
    site_profiles = {sites[0].name: []}  # empty profiles -> "no weather profiles" short-circuit, no jax needed

    cache = EvalCache(tmp_path / "cache.jsonl")
    key0 = design_vector_hash(xs[0], sites=(sites[0].name,))
    sentinel_result = _fake_result(xs[0], site_names=(sites[0].name,), sentinel=-999.0)
    cache.put(key0, sentinel_result)

    results = evaluate_batch(
        xs,
        cache=cache,
        sites=sites,
        site_profiles=site_profiles,
        econ=None,
    )

    # xs[0] was cached with a sentinel that no real (or empty-profile) run
    # would ever produce -- if it were recomputed, this would fail.
    assert results[0].combined_lcow == -999.0
    # xs[1] was uncached, so it ran for real (empty profiles -> FAIL_LCO -> penalty).
    assert results[1].combined_lcow == evaluator.PENALTY_LCOW_USD_PER_M3
    assert results[1].combined_lcow != -999.0
