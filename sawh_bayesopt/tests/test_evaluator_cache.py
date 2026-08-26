from __future__ import annotations

import numpy as np
import pytest

from sawh_bayesopt import evaluator
from sawh_bayesopt.design_space import DesignBounds, latin_hypercube_design
from sawh_bayesopt.evaluator import (
    DesignEvalResult,
    EvalCache,
    SiteResult,
    design_vector_hash,
    evaluate_batch,
    evaluate_requests,
)
from sawh_bayesopt.sites import SiteSpec
from solar_lumped.economics import LCOEconomicParams


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


def test_evaluate_batch_skips_cached_points(monkeypatch, tmp_path):
    bounds = DesignBounds()
    xs = list(latin_hypercube_design(2, bounds, seed=5))
    sites = _sites(1)
    # No usable day in the frame -> "no weather profiles" short-circuit, no jax needed.
    monkeypatch.setattr(evaluator, "_profiles_for_design", lambda df, x, complex_mode, stride=1: [])
    site_frames = {sites[0].name: object()}

    cache = EvalCache(tmp_path / "cache.jsonl")
    key0 = design_vector_hash(xs[0], sites=(sites[0].name,))
    sentinel_result = _fake_result(xs[0], site_names=(sites[0].name,), sentinel=-999.0)
    cache.put(key0, sentinel_result)

    results = evaluate_batch(
        xs,
        cache=cache,
        sites=sites,
        site_frames=site_frames,
        econ=None,
    )

    # xs[0] was cached with a sentinel that no real (or empty-profile) run
    # would ever produce -- if it were recomputed, this would fail.
    assert results[0].combined_lcow == -999.0
    # xs[1] was uncached, so it ran for real (empty profiles -> FAIL_LCO -> penalty).
    assert results[1].combined_lcow == evaluator.PENALTY_LCOW_USD_PER_M3
    assert results[1].combined_lcow != -999.0


def test_evaluate_requests_batches_every_site_into_one_call(monkeypatch, tmp_path):
    """The lockstep sweep's whole economy: many sites, ONE physics call, separate caches.

    A year is ~366 sequential day-steps whatever the batch width, so per-site calls are
    what a sweep must avoid. This pins the three properties the sweep relies on: one call
    for the whole request list, per-site cache files, and each site's own elevation on its
    own instances.
    """
    from solar_lumped.economics import LCOEconomicParams

    monkeypatch.setattr(evaluator, "_profiles_for_design", lambda df, x, complex_mode, stride=1: [(1, object())])
    calls: list[list[int]] = []
    elevations_seen: list[float] = []

    def fake_run_year(instance_profiles, instance_configs, owner, **_kwargs):
        calls.append(list(owner))
        elevations_seen.extend(c.site_elevation_m for c in instance_configs)
        return {i: 1.0 for i in owner}, {i: 0.5 for i in owner}, {i: False for i in owner}, {}

    monkeypatch.setattr(evaluator, "_run_jax_year", fake_run_year)

    a, b = SiteSpec("a", -23.0, -70.0), SiteSpec("b", 42.0, -71.0)
    caches = {"a": EvalCache(tmp_path / "a.jsonl"), "b": EvalCache(tmp_path / "b.jsonl")}
    x1, x2 = latin_hypercube_design(2, DesignBounds(), seed=1)
    # x1 at site a appears twice: a verification neighbour landing on an evaluated point
    # is normal, and paying for it twice inside one call would be silent waste.
    requests = [(a, x1), (b, x1), (a, x2), (b, x2), (a, x1)]

    results = evaluate_requests(
        requests,
        caches=caches,
        econ=LCOEconomicParams(),
        site_frames={"a": object(), "b": object()},
        site_elevations={"a": 0.0, "b": 1500.0},
    )

    assert len(calls) == 1, f"{len(calls)} physics calls for 2 sites -- the batching is gone"
    assert len(calls[0]) == 4, "the duplicate request should not have been run twice"
    assert [r.site_results[0].site_name for r in results] == ["a", "b", "a", "b", "a"]
    assert results[0].combined_lcow == results[4].combined_lcow
    assert sorted(elevations_seen) == [0.0, 0.0, 1500.0, 1500.0]
    # Each site's results land in its own cache.jsonl, because each site owns a run dir.
    assert len(caches["a"]) == 2 and len(caches["b"]) == 2
    assert (tmp_path / "a.jsonl").is_file() and (tmp_path / "b.jsonl").is_file()


def test_evaluate_batch_rejects_multi_site(tmp_path):
    """Optimization is single-site (site_lcow_or_penalty); many sites means many requests,
    not many sites per design."""
    import pytest

    with pytest.raises(ValueError, match="single-site"):
        evaluate_batch(
            [latin_hypercube_design(1, DesignBounds(), seed=0)[0]],
            cache=EvalCache(tmp_path / "c.jsonl"), sites=_sites(2), econ=None,
        )


def test_day_stride_is_part_of_the_cache_key_but_stride_one_is_unchanged():
    """A strided LCOW is a different objective, not a cheaper route to the same number, so
    it must not collide with a full-year entry. Stride 1 must keep the existing key, or
    every cache.jsonl on disk is retired."""
    x = latin_hypercube_design(1, DesignBounds(), seed=3)[0]
    full = design_vector_hash(x, sites=("a",))
    assert design_vector_hash(x, sites=("a",), day_stride=1) == full
    strided = design_vector_hash(x, sites=("a",), day_stride=5)
    assert strided != full
    assert design_vector_hash(x, sites=("a",), day_stride=7) not in (full, strided)


def test_evaluate_requests_forwards_day_stride_to_the_profile_build(monkeypatch, tmp_path):
    """The stride has to reach real_weather_days_from_df; a stride that only changed the
    cache key would silently return full-year physics under a strided key."""
    from solar_lumped.economics import LCOEconomicParams

    seen: list[int] = []

    def stub_profiles(df, x, complex_mode, stride=1):
        seen.append(stride)
        return [(1, object())]

    monkeypatch.setattr(evaluator, "_profiles_for_design", stub_profiles)
    monkeypatch.setattr(
        evaluator, "_run_jax_year",
        lambda profiles, configs, owner, **kw: (
            {i: 1.0 for i in owner}, {i: 0.5 for i in owner}, {i: False for i in owner}, {}
        ),
    )
    site = SiteSpec("a", 0.0, 0.0)
    evaluate_requests(
        [(site, latin_hypercube_design(1, DesignBounds(), seed=4)[0])],
        caches={"a": EvalCache(tmp_path / "a.jsonl")},
        econ=LCOEconomicParams(),
        site_frames={"a": object()},
        day_stride=5,
    )
    assert seen == [5]


def test_cpu_backend_scopes_a_failure_to_the_one_design_that_caused_it(monkeypatch, tmp_path):
    """_run_cpu_year used to `return` out of the whole function from inside its
    per-instance loop, so one unintegrable design took every design batched with it down
    with it -- the exact opposite of the "one bad design must not kill the batch" comment
    sitting on the handler. Batching must not change any other design's answer."""
    from solar_lumped.simulation import PhaseResult

    def fake_run_daily_cycle(prof, config, **_kw):
        if abs(config.hydrogel_thickness_m - 0.005) < 1e-9:
            raise RuntimeError("integration blew up")
        arr = np.array([0.0, 1.0])
        phase = PhaseResult(
            time_s=arr, c_w=arr, H=arr, t_cond_c=None, t_gel_c=arr,
            water_collected_kg_m2=2.0, m_des_kg_s_m2=arr,
        )
        return 2.0, 0.3, phase, phase

    monkeypatch.setattr(evaluator, "_profiles_for_design",
                        lambda df, x, complex_mode, day_stride=1: [(1, object())])
    monkeypatch.setattr("solar_lumped.simulation.find_cyclic_state", lambda *a, **k: (1.0, 0.001))
    monkeypatch.setattr("solar_lumped.simulation.run_daily_cycle", fake_run_daily_cycle)

    site = SiteSpec("a", 0.0, 0.0)
    good = np.array([0.003, 30.0, 0.0, 0.0])
    bad = np.array([0.005, 30.0, 0.0, 0.0])
    results = evaluate_requests(
        [(site, good), (site, bad), (site, good * 1.0 + np.array([1e-4, 0, 0, 0]))],
        caches={"a": EvalCache(tmp_path / "a.jsonl")}, econ=LCOEconomicParams(),
        site_frames={"a": object()}, site_elevations={"a": 0.0}, backend="cpu",
    )
    ok_a, failed, ok_b = (r.site("a") for r in results)
    assert failed.failure_reason == "integration blew up"
    assert not failed.feasible
    # The neighbours are untouched -- this is what the old whole-batch return destroyed.
    assert ok_a.feasible and ok_b.feasible
    assert ok_a.yield_kg_m2 == pytest.approx(2.0)
    assert ok_b.yield_kg_m2 == pytest.approx(2.0)
