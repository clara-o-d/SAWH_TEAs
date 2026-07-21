from __future__ import annotations

import numpy as np
import pytest

from sawh_bayesopt import evaluator
from sawh_bayesopt.design_space import DesignBounds, latin_hypercube_design
from sawh_bayesopt.evaluator import EvalCache
from sawh_bayesopt.sites import SiteSpec

_DUMMY_PROFILES = [(1, object(), 30)]


class _FakeJaxDailyCycle:
    """Stand-in for gpu_sweep/jax_daily_cycle.py -- lets these tests exercise
    evaluate_batch's failure-handling logic without needing jax/diffrax
    installed or running any real physics."""

    def __init__(self, *, water=None, eta=None, raises: Exception | None = None):
        self._water = water
        self._eta = eta
        self._raises = raises

    def build_batch_arrays(self, profiles, configs):
        if self._raises is not None:
            raise self._raises
        return {}, 0.0, 1, 1

    def make_batched_daily_cycle_fn(self, batch, dt, n_abs_max, n_des_max):
        n = len(self._water)
        water, eta = self._water, self._eta

        def fn(cw, h):
            return np.asarray(water), np.asarray(eta), np.asarray(cw[:n]), np.asarray(h[:n])

        return fn

    def find_cyclic_state_batched(self, daily_cycle_fn, *, c_w_initial, h_initial, max_rounds):
        return np.asarray(c_w_initial), np.asarray(h_initial)


@pytest.fixture
def one_site():
    return (SiteSpec("dummy", 0.0, 0.0),)


@pytest.fixture
def econ():
    from solar_lumped.economics.params import LCOEconomicParams

    return LCOEconomicParams()


def _one_x():
    bounds = DesignBounds()
    return latin_hypercube_design(1, bounds, seed=0)[0]


def test_evaluate_batch_penalizes_batched_call_failure(monkeypatch, tmp_path, one_site, econ):
    from solar_lumped.economics.lcow import FAIL_LCO

    fake = _FakeJaxDailyCycle(raises=RuntimeError("solve_ivp did not converge"))
    monkeypatch.setattr(evaluator, "_load_jax_daily_cycle", lambda: fake)

    x = _one_x()
    site_profiles = {"dummy": _DUMMY_PROFILES}
    cache = EvalCache(tmp_path / "cache.jsonl")
    [result] = evaluator.evaluate_batch(
        [x], cache=cache, sites=one_site, site_profiles=site_profiles, econ=econ, combine_rule="mean"
    )

    assert result.site_results[0].feasible is False
    assert result.site_results[0].lcow == FAIL_LCO
    assert "solve_ivp" in result.site_results[0].failure_reason


def test_evaluate_batch_penalizes_zero_yield(monkeypatch, tmp_path, one_site, econ):
    from solar_lumped.economics.lcow import FAIL_LCO

    fake = _FakeJaxDailyCycle(water=[0.0], eta=[0.0])
    monkeypatch.setattr(evaluator, "_load_jax_daily_cycle", lambda: fake)

    x = _one_x()
    site_profiles = {"dummy": _DUMMY_PROFILES}
    cache = EvalCache(tmp_path / "cache.jsonl")
    [result] = evaluator.evaluate_batch(
        [x], cache=cache, sites=one_site, site_profiles=site_profiles, econ=econ, combine_rule="mean"
    )

    assert result.site_results[0].feasible is False
    assert result.site_results[0].lcow == FAIL_LCO
    assert result.site_results[0].failure_reason == "zero or invalid yield"


def test_evaluate_batch_combined_lcow_uses_finite_penalty_not_fail_lco(
    monkeypatch, tmp_path, one_site, econ
):
    fake = _FakeJaxDailyCycle(raises=RuntimeError("boom"))
    monkeypatch.setattr(evaluator, "_load_jax_daily_cycle", lambda: fake)

    x = _one_x()
    site_profiles = {"dummy": _DUMMY_PROFILES}
    cache = EvalCache(tmp_path / "cache.jsonl")
    [result] = evaluator.evaluate_batch(
        [x], cache=cache, sites=one_site, site_profiles=site_profiles, econ=econ, combine_rule="mean"
    )

    assert result.site_results[0].feasible is False
    assert result.combined_lcow == evaluator.PENALTY_LCOW_USD_PER_M3
    assert result.combined_lcow < 1e29  # nowhere near solar_lumped's raw FAIL_LCO (1e30)


def test_evaluate_batch_penalizes_missing_weather(tmp_path, one_site, econ):
    from solar_lumped.economics.lcow import FAIL_LCO

    x = _one_x()
    site_profiles = {"dummy": []}  # no weather at all -> never touches jax
    cache = EvalCache(tmp_path / "cache.jsonl")
    [result] = evaluator.evaluate_batch(
        [x], cache=cache, sites=one_site, site_profiles=site_profiles, econ=econ, combine_rule="mean"
    )

    assert result.site_results[0].feasible is False
    assert result.site_results[0].lcow == FAIL_LCO
    assert result.site_results[0].failure_reason == "no weather profiles"
    assert result.combined_lcow == evaluator.PENALTY_LCOW_USD_PER_M3
