from __future__ import annotations

import numpy as np
import pytest

from sawh_bayesopt import evaluator
from sawh_bayesopt.design_space import DesignBounds, latin_hypercube_design
from sawh_bayesopt.evaluator import EvalCache
from sawh_bayesopt.sites import SiteSpec

_DUMMY_PROFILES = [(1, object())]
_DUMMY_FRAMES = {"dummy": object()}  # never read: _profiles_for_design is stubbed below


class _FakeJaxDailyCycle:
    """Stand-in for gpu_sweep/jax_daily_cycle.py -- lets these tests exercise
    evaluate_batch's failure-handling logic without needing jax/diffrax
    installed or running any real physics."""

    def __init__(self, *, water=None, eta=None, raises: Exception | None = None):
        self._water = water
        self._eta = eta
        self._raises = raises

    def year_padding(self, profiles_by_instance):
        if self._raises is not None:
            raise self._raises
        return 0.0, 1, 1

    # Mode flags (complex_mode, condenser_tracks_ambient, instant_equilibrium) are
    # accepted and ignored: these tests exercise evaluate_batch's failure handling,
    # which is identical in every mode.
    def build_system_arrays(self, configs, **_flags):
        return {}

    def build_day_weather(self, profiles, n_abs_max, n_des_max):
        return ()

    def make_year_step_fn(self, system, dt, n_abs_max, n_des_max, **_flags):
        return lambda cw, h, weather: (self._water, self._eta, cw, h)

    def run_year_batched(self, step_fn, day_weathers, *, c_w_initial, h_initial, aitken_max_rounds):
        return np.asarray(self._water), np.asarray(self._eta)


@pytest.fixture(autouse=True)
def stub_profiles(monkeypatch):
    """evaluate_batch rebuilds per-day profiles from the site's weather frame for every
    design (the schedule offsets are optimized dims), which these tests have no real
    frame for. Return a sentinel profile list instead; the failure paths under test never
    look inside it."""
    monkeypatch.setattr(evaluator, "_profiles_for_design", lambda df, x, complex_mode: _DUMMY_PROFILES)


@pytest.fixture
def one_site():
    return (SiteSpec("dummy", 0.0, 0.0),)


@pytest.fixture
def econ():
    from solar_lumped.economics import LCOEconomicParams

    return LCOEconomicParams()


def _one_x():
    bounds = DesignBounds()
    return latin_hypercube_design(1, bounds, seed=0)[0]


def test_evaluate_batch_penalizes_batched_call_failure(monkeypatch, tmp_path, one_site, econ):
    from solar_lumped.economics import FAIL_LCO

    fake = _FakeJaxDailyCycle(raises=RuntimeError("solve_ivp did not converge"))
    monkeypatch.setattr(evaluator, "_load_jax_daily_cycle", lambda: fake)

    x = _one_x()
    cache = EvalCache(tmp_path / "cache.jsonl")
    [result] = evaluator.evaluate_batch(
        [x], cache=cache, sites=one_site, site_frames=_DUMMY_FRAMES, econ=econ
    )

    assert result.site_results[0].feasible is False
    assert result.site_results[0].lcow == FAIL_LCO
    assert "solve_ivp" in result.site_results[0].failure_reason


def test_evaluate_batch_penalizes_zero_yield(monkeypatch, tmp_path, one_site, econ):
    from solar_lumped.economics import FAIL_LCO

    fake = _FakeJaxDailyCycle(water=[0.0], eta=[0.0])
    monkeypatch.setattr(evaluator, "_load_jax_daily_cycle", lambda: fake)

    x = _one_x()
    cache = EvalCache(tmp_path / "cache.jsonl")
    [result] = evaluator.evaluate_batch(
        [x], cache=cache, sites=one_site, site_frames=_DUMMY_FRAMES, econ=econ
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
    cache = EvalCache(tmp_path / "cache.jsonl")
    [result] = evaluator.evaluate_batch(
        [x], cache=cache, sites=one_site, site_frames=_DUMMY_FRAMES, econ=econ
    )

    assert result.site_results[0].feasible is False
    assert result.combined_lcow == evaluator.PENALTY_LCOW_USD_PER_M3
    assert result.combined_lcow < 1e29  # nowhere near solar_lumped's raw FAIL_LCO (1e30)


def test_evaluate_batch_propagates_bugs_instead_of_penalizing(monkeypatch, tmp_path, one_site, econ):
    """A NameError in the physics is broken code, not an infeasible design. Laundering it
    into the 1e4 penalty is how a missing import in jax_physics.py produced a sweep of
    penalties that read as a completed run (see evaluator._BUG_EXCEPTIONS)."""
    fake = _FakeJaxDailyCycle(raises=NameError("name '_M_DES_BRACKET_MAX' is not defined"))
    monkeypatch.setattr(evaluator, "_load_jax_daily_cycle", lambda: fake)

    cache = EvalCache(tmp_path / "cache.jsonl")
    with pytest.raises(NameError, match="_M_DES_BRACKET_MAX"):
        evaluator.evaluate_batch(
            [_one_x()], cache=cache, sites=one_site,
            site_frames=_DUMMY_FRAMES, econ=econ,
        )


def test_evaluate_batch_penalizes_missing_weather(monkeypatch, tmp_path, one_site, econ):
    from solar_lumped.economics import FAIL_LCO

    x = _one_x()
    # No usable day in the frame -> never touches jax.
    monkeypatch.setattr(evaluator, "_profiles_for_design", lambda df, x, complex_mode: [])
    cache = EvalCache(tmp_path / "cache.jsonl")
    [result] = evaluator.evaluate_batch(
        [x], cache=cache, sites=one_site, site_frames=_DUMMY_FRAMES, econ=econ
    )

    assert result.site_results[0].feasible is False
    assert result.site_results[0].lcow == FAIL_LCO
    assert result.site_results[0].failure_reason == "no weather profiles"
    assert result.combined_lcow == evaluator.PENALTY_LCOW_USD_PER_M3


def test_cpu_backend_rejects_multiple_sites(tmp_path, one_site, econ):
    """The CPU path is single-location only -- global/multi-site sweeping is JAX's job."""
    from sawh_bayesopt.sites import SiteSpec

    sites = (one_site[0], SiteSpec("second", 0.0, 0.0))
    with pytest.raises(ValueError, match="single-site"):
        evaluator.evaluate_batch(
            [(4.0, 40.0, 0.3, 0.0, 0.0)],
            cache=evaluator.EvalCache(tmp_path / "cache.jsonl"),
            sites=sites,
            site_frames={s.name: object() for s in sites},
            econ=econ,
            backend="cpu",
        )
