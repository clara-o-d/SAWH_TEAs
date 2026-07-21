"""Tiny end-to-end run against the real model -- confirms wiring (config ->
jax_daily_cycle's batched daily-cycle/Aitken pipeline -> lcow_from_daily_yield
-> cache -> GP fit) doesn't error and produces a finite LCOW. Not meant to
validate optimization quality (n_init/n_total are far too small for that) --
see scripts/run_bayesopt.py for a real run.

Guarded because it needs jax/diffrax installed and a real weather fetch: set
SAWH_BAYESOPT_SLOW_TESTS=1 to opt in, e.g.

    SAWH_BAYESOPT_SLOW_TESTS=1 pytest tests/test_integration_real_model.py -v
"""

from __future__ import annotations

import math
import os

import pytest


def _slow_tests_enabled() -> bool:
    if os.environ.get("SAWH_BAYESOPT_SLOW_TESTS") != "1":
        return False
    try:
        import diffrax  # noqa: F401
        import jax  # noqa: F401
        import solar_lumped  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _slow_tests_enabled(),
    reason="real-model integration test needing jax/diffrax; set SAWH_BAYESOPT_SLOW_TESTS=1 to run",
)


def test_tiny_real_bayesopt_run_produces_finite_lcow(tmp_path):
    from sawh_bayesopt.bayesopt import BayesOptConfig, run_bayesopt
    from sawh_bayesopt.sites import CAMBRIDGE

    cfg = BayesOptConfig(
        sites=(CAMBRIDGE,),
        n_init=2,
        n_total=3,
        batch_size=1,
        seed=0,
    )
    result = run_bayesopt(cfg, tmp_path / "real_run")

    assert len(result.history) >= cfg.n_init
    assert math.isfinite(result.best.combined_lcow)
    assert result.best.combined_lcow < 1e9  # nowhere near a penalty/FAIL_LCO value
