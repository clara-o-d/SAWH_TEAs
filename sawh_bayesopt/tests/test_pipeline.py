"""Checks for the pieces the Sherlock stages are built from, on synthetic inputs. The last
test runs real physics (a few days, CPU JAX), so it needs solar_lumped/.venv_gpu."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")


def test_greedy_maximin_matches_brute_force_algorithm_16_8():
    """Every pick must be the candidate minimizing d_max(X, S + {x}), recomputed from
    scratch -- the incremental nearest-distance bookkeeping is the only shortcut."""
    from scipy.spatial.distance import cdist

    from sawh_bayesopt.climate import greedy_maximin_subset

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    S, curve = greedy_maximin_subset(X, 8, rng=np.random.default_rng(1), initial=np.array([5, 17]), col_chunk=7)
    ref = [5, 17]
    for _ in range(6):
        d = [cdist(X, X[ref + [x]]).min(axis=1).max() if x not in ref else np.inf for x in range(len(X))]
        ref.append(int(np.argmin(d)))
    assert list(S) == ref
    assert np.all(np.diff(curve[1:]) <= 1e-6)  # coverage never gets worse


def test_select_cells_desert_stage_and_split():
    from sawh_bayesopt.climate import select_cells

    rng = np.random.default_rng(0)
    n = 400
    groups = rng.choice(["desert", "tropical", "temperate"], n)
    classes = np.where(groups == "desert", rng.choice(["BWh", "BWk"], n), "Af")
    sel, curves = select_cells(rng.normal(size=(n, 5)), groups, classes, n_space_filling=60, n_desert=30,
                               split_fracs=(0.7, 0.15, 0.15), seed=0)
    assert len(sel) == 90 and sel.cell.is_unique
    assert set(sel.loc[sel.stage == "desert", "koppen"]) <= {"BWh", "BWk"}
    assert len(curves["desert_d_max"]) == 30
    frac = sel.split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.7) < 0.05 and abs(frac["test"] - 0.15) < 0.05


def test_constrained_bo_finds_a_1d_optimum_next_to_an_infeasible_region():
    """Minimum at the edge of the feasible set: BO must get within one grid step of it in
    its budget on every problem, for EI and LCB."""
    from sawh_bayesopt.optimize import BOConfig, run_bo

    grid = np.linspace(0, 1, 181)[:, None]
    optima = np.array([0.3, 0.55, 0.7])

    def f(p, x):
        return 5.0 + 20.0 * (x - optima[p] - 0.1) ** 2, x - optima[p]

    def evaluate(requests):
        return [(float(f(p, grid[i, 0])[0]), float(f(p, grid[i, 0])[1]), None) for p, i in requests]

    for acq, param in (("ei", 0.01), ("lcb", 2.0)):
        hist = run_bo(evaluate, len(optima), grid, BOConfig(acquisition=acq, acq_param=param, budget=14),
                      seed=0, n_jobs=1, log_every=0)
        for p, h in enumerate(hist):
            y = np.where(h["feasible"], h["y"], np.inf)
            assert abs(grid[h["idx"][int(np.argmin(y))], 0] - optima[p]) <= 1 / 180 + 1e-9, (acq, p)


def test_decision_features_regimes():
    from sawh_bayesopt.optimize import decision_features

    feats = np.arange(366 * 2, dtype=float).reshape(1, 366, 2)
    month = pd.date_range("2024-01-01", periods=366).month.to_numpy()
    assert np.array_equal(decision_features(feats, month, "persistence")[0, 1], feats[0, 0])
    monthly = decision_features(feats, month, "monthly")
    assert np.allclose(monthly[0, :31], feats[0, :31].mean(axis=0))
    assert np.allclose(decision_features(feats, month, "yearly")[0], feats[0].mean(axis=0))


def _toy_surrogate(n_in: int, family: str):
    """A fitted family on synthetic data shaped like real rows (x[:, 6:8] is the state)."""
    from sawh_bayesopt import surrogates

    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, n_in))
    X[:, 6:8] = rng.uniform([1e4, 1.0], [5e4, 10.0], size=(3000, 2))
    Y = np.column_stack([np.log1p(np.abs(X[:, 1]) + 0.2 * X[:, 2] ** 2), X[:, 6] * 0.9, X[:, 7] * 0.95,
                         (X[:, 3] > 1.0).astype(float)])
    rows = {"water": np.expm1(Y[:, 0]), "state_end": Y[:, 1:3], "capped": Y[:, 3] > 0.5}
    split = {"X": X, "Y": Y, "rows": rows}
    hyper = {"alpha": 1.0} if family == "ridge" else {"width": 16, "depth": 2, "lr": 3e-3}
    if family == "mlp":
        norm = surrogates.normalization(X, Y)
        Xz, Yz = surrogates.standardize(norm, X, Y)
        model, _conv = surrogates._fit_mlp(hyper, norm, Xz, Yz, Xz, Yz, split, seed=0, steps=200, batch=256,
                                           eval_every=100)
        model["fit_seconds"] = 0.0
        return model, X
    model, _conv = surrogates.fit_family(family, hyper, {"train": split, "val": split}, seed=0)
    return model, X


@pytest.mark.parametrize("family", ["ridge", "mlp"])
def test_deployed_jax_predictor_matches_numpy_predict(family):
    import jax
    import jax.numpy as jnp

    from sawh_bayesopt import surrogates

    model, X = _toy_surrogate(12, family)
    net, norm, capped_logit = surrogates.deployable(model)
    z = jax.vmap(net)((jnp.asarray(X[:50], jnp.float32) - norm["x_mean"]) / norm["x_std"])
    y = np.asarray(z[:, :3]) * model["norm"]["y_std"] + model["norm"]["y_mean"]
    ref = surrogates.predict(model, X[:50])
    np.testing.assert_allclose(np.maximum(np.expm1(y[:, 0]), 0), ref["water"], rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(y[:, 1:3], ref["state"], rtol=2e-4)
    assert capped_logit == (family == "mlp")


@pytest.mark.parametrize("tilt_mode", ["daily", "seasonal", "three_per_day"])
def test_inner_year_controls_respect_the_tilt_mode(tilt_mode):
    """Shortened year (the scan is over whatever days it is given). Daily: equal tilts each
    day. Seasonal: one tilt per quarter. Three-per-day: drawn from the 15-degree grid."""
    import jax.numpy as jnp

    from sawh_bayesopt import surrogates
    from sawh_bayesopt.optimize import make_year_fn
    from sawh_bayesopt.simulate import TILT3_GRID

    n_days, n_feat = 40, 4
    model, _X = _toy_surrogate(8 + n_feat, "ridge")
    net, norm, capped_logit = surrogates.deployable(model)
    month = np.repeat([1, 4, 7, 10], 10)  # four quarters of 10 days
    year = make_year_fn(net, norm, capped_logit, tilt_mode, month)
    rng = np.random.default_rng(0)
    feats = jnp.asarray(rng.normal(size=(2, n_days, n_feat)), jnp.float32)
    water, capped, controls = year(jnp.asarray([0.003, 0.006], jnp.float32), feats, feats,
                                   jnp.asarray([[2e4, 3.0], [2e4, 6.0]], jnp.float32))
    u = np.asarray(controls)
    assert u.shape == (2, n_days, 5) and np.asarray(water).shape == (2, n_days)
    tilt = u[..., 2:]
    if tilt_mode != "three_per_day":
        assert np.all(tilt == tilt[..., :1])
    if tilt_mode == "seasonal":
        for q in range(4):
            assert len(np.unique(tilt[0, 10 * q:10 * q + 10, 0])) == 1
    if tilt_mode == "three_per_day":
        assert np.isin(tilt, TILT3_GRID).all()


def test_random_controls_mix_equal_and_independent_tilts():
    from sawh_bayesopt.simulate import SEAL_GRID, TILT_GRID, random_controls

    u = random_controls(np.random.default_rng(0), 50, 366)
    equal = (u[..., 2] == u[..., 3]) & (u[..., 3] == u[..., 4])
    assert 0.45 < equal.mean() < 0.6  # half by construction, plus chance coincidences
    assert np.isin(u[..., 0], SEAL_GRID).all() and np.isin(u[..., 2:], TILT_GRID).all()


def test_physics_years_run_through_the_tilt_channel():
    """Three real days on a synthetic frame: two instances at the same thickness under a
    flat and a steep three-segment schedule must both converge, chain, and differ."""
    pytest.importorskip("diffrax")
    from sawh_bayesopt.simulate import simulate_years

    idx = pd.date_range("2024-06-20 00:00", periods=96 * 3, freq="15min")
    hours = idx.hour + idx.minute / 60.0
    ghi = np.where((hours >= 6) & (hours <= 18), 900 * np.sin(np.pi * (hours - 6) / 12), 0.0)
    df = pd.DataFrame({"temperature_2m": 25 - 8 * np.cos(2 * np.pi * (hours - 3) / 24),
                       "relative_humidity_2m": 55 + 30 * np.cos(2 * np.pi * (hours - 3) / 24),
                       "shortwave_radiation": ghi, "latitude": 25.0, "longitude": 30.0, "utc_offset_s": 7200.0,
                       "elevation_m": 300.0}, index=idx)
    days = np.array([171, 172, 173])
    controls = np.zeros((2, 3, 5))
    controls[1, :, 2:] = (0.0, 30.0, 60.0)
    rows = simulate_years([df, df], np.array([0.003, 0.003]), controls, days, progress_every=0)
    assert set(rows["instance"]) == {0, 1}
    w = [rows["water"][rows["instance"] == i] for i in (0, 1)]
    assert np.all(w[0] > 0) and not np.allclose(w[0], w[1])
    for i in (0, 1):
        m = rows["instance"] == i
        np.testing.assert_array_equal(rows["state"][m][1:], rows["state_end"][m][:-1])
