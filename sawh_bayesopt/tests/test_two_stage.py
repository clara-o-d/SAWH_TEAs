"""Daily surrogate + two-stage inner/outer loops, on a synthetic yield function with a
known optimum, plus one real-physics check that the per-day tilt override is the same
physics as a fixed-tilt year. Needs jax/equinox (and diffrax for the physics check): run
with solar_lumped/.venv_gpu, where they are installed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")

from sawh_bayesopt import daily_surrogate as ds  # noqa: E402
from sawh_bayesopt import two_stage as ts  # noqa: E402


def _true_water(design, control, feats):
    """Known optimum: seal = today's feature 0, open = -1 h, tilt = 30; thicker is wetter."""
    return (2.0 + 0.05 * design[..., 0] * 1000.0
            - 0.05 * (control[..., 0] - feats[..., 0]) ** 2
            - 0.05 * (control[..., 1] + 1.0) ** 2
            - ((control[..., 2] - 30.0) / 40.0) ** 2)


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    n = 30_000
    bounds = ds.design_bounds()
    design = bounds[:, 0] + rng.random((n, 3)) * (bounds[:, 1] - bounds[:, 0])
    control = ds.random_controls(rng, n, 1)[:, 0]
    feats = np.column_stack([rng.uniform(-2, 2, n), rng.normal(size=n)])
    state = np.column_stack([rng.uniform(4e4, 5e4, n), rng.uniform(3, 6, n)])
    X = ds.assemble_inputs(design, control, state, feats)
    return ds.fit_ensemble(X, _true_water(design, control, feats), state, np.zeros(n, bool),
                           cell=rng.integers(0, 50, n), n_members=3, width=64, steps=4000, batch=1024)


def _cell(n_days=24, seed=1):
    rng = np.random.default_rng(seed)
    feats = np.column_stack([rng.choice([-2.0, 2.0], n_days), rng.normal(size=n_days)])[None]
    month = np.repeat(np.arange(1, 13), n_days // 12)
    return feats, month


def test_ensemble_learns_the_function_and_ranks_designs_like_the_physics(model):
    rng = np.random.default_rng(5)
    design = np.tile([0.004, 0.02, 4.0], (200, 1))
    control = ds.random_controls(rng, 200, 1)[:, 0]
    feats = np.column_stack([rng.uniform(-2, 2, 200), rng.normal(size=200)])
    state = np.tile([4.5e4, 4.5], (200, 1))
    pred = np.asarray(ds.predict(model, ds.assemble_inputs(design, control, state, feats).astype(np.float32))["water_mean"])
    assert np.sqrt(np.mean((pred - _true_water(design, control, feats)) ** 2)) < 0.15

    # Held-out report on known truth: 3 cells x 6 designs x 8 days, physics = _true_water.
    n_cells, n_designs, n_days = 3, 6, 8
    bounds = ds.design_bounds()
    designs = bounds[:, 0] + rng.random((n_designs, 3)) * (bounds[:, 1] - bounds[:, 0])
    day_feats = np.stack([np.column_stack([rng.uniform(-2, 2, n_days), rng.normal(size=n_days)])
                          for _ in range(n_cells)])
    cell, inst, day, dsg, ctrl = [], [], [], [], []
    for c in range(n_cells):
        for d in range(n_designs):
            u = ds.random_controls(rng, 1, n_days)[0]
            for t in range(n_days):
                cell.append(c); inst.append(c * n_designs + d); day.append(t); dsg.append(designs[d]); ctrl.append(u[t])
    cell, day, dsg, ctrl = map(np.array, (cell, day, dsg, ctrl))
    st = np.tile([4.5e4, 4.5], (len(cell), 1))
    rows = {"cell": cell, "instance": np.array(inst), "day": day, "design": dsg, "control": ctrl, "state": st,
            "water": _true_water(dsg, ctrl, day_feats[cell, day]), "state_end": st, "capped": np.zeros(len(cell), bool)}
    rep = ds.grouped_holdout_report(model, rows, day_feats, np.array(["other"] * n_cells))
    assert rep["annual"]["all"]["median_rel_err"] < 0.05
    assert rep["design_ranking"]["all"]["median_spearman"] > 0.8
    assert rep["design_ranking"]["all"]["thickness_sign_agreement"] == 1.0
    assert rep["direction"]["all"]["physics_thickness_positive"] == 1.0  # _true_water rises with thickness



def test_enumeration_recovers_the_known_optimum(model):
    from solar_lumped.economics import LCOEconomicParams

    feats, month = _cell()
    [r] = ts.evaluate_designs(model, [(0, np.array([0.004, 0.02, 4.0]))], feats, month,
                              tilt_mode="daily", schedule_mode="hindsight", econ=LCOEconomicParams())
    c = r["controls"]
    assert np.mean(np.abs(c[:, 0] - feats[0, :, 0])) < 0.5
    assert np.mean(np.abs(c[:, 1] + 1.0)) < 0.5
    assert np.mean(np.abs(c[:, 2] - 30.0)) <= 5.0
    assert r["feasible"] and np.isfinite(r["lcow"])


def test_more_control_freedom_and_more_information_never_lose(model):
    """Nested control spaces: daily tilt >= seasonal >= fixed, and choosing on the real day
    beats choosing on yesterday's. Holds exactly on the surrogate's own objective here
    because the synthetic yield ignores state, so greedy is optimal."""
    from solar_lumped.economics import LCOEconomicParams

    feats, month = _cell()
    x = np.array([0.004, 0.02, 4.0])
    water = {}
    for tilt_mode, schedule in [("daily", "hindsight"), ("seasonal", "hindsight"), ("fixed", "hindsight"),
                                ("fixed", "persistence"), ("fixed", "constant"), ("fixed", "climatological")]:
        [r] = ts.evaluate_designs(model, [(0, x)], feats, month, tilt_mode=tilt_mode,
                                  schedule_mode=schedule, econ=LCOEconomicParams())
        water[tilt_mode, schedule] = r["water"]
    tol = 1e-4
    assert water["daily", "hindsight"] >= water["seasonal", "hindsight"] - tol
    assert water["seasonal", "hindsight"] >= water["fixed", "hindsight"] - tol
    assert water["fixed", "hindsight"] >= water["fixed", "persistence"] - tol
    assert water["fixed", "hindsight"] >= water["fixed", "constant"] - tol
    # The feature flips +-2 day to day, so yesterday's weather is a real handicap here.
    assert water["fixed", "hindsight"] - water["fixed", "persistence"] > 0.05


def test_outer_loop_respects_budget_and_bounds(model):
    from solar_lumped.economics import LCOEconomicParams

    feats, month = _cell(n_days=12)
    feats = np.concatenate([feats, feats])
    best = ts.optimize_cells(model, np.array([0, 1]), feats, month, tilt_mode="fixed",
                             schedule_mode="hindsight", econ=LCOEconomicParams(), n_init=3, n_total=5,
                             batch_size=2, warm_starts={1: [np.array([0.005, 0.03, 5.0])]}, n_jobs=1,
                             de_maxiter=10, de_popsize=5)
    lo, hi = ds.design_bounds().T
    for c in (0, 1):
        assert best[c]["n_evals"] == 5
        assert np.all(best[c]["design"] >= lo - 1e-12) and np.all(best[c]["design"] <= hi + 1e-12)


def test_active_requests_center_on_optima_and_stay_on_grid():
    rng = np.random.default_rng(0)
    ctrl = np.tile([0.0, -1.0, 30.0], (10, 1))
    best = {3: {"design": np.array([0.004, 0.02, 4.0]), "controls": ctrl, "water": 2.0, "water_std": 0.2},
            7: {"design": np.array([0.006, 0.03, 6.0]), "controls": ctrl, "water": 2.0, "water_std": 0.0}}
    reqs = ts.active_requests(best, n_cells=1, n_perturb=2, rng=rng)
    assert len(reqs) == 3 and {c for c, _x, _u in reqs} == {3}  # zero spread is never drawn
    np.testing.assert_array_equal(reqs[0][1], best[3]["design"])
    for _c, _x, u in reqs:
        assert np.all(np.isin(u[:, 0], ds.SEAL_GRID)) and np.all(np.isin(u[:, 2], ds.TILT_GRID))


def test_simulate_years_per_day_tilt_is_the_fixed_tilt_physics():
    """Swapping the tilt array per day (make_day_step_fn) must be the same physics as a
    config built at that tilt and walked by make_year_step_fn."""
    pytest.importorskip("diffrax")
    import pandas as pd

    from sawh_bayesopt import design_space
    from sawh_bayesopt.evaluator import _load_jax_daily_cycle
    from solar_lumped.physics import initial_loading
    from solar_lumped.simulation import SystemConfig
    from solar_lumped.weather import profile_from_day_df

    idx = pd.date_range("2024-06-01", periods=2 * 96, freq="15min")
    hour = idx.hour + idx.minute / 60.0
    df = pd.DataFrame({
        "temperature_2m": 25 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi),
        "relative_humidity_2m": 50 - 20 * np.sin((hour - 9) / 24 * 2 * np.pi),
        "shortwave_radiation": np.clip(900 * np.sin((hour - 6) / 12 * np.pi), 0, None),
        "latitude": 25.0, "longitude": 0.0, "utc_offset_s": 0.0, "elevation_m": 0.0,
    }, index=idx)
    day_index = np.array([152, 153])
    design = np.array([[0.004, 0.02, 4.0]])
    controls = np.array([[[0.5, -0.5, 45.0], [0.5, -0.5, 45.0]]])
    rows = ds.simulate_years([df], design, controls, day_index, progress_every=0)

    jdc = _load_jax_daily_cycle()
    cfg = SystemConfig(**design_space.to_system_config_kwargs(np.array([0.004, 0.02, 4.0, 45.0, 0.5, -0.5])))
    profiles = [profile_from_day_df(g, poa_tilt_deg=45.0, seal_offset_h=0.5, open_offset_h=-0.5)
                for _d, g in df.groupby(df.index.date)]
    dt, n_abs, n_des = jdc.year_padding([profiles])
    step = jdc.make_year_step_fn(jdc.build_system_arrays([cfg]), dt, n_abs, n_des)
    days = jdc.run_year_batched(step, [jdc.build_day_weather([p], n_abs, n_des) for p in profiles],
                                c_w_initial=np.array([initial_loading(cfg)]),
                                h_initial=np.array([cfg.hydrogel_thickness_m]))
    # Different padding (simulate_years pads to the whole day) changes the adaptive
    # stepping slightly, never the answer.
    np.testing.assert_allclose(rows["water"], days["water"][:, 0], rtol=1e-4)
