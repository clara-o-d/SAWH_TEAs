"""Featurization, stratified selection and OOD flagging -- no jax needed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sawh_bayesopt import climate


def _synthetic_frame(n_days: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-01", periods=n_days * 96, freq="15min")
    hour = idx.hour + idx.minute / 60.0
    day = np.repeat(rng.normal(0, 3, n_days), 96)
    t = 20 + day + 8 * np.sin((hour - 9) / 24 * 2 * np.pi)
    return pd.DataFrame({
        "temperature_2m": t,
        "relative_humidity_2m": np.clip(60 - 2 * (t - 20) + np.repeat(rng.normal(0, 5, n_days), 96), 5, 100),
        "shortwave_radiation": np.clip(900 * np.sin((hour - 6) / 12 * np.pi), 0, None),
        "latitude": 20.0, "longitude": 10.0, "utc_offset_s": 3600.0, "elevation_m": 500.0,
    }, index=idx)


def test_full_rank_pca_round_trip_is_exact_and_frame_reconstruction_tracks():
    df = _synthetic_frame()
    matrix, dates = climate.daily_weather_matrix(df)
    assert matrix.shape == (40, 72) and len(dates) == 40
    pca = climate.fit_climate_pca(matrix)
    np.testing.assert_allclose(climate.reconstruct(pca, climate.project(pca, matrix), 72), matrix, atol=1e-8)
    # At full rank the only error left is 15-min -> hourly -> 15-min interpolation.
    rebuilt = climate.reconstruct_day_frame(df, pca, 72)
    assert (rebuilt["temperature_2m"] - df.loc[rebuilt.index, "temperature_2m"]).abs().mean() < 0.3
    assert rebuilt["shortwave_radiation"].min() >= 0.0


def test_day_scalars_are_physical():
    df = _synthetic_frame(n_days=5)
    matrix, dates = climate.daily_weather_matrix(df)
    s = climate.day_scalars(matrix, np.arange(len(matrix)), lat=-20.0, elevation_m=500.0)
    col = dict(zip(climate.SCALARS, s[0]))
    assert 14.0 < col["t_diurnal_amplitude_c"] < 17.0  # a +-8 C sine, sampled hourly
    assert col["dewpoint_depression_at_peak_sun_c"] > 0.0
    assert col["elevation_km"] == 0.5 and col["abs_latitude_deg"] == 20.0


def test_selection_meets_quotas_and_never_splits_a_cell_across_train_and_test():
    rng = np.random.default_rng(0)
    desc = rng.normal(size=(400, 6))
    regime = np.array(["other"] * 380 + ["hyper_arid"] * 20)
    sel = climate.select_representatives(desc, regime, n_clusters=30, regime_quota={"hyper_arid": 12}, seed=0)
    assert (sel["regime"] == "hyper_arid").sum() >= 12
    assert sel["cell"].is_unique
    assert set(sel.loc[sel.split == "train", "cell"]).isdisjoint(sel.loc[sel.split == "test", "cell"])
    assert (sel.loc[sel.regime == "hyper_arid", "split"] == "test").any()


def test_mahalanobis_flags_a_planted_outlier_only():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(500, 5))
    probe = np.vstack([np.zeros(5), np.full(5, 8.0)])
    _d2, ood, _thr = climate.mahalanobis_ood(train, probe)
    assert ood.tolist() == [False, True]


def test_regime_rules_first_match_wins():
    meta = {"lat": np.zeros(3), "elevation_m": np.array([3000.0, 100.0, 100.0]),
            "mean_rh_pct": np.array([10.0, 10.0, 80.0]), "rh_seasonal_amplitude_pct": np.array([0.0, 0.0, 5.0]),
            "t_diurnal_amplitude_c": np.array([10.0, 10.0, 4.0])}
    assert climate.label_regimes(meta).tolist() == ["high_altitude", "hyper_arid", "coastal_humid"]
