"""Tests for weather climate aggregation."""

from datetime import date

import pandas as pd
import pytest

from solar_lumped.weather.climate import (
    STEPS_PER_DAY,
    representative_mean_day_df,
)
from solar_lumped.weather.profiles import profile_from_day_df


def _synthetic_year_15min() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", "2024-01-02 23:45", freq="15min")[:-1]
    n = len(index)
    hours = index.hour + index.minute / 60.0
    return pd.DataFrame(
        {
            "relative_humidity_2m": 50.0 + 20.0 * (hours / 24.0),
            "temperature_2m": 15.0 + 10.0 * (hours / 24.0),
            "shortwave_radiation": [max(0.0, 800.0 * (h - 6) / 6.0) if 6 <= h <= 18 else 0.0 for h in hours],
            "wind_speed_10m": 1.0 + 0.1 * (hours / 24.0),
        },
        index=index,
    )


def test_representative_mean_day_has_96_slots():
    df = _synthetic_year_15min()
    mean_day = representative_mean_day_df(df, reference_day=date(2024, 6, 15))
    assert len(mean_day) == STEPS_PER_DAY
    assert mean_day.index[0].hour == 0
    assert mean_day.index[0].minute == 0


def test_representative_mean_day_averages_slots():
    df = _synthetic_year_15min()
    mean_day = representative_mean_day_df(df)
    slot0_rh = df.loc[(df.index.hour == 0) & (df.index.minute == 0), "relative_humidity_2m"].mean()
    assert mean_day["relative_humidity_2m"].iloc[0] == pytest.approx(slot0_rh)


def test_representative_mean_day_builds_valid_profile():
    df = _synthetic_year_15min()
    mean_day = representative_mean_day_df(df)
    profile = profile_from_day_df(mean_day)
    assert len(profile.absorption.temperature_c) == len(profile.desorption.temperature_c)
    assert max(profile.desorption.solar_w_m2) > 0.0
    assert max(profile.absorption.solar_w_m2) == 0.0
