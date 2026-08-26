"""The measured Stanford met-tower frame has to look like an Open-Meteo one, and it has
to be clean: the export carries blank cells and a -38.6 °F stuck-thermistor sentinel that
would otherwise ride straight into the ODE state."""

from __future__ import annotations

from solar_lumped.weather import (
    DEFAULT_VARIABLES,
    real_weather_days_from_df,
    stanford_year_weather,
)


def test_stanford_frame_is_a_clean_open_meteo_shaped_year() -> None:
    df = stanford_year_weather(2025)

    assert set(DEFAULT_VARIABLES).issubset(df.columns)
    assert {"latitude", "longitude", "utc_offset_s", "elevation_m"}.issubset(df.columns)
    assert not df.isna().to_numpy().any()
    assert (df.index.year == 2025).all()
    assert df.index.is_monotonic_increasing

    # Palo Alto, not a stuck sensor: 2025 never froze and never hit 60 C.
    assert -5.0 < df["temperature_2m"].min() < 10.0
    assert 25.0 < df["temperature_2m"].max() < 50.0
    assert df["relative_humidity_2m"].between(0.0, 100.0).all()
    assert df["shortwave_radiation"].between(0.0, 1500.0).all()

    # Every calendar day must survive the day/night split, or annual runs silently
    # integrate a short year.
    assert len(real_weather_days_from_df(df)) == 365
