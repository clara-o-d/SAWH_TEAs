"""Aggregate hourly / 15-min weather into representative diurnal profiles."""

from __future__ import annotations

from datetime import date

import pandas as pd

STEPS_PER_HOUR = 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR


def _slot_index(index: pd.DatetimeIndex) -> pd.Series:
    """Map each timestamp to its 15-min slot within the calendar day (0..95)."""
    return index.hour * STEPS_PER_HOUR + index.minute // 15


def _mean_by_slot(df: pd.DataFrame, col: str) -> tuple[float, ...]:
    grouped = df[col].groupby(_slot_index(df.index)).mean()
    fallback = float(grouped.mean()) if len(grouped) else 0.0
    return tuple(float(grouped.get(s, fallback)) for s in range(STEPS_PER_DAY))


def representative_kinetics_rh_from_minutely_15(df: pd.DataFrame) -> tuple[float, ...]:
    """Mean relative humidity (fraction 0–1) for each 15-min slot 0..95 within a day."""
    if "relative_humidity_2m" not in df.columns:
        raise KeyError("DataFrame must contain column 'relative_humidity_2m'")
    rh_pct = _mean_by_slot(df, "relative_humidity_2m")
    return tuple(r / 100.0 for r in rh_pct)


def representative_kinetics_temperature_from_minutely_15(df: pd.DataFrame) -> tuple[float, ...]:
    """Mean 2 m air temperature (deg C) for each 15-min slot 0..95 within a day."""
    col = "temperature_2m"
    if col not in df.columns:
        raise KeyError(f"DataFrame must contain column {col!r}")
    return _mean_by_slot(df, col)


def representative_kinetics_solar_from_minutely_15(df: pd.DataFrame) -> tuple[float, ...]:
    """Mean shortwave GHI (W/m²) for each 15-min slot 0..95 within a day."""
    col = "shortwave_radiation"
    if col not in df.columns:
        raise KeyError(f"DataFrame must contain column {col!r}")
    solar = _mean_by_slot(df, col)
    return tuple(max(0.0, s) for s in solar)


def representative_kinetics_wind_from_minutely_15(df: pd.DataFrame) -> tuple[float, ...]:
    """Mean 10 m wind speed (m/s) for each 15-min slot 0..95 within a day."""
    col = "wind_speed_10m"
    if col not in df.columns:
        return (0.5,) * STEPS_PER_DAY
    return _mean_by_slot(df, col)


def representative_hourly_rh_from_hourly(df: pd.DataFrame) -> tuple[float, ...]:
    """Mean relative humidity (fraction 0–1) for each hour-of-day 0..23."""
    if "relative_humidity_2m" not in df.columns:
        raise KeyError("DataFrame must contain column 'relative_humidity_2m'")
    hourly = df["relative_humidity_2m"].groupby(df.index.hour).mean() / 100.0
    return tuple(float(hourly.get(h, hourly.mean())) for h in range(24))


def representative_hourly_temperature_from_hourly(df: pd.DataFrame) -> tuple[float, ...]:
    """Mean 2 m air temperature (deg C) for each hour-of-day 0..23."""
    col = "temperature_2m"
    if col not in df.columns:
        raise KeyError(f"DataFrame must contain column {col!r}")
    hourly = df[col].groupby(df.index.hour).mean()
    return tuple(float(hourly.get(h, hourly.mean())) for h in range(24))


def representative_hourly_solar_from_hourly(df: pd.DataFrame) -> tuple[float, ...]:
    """Mean shortwave GHI (W/m²) for each hour-of-day 0..23."""
    col = "shortwave_radiation"
    if col not in df.columns:
        raise KeyError(f"DataFrame must contain column {col!r}")
    hourly = df[col].groupby(df.index.hour).mean()
    return tuple(float(max(0.0, hourly.get(h, hourly.mean()))) for h in range(24))


def representative_hourly_wind_from_hourly(df: pd.DataFrame) -> tuple[float, ...]:
    """Mean 10 m wind speed (m/s) for each hour-of-day 0..23."""
    col = "wind_speed_10m"
    if col not in df.columns:
        return (0.5,) * 24
    hourly = df[col].groupby(df.index.hour).mean()
    return tuple(float(hourly.get(h, hourly.mean())) for h in range(24))


def _expand_hourly_to_15min(hourly: tuple[float, ...]) -> tuple[float, ...]:
    out: list[float] = []
    for value in hourly:
        out.extend([value] * STEPS_PER_HOUR)
    return tuple(out[:STEPS_PER_DAY])


def representative_mean_day_df(
    df: pd.DataFrame,
    *,
    reference_day: date | None = None,
) -> pd.DataFrame:
    """Build one synthetic calendar day from mean slot values across *df*.

    Uses native 15-min resolution when timestamps are 15-min spaced; otherwise
    falls back to hourly means expanded to 96 slots.
    """
    if df.empty:
        raise ValueError("Cannot build representative mean day from empty weather data.")

    ref = reference_day or date(2024, 1, 1)
    base = pd.Timestamp(ref)
    if df.index.tz is not None:
        base = base.tz_localize(df.index.tz)

    median_delta_min = float(df.index.to_series().diff().dropna().dt.total_seconds().median() / 60.0)
    if median_delta_min <= 20.0:
        rh = representative_kinetics_rh_from_minutely_15(df)
        temp = representative_kinetics_temperature_from_minutely_15(df)
        solar = representative_kinetics_solar_from_minutely_15(df)
        wind = representative_kinetics_wind_from_minutely_15(df)
        freq = "15min"
    else:
        rh_h = representative_hourly_rh_from_hourly(df)
        temp_h = representative_hourly_temperature_from_hourly(df)
        solar_h = representative_hourly_solar_from_hourly(df)
        wind_h = representative_hourly_wind_from_hourly(df)
        rh = _expand_hourly_to_15min(rh_h)
        temp = _expand_hourly_to_15min(temp_h)
        solar = _expand_hourly_to_15min(solar_h)
        wind = _expand_hourly_to_15min(wind_h)
        freq = "15min"

    index = pd.date_range(base, periods=STEPS_PER_DAY, freq=freq)
    out = pd.DataFrame(
        {
            "relative_humidity_2m": [r * 100.0 for r in rh],
            "temperature_2m": temp,
            "shortwave_radiation": solar,
            "wind_speed_10m": wind,
        },
        index=index,
    )
    out.index.name = "time"
    if "latitude" in df.columns:
        out["latitude"] = float(df["latitude"].iloc[0])
    if "longitude" in df.columns:
        out["longitude"] = float(df["longitude"].iloc[0])
    return out
