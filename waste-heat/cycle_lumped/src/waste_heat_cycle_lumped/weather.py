"""Open-Meteo weather client, diurnal-profile aggregation, and data-center operating
profiles for the two-bed waste-heat SAWH.

Consolidated from the former weather/{client, climate, profiles}.py. Section headers
below mark each former module's boundary for traceability.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
import retry_requests

from waste_heat_cycle_lumped.physics import (
    H_AMB_W_M2_K,
    M_WH_KG_S_M2,
    RH_AMB,
    TAU_HALF_S,
    T_AMB_C,
    T_WH_IN_C,
    initial_bed_states,
)

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

DEFAULT_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "wind_speed_10m",
)


class WeatherClient:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        session_timeout: int = 60,
        *,
        max_retries: int = 5,
        retry_backoff_factor: float = 2.0,
    ) -> None:
        self._timeout = session_timeout
        self._session = self._build_session(
            cache_dir, max_retries=max_retries, retry_backoff_factor=retry_backoff_factor
        )

    def get_historical(
        self,
        latitude: float,
        longitude: float,
        start: str | date,
        end: str | date,
        timezone: str = "auto",
    ) -> pd.DataFrame:
        start_str = str(start) if isinstance(start, date) else start
        end_str = str(end) if isinstance(end, date) else end
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_str,
            "end_date": end_str,
            "timezone": timezone,
            "hourly": ",".join(DEFAULT_VARIABLES),
        }
        return self._fetch(_ARCHIVE_URL, params, latitude, longitude)

    def get_historical_forecast_site_weather(
        self,
        latitude: float,
        longitude: float,
        start: str | date,
        end: str | date,
        timezone: str = "auto",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        start_str = str(start) if isinstance(start, date) else start
        end_str = str(end) if isinstance(end, date) else end
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_str,
            "end_date": end_str,
            "timezone": timezone,
            "hourly": ",".join(DEFAULT_VARIABLES),
            "minutely_15": ",".join(DEFAULT_VARIABLES),
        }
        response = self._session.get(
            _HISTORICAL_FORECAST_URL, params=params, timeout=self._timeout
        )
        _raise_for_openmeteo_error(response)
        data = response.json()
        df_hourly = self._series_to_dataframe(data, "hourly", latitude, longitude)
        df_min15 = self._series_to_dataframe(data, "minutely_15", latitude, longitude)
        return df_hourly, df_min15

    def _build_session(
        self,
        cache_dir: str | Path | None,
        *,
        max_retries: int,
        retry_backoff_factor: float,
    ) -> requests.Session:
        if cache_dir is not None:
            try:
                import requests_cache

                session = requests_cache.CachedSession(
                    cache_name=str(Path(cache_dir) / "openmeteo_cache"),
                    backend="sqlite",
                    expire_after=timedelta(hours=6),
                )
            except ImportError:
                warnings.warn("requests-cache not installed; caching disabled.", stacklevel=2)
                session = requests.Session()
        else:
            session = requests.Session()
        return retry_requests.retry(
            session,
            retries=max_retries,
            backoff_factor=retry_backoff_factor,
            status_to_retry=(429, 500, 502, 503, 504),
        )

    def _fetch(
        self,
        url: str,
        params: dict,
        latitude: float,
        longitude: float,
    ) -> pd.DataFrame:
        response = self._session.get(url, params=params, timeout=self._timeout)
        _raise_for_openmeteo_error(response)
        data = response.json()
        return self._series_to_dataframe(data, "hourly", latitude, longitude)

    @staticmethod
    def _series_to_dataframe(
        data: dict,
        series_key: Literal["hourly", "minutely_15"],
        latitude: float,
        longitude: float,
    ) -> pd.DataFrame:
        series = dict(data.get(series_key, {}))
        if not series:
            raise ValueError(f"API returned no {series_key} data.")
        times = pd.to_datetime(series.pop("time"))
        df = pd.DataFrame(series, index=times)
        df.index.name = "time"
        tz = data.get("timezone")
        if tz and tz != "UTC":
            try:
                df.index = df.index.tz_localize(tz)
            except Exception:
                pass
        df["latitude"] = data.get("latitude", latitude)
        df["longitude"] = data.get("longitude", longitude)
        return df


def _raise_for_openmeteo_error(response: requests.Response) -> None:
    if response.status_code == 200:
        return
    try:
        detail = response.json().get("reason", response.text)
    except Exception:
        detail = response.text
    raise requests.HTTPError(
        f"Open-Meteo API error {response.status_code}: {detail}",
        response=response,
    )
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR


def _mean_by_slot(df: pd.DataFrame, col: str) -> tuple[float, ...]:
    slot = df.index.hour * STEPS_PER_HOUR + df.index.minute // 15  # 15-min slot 0..95
    grouped = df[col].groupby(slot).mean()
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
PROFILE_DT_S = 60.0


@dataclass(frozen=True, slots=True)
class HalfCycleProfile:
    """Weather / boundary conditions for one half-cycle (up to max duration τ_1/2,max)."""

    temperature_c: tuple[float, ...]
    relative_humidity: tuple[float, ...]
    h_amb_w_m2_k: tuple[float, ...]
    t_wh_in_c: tuple[float, ...]
    m_dot_wh_kg_s_m2: tuple[float, ...]
    dt_s: float = PROFILE_DT_S


def _steps_for_tau(tau_half_s: float, dt_s: float = PROFILE_DT_S) -> int:
    return max(4, int(round(tau_half_s / dt_s)))


def datacenter_baseline_profile(
    *,
    tau_half_s: float | None = None,
    dt_s: float = PROFILE_DT_S,
    t_amb_c: float = T_AMB_C,
    rh: float = RH_AMB,
    h_amb: float = H_AMB_W_M2_K,
    t_wh_in_c: float = T_WH_IN_C,
    m_dot_wh_kg_s_m2: float = M_WH_KG_S_M2,
) -> HalfCycleProfile:
    tau = tau_half_s if tau_half_s is not None else TAU_HALF_S
    n = _steps_for_tau(tau, dt_s)
    return HalfCycleProfile(
        temperature_c=(t_amb_c,) * n,
        relative_humidity=(rh,) * n,
        h_amb_w_m2_k=(h_amb,) * n,
        t_wh_in_c=(t_wh_in_c,) * n,
        m_dot_wh_kg_s_m2=(m_dot_wh_kg_s_m2,) * n,
        dt_s=dt_s,
    )


def datacenter_diurnal_profile(
    *,
    tau_half_s: float | None = None,
    dt_s: float = PROFILE_DT_S,
    t_amb_mean_c: float = T_AMB_C,
    t_amb_amp_c: float = 3.0,
    rh_mean: float = RH_AMB,
    rh_amp: float = 0.08,
    h_amb: float = H_AMB_W_M2_K,
    t_wh_in_c: float = T_WH_IN_C,
    m_dot_wh_kg_s_m2: float = M_WH_KG_S_M2,
) -> HalfCycleProfile:
    tau = tau_half_s if tau_half_s is not None else TAU_HALF_S
    n = _steps_for_tau(tau, dt_s)
    temps: list[float] = []
    rhs: list[float] = []
    for i in range(n):
        phase = 2.0 * math.pi * i / n
        temps.append(t_amb_mean_c + t_amb_amp_c * math.sin(phase))
        rhs.append(max(0.05, min(0.95, rh_mean + rh_amp * math.cos(phase))))
    return HalfCycleProfile(
        temperature_c=tuple(temps),
        relative_humidity=tuple(rhs),
        h_amb_w_m2_k=(h_amb,) * n,
        t_wh_in_c=(t_wh_in_c,) * n,
        m_dot_wh_kg_s_m2=(m_dot_wh_kg_s_m2,) * n,
        dt_s=dt_s,
    )


def initial_loadings(config) -> tuple[float, float]:
    """Default (loading_adsorbing, loading_desorbing) at cycle start."""
    bed_a, bed_d = initial_bed_states(config)
    return bed_a.loading, bed_d.loading
