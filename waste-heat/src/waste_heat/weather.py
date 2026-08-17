"""Open-Meteo weather client, diurnal-profile aggregation, and data-center operating
profiles for the two-bed waste-heat SAWH (no HTF loop -- direct waste-heat coupling).

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

from solar_lumped._parameters_xlsx import physics_value as _pv
from waste_heat.physics import (
    H_AMB_W_M2_K,
    M_WH_KG_S_M2,
    RH_AMB,
    TAU_HALF_S,
    T_AMB_C,
    T_WH_IN_C,
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
        self._session = retry_requests.retry(
            session,
            retries=max_retries,
            backoff_factor=retry_backoff_factor,
            status_to_retry=(429, 500, 502, 503, 504),
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
        response = self._session.get(_ARCHIVE_URL, params=params, timeout=self._timeout)
        _raise_for_openmeteo_error(response)
        data = response.json()
        return self._series_to_dataframe(data, "hourly", latitude, longitude)

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
PROFILE_DT_S = _pv("Waste-heat: weather profile time step")


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

