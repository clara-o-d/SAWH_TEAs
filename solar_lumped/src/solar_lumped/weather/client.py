"""Open-Meteo weather client (standalone, trimmed)."""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
import retry_requests

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
                    # Requests are always for a fixed, already-elapsed date range
                    # (a past calendar year), so the archive response never changes.
                    expire_after=requests_cache.NEVER_EXPIRE,
                    # WAL mode lets concurrent readers/writers not block each other
                    # (vs. SQLite's default rollback-journal mode, which serializes
                    # all writes); busy_timeout (ms) is how long a writer waits on a
                    # lock before raising "database is locked" instead of the
                    # 5s sqlite3 default -- both needed once many GPU-sweep array
                    # tasks share this one cache file concurrently (see
                    # gpu_sweep/FINDINGS.md/docs/gpu_sweep_handoff.md).
                    wal=True,
                    busy_timeout=60_000,
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
