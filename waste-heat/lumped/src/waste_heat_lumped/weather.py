"""Weather: Open-Meteo client, real/replay/baseline profile builders, and the
Note S1 Fig. S1D / Atacama Fig. 4 validation profiles.

Consolidated from the former weather/{client, climate, profiles, fig_s1,
atacama_figure}.py. Section headers below mark each former module's boundary
for traceability.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
import retry_requests

from waste_heat_lumped.physics import (
    FABRICATION_EQUILIBRIUM_RH,
    H0_M,
    H_AMB_W_M2_K,
    RH_AMB,
    T_AMB_C,
    c_w_from_water_in_gel_l_m2,
    equilibrium_c_w_from_dvs_at_rh,
)


# =============================================================================
# Open-Meteo weather API client (real historical + forecast retrieval)
# =============================================================================

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

# =============================================================================
# Real-weather day statistics
# =============================================================================

STEPS_PER_HOUR = 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR


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


def representative_kinetics_rh_from_minutely_15(df: pd.DataFrame) -> tuple[float, ...]:
    """Mean relative humidity (fraction 0–1) for each 15-min slot 0..95 within a day."""
    if "relative_humidity_2m" not in df.columns:
        raise KeyError("DataFrame must contain column 'relative_humidity_2m'")
    rh_pct = _mean_by_slot(df, "relative_humidity_2m")
    return tuple(r / 100.0 for r in rh_pct)


def _mean_by_slot(df: pd.DataFrame, col: str) -> tuple[float, ...]:
    slot = df.index.hour * STEPS_PER_HOUR + df.index.minute // 15  # 15-min slot 0..95
    grouped = df[col].groupby(slot).mean()
    fallback = float(grouped.mean()) if len(grouped) else 0.0
    return tuple(float(grouped.get(s, fallback)) for s in range(STEPS_PER_DAY))


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


def site_row_from_hourly(df: pd.DataFrame) -> dict[str, float]:
    """Mean daily diurnal extrema for SAWH site diagnostics."""
    rh_high, rh_low = diurnal_rh_from_hourly(df)
    out: dict[str, float] = {"rh_high_frac": rh_high, "rh_low_frac": rh_low}
    if "temperature_2m" in df.columns:
        t_high, t_low = diurnal_temperature_from_hourly(df)
        out["temperature_high_c"] = t_high
        out["temperature_low_c"] = t_low
    if "shortwave_radiation" in df.columns:
        out["solar_irradiance_w_per_m2"] = mean_daily_max_irradiance_from_hourly(df)
    return out


def diurnal_rh_from_hourly(df: pd.DataFrame) -> tuple[float, float]:
    """Mean of daily max RH and mean of daily min RH (fractions 0-1)."""
    return _diurnal_extrema(df, "relative_humidity_2m", scale=1.0 / 100.0)


def _diurnal_extrema(df: pd.DataFrame, col: str, scale: float = 1.0) -> tuple[float, float]:
    """Return ``(mean_daily_max * scale, mean_daily_min * scale)`` for *col*."""
    if col not in df.columns:
        raise KeyError(f"DataFrame must contain column {col!r}")
    r = df[col].resample("D")
    return (float(r.max().mean() * scale), float(r.min().mean() * scale))


def diurnal_temperature_from_hourly(df: pd.DataFrame) -> tuple[float, float]:
    """Mean of daily max and min 2 m air temperature (deg C)."""
    return _diurnal_extrema(df, "temperature_2m")


def mean_daily_max_irradiance_from_hourly(df: pd.DataFrame) -> float:
    """Mean of daily peak shortwave GHI irradiance (W/m²)."""
    col = "shortwave_radiation"
    if col not in df.columns:
        raise KeyError(f"DataFrame must contain column {col!r}")
    return float(df[col].resample("D").max().mean())

# =============================================================================
# Weather profiles for fluid-heated daily-cycle SAWH
# =============================================================================

PHASE_DT_S = 100.0
PHASE_HOURS = 12.0
STEPS_PER_PHASE = int(round(PHASE_HOURS * 3600.0 / PHASE_DT_S))

# Loop fluid runs on a fixed operating schedule (not solar), so the 24 h
# fetched weather day is split by clock hour rather than day/night irradiance.
DEFAULT_DESORPTION_START_HOUR = 8


@dataclass(frozen=True, slots=True)
class PhaseProfile:
    """One half-cycle (12 h) ambient boundary conditions."""

    temperature_c: tuple[float, ...]
    relative_humidity: tuple[float, ...]
    h_amb_w_m2_k: tuple[float, ...]
    dt_s: float = PHASE_DT_S


@dataclass(frozen=True, slots=True)
class DailyWeatherProfile:
    absorption: PhaseProfile
    desorption: PhaseProfile


def _constant_phase(
    *,
    n: int = STEPS_PER_PHASE,
    t_amb_c: float,
    rh: float,
    h_amb: float,
    dt_s: float = PHASE_DT_S,
) -> PhaseProfile:
    return PhaseProfile(
        temperature_c=(t_amb_c,) * n,
        relative_humidity=(rh,) * n,
        h_amb_w_m2_k=(h_amb,) * n,
        dt_s=dt_s,
    )


def datacenter_baseline_profile(
    *,
    t_amb_c: float = T_AMB_C,
    rh: float = RH_AMB,
    h_amb: float = H_AMB_W_M2_K,
    dt_s: float = PHASE_DT_S,
) -> DailyWeatherProfile:
    """Fixed 12 h absorption + 12 h desorption at data-center return-air conditions."""
    abs_prof = _constant_phase(t_amb_c=t_amb_c, rh=rh, h_amb=h_amb, dt_s=dt_s)
    des_prof = _constant_phase(t_amb_c=t_amb_c, rh=rh, h_amb=h_amb, dt_s=dt_s)
    return DailyWeatherProfile(absorption=abs_prof, desorption=des_prof)


def baseline_profile(
    *,
    temperature_c: float = T_AMB_C,
    relative_humidity: float = RH_AMB,
    h_amb_w_m2_k: float = H_AMB_W_M2_K,
) -> DailyWeatherProfile:
    return datacenter_baseline_profile(
        t_amb_c=temperature_c,
        rh=relative_humidity,
        h_amb=h_amb_w_m2_k,
    )


def baseline_initial_c_w(
    *,
    equilibrium_rh: float = FABRICATION_EQUILIBRIUM_RH,
    h0_m: float = H0_M,
) -> float:
    return equilibrium_c_w_from_dvs_at_rh(equilibrium_rh, h0_m)


def _resample_phase(df: pd.DataFrame, n: int = STEPS_PER_PHASE) -> PhaseProfile:
    if len(df) == 0:
        raise ValueError("Empty weather slice for phase profile.")
    if len(df) >= n:
        idx = np.linspace(0, len(df) - 1, n).astype(int)
        rh = df["relative_humidity_2m"].astype(float).values[idx] / 100.0
        temp = df["temperature_2m"].astype(float).values[idx]
    else:
        # The absorption/desorption clock split can wrap midnight, so the
        # slice's rows aren't contiguous in calendar time (e.g. 20:00-23:45
        # then 00:00-07:45). Interpolate by row position within the slice,
        # not by real timestamp -- a calendar reindex would silently drop
        # whichever chunk falls outside the first chunk's forward span.
        x_src = np.arange(len(df))
        x_tgt = np.linspace(0, len(df) - 1, n)
        rh = np.interp(x_tgt, x_src, df["relative_humidity_2m"].astype(float).values) / 100.0
        temp = np.interp(x_tgt, x_src, df["temperature_2m"].astype(float).values)

    return PhaseProfile(
        temperature_c=tuple(float(x) for x in temp),
        relative_humidity=tuple(float(x) for x in rh),
        h_amb_w_m2_k=(H_AMB_W_M2_K,) * n,
    )


def profile_from_day_df(
    day_df: pd.DataFrame,
    *,
    desorption_start_hour: int = DEFAULT_DESORPTION_START_HOUR,
) -> DailyWeatherProfile:
    """Split one calendar day into 12 h absorption + 12 h desorption by clock hour.

    The loop-fluid heater runs on a fixed operating schedule rather than
    sunlight, so the split point is a clock hour (default 08:00), not a
    day/night irradiance threshold.
    """
    # A plain boolean mask preserves calendar-day row order (00:00 first), not
    # elapsed real time within a phase that wraps midnight (e.g. absorption
    # 20:00-08:00 is really 20:00-23:45 then 00:00-07:45). Rotate rows into
    # true chronological order starting at desorption_start_hour first, so a
    # phase spanning midnight resamples as one contiguous span instead of
    # jumping mid-phase from 07:45 back to 20:00.
    hour_frac = day_df.index.hour + day_df.index.minute / 60.0
    rel_hour = (hour_frac - desorption_start_hour) % 24
    order = np.argsort(rel_hour, kind="stable")
    rotated = day_df.iloc[order]
    rel_sorted = rel_hour[order]
    desorption_df = rotated[rel_sorted < 12]
    absorption_df = rotated[rel_sorted >= 12]
    if len(absorption_df) < 4:
        absorption_df = day_df.nsmallest(max(STEPS_PER_PHASE, len(day_df) // 2), "temperature_2m")
    if len(desorption_df) < 4:
        desorption_df = day_df.nlargest(max(STEPS_PER_PHASE, len(day_df) // 2), "temperature_2m")
    return DailyWeatherProfile(
        absorption=_resample_phase(absorption_df),
        desorption=_resample_phase(desorption_df),
    )


def _single_day_df(df: pd.DataFrame, day: date) -> pd.DataFrame:
    if df.index.tz is not None:
        mask = df.index.date == day
    else:
        mask = df.index.normalize() == pd.Timestamp(day)
    return df.loc[mask].copy()


def _fetch_year(lat: float, lon: float, year: int, *, cache_dir: str | None) -> pd.DataFrame:
    client = WeatherClient(cache_dir=cache_dir)
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    try:
        _, df_min15 = client.get_historical_forecast_site_weather(lat, lon, start, end)
        return df_min15
    except Exception:
        return client.get_historical(lat, lon, start, end)


def representative_mean_day_profile(
    lat: float,
    lon: float,
    year: int,
    *,
    cache_dir: str | None = None,
    desorption_start_hour: int = DEFAULT_DESORPTION_START_HOUR,
) -> DailyWeatherProfile:
    """Fetch one calendar year and return a single mean diurnal profile."""
    df = _fetch_year(lat, lon, year, cache_dir=cache_dir)
    mean_day = representative_mean_day_df(df, reference_day=date(year, 6, 15))
    return profile_from_day_df(mean_day, desorption_start_hour=desorption_start_hour)


def monthly_mean_day_profiles(
    lat: float,
    lon: float,
    year: int,
    *,
    cache_dir: str | None = None,
    desorption_start_hour: int = DEFAULT_DESORPTION_START_HOUR,
) -> list[tuple[int, DailyWeatherProfile, int]]:
    """One representative mean-day profile per calendar month present in the fetched year.

    Returns (month, profile, n_days_in_month) so callers can day-weight the average.
    """
    df = _fetch_year(lat, lon, year, cache_dir=cache_dir)
    out: list[tuple[int, DailyWeatherProfile, int]] = []
    for m in sorted(set(df.index.month)):
        month_df = df[df.index.month == m]
        if month_df.empty:
            continue
        ref_day = month_df.index[len(month_df) // 2].date()
        mean_day = representative_mean_day_df(month_df, reference_day=ref_day)
        profile = profile_from_day_df(mean_day, desorption_start_hour=desorption_start_hour)
        n_days = len(pd.unique(month_df.index.date))
        out.append((m, profile, n_days))
    return out


def real_weather_days_from_df(
    df: pd.DataFrame,
    *,
    stride: int = 1,
    desorption_start_hour: int = DEFAULT_DESORPTION_START_HOUR,
) -> list[tuple[date, DailyWeatherProfile, pd.DataFrame]]:
    """Build per-day profiles from a pre-fetched year of Open-Meteo data."""
    days_out: list[tuple[date, DailyWeatherProfile, pd.DataFrame]] = []
    for idx, (day_key, group) in enumerate(df.groupby(df.index.date)):
        if stride > 1 and idx % stride != 0:
            continue
        try:
            prof = profile_from_day_df(group, desorption_start_hour=desorption_start_hour)
            days_out.append((day_key, prof, group))
        except (ValueError, KeyError):
            continue
    return days_out

# =============================================================================
# Wilson Note S1 Fig. S1D mass-transfer validation profile (24 h cycle)
# =============================================================================

FIG_S1_TEMPERATURE_C = 25.0
FIG_S1_ABSORPTION_RH = 0.5
FIG_S1_DESORPTION_SOLAR_W_M2 = 800.0
FIG_S1_H_AMB_W_M2_K = 10.0
FIG_S1_ABSORPTION_HOURS = 16.0
FIG_S1_DESORPTION_HOURS = 8.0
FIG_S1_DT_S = PHASE_DT_S
FIG_S1_ABSORPTION_STEPS = int(round(FIG_S1_ABSORPTION_HOURS * 3600.0 / PHASE_DT_S))
FIG_S1_DESORPTION_STEPS = int(round(FIG_S1_DESORPTION_HOURS * 3600.0 / PHASE_DT_S))

# Experimental endpoints from Fig. S1D (water in gel, L/m²).
FIG_S1_INITIAL_WATER_L_M2 = 1.2
FIG_S1_PEAK_WATER_L_M2 = 2.2
FIG_S1_FINAL_WATER_L_M2 = 1.2


def water_in_gel_l_m2(
    c_w: float,
    h_m: float,
    *,
    h0_ref_m: float = 0.004,
    dvs_basis: bool = True,
) -> float:
    """Water in gel (L/m²). Paper Fig. S1D uses DVS gravimetric basis (g/g × m_dry)."""
    if dvs_basis:
        u = pam_licl_gravimetric_uptake_g_g(c_w, h_m, h0_ref_m=h0_ref_m)
        return u * pam_licl_dry_mass_kg_m2(h0_ref_m)
    return max(0.0, c_w) * h_m * WATER_MOLAR_MASS_KG_MOL


def c_w_from_water_in_gel_l_m2(water_l_m2: float, h_m: float) -> float:
    """Invert water-in-gel inventory to uniform c_w (mol/m³) at thickness h_m."""
    if h_m <= 0.0:
        return 0.0
    return max(0.0, water_l_m2) / (h_m * WATER_MOLAR_MASS_KG_MOL)


def fig_s1_initial_c_w(*, h_m: float = 0.004) -> float:
    """Initial brine state matching Fig. S1D start (~1.2 L/m² at H₀)."""
    return c_w_from_water_in_gel_l_m2(FIG_S1_INITIAL_WATER_L_M2, h_m)


def fig_s1_profile() -> DailyWeatherProfile:
    """Build Note S1 Fig. S1D replay: 16 h @ 50% RH, 8 h @ 800 W/m² solar."""
    n_abs = FIG_S1_ABSORPTION_STEPS
    n_des = FIG_S1_DESORPTION_STEPS
    t = FIG_S1_TEMPERATURE_C
    return DailyWeatherProfile(
        absorption=PhaseProfile(
            temperature_c=(t,) * n_abs,
            relative_humidity=(FIG_S1_ABSORPTION_RH,) * n_abs,
            solar_w_m2=(0.0,) * n_abs,
            h_amb_w_m2_k=(FIG_S1_H_AMB_W_M2_K,) * n_abs,
            dt_s=FIG_S1_DT_S,
        ),
        desorption=PhaseProfile(
            temperature_c=(t,) * n_des,
            relative_humidity=(FIG_S1_ABSORPTION_RH,) * n_des,
            solar_w_m2=(FIG_S1_DESORPTION_SOLAR_W_M2,) * n_des,
            h_amb_w_m2_k=(FIG_S1_H_AMB_W_M2_K,) * n_des,
            dt_s=FIG_S1_DT_S,
        ),
    )

# =============================================================================
# Wilson Fig. 4 Atacama field-test weather (24 h from 18:00)
# =============================================================================

_PACKAGE_DIR = Path(__file__).resolve().parent
ATACAMA_RH_CSV = _PACKAGE_DIR / "Atacama_RH.csv"
ATACAMA_TEMP_CSV = _PACKAGE_DIR / "Atacama_Temp.csv"
ATACAMA_AMB_CSV = _PACKAGE_DIR / "Atacama_amb.csv"
ATACAMA_SOLAR_CSV = _PACKAGE_DIR / "Atacama_solar_kW_m2.csv"

# 24 h timeline origin: 18:00 (6 pm) on absorption night, matching paper figure.
CYCLE_ORIGIN_HOUR = 18.0
ABSORPTION_HOURS = 12.0
DESORPTION_HOURS = 12.0

# Atacama field protocol (Methods): install at 8 a.m., ~8 h desorption in sun.
ATACAMA_INSTALL_HOUR_FROM_ORIGIN = 14.0  # 18:00 + 14 h = 08:00
ATACAMA_WIND_STEP_HOUR_FROM_ORIGIN = 20.0  # 18:00 + 20 h = 14:00 (2 p.m.)
# Digitized Fig. 4 curves begin ~0.15 h after 8 a.m. (first data point ≈ 8:09 a.m.),
# so the model desorption is started there rather than exactly at 8:00 a.m. The
# window still ends at 4 p.m. (8 h from 8 a.m.).
ATACAMA_DESORPTION_START_OFFSET_H = 0.15
ATACAMA_FIELD_DESORPTION_HOURS = 8.0 - ATACAMA_DESORPTION_START_OFFSET_H
ATACAMA_FIELD_DESORPTION_STEPS = int(
    round(ATACAMA_FIELD_DESORPTION_HOURS * 3600.0 / PHASE_DT_S)
)
# Wilson Fig. S2: fan-forced condenser cooling ≈ 0.5 m/s → h ≈ 10 W/m²K, decoupled
# from the variable ambient wind that drives the absorber/glass h_amb schedule.
ATACAMA_CONDENSER_FAN_H_AMB_W_M2_K = 10.0


def _load_figure_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    hours: list[float] = []
    values: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            hours.append(float(parts[0].strip()))
            values.append(float(parts[1].strip()))
    if not hours:
        raise ValueError(f"No data in {path}")
    h = np.array(hours, dtype=float)
    v = np.array(values, dtype=float)
    order = np.argsort(h)
    return h[order], v[order]


def _phase_hour_grid(
    phase_start_h: float,
    phase_hours: float,
    n_steps: int,
) -> np.ndarray:
    """Hours from 18:00 origin at cell-centers for a phase."""
    dt_h = phase_hours / n_steps
    return phase_start_h + (np.arange(n_steps, dtype=float) + 0.5) * dt_h


def _interp_clamped(
    h_data: np.ndarray,
    v_data: np.ndarray,
    h_query: np.ndarray,
) -> np.ndarray:
    h_min, h_max = float(h_data.min()), float(h_data.max())
    hq = np.clip(h_query, h_min, h_max)
    return np.interp(hq, h_data, v_data)


def _atacama_h_amb_w_m2_k(hours_from_6pm: float) -> float:
    """Wilson Atacama Methods: h_amb = 1 W/m²K, stepped to 10 W/m²K at 2 p.m.

    Fig. 4 timeline origin is 18:00; 2 p.m. local = hour 20 on that axis.
    """
    return (
        10.0
        if hours_from_6pm >= ATACAMA_WIND_STEP_HOUR_FROM_ORIGIN
        else 1.0
    )


def _build_atacama_profile(
    *,
    desorption_start_h: float,
    desorption_hours: float,
    desorption_steps: int,
) -> DailyWeatherProfile:
    h_rh, rh = _load_figure_csv(ATACAMA_RH_CSV)
    h_t, temp_c = _load_figure_csv(ATACAMA_TEMP_CSV)
    h_amb_fig, amb_c = _load_figure_csv(ATACAMA_AMB_CSV)
    h_s, solar_kw = _load_figure_csv(ATACAMA_SOLAR_CSV)

    abs_h = _phase_hour_grid(0.0, ABSORPTION_HOURS, STEPS_PER_PHASE)
    des_h = _phase_hour_grid(desorption_start_h, desorption_hours, desorption_steps)
    # Fig. 4 ambient curve is digitized on hours from 8 a.m. install (0–8 h).
    des_h_from_install = des_h - ATACAMA_INSTALL_HOUR_FROM_ORIGIN

    abs_rh = _interp_clamped(h_rh, rh, abs_h)
    des_rh = _interp_clamped(h_rh, rh, des_h)
    abs_t = _interp_clamped(h_t, temp_c, abs_h)
    des_t = _interp_clamped(h_amb_fig, amb_c, des_h_from_install)
    des_solar = _interp_clamped(h_s, solar_kw, des_h) * 1000.0  # kW/m² → W/m²

    abs_hamb = tuple(_atacama_h_amb_w_m2_k(float(h)) for h in abs_h)
    des_hamb = tuple(_atacama_h_amb_w_m2_k(float(h)) for h in des_h)
    des_hcond = (ATACAMA_CONDENSER_FAN_H_AMB_W_M2_K,) * desorption_steps

    return DailyWeatherProfile(
        absorption=PhaseProfile(
            temperature_c=tuple(float(x) for x in abs_t),
            relative_humidity=tuple(float(x) for x in abs_rh),
            solar_w_m2=(0.0,) * STEPS_PER_PHASE,
            h_amb_w_m2_k=abs_hamb,
            dt_s=PHASE_DT_S,
        ),
        desorption=PhaseProfile(
            temperature_c=tuple(float(x) for x in des_t),
            relative_humidity=tuple(float(x) for x in des_rh),
            solar_w_m2=tuple(max(0.0, float(x)) for x in des_solar),
            h_amb_w_m2_k=des_hamb,
            h_amb_cond_w_m2_k=des_hcond,
            dt_s=PHASE_DT_S,
        ),
    )


def atacama_figure_profile() -> DailyWeatherProfile:
    """Fig. 4 symmetric 12 h + 12 h replay (legacy)."""
    return _build_atacama_profile(
        desorption_start_h=ABSORPTION_HOURS,
        desorption_hours=DESORPTION_HOURS,
        desorption_steps=STEPS_PER_PHASE,
    )


def atacama_field_profile() -> DailyWeatherProfile:
    """Atacama field validation: 12 h open absorption, desorption from ~8:09 a.m.

    Desorption begins where the digitized Fig. 4 curves start (≈0.15 h after
    8 a.m.) and runs through 4 p.m.
    """
    return _build_atacama_profile(
        desorption_start_h=(
            ATACAMA_INSTALL_HOUR_FROM_ORIGIN + ATACAMA_DESORPTION_START_OFFSET_H
        ),
        desorption_hours=ATACAMA_FIELD_DESORPTION_HOURS,
        desorption_steps=ATACAMA_FIELD_DESORPTION_STEPS,
    )
