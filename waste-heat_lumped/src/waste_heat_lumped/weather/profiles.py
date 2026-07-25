"""Weather profiles for fluid-heated daily-cycle SAWH."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from waste_heat_lumped.physics import device_defaults as dd
from waste_heat_lumped.physics.salt_properties import (
    FABRICATION_EQUILIBRIUM_RH,
    equilibrium_c_w_from_dvs_at_rh,
)
from waste_heat_lumped.weather.client import WeatherClient

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
    t_amb_c: float = dd.T_AMB_C,
    rh: float = dd.RH_AMB,
    h_amb: float = dd.H_AMB_W_M2_K,
    dt_s: float = PHASE_DT_S,
) -> DailyWeatherProfile:
    """Fixed 12 h absorption + 12 h desorption at data-center return-air conditions."""
    abs_prof = _constant_phase(t_amb_c=t_amb_c, rh=rh, h_amb=h_amb, dt_s=dt_s)
    des_prof = _constant_phase(t_amb_c=t_amb_c, rh=rh, h_amb=h_amb, dt_s=dt_s)
    return DailyWeatherProfile(absorption=abs_prof, desorption=des_prof)


def baseline_profile(
    *,
    temperature_c: float = dd.T_AMB_C,
    relative_humidity: float = dd.RH_AMB,
    h_amb_w_m2_k: float = dd.H_AMB_W_M2_K,
) -> DailyWeatherProfile:
    return datacenter_baseline_profile(
        t_amb_c=temperature_c,
        rh=relative_humidity,
        h_amb=h_amb_w_m2_k,
    )


def baseline_initial_c_w(
    *,
    equilibrium_rh: float = FABRICATION_EQUILIBRIUM_RH,
    h0_m: float = dd.H0_M,
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
        h_amb_w_m2_k=(dd.H_AMB_W_M2_K,) * n,
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
    from waste_heat_lumped.weather.climate import representative_mean_day_df

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
    from waste_heat_lumped.weather.climate import representative_mean_day_df

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
