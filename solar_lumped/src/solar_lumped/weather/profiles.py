"""Weather profile builders: baseline, replay, and real per-day series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from solar_lumped.weather.client import WeatherClient

from solar_lumped.physics.salt_properties import (
    FABRICATION_EQUILIBRIUM_RH,
    equilibrium_c_w_from_dvs_at_rh,
)

PHASE_DT_S = 100.0  # Wilson Note S1 / COMSOL time step (s)
PHASE_HOURS = 12.0
STEPS_PER_PHASE = int(round(PHASE_HOURS * 3600.0 / PHASE_DT_S))
SOLAR_NIGHT_THRESHOLD_W_M2 = 5.0


@dataclass(frozen=True, slots=True)
class PhaseProfile:
    """One half-cycle (12 h) weather at ``PHASE_DT_S`` resolution."""

    temperature_c: tuple[float, ...]
    relative_humidity: tuple[float, ...]
    solar_w_m2: tuple[float, ...]
    h_amb_w_m2_k: tuple[float, ...]
    dt_s: float = PHASE_DT_S
    # Optional separate convection coefficient for the condenser backing. When set,
    # it decouples condenser cooling from the ambient h_amb that drives the
    # absorber/glass. Wilson's Atacama device forces ~0.5 m/s over the condenser with
    # fans (Fig. S2) regardless of the variable ambient wind. None → use h_amb.
    h_amb_cond_w_m2_k: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class DailyWeatherProfile:
    absorption: PhaseProfile
    desorption: PhaseProfile
    cooling: PhaseProfile | None = None


FIXED_H_AMB_W_M2_K = 10.0


def _wind_series(_df: pd.DataFrame, n: int) -> tuple[float, ...]:
    """Ambient convection coefficient per timestep -- fixed, not wind-derived."""
    return (FIXED_H_AMB_W_M2_K,) * n


def _resample_phase(df: pd.DataFrame, n: int = STEPS_PER_PHASE) -> PhaseProfile:
    if len(df) == 0:
        raise ValueError("Empty weather slice for phase profile.")
    if len(df) >= n:
        idx = np.linspace(0, len(df) - 1, n).astype(int)
        rh = df["relative_humidity_2m"].astype(float).values[idx] / 100.0
        temp = df["temperature_2m"].astype(float).values[idx]
        solar_src = df.get("shortwave_radiation", pd.Series(0.0, index=df.index))
        solar = solar_src.astype(float).values[idx]
    else:
        # The absorption/desorption day/night split can wrap midnight, so the
        # slice's rows aren't contiguous in calendar time (e.g. 20:00-23:45
        # then 00:00-05:45). Interpolate by row position within the slice,
        # not by real timestamp -- a calendar reindex would silently drop
        # whichever chunk falls outside the first chunk's forward span.
        x_src = np.arange(len(df))
        x_tgt = np.linspace(0, len(df) - 1, n)
        rh = np.interp(x_tgt, x_src, df["relative_humidity_2m"].astype(float).values) / 100.0
        temp = np.interp(x_tgt, x_src, df["temperature_2m"].astype(float).values)
        solar_src = df.get("shortwave_radiation", pd.Series(0.0, index=df.index))
        solar = np.interp(x_tgt, x_src, solar_src.astype(float).values)
    solar = np.maximum(0.0, solar)
    h_amb = _wind_series(df, n)
    return PhaseProfile(
        temperature_c=tuple(float(x) for x in temp),
        relative_humidity=tuple(float(x) for x in rh),
        solar_w_m2=tuple(float(x) for x in solar),
        h_amb_w_m2_k=h_amb,
    )


def _native_dt_s(df: pd.DataFrame) -> float:
    deltas = df.index.to_series().diff().dropna().dt.total_seconds()
    return float(deltas.median()) if len(deltas) else PHASE_DT_S


def _steps_for(n_rows: int, native_dt_s: float) -> int:
    """Step count at PHASE_DT_S resolution matching a slice's real elapsed duration.

    Absorption and desorption aren't equal-length in real time (day/night varies by
    latitude and season), so each phase gets its own step count rather than being
    forced onto a fixed 12 h grid -- that would silently speed up or slow down real
    time within the ODE integration.
    """
    return max(4, int(round(n_rows * native_dt_s / PHASE_DT_S)))


def _rotate_chronological(df: pd.DataFrame, *, pivot_hour: float) -> pd.DataFrame:
    """Reorder rows into real elapsed-time order for a slice that may wrap around pivot_hour.

    A boolean mask (e.g. solar < threshold) preserves calendar-day row order --
    00:00 first, then whatever comes after sunset appended at the end -- not real
    elapsed time within the night, which actually runs sunset -> midnight ->
    sunrise. ``pivot_hour`` must be an hour the slice never contains (noon for
    night, midnight for day), so rotating the axis to start there never splits
    the slice's one real contiguous span in two.
    """
    hour_frac = df.index.hour + df.index.minute / 60.0
    rel_hour = (hour_frac - pivot_hour) % 24
    order = np.argsort(rel_hour, kind="stable")
    return df.iloc[order]


def profile_from_day_df(day_df: pd.DataFrame) -> DailyWeatherProfile:
    """Split one calendar day into absorption (night) + desorption (day).

    The two halves run for their true real-time duration (via each phase's own
    step count at PHASE_DT_S resolution), not a fixed 12 h/12 h split.
    """
    native_dt_s = _native_dt_s(day_df)
    solar = day_df.get("shortwave_radiation", pd.Series(0.0, index=day_df.index)).astype(float)
    night = day_df[solar < SOLAR_NIGHT_THRESHOLD_W_M2]
    day = day_df[solar >= SOLAR_NIGHT_THRESHOLD_W_M2]
    if len(night) < 4:
        night = day_df.nsmallest(max(STEPS_PER_PHASE, len(day_df) // 2), "shortwave_radiation")
    if len(day) < 4:
        day = day_df.nlargest(max(STEPS_PER_PHASE, len(day_df) // 2), "shortwave_radiation")
    night = _rotate_chronological(night, pivot_hour=12.0)
    day = _rotate_chronological(day, pivot_hour=0.0)
    return DailyWeatherProfile(
        absorption=_resample_phase(night, _steps_for(len(night), native_dt_s)),
        desorption=_resample_phase(day, _steps_for(len(day), native_dt_s)),
    )


def baseline_profile(
    *,
    temperature_c: float = 25.0,
    relative_humidity: float = 0.5,
    solar_w_m2: float = 600.0,
    h_amb_w_m2_k: float = 10.0,
) -> DailyWeatherProfile:
    abs_prof = PhaseProfile(
        temperature_c=(temperature_c,) * STEPS_PER_PHASE,
        relative_humidity=(relative_humidity,) * STEPS_PER_PHASE,
        solar_w_m2=(0.0,) * STEPS_PER_PHASE,
        h_amb_w_m2_k=(h_amb_w_m2_k,) * STEPS_PER_PHASE,
    )
    des_prof = PhaseProfile(
        temperature_c=(temperature_c,) * STEPS_PER_PHASE,
        relative_humidity=(relative_humidity,) * STEPS_PER_PHASE,
        solar_w_m2=(solar_w_m2,) * STEPS_PER_PHASE,
        h_amb_w_m2_k=(h_amb_w_m2_k,) * STEPS_PER_PHASE,
    )
    return DailyWeatherProfile(absorption=abs_prof, desorption=des_prof)


COMSOL_DESORPTION_HOURS = 8.0
COMSOL_COOLING_HOURS = 12.0
STEPS_COMSOL_DES = int(round(COMSOL_DESORPTION_HOURS * 3600.0 / PHASE_DT_S))
STEPS_COMSOL_COOL = int(round(COMSOL_COOLING_HOURS * 3600.0 / PHASE_DT_S))


def comsol_fig2_profile(
    *,
    tint_c: float = 23.0,
    rh_high: float = 0.5,
    solar_w_m2: float = 1000.0,
    h_front_w_m2_k: float = 10.0,
    include_cooling: bool = False,
) -> DailyWeatherProfile:
    """Wilson COMSOL lumped prototype: 12 h absorption, 8 h desorption, optional 12 h cool."""
    abs_prof = PhaseProfile(
        temperature_c=(tint_c,) * STEPS_PER_PHASE,
        relative_humidity=(rh_high,) * STEPS_PER_PHASE,
        solar_w_m2=(0.0,) * STEPS_PER_PHASE,
        h_amb_w_m2_k=(h_front_w_m2_k,) * STEPS_PER_PHASE,
    )
    des_prof = PhaseProfile(
        temperature_c=(tint_c,) * STEPS_COMSOL_DES,
        relative_humidity=(rh_high,) * STEPS_COMSOL_DES,
        solar_w_m2=(solar_w_m2,) * STEPS_COMSOL_DES,
        h_amb_w_m2_k=(h_front_w_m2_k,) * STEPS_COMSOL_DES,
    )
    cooling = None
    if include_cooling:
        cooling = PhaseProfile(
            temperature_c=(tint_c,) * STEPS_COMSOL_COOL,
            relative_humidity=(rh_high,) * STEPS_COMSOL_COOL,
            solar_w_m2=(0.0,) * STEPS_COMSOL_COOL,
            h_amb_w_m2_k=(h_front_w_m2_k,) * STEPS_COMSOL_COOL,
        )
    return DailyWeatherProfile(absorption=abs_prof, desorption=des_prof, cooling=cooling)


# Wilson Methods: hydrogel cast at DVS equilibrium with ~20% RH before cycling.
BASELINE_INITIAL_EQUILIBRIUM_RH = FABRICATION_EQUILIBRIUM_RH


def baseline_initial_c_w(*, h_m: float = 0.004) -> float:
    """Initial brine state for baseline / Fig. 2 replay (fabrication at ~20% RH)."""
    return equilibrium_c_w_from_dvs_at_rh(
        BASELINE_INITIAL_EQUILIBRIUM_RH,
        h_m=h_m,
        h0_ref_m=h_m,
    )


def _single_day_df(
    df: pd.DataFrame,
    day: date,
) -> pd.DataFrame:
    if df.index.tz is not None:
        mask = df.index.date == day
    else:
        mask = df.index.normalize() == pd.Timestamp(day)
    return df.loc[mask].copy()


def replay_profile(
    mode: Literal["atacama-replay", "cambridge-replay", "fig-s1-replay"],
    *,
    cache_dir: str | None = None,
) -> DailyWeatherProfile:
    if mode == "atacama-replay":
        from solar_lumped.weather.atacama_figure import atacama_field_profile

        return atacama_field_profile()
    if mode == "fig-s1-replay":
        from solar_lumped.weather.fig_s1 import fig_s1_profile

        return fig_s1_profile()

    day = date(2024, 6, 3)
    lat, lon = 42.36, -71.09

    client = WeatherClient(cache_dir=cache_dir)
    _, df_min15 = client.get_historical_forecast_site_weather(
        lat, lon, day.isoformat(), day.isoformat()
    )
    day_df = _single_day_df(df_min15, day)
    if day_df.empty:
        _, df_h = client.get_historical_forecast_site_weather(
            lat, lon, day.isoformat(), day.isoformat()
        )
        day_df = _single_day_df(df_h, day)
    return profile_from_day_df(day_df)


def representative_mean_day_profile(
    lat: float,
    lon: float,
    year: int,
    *,
    cache_dir: str | None = None,
) -> DailyWeatherProfile:
    """Fetch one calendar year and return a single mean diurnal profile."""
    from solar_lumped.weather.climate import representative_mean_day_df

    client = WeatherClient(cache_dir=cache_dir)
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    try:
        _, df_min15 = client.get_historical_forecast_site_weather(lat, lon, start, end)
        df = df_min15
    except Exception:
        df = client.get_historical(lat, lon, start, end)
    mean_day = representative_mean_day_df(df, reference_day=date(year, 6, 15))
    return profile_from_day_df(mean_day)


def real_weather_days_from_df(
    df: pd.DataFrame,
    *,
    stride: int = 1,
) -> list[tuple[date, DailyWeatherProfile, pd.DataFrame]]:
    """Build per-day profiles from a pre-fetched year of Open-Meteo data."""
    days_out: list[tuple[date, DailyWeatherProfile, pd.DataFrame]] = []
    for idx, (day_key, group) in enumerate(df.groupby(df.index.date)):
        if stride > 1 and idx % stride != 0:
            continue
        try:
            prof = profile_from_day_df(group)
            days_out.append((day_key, prof, group))
        except (ValueError, KeyError):
            continue
    return days_out


def real_weather_days(
    lat: float,
    lon: float,
    year: int,
    *,
    cache_dir: str | None = None,
    stride: int = 1,
    df: pd.DataFrame | None = None,
) -> list[tuple[date, DailyWeatherProfile]]:
    """Build per-day profiles for a full year from minutely_15 (or hourly fallback)."""
    if df is None:
        client = WeatherClient(cache_dir=cache_dir)
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        try:
            _, df_min15 = client.get_historical_forecast_site_weather(lat, lon, start, end)
            df = df_min15
        except Exception:
            df = client.get_historical(lat, lon, start, end)

    return [(day_key, prof) for day_key, prof, _ in real_weather_days_from_df(df, stride=stride)]
