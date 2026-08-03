"""Weather: Open-Meteo client, real/replay/baseline profiles, land-grid sampling,
and the Note S1 Fig. S1D / Atacama Fig. 4 validation profiles."""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
import retry_requests

from solar_lumped._parameters_xlsx import physics_value as _pv
from solar_lumped.physics import (
    FABRICATION_EQUILIBRIUM_RH,
    H0_M,
    H_AMB_W_M2_K,
    c_w_from_water_in_gel_l_m2,
    equilibrium_c_w_from_dvs_at_rh,
    wind_to_h_amb_w_m2_k,
)

# Wilson et al. (2025) Methods baseline scenario (docs/parameters.xlsx Physics
# sheet) -- used as the ``baseline_profile()`` synthetic-weather defaults below.
BASELINE_T_AMB_C: float = _pv("Ambient temperature (T_amb)")
BASELINE_RH_AMB: float = _pv("Uptake RH / ambient RH (RH_amb)")
BASELINE_Q_SOLAR_W_M2: float = _pv("Solar irradiance (Q_solar)")
BASELINE_H_AMB_W_M2_K: float = H_AMB_W_M2_K


# --- Open-Meteo weather API client (real historical + forecast retrieval) ---

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
                    # Requests are always for a fixed, already-elapsed date range
                    # (a past calendar year), so the archive response never changes.
                    expire_after=requests_cache.NEVER_EXPIRE,
                    # WAL + long busy_timeout so concurrent GPU-sweep array tasks
                    # sharing this cache file don't serialize or hit "database is locked".
                    wal=True,
                    busy_timeout=60_000,
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


def fetch_year_weather(
    lat: float,
    lon: float,
    year: int,
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """One calendar year of minutely-15 weather, falling back to hourly archive data."""
    client = WeatherClient(cache_dir=cache_dir)
    start, end = f"{year}-01-01", f"{year}-12-31"
    try:
        _, df = client.get_historical_forecast_site_weather(lat, lon, start, end)
        return df
    except Exception:
        return client.get_historical(lat, lon, start, end)

# --- Real-weather day statistics (solar/temperature/RH day summaries) ---


def day_weather_stats(day_df: pd.DataFrame) -> dict[str, float]:
    """Mean and peak weather for one calendar day of raw Open-Meteo data."""
    if day_df.empty:
        raise ValueError("Cannot compute weather stats from empty day DataFrame.")
    out: dict[str, float] = {}
    if "relative_humidity_2m" in day_df.columns:
        rh = day_df["relative_humidity_2m"].astype(float)
        out["rh_avg_frac"] = float(rh.mean() / 100.0)
        out["rh_peak_frac"] = float(rh.max() / 100.0)
    if "temperature_2m" in day_df.columns:
        temp = day_df["temperature_2m"].astype(float)
        out["temp_avg_c"] = float(temp.mean())
        out["temp_peak_c"] = float(temp.max())
    if "shortwave_radiation" in day_df.columns:
        solar = day_df["shortwave_radiation"].astype(float).clip(lower=0.0)
        out["solar_avg_w_m2"] = float(solar.mean())
        out["solar_peak_w_m2"] = float(solar.max())
    return out

# --- Land-grid site sampling for global maps ---

# Excluded by default: polar territory with no deployment demand that reaches further
# south than the global lat_hi cutoff catches.
DEFAULT_EXCLUDE_COUNTRY_ABOVE_LAT: dict[str, float] = {"Canada": 60.0, "Russia": 60.0}


def grid_land_points(
    step_deg: float = 5.0,
    *,
    lat_lo: float = -56.0,
    lat_hi: float = 72.0,
    exclude_country_above_lat: dict[str, float] | None = DEFAULT_EXCLUDE_COUNTRY_ABOVE_LAT,
) -> list[tuple[float, float]]:
    """All (lat, lon) land grid nodes at ``step_deg`` spacing (WGS84).
    ``exclude_country_above_lat`` drops points in a Natural Earth ADMIN country above a
    latitude the global ``lat_hi`` cutoff misses; pass ``{}``/``None`` to disable."""
    from shapely import geometry as sh_geom
    from shapely.ops import unary_union
    from shapely.prepared import prep

    import cartopy.io.shapereader as shpreader

    print(
        "  Loading Natural Earth land polygons (first run may download shapefiles)…",
        flush=True,
    )
    t0 = time.perf_counter()
    land_path = shpreader.natural_earth(resolution="110m", category="physical", name="land")
    land = prep(unary_union(list(shpreader.Reader(land_path).geometries())))
    print(f"  Land geometry ready in {time.perf_counter() - t0:.2f}s.", flush=True)

    exclude_country_above_lat = exclude_country_above_lat or {}
    country_path = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
    exclude_masks = [
        (
            prep(
                unary_union(
                    [
                        r.geometry
                        for r in shpreader.Reader(country_path).records()
                        if r.attributes.get("ADMIN") == name
                    ]
                )
            ),
            thresh,
        )
        for name, thresh in exclude_country_above_lat.items()
    ]

    lat_start = math.ceil(lat_lo / step_deg) * step_deg
    lats = np.arange(lat_start, lat_hi + 1e-9, step_deg)
    lons = np.arange(-180.0, 180.0, step_deg)
    n_grid = int(lats.size * lons.size)

    out: list[tuple[float, float]] = []
    n_excluded = 0
    for lat in lats:
        for lon in lons:
            pt = sh_geom.Point(float(lon), float(lat))
            if not land.contains(pt):
                continue
            if any(lat > thresh and mask.contains(pt) for mask, thresh in exclude_masks):
                n_excluded += 1
                continue
            out.append((float(lat), float(lon)))

    excl_note = f"; {n_excluded} excluded (country/lat rule)" if exclude_masks else ""
    print(
        f"  Grid {step_deg:g}°: {len(lats)} lat × {len(lons)} lon = {n_grid} nodes; "
        f"{len(out)} on land (lat ∈ [{lat_start:g}, {lats[-1]:g}]){excl_note}.",
        flush=True,
    )
    return out

# --- Weather profile builders: baseline, replay, and real per-day series ---

PHASE_DT_S = 100.0  # Wilson Note S1 / COMSOL time step (s)
PHASE_HOURS = 12.0
STEPS_PER_PHASE = int(round(PHASE_HOURS * 3600.0 / PHASE_DT_S))
SOLAR_NIGHT_THRESHOLD_W_M2 = 5.0

# --- Optional plane-of-array (POA) irradiance transposition ---
#
# OFF BY DEFAULT. Wilson's Eq. 4 drives the absorber with Q_solar taken straight
# from the site's horizontal irradiance (GHI), and ``tilt_deg`` enters the model
# *only* through the Hollands tilted-plate Nusselt correlation for the two air
# gaps (physics.py::hollands_vapor_gap_h_conv_w_m2_k). That makes tilt a pure
# internal-natural-convection knob: tilting the system changes gap convection but
# not the solar energy it collects, which is backwards for a solar collector.
#
# Enabling POA (``profile_from_day_df(..., poa_tilt_deg=...)``) transposes GHI
# onto the tilted aperture, so a single ``tilt_deg`` then trades solar gain
# against gap convection simultaneously -- the coupling needed to optimize tilt
# for real. Left opt-in because switching it on changes every real-weather yield
# number, so all existing GHI-based results stay reproducible untouched.
#
# Method: Erbs et al. (1982) diffuse-fraction decomposition of GHI into DNI/DHI,
# then Liu & Jordan isotropic-sky transposition. Isotropic is the conservative
# choice (it ignores circumsolar/horizon brightening, so it slightly *under*-
# predicts POA for tilted surfaces facing the sun) and needs no extra inputs.
POA_DEFAULT_ALBEDO: float = 0.2  # generic ground reflectance (Duffie & Beckman)
_SOLAR_CONSTANT_W_M2: float = 1367.0
# Below ~3 deg solar elevation the DNI = (GHI - DHI)/cos(zenith) division blows
# up on near-zero cos(zenith); clamp rather than emit a spurious POA spike.
_MIN_COS_ZENITH: float = 0.05


def plane_of_array_w_m2(
    ghi_w_m2: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    latitude_deg: float,
    longitude_deg: float,
    tilt_deg: float,
    surface_azimuth_deg: float | None = None,
    albedo: float = POA_DEFAULT_ALBEDO,
) -> np.ndarray:
    """Transpose horizontal irradiance (GHI) onto a tilted aperture.

    ``surface_azimuth_deg`` is measured clockwise from north (180 = due south);
    ``None`` picks the equator-facing orientation from the latitude's sign, which
    is what both study sites want (Cambridge +42 -> south, Atacama -23 -> north).

    Returns POA in W/m^2, same shape as ``ghi_w_m2``. At ``tilt_deg == 0`` this is
    an exact identity (POA == GHI): the beam term collapses to DNI*cos(zenith) =
    GHI - DHI, the isotropic sky term to DHI, and the ground term to zero.
    """
    ghi = np.asarray(ghi_w_m2, dtype=float)
    if surface_azimuth_deg is None:
        surface_azimuth_deg = 180.0 if latitude_deg >= 0.0 else 0.0

    # --- Solar position (Duffie & Beckman ch. 1) ---
    doy = np.asarray(index.dayofyear, dtype=float)
    b = np.radians(360.0 / 365.0 * (doy - 81.0))
    # Equation of time (min) and longitude correction give *true solar* time; using
    # raw clock time would bias the apparent solar noon by up to ~1 h and therefore
    # bias the optimal tilt/azimuth.
    eot_min = 9.87 * np.sin(2.0 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    utc_offset_h = np.asarray(
        [(ts.utcoffset().total_seconds() / 3600.0) if ts.utcoffset() is not None else 0.0 for ts in index],
        dtype=float,
    )
    clock_h = np.asarray(index.hour, dtype=float) + np.asarray(index.minute, dtype=float) / 60.0
    solar_h = clock_h + (4.0 * (longitude_deg - 15.0 * utc_offset_h) + eot_min) / 60.0
    hour_angle = np.radians(15.0 * (solar_h - 12.0))
    declination = np.radians(23.45) * np.sin(np.radians(360.0 / 365.0 * (doy + 284.0)))

    phi = np.radians(latitude_deg)
    beta = np.radians(tilt_deg)
    gamma = np.radians(surface_azimuth_deg - 180.0)  # D&B convention: 0 = equator-facing
    cos_zen = np.sin(declination) * np.sin(phi) + np.cos(declination) * np.cos(phi) * np.cos(hour_angle)
    cos_zen_eff = np.maximum(cos_zen, _MIN_COS_ZENITH)

    # --- Erbs diffuse fraction from the clearness index ---
    e0 = _SOLAR_CONSTANT_W_M2 * (1.0 + 0.033 * np.cos(np.radians(360.0 * doy / 365.0)))
    kt = np.clip(np.divide(ghi, e0 * cos_zen_eff, out=np.zeros_like(ghi), where=e0 * cos_zen_eff > 0.0), 0.0, 1.0)
    diffuse_frac = np.where(
        kt <= 0.22,
        1.0 - 0.09 * kt,
        np.where(
            kt <= 0.80,
            0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
            0.165,
        ),
    )
    dhi = np.clip(diffuse_frac, 0.0, 1.0) * ghi
    dni = np.maximum(0.0, (ghi - dhi) / cos_zen_eff)

    # --- Liu & Jordan isotropic transposition (D&B Eq. 1.6.2 for cos(AOI)) ---
    cos_aoi = (
        np.sin(declination) * np.sin(phi) * np.cos(beta)
        - np.sin(declination) * np.cos(phi) * np.sin(beta) * np.cos(gamma)
        + np.cos(declination) * np.cos(phi) * np.cos(beta) * np.cos(hour_angle)
        + np.cos(declination) * np.sin(phi) * np.sin(beta) * np.cos(gamma) * np.cos(hour_angle)
        + np.cos(declination) * np.sin(beta) * np.sin(gamma) * np.sin(hour_angle)
    )
    poa = (
        dni * np.maximum(0.0, cos_aoi)
        + dhi * (1.0 + np.cos(beta)) / 2.0
        + ghi * albedo * (1.0 - np.cos(beta)) / 2.0
    )
    # Night (and the clamped near-horizon band) collect nothing.
    return np.where(cos_zen > 0.0, np.maximum(0.0, poa), 0.0)


@dataclass(frozen=True, slots=True)
class PhaseProfile:
    """One half-cycle (12 h) weather at ``PHASE_DT_S`` resolution."""

    temperature_c: tuple[float, ...]
    relative_humidity: tuple[float, ...]
    solar_w_m2: tuple[float, ...]
    h_amb_w_m2_k: tuple[float, ...]
    dt_s: float = PHASE_DT_S
    # Decouples condenser cooling from the absorber/glass h_amb (Wilson's Atacama system
    # fan-forces ~0.5 m/s over the condenser, Fig. S2). None → use h_amb.
    h_amb_cond_w_m2_k: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class DailyWeatherProfile:
    absorption: PhaseProfile
    desorption: PhaseProfile
    cooling: PhaseProfile | None = None


FIXED_H_AMB_W_M2_K = BASELINE_H_AMB_W_M2_K


def profile_from_day_df(
    day_df: pd.DataFrame,
    *,
    poa_tilt_deg: float | None = None,
    poa_surface_azimuth_deg: float | None = None,
    poa_albedo: float = POA_DEFAULT_ALBEDO,
    seal_offset_h: float = 0.0,
    open_offset_h: float = 0.0,
    condenser_air_speed_m_s: float = 0.0,
) -> DailyWeatherProfile:
    """Split one calendar day into absorption (night) + desorption (day), each running its
    true real-time duration at PHASE_DT_S resolution rather than a fixed 12 h/12 h split.

    ``poa_tilt_deg`` is opt-in (see the POA block above): when set, the profile's solar
    series becomes plane-of-array irradiance on an aperture at that tilt instead of raw
    GHI, so ``tilt_deg`` drives solar gain as well as gap natural convection. Requires the
    ``latitude``/``longitude`` columns Open-Meteo responses already carry. The day/night
    split itself stays keyed on GHI, so which samples land in which phase is unchanged.

    Complex mode (A1) shifts that split: ``seal_offset_h`` moves the moment the system is
    sealed and desorption begins relative to GHI sunrise, and ``open_offset_h`` moves the
    moment it re-opens relative to GHI sunset (both in hours; positive delays). Sealing
    late keeps harvesting the pre-dawn RH peak but leaves fewer solar hours to desorb --
    the trade-off A1 exists to expose. Both zero (default) is the unshifted GHI split.

    Complex mode (B4) also sets ``condenser_air_speed_m_s``, which fills the profile's
    ``h_amb_cond_w_m2_k`` channel so the condenser is cooled by forced air independent of
    ambient wind, rather than sharing the absorber's h_amb.
    """
    deltas = day_df.index.to_series().diff().dropna().dt.total_seconds()
    native_dt_s = float(deltas.median()) if len(deltas) else PHASE_DT_S
    solar = day_df.get("shortwave_radiation", pd.Series(0.0, index=day_df.index)).astype(float)
    if poa_tilt_deg is not None:
        if "latitude" not in day_df or "longitude" not in day_df:
            raise ValueError(
                "POA transposition needs 'latitude'/'longitude' columns on the weather "
                "frame (Open-Meteo responses carry them; synthetic frames may not)."
            )
        day_df = day_df.assign(
            poa_w_m2=plane_of_array_w_m2(
                solar.to_numpy(),
                day_df.index,
                latitude_deg=float(day_df["latitude"].iloc[0]),
                longitude_deg=float(day_df["longitude"].iloc[0]),
                tilt_deg=float(poa_tilt_deg),
                surface_azimuth_deg=poa_surface_azimuth_deg,
                albedo=poa_albedo,
            )
        )
    if seal_offset_h or open_offset_h:
        is_day = _shifted_desorption_mask(day_df, solar, seal_offset_h, open_offset_h)
    else:
        is_day = solar >= SOLAR_NIGHT_THRESHOLD_W_M2
    night = day_df[~is_day]
    day = day_df[is_day]
    if len(night) < 4:
        night = day_df.nsmallest(max(STEPS_PER_PHASE, len(day_df) // 2), "shortwave_radiation")
    if len(day) < 4:
        day = day_df.nlargest(max(STEPS_PER_PHASE, len(day_df) // 2), "shortwave_radiation")
    night = _rotate_chronological(night, pivot_hour=12.0)
    day = _rotate_chronological(day, pivot_hour=0.0)
    return DailyWeatherProfile(
        absorption=_resample_phase(night, _steps_for(len(night), native_dt_s)),
        desorption=_resample_phase(
            day,
            _steps_for(len(day), native_dt_s),
            condenser_air_speed_m_s=condenser_air_speed_m_s,
        ),
    )


def _shifted_desorption_mask(
    day_df: pd.DataFrame,
    solar: pd.Series,
    seal_offset_h: float,
    open_offset_h: float,
) -> np.ndarray:
    """Desorption-phase mask with the seal/open boundaries shifted off GHI sunrise/sunset.

    The unshifted phase is every sample at or above ``SOLAR_NIGHT_THRESHOLD_W_M2``; its
    first and last samples are taken as sunrise and sunset. Offsets move those two edges
    independently along the clock, so desorption can start before dawn (negative
    ``seal_offset_h``, sealing the system early) or run past dusk (positive
    ``open_offset_h``, holding it shut into the evening).

    Both edges are clamped to keep at least one sample in each phase: a design that
    starves absorption or desorption entirely should score badly on yield, not crash the
    integrator with an empty slice.
    """
    lit = np.asarray(solar >= SOLAR_NIGHT_THRESHOLD_W_M2)
    if not lit.any() or lit.all():
        return lit
    hour = np.asarray(day_df.index.hour + day_df.index.minute / 60.0, dtype=float)
    lit_hours = hour[lit]
    sunrise, sunset = float(lit_hours.min()), float(lit_hours.max())
    start = sunrise + float(seal_offset_h)
    end = sunset + float(open_offset_h)
    if end <= start:  # offsets crossed over; keep a minimal desorption window
        end = start + 1.0
    mask = (hour >= start) & (hour <= end)
    if not mask.any():
        mask = lit
    elif mask.all():
        mask = mask & (hour < float(hour.max()))
    return mask


def _rotate_chronological(df: pd.DataFrame, *, pivot_hour: float) -> pd.DataFrame:
    """Reorder a mask-selected slice into real elapsed-time order (night runs sunset →
    midnight → sunrise). ``pivot_hour`` must be an hour the slice never contains."""
    hour_frac = df.index.hour + df.index.minute / 60.0
    rel_hour = (hour_frac - pivot_hour) % 24
    order = np.argsort(rel_hour, kind="stable")
    return df.iloc[order]


def _steps_for(n_rows: int, native_dt_s: float) -> int:
    """Step count at PHASE_DT_S matching a slice's real elapsed duration -- day/night
    lengths vary, so a fixed 12 h grid would distort real time inside the ODE."""
    return max(4, int(round(n_rows * native_dt_s / PHASE_DT_S)))


def _resample_phase(
    df: pd.DataFrame,
    n: int = STEPS_PER_PHASE,
    *,
    condenser_air_speed_m_s: float = 0.0,
) -> PhaseProfile:
    if len(df) == 0:
        raise ValueError("Empty weather slice for phase profile.")
    # "poa_w_m2" is present only when profile_from_day_df ran with POA enabled; it
    # already carries the tilt/site geometry, so it supersedes raw GHI here.
    solar_col = "poa_w_m2" if "poa_w_m2" in df else "shortwave_radiation"
    if len(df) >= n:
        idx = np.linspace(0, len(df) - 1, n).astype(int)
        rh = df["relative_humidity_2m"].astype(float).values[idx] / 100.0
        temp = df["temperature_2m"].astype(float).values[idx]
        solar_src = df.get(solar_col, pd.Series(0.0, index=df.index))
        solar = solar_src.astype(float).values[idx]
    else:
        # The day/night split can wrap midnight, so rows aren't contiguous in calendar
        # time -- interpolate by row position; a calendar reindex would drop a chunk.
        x_src = np.arange(len(df))
        x_tgt = np.linspace(0, len(df) - 1, n)
        rh = np.interp(x_tgt, x_src, df["relative_humidity_2m"].astype(float).values) / 100.0
        temp = np.interp(x_tgt, x_src, df["temperature_2m"].astype(float).values)
        solar_src = df.get(solar_col, pd.Series(0.0, index=df.index))
        solar = np.interp(x_tgt, x_src, solar_src.astype(float).values)
    solar = np.maximum(0.0, solar)
    h_amb = (FIXED_H_AMB_W_M2_K,) * n  # ambient convection coefficient -- fixed, not wind-derived
    # B4: forced condenser air decouples condenser cooling from the absorber's h_amb.
    # Wilson's own fans hold ~0.5 m/s regardless of ambient wind (Note S2), so this is a
    # floor, not a replacement -- a windy site still gets the better of the two.
    h_amb_cond = None
    if condenser_air_speed_m_s > 0.0:
        forced = wind_to_h_amb_w_m2_k(condenser_air_speed_m_s)
        h_amb_cond = (max(forced, FIXED_H_AMB_W_M2_K),) * n
    return PhaseProfile(
        temperature_c=tuple(float(x) for x in temp),
        relative_humidity=tuple(float(x) for x in rh),
        solar_w_m2=tuple(float(x) for x in solar),
        h_amb_w_m2_k=h_amb,
        h_amb_cond_w_m2_k=h_amb_cond,
    )


def baseline_profile(
    *,
    temperature_c: float = BASELINE_T_AMB_C,
    relative_humidity: float = BASELINE_RH_AMB,
    solar_w_m2: float = BASELINE_Q_SOLAR_W_M2,
    h_amb_w_m2_k: float = BASELINE_H_AMB_W_M2_K,
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


# Wilson Methods: hydrogel cast at DVS equilibrium with ~20% RH before cycling.
BASELINE_INITIAL_EQUILIBRIUM_RH = FABRICATION_EQUILIBRIUM_RH


def baseline_initial_c_w(*, h_m: float = H0_M) -> float:
    """Initial brine state for baseline / Fig. 2 replay (fabrication at ~20% RH)."""
    return equilibrium_c_w_from_dvs_at_rh(
        BASELINE_INITIAL_EQUILIBRIUM_RH,
        h_m=h_m,
        h0_ref_m=h_m,
    )


def replay_profile(
    mode: Literal["atacama-replay", "cambridge-replay", "fig-s1-replay"],
    *,
    cache_dir: str | None = None,
) -> DailyWeatherProfile:
    if mode == "atacama-replay":
        return atacama_field_profile()
    if mode == "fig-s1-replay":
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


def _single_day_df(
    df: pd.DataFrame,
    day: date,
) -> pd.DataFrame:
    if df.index.tz is not None:
        mask = df.index.date == day
    else:
        mask = df.index.normalize() == pd.Timestamp(day)
    return df.loc[mask].copy()


def real_day_profile(
    lat: float,
    lon: float,
    day: date,
    *,
    cache_dir: str | None = None,
    poa_tilt_deg: float | None = None,
) -> DailyWeatherProfile:
    """One real calendar day's weather at a site -- no averaging of any kind.

    Single-day CPU runs use a real day rather than a synthetic mean one: mean days
    smear out the diurnal extremes that actually drive uptake and desorption.
    """
    df = fetch_year_weather(lat, lon, day.year, cache_dir=cache_dir)
    day_df = _single_day_df(df, day)
    if day_df.empty:
        raise ValueError(f"No weather data for {day.isoformat()} at ({lat:.4f}, {lon:.4f}).")
    return profile_from_day_df(day_df, poa_tilt_deg=poa_tilt_deg)


def real_weather_days_from_df(
    df: pd.DataFrame,
    *,
    stride: int = 1,
    seal_offset_h: float = 0.0,
    open_offset_h: float = 0.0,
    condenser_air_speed_m_s: float = 0.0,
    poa_tilt_deg: float | None = None,
) -> list[tuple[date, DailyWeatherProfile, pd.DataFrame]]:
    """Build per-day profiles from a pre-fetched year of Open-Meteo data.

    The keyword arguments are complex mode's profile-level design variables (A1
    schedule shift, B4 forced condenser air, and opt-in POA transposition so tilt
    drives solar gain). They make the profile set design-dependent, so a caller
    sweeping designs must rebuild per design point rather than fetching once --
    see ``sawh_bayesopt.sites.profiles_for_design``.
    """
    days_out: list[tuple[date, DailyWeatherProfile, pd.DataFrame]] = []
    for idx, (day_key, group) in enumerate(df.groupby(df.index.date)):
        if stride > 1 and idx % stride != 0:
            continue
        try:
            prof = profile_from_day_df(
                group,
                seal_offset_h=seal_offset_h,
                open_offset_h=open_offset_h,
                condenser_air_speed_m_s=condenser_air_speed_m_s,
                poa_tilt_deg=poa_tilt_deg,
            )
            days_out.append((day_key, prof, group))
        except (ValueError, KeyError):
            continue
    return days_out

# --- Wilson Note S1 Fig. S1D mass-transfer validation profile (24 h cycle) ---

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


def fig_s1_initial_c_w(*, h_m: float = H0_M) -> float:
    """Initial brine state matching Fig. S1D start (~1.2 L/m² at H₀)."""
    return c_w_from_water_in_gel_l_m2(FIG_S1_INITIAL_WATER_L_M2, h_m)

# --- Wilson Fig. 4 Atacama field-test weather (24 h from 18:00) ---

_WEATHER_DATA_DIR = Path(__file__).resolve().parent / "data" / "weather"
ATACAMA_RH_CSV = _WEATHER_DATA_DIR / "Atacama_RH.csv"
ATACAMA_TEMP_CSV = _WEATHER_DATA_DIR / "Atacama_Temp.csv"
ATACAMA_AMB_CSV = _WEATHER_DATA_DIR / "Atacama_amb.csv"
ATACAMA_SOLAR_CSV = _WEATHER_DATA_DIR / "Atacama_solar_kW_m2.csv"

# 24 h timeline origin: 18:00 (6 pm) on absorption night, matching paper figure.
ABSORPTION_HOURS = 12.0

# Atacama field protocol (Methods): install at 8 a.m., ~8 h desorption in sun.
ATACAMA_INSTALL_HOUR_FROM_ORIGIN = 14.0  # 18:00 + 14 h = 08:00
ATACAMA_WIND_STEP_HOUR_FROM_ORIGIN = 20.0  # 18:00 + 20 h = 14:00 (2 p.m.)
# Digitized Fig. 4 curves start ≈8:09 a.m., so desorption starts there, not 8:00 a.m.;
# the window still ends at 4 p.m.
ATACAMA_DESORPTION_START_OFFSET_H = 0.15
ATACAMA_FIELD_DESORPTION_HOURS = 8.0 - ATACAMA_DESORPTION_START_OFFSET_H
ATACAMA_FIELD_DESORPTION_STEPS = int(
    round(ATACAMA_FIELD_DESORPTION_HOURS * 3600.0 / PHASE_DT_S)
)
# The device carries 5 dc fans (≈0.5 m/s) on the condenser backing, but Note S4 states
# that test-day wind ≫ 0.5 m/s, "so the inclusion of active cooling is negligible for
# environmental testing presented in this work" -- their model runs one h_amb for the
# whole device. Decoupling the condenser at h=10 while the absorber sits at h=1 kept the
# condenser artificially cold all morning and inflated the yield by ~40%.


def atacama_field_profile() -> DailyWeatherProfile:
    """Atacama field validation: 12 h open absorption, then desorption from where the
    digitized Fig. 4 curves start (≈8:09 a.m.) through 4 p.m."""
    return _build_atacama_profile(
        desorption_start_h=(
            ATACAMA_INSTALL_HOUR_FROM_ORIGIN + ATACAMA_DESORPTION_START_OFFSET_H
        ),
        desorption_hours=ATACAMA_FIELD_DESORPTION_HOURS,
        desorption_steps=ATACAMA_FIELD_DESORPTION_STEPS,
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

    # Absorption is 18:00–06:00, NOT the 12 h ending at the 8 a.m. install: the Methods
    # quote 11 °C / 38% RH as the phase averages, and this window gives 11.5 °C / 38.3%
    # where 20:09–08:09 gives 8.3 °C / 42.1%. The stage stays outside through the
    # 06:00–08:09 gap and would keep absorbing, but the paper's 12 h stops at 06:00.
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
            dt_s=PHASE_DT_S,
        ),
    )


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
    """Wilson Atacama Methods: h_amb = 1 W/m²K stepped to 10 W/m²K at 2 p.m. (Fig. 4
    timeline origin is 18:00, so 2 p.m. local = hour 20)."""
    return (
        10.0
        if hours_from_6pm >= ATACAMA_WIND_STEP_HOUR_FROM_ORIGIN
        else 1.0
    )
