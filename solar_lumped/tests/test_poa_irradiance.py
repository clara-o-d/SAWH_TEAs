"""Plane-of-array transposition (weather.py::plane_of_array_w_m2), the tilt-vs-solar-gain
coupling. On by default; ``poa_tilt_deg=None`` must still give back raw GHI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_lumped.weather import (
    POA_DEFAULT_ALBEDO,
    POA_DEFAULT_TILT_DEG,
    plane_of_array_w_m2,
    profile_from_day_df,
    real_weather_days_from_df,
)

# (lat, lon, utc_offset_h). Frames are built tz-naive with an explicit offset because
# that is the shape Open-Meteo actually returns: a fixed-offset local clock whose index
# cannot be tz_localize'd across a DST transition (weather.py::_series_to_dataframe).
STANFORD = (37.4275, -122.1697, -8.0)
ATACAMA = (-23.65, -70.40, -4.0)

# Hours (local clock) over which the sun is unambiguously up at both sites on both
# solstices. The zero-tilt identity is asserted only here: near sunrise/sunset the
# module deliberately zeroes POA below its near-horizon cos(zenith) clamp, so a
# synthetic GHI that ignores real sunrise times would disagree there for a reason
# that has nothing to do with the transposition being tested.
_SOLID_DAYLIGHT = (9.0, 15.0)


def _day_df(
    lat: float, lon: float, utc_offset_h: float, day: str = "2024-06-21", ghi_peak: float = 900.0
) -> pd.DataFrame:
    """One synthetic clear-ish day at 15-min *local* resolution, GHI a half-sine 06:00-18:00."""
    idx = pd.date_range(f"{day} 00:00", periods=96, freq="15min")
    hours = idx.hour + idx.minute / 60.0
    ghi = np.where(
        (hours >= 6.0) & (hours <= 18.0),
        ghi_peak * np.sin(np.pi * (hours - 6.0) / 12.0),
        0.0,
    )
    return pd.DataFrame(
        {
            "temperature_2m": 25.0,
            "relative_humidity_2m": 45.0,
            "shortwave_radiation": ghi,
            "latitude": lat,
            "longitude": lon,
            "utc_offset_s": utc_offset_h * 3600.0,
        },
        index=idx,
    )


@pytest.mark.parametrize("site", [STANFORD, ATACAMA], ids=["stanford", "atacama"])
def test_zero_tilt_is_exactly_ghi(site: tuple[float, float, float]) -> None:
    """At tilt=0 the three POA terms collapse to DNI*cos(zen) + DHI + 0 == GHI exactly.
    This is the correctness anchor for the whole decomposition/transposition chain: it
    only holds if the Erbs split and the cos(AOI) geometry are mutually consistent."""
    lat, lon, off = site
    df = _day_df(lat, lon, off)
    ghi = df["shortwave_radiation"].to_numpy()
    poa = plane_of_array_w_m2(
        ghi, df.index, latitude_deg=lat, longitude_deg=lon, utc_offset_h=off,
        tilt_deg=0.0, albedo=POA_DEFAULT_ALBEDO,
    )
    hours = df.index.hour + df.index.minute / 60.0
    core = (hours >= _SOLID_DAYLIGHT[0]) & (hours <= _SOLID_DAYLIGHT[1])
    assert core.sum() > 0
    assert np.allclose(poa[core], ghi[core], rtol=1e-9, atol=1e-9)


def test_night_collects_nothing() -> None:
    lat, lon, off = STANFORD
    df = _day_df(lat, lon, off)
    poa = plane_of_array_w_m2(
        df["shortwave_radiation"].to_numpy(),
        df.index,
        latitude_deg=lat,
        longitude_deg=lon,
        utc_offset_h=off,
        tilt_deg=35.0,
    )
    assert np.all(poa[df["shortwave_radiation"].to_numpy() <= 0.0] == 0.0)


def test_utc_offset_places_the_sun_on_the_local_clock() -> None:
    """Regression: Open-Meteo frames are tz-naive, so the offset must be passed in. Reading
    it off the index gave zero -- the model put solar noon at 07:00 EST and zeroed every
    morning sample as 'sun below horizon' while GHI said the sun was clearly up."""
    lat, lon, off = STANFORD
    df = _day_df(lat, lon, off)
    ghi = df["shortwave_radiation"].to_numpy()
    hours = df.index.hour + df.index.minute / 60.0
    morning = (hours >= 8.0) & (hours <= 11.0)
    kw = dict(latitude_deg=lat, longitude_deg=lon, tilt_deg=30.0)

    good = plane_of_array_w_m2(ghi, df.index, utc_offset_h=off, **kw)
    assert np.all(good[morning] > 0.0)
    # Treating the local clock as UTC is exactly the old bug; it must not look the same.
    wrong = plane_of_array_w_m2(ghi, df.index, utc_offset_h=0.0, **kw)
    assert np.any(wrong[morning] == 0.0)


def test_winter_tilt_beats_flat_in_both_hemispheres() -> None:
    """Equator-facing tilt near |latitude| must out-collect a flat plate in local winter --
    and the default azimuth has to flip with the hemisphere for that to hold in Atacama."""
    for (lat, lon, off), winter_day in ((STANFORD, "2024-12-21"), (ATACAMA, "2024-06-21")):
        df = _day_df(lat, lon, off, day=winter_day)
        ghi = df["shortwave_radiation"].to_numpy()
        kw = dict(latitude_deg=lat, longitude_deg=lon, utc_offset_h=off)
        flat = plane_of_array_w_m2(ghi, df.index, tilt_deg=0.0, **kw).sum()
        tilted = plane_of_array_w_m2(ghi, df.index, tilt_deg=abs(lat), **kw).sum()
        assert tilted > flat, f"lat={lat}: tilted {tilted:.0f} !> flat {flat:.0f}"


def test_profile_defaults_to_poa_and_none_opts_out() -> None:
    """POA is the default; ``poa_tilt_deg=None`` is the Wilson-recreation GHI escape."""
    lat, lon, off = STANFORD
    df = _day_df(lat, lon, off)
    default = profile_from_day_df(df)
    assert (
        default.desorption.solar_w_m2
        == profile_from_day_df(df, poa_tilt_deg=POA_DEFAULT_TILT_DEG).desorption.solar_w_m2
    )

    ghi = profile_from_day_df(df, poa_tilt_deg=None)
    assert default.desorption.solar_w_m2 != ghi.desorption.solar_w_m2
    # Day/night split stays keyed on GHI, so phase lengths must not shift.
    assert len(default.desorption.solar_w_m2) == len(ghi.desorption.solar_w_m2)
    assert len(default.absorption.solar_w_m2) == len(ghi.absorption.solar_w_m2)

    # tilt=0 recovers the GHI profile to within the near-horizon clamp, which zeroes a
    # few sunrise/sunset samples the GHI-keyed day split still counts as daytime.
    flat = np.array(profile_from_day_df(df, poa_tilt_deg=0.0).desorption.solar_w_m2)
    base = np.array(ghi.desorption.solar_w_m2)
    assert flat.sum() == pytest.approx(base.sum(), rel=0.02)


def test_days_from_df_reports_missing_coordinates_instead_of_dropping_days() -> None:
    """The per-day loop skips malformed days; a coordinate-less frame must not read as
    'no usable weather days' when POA is on by default."""
    lat, lon, off = STANFORD
    df = _day_df(lat, lon, off).drop(columns=["latitude", "longitude"])
    with pytest.raises(ValueError, match="latitude"):
        real_weather_days_from_df(df)
    assert len(real_weather_days_from_df(df, poa_tilt_deg=None)) == 1


def test_poa_requires_site_coordinates() -> None:
    lat, lon, off = STANFORD
    df = _day_df(lat, lon, off).drop(columns=["latitude", "longitude"])
    with pytest.raises(ValueError, match="latitude"):
        profile_from_day_df(df, poa_tilt_deg=35.0)
