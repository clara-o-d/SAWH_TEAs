"""Plane-of-array transposition (weather.py::plane_of_array_w_m2), the opt-in
tilt-vs-solar-gain coupling. Default (GHI) behaviour must be untouched."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_lumped.weather import (
    POA_DEFAULT_ALBEDO,
    plane_of_array_w_m2,
    profile_from_day_df,
)

CAMBRIDGE = (42.36, -71.09, "America/New_York")
ATACAMA = (-23.65, -70.40, "America/Santiago")

# Hours (local clock) over which the sun is unambiguously up at both sites on both
# solstices. The zero-tilt identity is asserted only here: near sunrise/sunset the
# module deliberately zeroes POA below its near-horizon cos(zenith) clamp, so a
# synthetic GHI that ignores real sunrise times would disagree there for a reason
# that has nothing to do with the transposition being tested.
_SOLID_DAYLIGHT = (9.0, 15.0)


def _day_df(
    lat: float, lon: float, tz: str, day: str = "2024-06-21", ghi_peak: float = 900.0
) -> pd.DataFrame:
    """One synthetic clear-ish day at 15-min *local* resolution, GHI a half-sine 06:00-18:00."""
    idx = pd.date_range(f"{day} 00:00", periods=96, freq="15min", tz=tz)
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
        },
        index=idx,
    )


@pytest.mark.parametrize("site", [CAMBRIDGE, ATACAMA], ids=["cambridge", "atacama"])
def test_zero_tilt_is_exactly_ghi(site: tuple[float, float, str]) -> None:
    """At tilt=0 the three POA terms collapse to DNI*cos(zen) + DHI + 0 == GHI exactly.
    This is the correctness anchor for the whole decomposition/transposition chain: it
    only holds if the Erbs split and the cos(AOI) geometry are mutually consistent."""
    lat, lon, tz = site
    df = _day_df(lat, lon, tz)
    ghi = df["shortwave_radiation"].to_numpy()
    poa = plane_of_array_w_m2(
        ghi, df.index, latitude_deg=lat, longitude_deg=lon, tilt_deg=0.0, albedo=POA_DEFAULT_ALBEDO
    )
    hours = df.index.hour + df.index.minute / 60.0
    core = (hours >= _SOLID_DAYLIGHT[0]) & (hours <= _SOLID_DAYLIGHT[1])
    assert core.sum() > 0
    assert np.allclose(poa[core], ghi[core], rtol=1e-9, atol=1e-9)


def test_night_collects_nothing() -> None:
    lat, lon, tz = CAMBRIDGE
    df = _day_df(lat, lon, tz)
    poa = plane_of_array_w_m2(
        df["shortwave_radiation"].to_numpy(),
        df.index,
        latitude_deg=lat,
        longitude_deg=lon,
        tilt_deg=35.0,
    )
    assert np.all(poa[df["shortwave_radiation"].to_numpy() <= 0.0] == 0.0)


def test_winter_tilt_beats_flat_in_both_hemispheres() -> None:
    """Equator-facing tilt near |latitude| must out-collect a flat plate in local winter --
    and the default azimuth has to flip with the hemisphere for that to hold in Atacama."""
    for (lat, lon, tz), winter_day in ((CAMBRIDGE, "2024-12-21"), (ATACAMA, "2024-06-21")):
        df = _day_df(lat, lon, tz, day=winter_day)
        ghi = df["shortwave_radiation"].to_numpy()
        kw = dict(latitude_deg=lat, longitude_deg=lon)
        flat = plane_of_array_w_m2(ghi, df.index, tilt_deg=0.0, **kw).sum()
        tilted = plane_of_array_w_m2(ghi, df.index, tilt_deg=abs(lat), **kw).sum()
        assert tilted > flat, f"lat={lat}: tilted {tilted:.0f} !> flat {flat:.0f}"


def test_profile_default_is_ghi_and_poa_is_opt_in() -> None:
    """Existing callers must see unchanged profiles; POA applies only when asked for."""
    lat, lon, tz = CAMBRIDGE
    df = _day_df(lat, lon, tz)
    default = profile_from_day_df(df)
    assert default.desorption.solar_w_m2 == profile_from_day_df(df).desorption.solar_w_m2

    tilted = profile_from_day_df(df, poa_tilt_deg=35.0)
    assert tilted.desorption.solar_w_m2 != default.desorption.solar_w_m2
    # Day/night split stays keyed on GHI, so phase lengths must not shift.
    assert len(tilted.desorption.solar_w_m2) == len(default.desorption.solar_w_m2)
    assert len(tilted.absorption.solar_w_m2) == len(default.absorption.solar_w_m2)

    # tilt=0 recovers the GHI profile to within the near-horizon clamp, which zeroes a
    # few sunrise/sunset samples the GHI-keyed day split still counts as daytime.
    flat = np.array(profile_from_day_df(df, poa_tilt_deg=0.0).desorption.solar_w_m2)
    base = np.array(default.desorption.solar_w_m2)
    assert flat.sum() == pytest.approx(base.sum(), rel=0.02)


def test_poa_requires_site_coordinates() -> None:
    lat, lon, tz = CAMBRIDGE
    df = _day_df(lat, lon, tz).drop(columns=["latitude", "longitude"])
    with pytest.raises(ValueError, match="latitude"):
        profile_from_day_df(df, poa_tilt_deg=35.0)
