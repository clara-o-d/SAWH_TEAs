"""The two experimentally field-validated SAWH sites: Cambridge, MA and the Atacama
Desert, Chile (Wilson et al. 2025). Weather is fetched once per site per optimization run
and reused across every design-point evaluation -- only the design changes per point.

Every evaluation runs all 365 real days, so there is no mean-day approximation error: the
year starts from a steady periodic state found by Aitken extrapolation on Jan 1, then each
subsequent day warm-starts from the previous day's end state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from solar_lumped.weather import WeatherClient, fetch_year_weather

# One entry per real calendar day: (day_of_year, DailyWeatherProfile).
DailyProfiles = list[tuple[int, object]]


@dataclass(frozen=True, slots=True)
class SiteSpec:
    name: str
    lat: float
    lon: float
    year: int = 2024


# Cambridge, MA -- MIT rooftop field test (Wilson et al. 2025, Fig. 3).
CAMBRIDGE = SiteSpec("cambridge", 42.36, -71.09)
# Atacama Desert, Chile, near Antofagasta -- field test (Wilson et al. 2025, Fig. 4).
ATACAMA = SiteSpec("atacama", -23.65, -70.40)

DEFAULT_SITES: tuple[SiteSpec, ...] = (CAMBRIDGE, ATACAMA)


def daily_profiles(df) -> DailyProfiles:
    """Every calendar day in *df* as its own profile, in chronological order."""
    from solar_lumped.weather import real_weather_days_from_df

    days = real_weather_days_from_df(df)
    return [(d.timetuple().tm_yday, prof) for d, prof, _group in days]


def fetch_daily_profiles(site: SiteSpec, *, cache_dir: str | Path) -> DailyProfiles:
    """Fetch *site*'s full year of weather and split it into per-day profiles."""
    df = fetch_year_weather(site.lat, site.lon, site.year, cache_dir=str(cache_dir))
    return daily_profiles(df)


__all__ = [
    "ATACAMA",
    "CAMBRIDGE",
    "DEFAULT_SITES",
    "DailyProfiles",
    "SiteSpec",
    "WeatherClient",
    "daily_profiles",
    "fetch_daily_profiles",
]
