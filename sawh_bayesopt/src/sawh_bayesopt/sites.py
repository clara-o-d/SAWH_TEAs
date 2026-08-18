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



def fetch_site_frame(site: SiteSpec, *, cache_dir: str | Path):
    """Fetch *site*'s raw year of weather, unsplit.

    The evaluator needs the frame rather than prebuilt profiles: A1's schedule offsets and
    POA tilt (both fidelities) plus complex mode's condenser airflow are design variables
    that live inside the profile, so profiles are rebuilt per design point (see
    ``evaluator._profiles_for_design``).
    """
    return fetch_year_weather(site.lat, site.lon, site.year, cache_dir=str(cache_dir))


def site_from_lat_lon(lat: float, lon: float, *, year: int = 2024, name: str | None = None) -> SiteSpec:
    """A SiteSpec at arbitrary coordinates, named from them unless given a name."""
    return SiteSpec(name or f"lat{lat:+.4f}_lon{lon:+.4f}", float(lat), float(lon), year)


__all__ = [
    "ATACAMA",
    "CAMBRIDGE",
    "DailyProfiles",
    "SiteSpec",
    "WeatherClient",
]
