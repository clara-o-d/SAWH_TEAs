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



def daily_profiles(df) -> DailyProfiles:
    """Every calendar day in *df* as its own profile, in chronological order."""
    from solar_lumped.weather import real_weather_days_from_df

    # No POA here even though it is the weather.py default: these profiles are shared by
    # every design in a batch, but ``tilt_deg`` is a swept dimension, so a single POA tilt
    # would be wrong for all but one of them. Complex mode rebuilds per design and does
    # transpose -- see ``evaluator._profiles_for_design``.
    days = real_weather_days_from_df(df, poa_tilt_deg=None)
    return [(d.timetuple().tm_yday, prof) for d, prof, _group in days]


def fetch_daily_profiles(site: SiteSpec, *, cache_dir: str | Path) -> DailyProfiles:
    """Fetch *site*'s full year of weather and split it into per-day profiles."""
    return daily_profiles(fetch_site_frame(site, cache_dir=cache_dir))


def fetch_site_elevations(sites, *, cache_dir: str | Path) -> dict[str, float]:
    """name -> site elevation (m), off the same cached year frames the profiles come from.

    Elevation is a site property, not a design variable, so it is resolved once per run
    rather than per design point. Costs no extra request: the frames are already in the
    requests-cache, and archive responses never expire.
    """
    from solar_lumped.weather import site_elevation_m

    return {
        s.name: site_elevation_m(fetch_site_frame(s, cache_dir=cache_dir)) for s in sites
    }


def fetch_site_frame(site: SiteSpec, *, cache_dir: str | Path):
    """Fetch *site*'s raw year of weather, unsplit.

    Complex mode needs the frame rather than prebuilt profiles: A1's schedule
    offsets, B4's condenser airflow, and POA tilt are design variables that change
    the split itself, so profiles are rebuilt per design point (see
    ``evaluator._profiles_for_design``).
    """
    return fetch_year_weather(site.lat, site.lon, site.year, cache_dir=str(cache_dir))


def site_from_lat_lon(lat: float, lon: float, *, year: int = 2024, name: str | None = None) -> SiteSpec:
    """A SiteSpec at arbitrary coordinates, named from them unless given a name."""
    return SiteSpec(name or f"lat{lat:+.4f}_lon{lon:+.4f}", float(lat), float(lon), year)


def land_grid_sites(
    *, step_deg: float, indices: list[int] | None = None, year: int = 2024
) -> tuple[SiteSpec, ...]:
    """Sites on solar_lumped's land grid, for global sweeps.

    Reuses ``weather.grid_land_points`` -- the same land mask the gpu_sweep global
    sweep already runs on -- so a global BayesOpt run and a global grid sweep cover
    an identical site list and stay comparable.
    """
    from solar_lumped.weather import grid_land_points

    points = list(grid_land_points(step_deg=step_deg))
    picked = list(range(len(points))) if indices is None else list(indices)
    return tuple(
        SiteSpec(
            f"grid{i:05d}_lat{points[i][0]:+.3f}_lon{points[i][1]:+.3f}",
            float(points[i][0]),
            float(points[i][1]),
            year,
        )
        for i in picked
    )


__all__ = [
    "ATACAMA",
    "CAMBRIDGE",
    "DailyProfiles",
    "SiteSpec",
    "WeatherClient",
    "daily_profiles",
    "fetch_daily_profiles",
]
