"""The two experimentally field-validated SAWH sites: Cambridge, MA and the
Atacama Desert, Chile (Wilson et al. 2025). Weather is fetched once per site
per optimization run and reused across every design-point evaluation -- only
the device design changes per point, not the weather (same pattern as
scripts/grid_param_sweep.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from solar_lumped.weather.client import WeatherClient

# Monthly-mean-day profiles: (month, DailyWeatherProfile, n_days_in_month).
MonthlyProfiles = list[tuple[int, object, int]]


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


def monthly_mean_profiles(df) -> MonthlyProfiles:
    """One representative mean-day profile per calendar month present in *df*.

    Verbatim logic from scripts/grid_param_sweep.py::monthly_mean_profiles --
    kept as a local copy (rather than importing the script) since scripts/ in
    solar_lumped isn't part of its installed package surface.
    """
    import pandas as pd

    from solar_lumped.weather.climate import representative_mean_day_df
    from solar_lumped.weather.profiles import profile_from_day_df

    out: MonthlyProfiles = []
    for m in sorted(set(df.index.month)):
        month_df = df[df.index.month == m]
        if month_df.empty:
            continue
        ref_day = month_df.index[len(month_df) // 2].date()
        mean_day_df = representative_mean_day_df(month_df, reference_day=ref_day)
        profile = profile_from_day_df(mean_day_df)
        n_days = len(pd.unique(month_df.index.date))
        out.append((m, profile, n_days))
    return out


def fetch_monthly_profiles(site: SiteSpec, *, cache_dir: str | Path) -> MonthlyProfiles:
    """Fetch *site*'s full-year weather and derive its monthly mean-day profiles.

    Falls back from the historical-forecast endpoint to the plain historical
    one on failure, matching scripts/grid_param_sweep.py's try/except.
    """
    client = WeatherClient(cache_dir=str(cache_dir))
    start = f"{site.year}-01-01"
    end = f"{site.year}-12-31"
    try:
        _, df = client.get_historical_forecast_site_weather(site.lat, site.lon, start, end)
    except Exception:
        df = client.get_historical(site.lat, site.lon, start, end)
    return monthly_mean_profiles(df)
