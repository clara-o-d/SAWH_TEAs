"""Land grid generation: (lat, lon) nodes on land, with optional country/latitude exclusion.

Deliberately has no matplotlib dependency (only numpy/shapely/cartopy.io.shapereader),
so headless callers (job-array workers, weather prefetch) don't need the plotting stack
just to build a grid.
"""

from __future__ import annotations

import math
import time

import numpy as np


def _prepared_land_union():
    from shapely import geometry as sh_geom
    from shapely.ops import unary_union
    from shapely.prepared import prep

    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(resolution="110m", category="physical", name="land")
    geoms = list(shpreader.Reader(path).geometries())
    u = prep(unary_union(geoms))
    return u, sh_geom


def _prepared_country_union(names: set[str]):
    """Prepared union of country polygons matching ADMIN in *names* (Natural Earth)."""
    from shapely.ops import unary_union
    from shapely.prepared import prep

    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
    geoms = [r.geometry for r in shpreader.Reader(path).records() if r.attributes.get("ADMIN") in names]
    return prep(unary_union(geoms))


# Land above this latitude within these countries is excluded by default: no realistic
# deployment demand, and it's mostly the same polar-day/night territory the lat_hi cutoff
# is already trying to avoid, just reaching further south here than the Arctic circle.
DEFAULT_EXCLUDE_COUNTRY_ABOVE_LAT: dict[str, float] = {"Canada": 60.0, "Russia": 60.0}


def grid_land_points(
    step_deg: float = 5.0,
    *,
    lat_lo: float = -56.0,
    lat_hi: float = 72.0,
    exclude_country_above_lat: dict[str, float] | None = DEFAULT_EXCLUDE_COUNTRY_ABOVE_LAT,
) -> list[tuple[float, float]]:
    """All (lat, lon) grid nodes on land at ``step_deg`` spacing (WGS84).

    ``exclude_country_above_lat`` additionally drops points inside a named country
    (Natural Earth ADMIN name) above a given latitude -- e.g. the default drops northern
    Canada and Russia above 60°N, which the global ``lat_hi`` cutoff alone doesn't reach
    (Canada/Russia extend well south of 72°N at those longitudes). Pass ``{}`` or ``None``
    to disable.
    """
    print(
        "  Loading Natural Earth land polygons (first run may download shapefiles)…",
        flush=True,
    )
    t0 = time.perf_counter()
    land, sh_geom = _prepared_land_union()
    print(f"  Land geometry ready in {time.perf_counter() - t0:.2f}s.", flush=True)

    exclude_country_above_lat = exclude_country_above_lat or {}
    exclude_masks = [
        (_prepared_country_union({name}), thresh) for name, thresh in exclude_country_above_lat.items()
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
