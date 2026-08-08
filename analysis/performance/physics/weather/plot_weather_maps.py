#!/usr/bin/env python3
"""Map the cached Open-Meteo weather: annual-mean RH and annual-mean solar irradiance.

Reads every response already sitting in the requests-cache sqlite (no network), reduces
each site to its annual means, and interpolates onto a 1 deg land-masked grid.

  python plot_weather_maps.py
  python plot_weather_maps.py --cache-dir ../../../sawh_bayesopt/.weather_cache --refresh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "analysis" / "global"))
from plot_map import interpolate_to_grid, world_ax  # noqa: E402

_PANELS = (
    ("mean_rh_pct", "Annual mean relative humidity (%)", "YlGnBu"),
    ("mean_solar_w_m2", "Annual mean shortwave irradiance (W m$^{-2}$)", "inferno"),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=_REPO / "solar_lumped" / ".weather_cache")
    ap.add_argument("--step-deg", type=float, default=1.0, help="Interpolation grid resolution")
    ap.add_argument("--refresh", action="store_true", help="Re-read the sqlite cache instead of the summary CSV")
    ap.add_argument("--output", type=Path, default=Path(__file__).with_name("weather_maps.png"))
    args = ap.parse_args()

    summary_csv = args.cache_dir / "site_means.csv"
    if summary_csv.exists() and not args.refresh:
        df = pd.read_csv(summary_csv)
    else:
        df = _site_means(args.cache_dir)
        df.to_csv(summary_csv, index=False)
        print(f"Wrote {summary_csv}")
    if df.empty:
        sys.exit(f"No cached responses under {args.cache_dir}.")

    lons, lats = df["lon"].to_numpy(float), df["lat"].to_numpy(float)
    fig, axes = plt.figure(figsize=(13, 12)), []
    for i, (col, label, cmap) in enumerate(_PANELS, start=1):
        ax = world_ax(fig, (2, 1, i))
        lon_v, lat_v, grid = interpolate_to_grid(
            lons, lats, df[col].to_numpy(float),
            sample_step_deg=_nn_spacing(lons, lats), fine_step_deg=args.step_deg,
        )
        import cartopy.crs as ccrs

        mesh = ax.pcolormesh(lon_v, lat_v, grid, cmap=cmap, shading="auto",
                             transform=ccrs.PlateCarree(), zorder=2)
        fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02).set_label(label)
        ax.set_title(f"{label} -- {len(df)} cached sites")
        axes.append(ax)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")
    return 0


def _site_means(cache_dir: Path) -> pd.DataFrame:
    """One row per cached site: annual mean RH and shortwave from the hourly block."""
    import requests_cache

    session = requests_cache.CachedSession(cache_name=str(cache_dir / "openmeteo_cache"), backend="sqlite")
    rows: dict[tuple[float, float], dict] = {}
    for n, response in enumerate(session.cache.responses.values(), start=1):
        try:
            data = json.loads(response.content)
        except (ValueError, UnicodeDecodeError):
            continue
        hourly = data.get("hourly")
        if not hourly or "relative_humidity_2m" not in hourly:
            continue
        lat, lon = float(data["latitude"]), float(data["longitude"])
        rows[(lat, lon)] = {
            "lat": lat,
            "lon": lon,
            "mean_rh_pct": float(np.nanmean(np.asarray(hourly["relative_humidity_2m"], dtype=float))),
            "mean_solar_w_m2": float(np.nanmean(np.asarray(hourly["shortwave_radiation"], dtype=float))),
        }
        if n % 200 == 0:
            print(f"  parsed {n} cached responses, {len(rows)} sites", flush=True)
    return pd.DataFrame(rows.values())


def _nn_spacing(lons: np.ndarray, lats: np.ndarray) -> float:
    """Median nearest-neighbour distance -- the masking radius for interpolate_to_grid."""
    from scipy.spatial import cKDTree

    pts = np.column_stack([lons, lats])
    if len(pts) < 2:
        return 1.0
    return float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1]))


if __name__ == "__main__":
    raise SystemExit(main())
