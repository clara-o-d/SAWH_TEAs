#!/usr/bin/env python3
"""Global maps of delivered desalination cost, SAWH cost, and the cheaper of the two.

Desalination follows ``desalination_cost_methods.tex`` (Kocher & Menon 2023): coastal RO at
$1/m3 plus levelized conveyance from the nearest ocean coastline to the site,

    LCOW_delivered = 1.0 + c_vert * dz + c_horiz_eff * d

with dz the rise above sea level (m) and d the great-circle distance to the coast (km).
``--transport-multiplier 10`` is the paper's high-cost transport scenario.

SAWH costs come from a per-site sweep CSV (BayesOpt summary by default). Those results were
deleted from the working tree, so the default is read straight out of git history.

Examples::

  python desal_vs_sawh_map.py
  python desal_vs_sawh_map.py --transport-multiplier 10 --output plots/desal_high_transport.png
  python desal_vs_sawh_map.py --sawh-csv my_sweep.csv --sawh-col lcow_usd_per_m3
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "global"))
from plot_map import LogPowerNorm, _save, infer_step, interpolate_to_grid, world_ax  # noqa: E402

LCOW_RO_USD_M3 = 1.0  # coastal seawater RO, CAPEX + OPEX incl. LCOE and brine disposal
C_VERT_USD_M3_PER_M = 5e-4  # levelized vertical lift
C_HORIZ_USD_M3_PER_KM = 13.58e-4  # equal-weighted canal/tunnel/pipe blend
EARTH_R_KM = 6371.0

# The latest per-site SAWH sweep, deleted from the tree in 957b243 ("Delete all results").
GIT_SAWH_CSV = "f753de4:analysis/sawh_bayesopt/full_sweep_summary_simplified_tea.csv"
DEFAULT_SAWH_COL = "simplified_best_combined_lcow_usd_m3"
ELEV_CACHE = Path(__file__).with_name("site_elevation_cache.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sawh-csv", type=Path, default=None,
                    help=f"Per-site SAWH LCOW CSV (default: {GIT_SAWH_CSV} out of git history)")
    ap.add_argument("--sawh-col", default=DEFAULT_SAWH_COL)
    ap.add_argument("--transport-multiplier", type=float, default=1.0,
                    help="Scale both conveyance unit costs (10 = paper's high-cost scenario)")
    ap.add_argument("--points", action="store_true", help="Plot site markers instead of interpolating")
    ap.add_argument("--output", type=Path, default=Path(__file__).with_name("desal_vs_sawh_map.png"))
    ap.add_argument("--selftest", action="store_true", help="Check the coast-distance and cost math, then exit")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()

    df = _load_sawh(args.sawh_csv, args.sawh_col)
    lats, lons = df["lat"].to_numpy(float), df["lon"].to_numpy(float)

    df["coast_km"] = coast_distance_km(lats, lons)
    df["elev_m"] = elevation_m(lats, lons)
    df["desal_lcow"] = (LCOW_RO_USD_M3
                        + args.transport_multiplier * C_VERT_USD_M3_PER_M * df["elev_m"].clip(lower=0.0)
                        + args.transport_multiplier * C_HORIZ_USD_M3_PER_KM * df["coast_km"])
    df["sawh_cheaper"] = df["sawh_lcow"] < df["desal_lcow"]

    n_sawh = int(df["sawh_cheaper"].sum())
    print(f"{len(df)} sites: SAWH cheaper at {n_sawh} ({100 * n_sawh / len(df):.1f}%), "
          f"desalination at {len(df) - n_sawh}")
    print(f"  desal LCOW  median {df.desal_lcow.median():.2f}, max {df.desal_lcow.max():.2f} $/m3")
    print(f"  SAWH  LCOW  median {df.sawh_lcow.median():.2f}, min {df.sawh_lcow.min():.2f} $/m3")
    if n_sawh == 0:  # how far off is SAWH? report the site needing the least transport inflation
        base = (C_VERT_USD_M3_PER_M * df.elev_m.clip(lower=0) + C_HORIZ_USD_M3_PER_KM * df.coast_km)
        need = ((df.sawh_lcow - LCOW_RO_USD_M3) / base).idxmin()
        r = df.loc[need]
        print(f"  SAWH first wins at ({r.lat:g}, {r.lon:g}) if transport costs are "
              f"{(r.sawh_lcow - LCOW_RO_USD_M3) / base[need]:.1f}x the baseline")
    csv_out = args.output.with_suffix(".csv")
    df.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")

    _plot(df, args)
    print(f"Wrote {args.output}")
    return 0


def _selftest() -> int:
    # Rotterdam is on the coast; Denver is ~1200 km from the Gulf of California, at ~1600 m.
    d = coast_distance_km(np.array([51.9, 39.7]), np.array([4.1, -105.0]))
    assert d[0] < 50, d
    assert 1000 < d[1] < 1500, d
    assert abs((LCOW_RO_USD_M3 + C_VERT_USD_M3_PER_M * 1600 + C_HORIZ_USD_M3_PER_KM * d[1]) - 3.42) < 0.15
    print(f"ok: coast distances {d.round(0)} km, Denver delivered cost ~$3.4/m3")
    return 0


def _load_sawh(path: Path | None, col: str) -> pd.DataFrame:
    if path is None:
        blob = subprocess.run(["git", "show", GIT_SAWH_CSV], capture_output=True, check=True,
                              cwd=Path(__file__).resolve().parents[2]).stdout
        df = pd.read_csv(io.BytesIO(blob))
    else:
        df = pd.read_csv(path)
    if col not in df.columns:
        sys.exit(f"Column {col!r} not found. Available: {', '.join(df.columns)}")
    df = df[["lat", "lon", col]].rename(columns={col: "sawh_lcow"})
    df["sawh_lcow"] = pd.to_numeric(df["sawh_lcow"], errors="coerce")
    return df[df["sawh_lcow"].between(0, 1e20)].reset_index(drop=True)


def coast_distance_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance to the nearest ocean coastline vertex (Natural Earth 50m).

    Segmentized at 0.2 deg so long straight coast segments still get sampled, then nearest
    neighbour via a KD-tree on unit-sphere chords (chord ordering matches arc ordering)."""
    import cartopy.feature as cfeature
    import shapely
    from scipy.spatial import cKDTree

    coast = shapely.get_coordinates(shapely.segmentize(
        shapely.geometrycollections(list(cfeature.NaturalEarthFeature("physical", "coastline", "50m")
                                         .geometries())), 0.2))
    chord, _ = cKDTree(_ecef(coast[:, 1], coast[:, 0])).query(_ecef(lats, lons))
    return 2.0 * EARTH_R_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def _ecef(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lats), np.radians(lons)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def elevation_m(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Site elevations from the Open-Meteo elevation API (Copernicus DEM), cached to CSV."""
    import requests

    key = pd.DataFrame({"lat": np.round(lats, 6), "lon": np.round(lons, 6)})
    cache = pd.read_csv(ELEV_CACHE) if ELEV_CACHE.exists() else pd.DataFrame(columns=["lat", "lon", "elev_m"])
    todo = key.merge(cache, on=["lat", "lon"], how="left")
    missing = todo[todo["elev_m"].isna()][["lat", "lon"]].drop_duplicates()

    for i in range(0, len(missing), 100):  # API caps a request at 100 coordinates
        chunk = missing.iloc[i:i + 100]
        for attempt in range(6):  # free tier rate-limits; back off and retry
            r = requests.get("https://api.open-meteo.com/v1/elevation", timeout=60, params={
                "latitude": ",".join(f"{v:g}" for v in chunk.lat),
                "longitude": ",".join(f"{v:g}" for v in chunk.lon)})
            if r.status_code != 429:
                break
            time.sleep(5 * 2 ** attempt)
        r.raise_for_status()
        cache = pd.concat([cache, chunk.assign(elev_m=r.json()["elevation"])], ignore_index=True)
        cache.drop_duplicates(["lat", "lon"], keep="last").to_csv(ELEV_CACHE, index=False)
        print(f"  fetched elevation for {min(i + 100, len(missing))}/{len(missing)} new sites")
        time.sleep(1.0)
    return key.merge(cache, on=["lat", "lon"], how="left")["elev_m"].to_numpy(float)


def _plot(df: pd.DataFrame, args) -> None:
    lons, lats = df["lon"].to_numpy(float), df["lat"].to_numpy(float)
    step = infer_step(lons)
    vmin = float(min(df.desal_lcow.min(), df.sawh_lcow.min()))
    vmax = float(np.percentile(np.concatenate([df.desal_lcow, df.sawh_lcow]), 99))
    norm = LogPowerNorm(vmin=vmin, vmax=vmax, gamma=0.7)

    fig = plt.figure(figsize=(13, 15))
    suffix = "" if args.transport_multiplier == 1.0 else f" (transport x{args.transport_multiplier:g})"
    ticks = [t for t in (1, 2, 3, 5, 7, 10, 15, 20, 30, 50) if vmin <= t <= vmax]
    # Each cost panel shows only the sites that technology wins, so the two tile the land together.
    for i, (col, label, wins) in enumerate([("desal_lcow", "Desalination, delivered", ~df.sawh_cheaper),
                                            ("sawh_lcow", "SAWH, optimized per site", df.sawh_cheaper)]):
        ax = world_ax(fig, (3, 1, i + 1))
        won = df[wins]
        if not won.empty:
            _draw(ax, won.lon.to_numpy(float), won.lat.to_numpy(float), won[col].to_numpy(float), step, args,
                  cmap="viridis", norm=norm)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
                            shrink=0.85, pad=0.02, ticks=ticks)
        cbar.ax.set_yticklabels([f"{t:g}" for t in ticks])
        cbar.set_label("LCOW ($/m³)")
        ax.set_title(f"{label}{suffix} — where cheapest ({len(won)} sites)")

    ax = world_ax(fig, (3, 1, 3))
    cmap = mcolors.ListedColormap(["#3b76af", "#d1802f"])  # desal, SAWH
    mesh = _draw(ax, lons, lats, df["sawh_cheaper"].to_numpy(float), step, args,
                 cmap=cmap, norm=mcolors.BoundaryNorm([-0.5, 0.5, 1.5], 2))
    cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1], shrink=0.85, pad=0.02)
    cbar.ax.set_yticklabels(["Desalination", "SAWH"])
    ax.set_title(f"Cheaper technology{suffix}")
    _save(fig, args.output)


def _draw(ax, lons, lats, vals, step, args, **kw):
    import cartopy.crs as ccrs

    if args.points:
        return ax.scatter(lons, lats, c=vals, s=14, linewidths=0, transform=ccrs.PlateCarree(), zorder=3, **kw)
    lon_v, lat_v, grid = interpolate_to_grid(lons, lats, vals, sample_step_deg=step)
    return ax.pcolormesh(lon_v, lat_v, grid, shading="auto", transform=ccrs.PlateCarree(), zorder=2, **kw)


if __name__ == "__main__":
    raise SystemExit(main())
