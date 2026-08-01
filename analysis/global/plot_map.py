#!/usr/bin/env python3
"""Plot any per-site variable from any results CSV on a world map.

Source-agnostic: the CSV needs a latitude column, a longitude column, and the column you
want to colour by. That covers gpu_sweep output (any case), BayesOpt sweep summaries, and
a single-configuration run alike.

Examples::

  python plot_map.py --csv full_sweep.csv --value lcow_usd_per_m3
  python plot_map.py --csv summary.csv --value mean_yield_kg_m2 --interpolate --scale linear
  python plot_map.py --csv sweep.csv --value hydrogel_thickness_mm --categorical
  python plot_map.py --csv sweep.csv --value lcow_usd_per_m3 --best-by lcow_usd_per_m3
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_LAT_CANDIDATES = ("lat", "latitude", "site_lat")
_LON_CANDIDATES = ("lon", "lng", "longitude", "site_lon")
# Values at or above this are solver sentinels (FAIL_LCO = 1e30), never real results.
_SENTINEL = 1e20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--value", required=True, help="Column to colour by")
    ap.add_argument("--lat-col", default=None, help="Override latitude column autodetection")
    ap.add_argument("--lon-col", default=None, help="Override longitude column autodetection")
    ap.add_argument("--best-by", default=None, metavar="COL",
                    help="Keep only the row minimizing COL at each site (for full-factorial sweeps)")
    ap.add_argument("--interpolate", action="store_true",
                    help="Fill between sites with land-masked interpolation instead of plotting points")
    ap.add_argument("--scale", choices=("linear", "log"), default="linear")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Log-scale contrast exponent; <1 spreads the low end")
    ap.add_argument("--categorical", action="store_true",
                    help="Treat the value as discrete levels with one colour each")
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--exclude-top-pct", type=float, default=0.0,
                    help="Clip the colour scale at this upper percentile")
    ap.add_argument("--title", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    lat_col = args.lat_col or _find_col(df, _LAT_CANDIDATES, "latitude")
    lon_col = args.lon_col or _find_col(df, _LON_CANDIDATES, "longitude")
    if args.value not in df.columns:
        sys.exit(f"Column {args.value!r} not in {args.csv}. Available: {', '.join(df.columns)}")

    if args.best_by:
        if args.best_by not in df.columns:
            sys.exit(f"--best-by column {args.best_by!r} not in {args.csv}")
        df = df.loc[df.groupby([lat_col, lon_col])[args.best_by].idxmin()]

    df = df[[lat_col, lon_col, args.value]].copy()
    df[args.value] = pd.to_numeric(df[args.value], errors="coerce")
    n_raw = len(df)
    df = df.dropna()
    df = df[df[args.value].abs() < _SENTINEL]
    if df.empty:
        sys.exit(f"No finite {args.value!r} values to plot.")
    if len(df) < n_raw:
        print(f"Dropped {n_raw - len(df)} non-finite/sentinel row(s); {len(df)} sites remain.")

    lons = df[lon_col].to_numpy(float)
    lats = df[lat_col].to_numpy(float)
    vals = df[args.value].to_numpy(float)

    out = args.output or args.csv.parent / f"map_{args.value}.png"
    title = args.title or f"{args.value} ({len(vals)} sites)"
    if args.categorical:
        _plot_categorical(lons, lats, vals, out, title, args)
    else:
        _plot_continuous(lons, lats, vals, out, title, args)
    print(f"Wrote {out}")
    return 0


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    sys.exit(f"No {what} column found (tried {', '.join(candidates)}). Pass --{what[:3]}-col.")


def _plot_continuous(lons, lats, vals, out: Path, title: str, args) -> None:
    vmin = args.vmin if args.vmin is not None else float(np.min(vals))
    vmax = args.vmax
    if vmax is None:
        vmax = float(np.percentile(vals, 100.0 - args.exclude_top_pct)) if args.exclude_top_pct > 0 \
            else float(np.max(vals))
    if args.scale == "log":
        vmin = max(vmin, 1e-12)
        norm = LogPowerNorm(vmin=vmin, vmax=max(vmax, vmin * 10), gamma=args.gamma)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = args.cmap or "viridis"

    fig = plt.figure(figsize=(13, 6.5))
    ax = world_ax(fig, 111)
    if args.interpolate:
        lon_v, lat_v, grid = interpolate_to_grid(lons, lats, vals, sample_step_deg=infer_step(lons))
        mesh = ax.pcolormesh(lon_v, lat_v, grid, cmap=cmap, norm=norm,
                             shading="auto", transform=_ccrs().PlateCarree(), zorder=2)
    else:
        mesh = ax.scatter(lons, lats, c=vals, cmap=cmap, norm=norm, s=14, linewidths=0,
                          transform=_ccrs().PlateCarree(), zorder=3)
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.75, pad=0.02)
    cbar.set_label(args.value)
    if args.scale == "log":
        _log_decade_ticks(cbar, vmin, vmax, norm)
    ax.set_title(title)
    _save(fig, out)


def _plot_categorical(lons, lats, vals, out: Path, title: str, args) -> None:
    levels = np.unique(vals)
    if len(levels) > 20:
        sys.exit(f"--categorical needs <=20 distinct values, found {len(levels)}.")
    base = plt.get_cmap(args.cmap or "tab10")
    colors = [base(i % base.N) for i in range(len(levels))]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(len(levels) + 1) - 0.5
    codes = np.searchsorted(levels, vals).astype(float)

    fig = plt.figure(figsize=(13, 6.5))
    ax = world_ax(fig, 111)
    mesh = ax.scatter(lons, lats, c=codes, cmap=cmap, norm=mcolors.BoundaryNorm(bounds, cmap.N),
                      s=14, linewidths=0, transform=_ccrs().PlateCarree(), zorder=3)
    cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(len(levels)), shrink=0.75, pad=0.02)
    cbar.ax.set_yticklabels([f"{v:g}" for v in levels])
    cbar.set_label(args.value)
    ax.set_title(title)
    _save(fig, out)


def world_ax(fig, pos):
    """A PlateCarree world axis with land/ocean/coastline basemap."""
    ccrs, cfeature = _map_stack()
    args = pos if isinstance(pos, tuple) else (pos,)
    ax = fig.add_subplot(*args, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "land", "110m", facecolor="0.88", edgecolor="0.4", linewidth=0.3, zorder=0))
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="0.92", zorder=0))
    ax.coastlines(resolution="110m", color="0.35", linewidth=0.4, zorder=1)
    return ax


def interpolate_to_grid(lons, lats, vals, *, sample_step_deg: float, fine_step_deg: float = 0.5):
    """Scattered (lon, lat, val) onto a fine grid: linear barycentric, holes filled nearest.
    Cells farther than one sample spacing away, or over water, are masked so colour stays on land."""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    lon_v = np.arange(lons.min(), lons.max() + fine_step_deg, fine_step_deg)
    lat_v = np.arange(lats.min(), lats.max() + fine_step_deg, fine_step_deg)
    lon_g, lat_g = np.meshgrid(lon_v, lat_v)
    pts = np.column_stack([lons, lats])

    grid = griddata(pts, vals, (lon_g, lat_g), method="linear")
    holes = np.isnan(grid)
    if np.any(holes):
        grid[holes] = griddata(pts, vals, (lon_g[holes], lat_g[holes]), method="nearest")
    dist, _ = cKDTree(pts).query(np.column_stack([lon_g.ravel(), lat_g.ravel()]))
    grid[(dist > sample_step_deg * 1.05).reshape(grid.shape)] = np.nan
    grid[~_land_mask(lon_g, lat_g)] = np.nan
    return lon_v, lat_v, grid


def infer_step(coord: np.ndarray) -> float:
    """Nominal spacing of a regular (but land-masked, so incomplete) coordinate grid."""
    d = np.diff(np.sort(np.unique(np.round(coord, 6))))
    return float(np.min(d[d > 1e-9])) if np.any(d > 1e-9) else 1.0


def _land_mask(lon_g: np.ndarray, lat_g: np.ndarray) -> np.ndarray:
    import shapely.vectorized
    from shapely.ops import unary_union

    _, cfeature = _map_stack()
    land = unary_union(list(cfeature.NaturalEarthFeature("physical", "land", "110m").geometries()))
    return shapely.vectorized.contains(land, lon_g, lat_g)


class LogPowerNorm(mcolors.Normalize):
    """Log-normalize to [0, 1], then apply ``t ** gamma``: gamma < 1 spreads the low end
    (more contrast among cheap sites) and compresses the high end; gamma = 1 is plain log."""

    def __init__(self, vmin: float, vmax: float, gamma: float = 1.0, clip: bool = True):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.gamma = float(gamma)

    def _bounds(self) -> tuple[float, float]:
        vmin = float(self.vmin) if self.vmin is not None else 1e-12
        vmax = float(self.vmax) if self.vmax is not None else vmin * 10
        return math.log(max(vmin, 1e-12)), math.log(max(vmax, vmin * 10))

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        lo, hi = self._bounds()
        result = np.ma.masked_less_equal(result, 0, copy=False)
        t = (np.ma.log(result) - lo) / (hi - lo)
        if self.clip if clip is None else clip:
            t = np.ma.clip(t, 0.0, 1.0)
        t = np.ma.power(np.ma.clip(t, 0.0, None), self.gamma)
        return t[0] if is_scalar else t

    def inverse(self, value):
        lo, hi = self._bounds()
        t = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
        return np.exp(lo + np.power(t, 1.0 / self.gamma) * (hi - lo))


def _log_decade_ticks(cbar, vmin: float, vmax: float, norm) -> None:
    """Tick at each base-e decade, placed at norm(value) so uneven spacing shows the gamma stretch."""
    k0, k1 = math.floor(math.log(vmin)), math.ceil(math.log(vmax))
    ticks = [math.exp(k) for k in range(k0, k1 + 1) if vmin <= math.exp(k) <= vmax]
    if len(ticks) >= 2:
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([f"{t:g}" for t in ticks])


def _map_stack():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        sys.exit('Maps need cartopy/shapely. Install with: pip install -e "solar_lumped[maps]"')
    return ccrs, cfeature


def _ccrs():
    return _map_stack()[0]


def _save(fig, out: Path) -> None:
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
