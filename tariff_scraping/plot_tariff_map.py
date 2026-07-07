#!/usr/bin/env python3
"""Choropleth world map of water tariff cost from tariff_results.csv.

Country fill hue encodes mean volumetric tariff at a chosen consumption tier (viridis).
Hatch density encodes the number of utility-level data points per country
(denser hatching = more data).

Tariff values in the CSV are monthly bills; they are divided by the tier volume
to obtain USD per m³ before aggregation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "cartopy is required for this script. Install with: pip install cartopy"
    ) from exc

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_CSV = _REPO / "solar_lumped" / "tariff_results.csv"
VOLUME_COLUMNS = {"6": "6M3", "15": "15M3", "50": "50M3"}

# CSV country names -> Natural Earth ``NAME`` field (50m admin_0_countries).
COUNTRY_TO_NE: dict[str, str | None] = {
    "American Samoa": "American Samoa",
    "Antigua and Barbuda": "Antigua and Barb.",
    "Bahamas": "Bahamas",
    "Bosnia and Herzegovina": "Bosnia and Herz.",
    "British Virgin Islands": "British Virgin Is.",
    "Cape Verde": "Cabo Verde",
    "Cayman Islands": "Cayman Is.",
    "Central African Republic": "Central African Rep.",
    "Congo": "Congo",
    "Congo, Dem. Rep.": "Dem. Rep. Congo",
    "Cook Islands": "Cook Is.",
    "Czech Republic": "Czechia",
    "Dominican Republic": "Dominican Rep.",
    "Federated States Of Micronesia": "Micronesia",
    "French Polynesia": "Fr. Polynesia",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Lao PDR": "Laos",
    "Macau, China": "Macao",
    "Marshall Islands": "Marshall Is.",
    "Netherlands Antilles": None,
    "Republic Of Kiribati": "Kiribati",
    "Republic Of Nauru": "Nauru",
    "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "Saint Lucia": "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vin. and Gren.",
    "St. Vincent and the Grenadines": "St. Vin. and Gren.",
    "The Gambia": "Gambia",
    "Slovak Republic": "Slovakia",
    "Solomon Islands": "Solomon Is.",
    "South Korea": "South Korea",
    "Swaziland": "eSwatini",
    "Syrian Arab Republic": "Syria",
    "Trinidad and Tobago": "Trinidad and Tobago",
    "Turks and Caicos Islands": "Turks and Caicos Is.",
    "U.S. Virgin Islands": "U.S. Virgin Is.",
    "UK, England and Wales": "United Kingdom",
    "UK, Scotland": "United Kingdom",
    "United States": "United States of America",
    "United States of America": "United States of America",
    "Wallis and Futuna": "Wallis and Futuna Is.",
    "West Bank and Gaza": "Palestine",
}


def _load_ne_name_index() -> dict[str, object]:
    shp = shpreader.natural_earth("50m", "cultural", "admin_0_countries")
    reader = shpreader.Reader(shp)
    index: dict[str, object] = {}
    for record in reader.records():
        name = record.attributes["NAME"]
        index[name] = record.geometry
    return index


def _resolve_ne_name(country: str, ne_names: set[str]) -> str | None:
    if country in COUNTRY_TO_NE:
        mapped = COUNTRY_TO_NE[country]
        if mapped is None:
            return None
        return mapped if mapped in ne_names else None
    if country in ne_names:
        return country
    for name in ne_names:
        if country.lower() == name.lower():
            return name
    return None


def aggregate_by_country(
    df: pd.DataFrame,
    ne_names: set[str],
    *,
    cost_column: str,
    volume_m3: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for country, group in df.groupby("Country", sort=False):
        ne_name = _resolve_ne_name(country, ne_names)
        if ne_name is None:
            continue
        bill = pd.to_numeric(group[cost_column], errors="coerce")
        rate = bill / volume_m3
        rows.append(
            {
                "ne_name": ne_name,
                "mean_cost": float(rate.mean()),
                "n_points": int(rate.notna().sum()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["ne_name", "mean_cost", "n_points"])

    agg = pd.DataFrame(rows).groupby("ne_name", as_index=False).agg(
        mean_cost=("mean_cost", "mean"),
        n_points=("n_points", "sum"),
    )
    return agg


HATCH_LEVELS = 7


def _hatch_for_count(count: int, count_min: int, count_max: int) -> str:
    if count_max <= count_min or count <= count_min:
        return ""
    norm = (count - count_min) / (count_max - count_min)
    level = min(int(np.ceil(norm * (HATCH_LEVELS - 1))), HATCH_LEVELS - 1)
    if level <= 0:
        return ""
    return "/" * level


def _hatch_legend_handles(count_min: int, count_max: int) -> list[mpatches.Patch]:
    if count_max <= count_min:
        ticks = [count_min]
    else:
        ticks = [int(v) for v in np.linspace(count_min, count_max, HATCH_LEVELS)]
        ticks = sorted(set(ticks))
    handles: list[mpatches.Patch] = []
    for tick in ticks:
        hatch = _hatch_for_count(tick, count_min, count_max)
        handles.append(
            mpatches.Patch(
                facecolor="0.85",
                edgecolor="0.2",
                hatch=hatch,
                linewidth=0.4,
                label=str(tick),
            )
        )
    return handles


def plot_tariff_map(
    df: pd.DataFrame,
    out_path: Path,
    *,
    volume_m3: int,
    cost_column: str,
) -> None:
    ne_geoms = _load_ne_name_index()
    ne_names = set(ne_geoms)
    country_stats = aggregate_by_country(
        df, ne_names, cost_column=cost_column, volume_m3=volume_m3
    )
    if country_stats.empty:
        raise ValueError("No countries could be matched to Natural Earth boundaries.")

    valid = country_stats["mean_cost"].notna() & (country_stats["mean_cost"] > 0)
    costs = country_stats.loc[valid, "mean_cost"].to_numpy()
    counts = country_stats["n_points"].to_numpy(dtype=int)
    vmin = max(float(np.nanmin(costs)) * 0.8, 0.01)
    vmax = float(np.nanmax(costs))
    count_min = int(counts.min())
    count_max = int(counts.max())
    highlight_threshold = float(np.percentile(costs, 90))

    cmap = plt.get_cmap("viridis")
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    stats_by_name = country_stats.set_index("ne_name")

    plt.rcParams["hatch.color"] = "0.15"
    plt.rcParams["hatch.linewidth"] = 0.45

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "ocean", "50m", facecolor="#eef2f6", edgecolor="none", zorder=0
        )
    )
    ax.add_feature(cfeature.LAKES, alpha=0.35, zorder=1)
    ax.coastlines(resolution="50m", color="0.35", linewidth=0.35, zorder=2)

    highlight_points: list[tuple[float, float, str, float]] = []

    for ne_name, geometry in ne_geoms.items():
        if ne_name in stats_by_name.index:
            row = stats_by_name.loc[ne_name]
            cost = float(row["mean_cost"])
            n_points = int(row["n_points"])
            if not np.isfinite(cost) or cost <= 0:
                facecolor = "0.92"
                edgecolor = "0.75"
                linewidth = 0.15
                hatch = None
            else:
                facecolor = cmap(norm(cost))
                edgecolor = "0.25"
                linewidth = 0.25
                hatch = _hatch_for_count(n_points, count_min, count_max)
                if cost >= highlight_threshold:
                    centroid = geometry.centroid
                    highlight_points.append((centroid.x, centroid.y, ne_name, cost))
        else:
            facecolor = "0.92"
            edgecolor = "0.75"
            linewidth = 0.15
            hatch = None

        ax.add_geometries(
            [geometry],
            ccrs.PlateCarree(),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
            zorder=3,
        )

    for lon, lat, label, cost in highlight_points:
        ax.plot(
            lon,
            lat,
            marker="*",
            markersize=10,
            color="#f4d03f",
            markeredgecolor="0.15",
            markeredgewidth=0.4,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        ax.text(
            lon,
            lat + 2.5,
            f"{label}\n${cost:.2f}/m³",
            transform=ccrs.PlateCarree(),
            fontsize=6.5,
            ha="center",
            va="bottom",
            color="0.1",
            zorder=7,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label(
        f"Mean volumetric tariff at {volume_m3} m³/month (USD per m³, log scale)",
        fontsize=11,
    )

    hatch_handles = _hatch_legend_handles(count_min, count_max)
    ax.legend(
        handles=hatch_handles,
        title="Utility data points\n(sparse → dense hatch)",
        loc="lower left",
        framealpha=0.92,
        fontsize=8,
        title_fontsize=9,
        ncol=min(len(hatch_handles), 4),
    )

    ax.set_title(
        f"Global water tariffs ({volume_m3} m³/month consumption, USD per m³)",
        fontsize=13,
        pad=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _default_out(volume_m3: int) -> Path:
    return _REPO / "solar_lumped" / f"tariff_map_{volume_m3}m3.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="tariff_results.csv path")
    parser.add_argument(
        "--volume",
        type=int,
        choices=[6, 15, 50],
        default=15,
        help="Monthly consumption tier in m³ (default: 15)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate maps for 6, 15, and 50 m³",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    required = {"Country", *VOLUME_COLUMNS.values()}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV must contain columns: {sorted(required)}")

    volumes = [6, 15, 50] if args.all else [args.volume]
    for volume in volumes:
        out_path = args.out if args.out and len(volumes) == 1 else _default_out(volume)
        cost_column = VOLUME_COLUMNS[str(volume)]
        plot_tariff_map(
            df,
            out_path,
            volume_m3=volume,
            cost_column=cost_column,
        )
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
