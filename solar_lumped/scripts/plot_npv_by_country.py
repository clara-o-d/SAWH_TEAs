#!/usr/bin/env python3
"""Choropleth world maps of solar AWH NPV and payback period by country.

Reads the per-country summary written by ``estimate_npv_by_country.py``
(``outputs/npv_global/npv_by_country.csv``) and renders two maps:
  * NPV per m2 of device footprint (diverging colormap centered on zero).
  * Payback period in years (viridis; countries that never pay back within
    the device lifetime are hatched gray).

Requires optional deps: pip install -e ".[maps]"  (Cartopy)

Examples::

  python scripts/plot_npv_by_country.py
  python scripts/plot_npv_by_country.py --csv outputs/npv_global/npv_by_country.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_ROOT = _REPO.parent
_TARIFF_SCRAPING = _ROOT / "tariff_scraping"
if str(_TARIFF_SCRAPING) not in sys.path:
    sys.path.insert(0, str(_TARIFF_SCRAPING))

from plot_tariff_map import _load_ne_name_index, _resolve_ne_name  # noqa: E402

_DEFAULT_CSV = _REPO / "outputs" / "npv_global" / "npv_by_country.csv"
_OUT_DIR = _REPO / "outputs" / "npv_global"


def _country_stats_by_ne(df: pd.DataFrame, ne_names: set[str], value_column: str) -> dict[str, float]:
    """Matched country -> raw value, including +inf ("never"). NaN/missing are dropped."""
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        ne_name = _resolve_ne_name(str(row["country"]), ne_names)
        if ne_name is None:
            continue
        value = row[value_column]
        if pd.notna(value):
            out[ne_name] = float(value)
    return out


def plot_npv_map(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ne_geoms = _load_ne_name_index()
    ne_names = set(ne_geoms)
    npv_by_ne = _country_stats_by_ne(df, ne_names, "median_npv_usd_per_m2")
    npv_by_ne = {k: v for k, v in npv_by_ne.items() if np.isfinite(v)}
    if not npv_by_ne:
        raise ValueError("No countries could be matched to Natural Earth boundaries.")

    values = np.array(list(npv_by_ne.values()))
    vmin = float(values.min())
    vmax = float(values.max())
    # Center the diverging colormap on zero but let each side span only the
    # data actually present, so an all-negative (or all-positive) result
    # still shows its full contrast instead of collapsing into one hue.
    vmin = min(vmin, -1e-6)
    vmax = max(vmax, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn")

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "50m", facecolor="#eef2f6", edgecolor="none", zorder=0)
    )
    ax.coastlines(resolution="50m", color="0.35", linewidth=0.35, zorder=2)

    for ne_name, geometry in ne_geoms.items():
        if ne_name in npv_by_ne:
            facecolor = cmap(norm(npv_by_ne[ne_name]))
            edgecolor, linewidth = "0.25", 0.25
        else:
            facecolor, edgecolor, linewidth = "0.92", "0.75", 0.15
        ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label("Median NPV (USD per m² device footprint)", fontsize=11)
    ax.set_title(
        "Solar AWH NPV by country\n(revenue priced at each country's mean tap-water tariff)",
        fontsize=13,
        pad=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_payback_map(df: pd.DataFrame, out_path: Path, *, device_lifetime_years: float | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ne_geoms = _load_ne_name_index()
    ne_names = set(ne_geoms)
    payback_by_ne = _country_stats_by_ne(df, ne_names, "median_payback_years_simple")
    if not payback_by_ne:
        raise ValueError("No countries could be matched to Natural Earth boundaries.")

    finite_paybacks = [v for v in payback_by_ne.values() if np.isfinite(v)]
    lifetime = device_lifetime_years or (float(np.percentile(finite_paybacks, 95)) if finite_paybacks else 20.0)
    vmax = max(lifetime, 1e-6)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax, clip=True)
    cmap = plt.get_cmap("viridis_r")

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "50m", facecolor="#eef2f6", edgecolor="none", zorder=0)
    )
    ax.coastlines(resolution="50m", color="0.35", linewidth=0.35, zorder=2)

    for ne_name, geometry in ne_geoms.items():
        if ne_name in payback_by_ne:
            years = payback_by_ne[ne_name]
            if not np.isfinite(years) or (device_lifetime_years is not None and years > device_lifetime_years):
                facecolor, edgecolor, linewidth, hatch = "0.6", "0.2", 0.25, "///"
            else:
                facecolor, edgecolor, linewidth, hatch = cmap(norm(years)), "0.25", 0.25, None
        else:
            facecolor, edgecolor, linewidth, hatch = "0.92", "0.75", 0.15, None
        ax.add_geometries(
            [geometry],
            ccrs.PlateCarree(),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
            zorder=3,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label("Median simple payback period (years)", fontsize=11)
    title_lines = ["Solar AWH payback period by country"]
    if device_lifetime_years is not None:
        title_lines.append(f"Hatched = never pays back within {device_lifetime_years:g}-year device lifetime")
    ax.set_title("\n".join(title_lines), fontsize=13, pad=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=_DEFAULT_CSV, help="npv_by_country.csv path")
    p.add_argument("--out-npv-png", type=Path, default=None)
    p.add_argument("--out-payback-png", type=Path, default=None)
    p.add_argument(
        "--device-lifetime-years",
        type=float,
        default=None,
        help="Cap/hatch payback map at this many years (default: from LCOEconomicParams)",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    if df.empty:
        print("No rows in CSV.", file=sys.stderr)
        return 1

    device_lifetime_years = args.device_lifetime_years
    if device_lifetime_years is None:
        _src = str(_REPO / "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from solar_lumped.economics.params import LCOEconomicParams

        device_lifetime_years = float(LCOEconomicParams().device_lifetime_years)

    out_npv_png = args.out_npv_png or (_OUT_DIR / "npv_map.png")
    out_payback_png = args.out_payback_png or (_OUT_DIR / "payback_map.png")

    print("=== plot_npv_by_country.py ===", flush=True)
    print(f"  {len(df)} countries loaded from {args.csv}", flush=True)

    plot_npv_map(df, out_npv_png)
    print(f"Wrote {out_npv_png}", flush=True)

    plot_payback_map(df, out_payback_png, device_lifetime_years=device_lifetime_years)
    print(f"Wrote {out_payback_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
