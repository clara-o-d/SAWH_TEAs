#!/usr/bin/env python3
"""Estimate NPV and payback period of solar AWH devices, country by country.

Combines the global SAWH yield grid (``lcow_grid_1deg_sites.csv``, one land
site per degree from ``lcow_full_global_map.py``) with tap-water tariffs
(``global_tap/tariff_results.csv``) to price the water an AWH device would
displace: each grid site is assigned to the country whose polygon contains it
(Natural Earth admin_0 boundaries), and that country's mean tap-water tariff
(USD/m³, scraped utility bills divided by the tier volume) is used as the
device's assumed revenue per m³ produced. Cash flows (revenue − OPEX each
year, CAPEX at year 0) are then discounted over the device lifetime to get
NPV and payback period per site, and sites are aggregated to a per-country
summary.

Requires optional deps: pip install -e ".[maps]"  (Shapely/Cartopy)

Examples::

  python scripts/estimate_npv_by_country.py
  python scripts/estimate_npv_by_country.py --tariff-tier 50 --grid-csv outputs/lcow_global/lcow_grid_sites.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_ROOT = _REPO.parent
_SRC = _REPO / "src"
_TARIFF_SCRAPING = _ROOT / "tariff_scraping"
for _p in (_SRC, _TARIFF_SCRAPING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plot_tariff_map import VOLUME_COLUMNS, _load_ne_name_index, aggregate_by_country  # noqa: E402
from solar_lumped.economics.npv import npv_from_daily_yield  # noqa: E402
from solar_lumped.economics.params import HYDROGEL_THICKNESS_M, KG_WATER_PER_M3, LCOEconomicParams  # noqa: E402

_DEFAULT_GRID_CSV = _REPO / "outputs" / "lcow_global" / "lcow_grid_1deg_sites_div1p1.csv"
_DEFAULT_TARIFF_CSV = _ROOT / "global_tap" / "tariff_results.csv"
_OUT_DIR = _REPO / "outputs" / "npv_global"


def _country_lookup_items(ne_geoms: dict[str, object]) -> list[tuple[str, object, tuple[float, float, float, float]]]:
    return [(name, geom, geom.bounds) for name, geom in ne_geoms.items()]


def _country_for_point(
    lon: float,
    lat: float,
    items: list[tuple[str, object, tuple[float, float, float, float]]],
    point_cls,
) -> str | None:
    pt = point_cls(lon, lat)
    for name, geom, (minx, miny, maxx, maxy) in items:
        if minx - 1e-6 <= lon <= maxx + 1e-6 and miny - 1e-6 <= lat <= maxy + 1e-6:
            if geom.contains(pt):
                return name
    return None


def assign_countries(lats: np.ndarray, lons: np.ndarray, ne_geoms: dict[str, object]) -> list[str | None]:
    from shapely.geometry import Point

    items = _country_lookup_items(ne_geoms)
    out: list[str | None] = []
    n = len(lats)
    for k in range(n):
        out.append(_country_for_point(float(lons[k]), float(lats[k]), items, Point))
        if (k + 1) % 2000 == 0:
            print(f"  Reverse-geocoded {k + 1}/{n} sites…", flush=True)
    return out


def load_country_tariffs(tariff_csv: Path, ne_names: set[str], *, tier: int) -> dict[str, float]:
    df = pd.read_csv(tariff_csv)
    cost_column = VOLUME_COLUMNS[str(tier)]
    agg = aggregate_by_country(df, ne_names, cost_column=cost_column, volume_m3=tier)
    agg = agg[np.isfinite(agg["mean_cost"]) & (agg["mean_cost"] > 0.0)]
    return dict(zip(agg["ne_name"], agg["mean_cost"]))


def compute_site_npv(
    grid_df: pd.DataFrame,
    country_by_site: list[str | None],
    tariff_by_country: dict[str, float],
    econ: LCOEconomicParams,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (_, row), country in zip(grid_df.iterrows(), country_by_site):
        if country is None or country not in tariff_by_country:
            continue
        infeasible = str(row["infeasible"]).strip().lower() == "true"
        if infeasible:
            continue
        daily_yield_m3_per_m2 = float(row["daily_yield_m3_per_m2"])
        if not np.isfinite(daily_yield_m3_per_m2) or daily_yield_m3_per_m2 <= 0.0:
            continue
        salt_name = str(row["best_salt"])
        salt_to_polymer_ratio = float(row["best_sl"])
        water_price = float(tariff_by_country[country])

        result = npv_from_daily_yield(
            daily_yield_m3_per_m2 * KG_WATER_PER_M3,
            water_price,
            salt_name=salt_name,
            salt_to_polymer_ratio=salt_to_polymer_ratio,
            hydrogel_thickness_m=HYDROGEL_THICKNESS_M,
            econ=econ,
            sorbent="hydrogel",
        )
        if result is None:
            continue

        rows.append(
            {
                "country": country,
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "water_price_usd_per_m3": water_price,
                "daily_yield_m3_per_m2": daily_yield_m3_per_m2,
                "best_salt": salt_name,
                "best_lcow_usd_per_m3": float(row["best_lcow"]),
                "capex_usd_per_m2": result.capex_usd_per_m2,
                "annual_revenue_usd_per_m2": result.annual_revenue_usd_per_m2,
                "annual_opex_usd_per_m2": result.annual_opex_usd_per_m2,
                "annual_net_cash_flow_usd_per_m2": result.annual_net_cash_flow_usd_per_m2,
                "npv_usd_per_m2": result.npv_usd_per_m2,
                "payback_years_simple": result.payback_years_simple,
                "payback_years_discounted": result.payback_years_discounted,
            }
        )
    return pd.DataFrame(rows)


def aggregate_by_country_npv(site_df: pd.DataFrame, econ: LCOEconomicParams) -> pd.DataFrame:
    def _median_with_inf(s: pd.Series) -> float:
        """Median treating +inf ("never pays back") as a real, worst-case value."""
        return float(np.median(s.to_numpy(dtype=float)))

    rows: list[dict] = []
    for country, group in site_df.groupby("country", sort=True):
        n_sites = len(group)
        n_profitable = int((group["npv_usd_per_m2"] > 0.0).sum())
        rows.append(
            {
                "country": country,
                "water_price_usd_per_m3": float(group["water_price_usd_per_m3"].iloc[0]),
                "n_sites": n_sites,
                "n_profitable_sites": n_profitable,
                "share_profitable": n_profitable / n_sites,
                "mean_npv_usd_per_m2": float(group["npv_usd_per_m2"].mean()),
                "median_npv_usd_per_m2": float(group["npv_usd_per_m2"].median()),
                "best_npv_usd_per_m2": float(group["npv_usd_per_m2"].max()),
                "worst_npv_usd_per_m2": float(group["npv_usd_per_m2"].min()),
                "median_payback_years_simple": _median_with_inf(group["payback_years_simple"]),
                "median_payback_years_discounted": _median_with_inf(group["payback_years_discounted"]),
                "best_payback_years_simple": float(group["payback_years_simple"].min()),
            }
        )
    out = pd.DataFrame(rows).sort_values("median_npv_usd_per_m2", ascending=False)
    out.attrs["device_lifetime_years"] = econ.device_lifetime_years
    out.attrs["discount_rate"] = econ.discount_rate
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-csv", type=Path, default=_DEFAULT_GRID_CSV, help="SAWH yield/LCOW grid CSV")
    p.add_argument("--tariff-csv", type=Path, default=_DEFAULT_TARIFF_CSV, help="global_tap tariff_results.csv")
    p.add_argument(
        "--tariff-tier",
        type=int,
        choices=[6, 15, 50],
        default=15,
        help="Monthly consumption tier (m3) used to price water (default: 15)",
    )
    p.add_argument("--out-sites-csv", type=Path, default=None)
    p.add_argument("--out-country-csv", type=Path, default=None)
    args = p.parse_args()

    if not args.grid_csv.is_file():
        print(f"Grid CSV not found: {args.grid_csv}", file=sys.stderr)
        return 1
    if not args.tariff_csv.is_file():
        print(f"Tariff CSV not found: {args.tariff_csv}", file=sys.stderr)
        return 1

    out_sites_csv = args.out_sites_csv or (_OUT_DIR / "npv_sites.csv")
    out_country_csv = args.out_country_csv or (_OUT_DIR / "npv_by_country.csv")

    print("=== estimate_npv_by_country.py ===", flush=True)
    econ = LCOEconomicParams()
    print(
        f"  discount_rate={econ.discount_rate}  device_lifetime_years={econ.device_lifetime_years}  "
        f"tariff_tier={args.tariff_tier}m3/month",
        flush=True,
    )

    print("--- Step 1: load Natural Earth country boundaries ---", flush=True)
    ne_geoms = _load_ne_name_index()
    ne_names = set(ne_geoms)
    print(f"  {len(ne_names)} countries loaded.", flush=True)

    print("--- Step 2: aggregate tap-water tariffs by country ---", flush=True)
    tariff_by_country = load_country_tariffs(args.tariff_csv, ne_names, tier=args.tariff_tier)
    print(f"  {len(tariff_by_country)} countries with a usable tariff.", flush=True)

    print("--- Step 3: load SAWH yield grid and reverse-geocode sites ---", flush=True)
    grid_df = pd.read_csv(args.grid_csv)
    lats = grid_df["lat"].to_numpy(dtype=float)
    lons = grid_df["lon"].to_numpy(dtype=float)
    country_by_site = assign_countries(lats, lons, ne_geoms)
    n_matched = sum(1 for c in country_by_site if c is not None)
    print(f"  {n_matched}/{len(grid_df)} sites matched to a country.", flush=True)

    print("--- Step 4: compute per-site NPV and payback period ---", flush=True)
    site_df = compute_site_npv(grid_df, country_by_site, tariff_by_country, econ)
    print(f"  {len(site_df)} sites with a feasible yield + matched tariff.", flush=True)

    out_sites_csv.parent.mkdir(parents=True, exist_ok=True)
    site_df.to_csv(out_sites_csv, index=False)
    print(f"Wrote {out_sites_csv}", flush=True)

    print("--- Step 5: aggregate to per-country summary ---", flush=True)
    country_df = aggregate_by_country_npv(site_df, econ)
    out_country_csv.parent.mkdir(parents=True, exist_ok=True)
    country_df.to_csv(out_country_csv, index=False)
    print(f"Wrote {out_country_csv}", flush=True)

    print("--- Summary ---", flush=True)
    print(f"  Countries covered: {len(country_df)}", flush=True)
    if len(country_df):
        top = country_df.head(5)[["country", "median_npv_usd_per_m2", "median_payback_years_simple"]]
        print("  Top 5 by median NPV (USD/m2):", flush=True)
        print(top.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
