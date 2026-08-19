#!/usr/bin/env python3
"""Global maps of the scenario sweep: per-scenario yield, and scenario-vs-scenario ratios.

Reuses plot_map.py's projection, land-masked interpolation and save path rather than
re-deriving them -- this script only adds what a single-column plotter cannot do: one
shared colour scale across scenarios, and derived ratio fields.

Two deliberate choices about colour, because they change what the reader concludes:

* **Every yield map shares one scale.** Per-panel autoscaling makes a 0.35 kg/m2 site and
  a 4.9 kg/m2 site the same shade of yellow in different panels, which is the fastest way
  to make five maps say nothing. Sequential, perceptually uniform, dark = low (viridis:
  monotonic in lightness and colourblind-safe, unlike any rainbow).
* **Ratio maps get a diverging ramp with a neutral midpoint pinned at 1.0 -- but only
  where the data actually straddles 1.** A diverging ramp on all-positive data invents a
  polarity that is not there, so a ratio bounded above 1 gets a sequential ramp instead.
  The choice is made from the data, per map, not by hand.

    python3 analysis/global/plot_scenario_maps.py \\
        --csv solar_lumped/outputs/gpu_scenario_sweep/finite_g_5scenarios.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_map as pm  # noqa: E402

# Physical order, cheapest device to most idealized, so the panel grid reads as a ladder.
SCENARIO_ORDER = (
    "wilson",
    "improved",
    "improved_perfect_cond",
    "optical_limits",
    "optical_limits_perfect_cond",
)
LABELS = {
    "wilson": "Wilson et al. (blackbody IR)",
    "improved": "Scenario 2: selective surface",
    "improved_perfect_cond": "Scenario 2 + perfect condenser",
    "optical_limits": "Optical material limits",
    "optical_limits_perfect_cond": "Optical limits + perfect condenser",
}
# Each pair asks "what does this one change buy?", which is why they are chained rather
# than all differenced against Wilson.
RATIOS = (
    ("improved", "wilson", "Selective surface vs Wilson"),
    ("optical_limits", "improved", "Perfect optics vs selective surface"),
    ("improved_perfect_cond", "improved", "Perfect condenser (scenario 2)"),
    ("optical_limits_perfect_cond", "optical_limits", "Perfect condenser (optical limits)"),
    ("optical_limits", "wilson", "Total optical headroom vs Wilson"),
)
VALUE = "mean_yield_kg_m2"
UNITS = "annual mean yield (kg m$^{-2}$ day$^{-1}$)"
LCOW_COL = "lcow_usd_per_m3"
LCOW_UNITS = "LCOW (USD m$^{-3}$)"
# Affordability bands rather than one more continuous ramp. With a fixed design the cost
# model has no site-dependent term, so LCOW is exactly K / yield -- the LCOW map's PATTERN
# is the yield map's reciprocal and carries nothing new. What a cost map adds is the
# absolute level and where it crosses a threshold, which a banded scale shows and a smooth
# ramp hides.
LCOW_BANDS = (0.0, 5.0, 10.0, 20.0, 50.0)

# The land mask is the same grid for every map here and costs a shapely union over the
# 110m land polygons each time. Ten maps of that is minutes of nothing; memoize it.
_MASK_CACHE: dict = {}
_ORIGINAL_LAND_MASK = pm._land_mask


def _cached_land_mask(lon_g, lat_g):
    key = (lon_g.shape, float(lon_g[0, 0]), float(lat_g[0, 0]), float(lon_g[-1, -1]), float(lat_g[-1, -1]))
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _ORIGINAL_LAND_MASK(lon_g, lat_g)
    return _MASK_CACHE[key]


pm._land_mask = _cached_land_mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--lcow", action="store_true",
                    help="Also compute LCOW from each row's yield and map it (absolute "
                         "levels + affordability bands; the pattern is 1/yield).")
    ap.add_argument("--salt-loading", type=float, default=None,
                    help="g salt / g polymer the sweep ran at. The sweep CSV does not "
                         "record it, so LCOW needs it stated; defaults to the "
                         "parameters.xlsx baseline.")
    args = ap.parse_args()
    out_dir = args.out_dir or args.csv.parent / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    wide = df.pivot(index=["lat", "lon"], columns="scenario", values=VALUE)
    present = [s for s in SCENARIO_ORDER if s in wide.columns]
    missing = [s for s in SCENARIO_ORDER if s not in wide.columns]
    if missing:
        print(f"note: not in this CSV, skipping: {', '.join(missing)}")
    lats = wide.index.get_level_values("lat").to_numpy(float)
    lons = wide.index.get_level_values("lon").to_numpy(float)
    step = pm.infer_step(lons)

    # One scale for all scenarios, from the pooled distribution.
    vmin = float(np.nanmin(wide[present].to_numpy()))
    vmax = float(np.nanmax(wide[present].to_numpy()))
    print(f"{len(wide)} sites, {len(present)} scenarios, shared scale {vmin:.3f}-{vmax:.3f} kg/m2/day")

    for name in present:
        _one_map(
            lons, lats, wide[name].to_numpy(float), step,
            out=out_dir / f"map_yield_{name}.png",
            title=f"{LABELS.get(name, name)}\n{UNITS}",
            cmap="viridis", norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cbar_label=UNITS,
        )
    _panel(lons, lats, wide, present, step, vmin, vmax, out_dir / "panel_yield_scenarios.png")

    for hi, lo, label in RATIOS:
        if hi not in wide.columns or lo not in wide.columns:
            continue
        ratio = (wide[hi] / wide[lo]).to_numpy(float)
        finite = ratio[np.isfinite(ratio)]
        straddles = bool(finite.min() < 1.0 < finite.max())
        if straddles:
            # Neutral pinned at 1.0, but the two halves scaled to the data's own reach.
            # Forcing symmetry here (half-width = the larger excursion) left the entire
            # blue half of the ramp unused when gains ran to +49% and losses only to -5%,
            # squashing all the real variation into the warm half. TwoSlopeNorm keeps
            # white at "no change" while spending both halves; the colourbar ticks carry
            # the resulting asymmetry, so nothing is hidden.
            norm = mcolors.TwoSlopeNorm(
                vcenter=1.0, vmin=float(finite.min()), vmax=float(finite.max())
            )
            cmap = "RdBu_r"
        else:
            norm = mcolors.Normalize(vmin=float(finite.min()), vmax=float(finite.max()))
            cmap = "YlOrRd"
        pct = 100.0 * (np.median(finite) - 1.0)
        _one_map(
            lons, lats, ratio, step,
            out=out_dir / f"map_ratio_{hi}_over_{lo}.png",
            title=f"{label}\nyield ratio, median {pct:+.1f}%"
                  + ("" if straddles else "  (bounded above 1: sequential scale)"),
            cmap=cmap, norm=norm, cbar_label=f"{hi} / {lo}  (yield ratio)",
        )
        print(f"  {hi}/{lo}: median {pct:+.1f}%, range {finite.min():.3f}-{finite.max():.3f}, "
              f"{'diverging' if straddles else 'sequential'}")

    if args.lcow:
        _lcow_maps(df, wide, present, lons, lats, step, out_dir, args.salt_loading)

    print(f"\nWrote maps to {out_dir}")
    return 0


def _lcow_maps(df, wide, present, lons, lats, step, out_dir: Path, salt_loading) -> None:
    """LCOW per scenario: shared log scale, plus banded affordability maps.

    No ratio maps here. LCOW = annual_cost / (utilization * 365 * yield) and nothing in
    the cost model varies by site, so an LCOW ratio is exactly the inverse of the yield
    ratio already mapped -- plotting it again would be the same figure with reciprocal
    ticks.
    """
    from solar_lumped import site_sweep as ss
    from solar_lumped.economics import LCOEconomicParams, lcow_from_daily_yield

    econ = LCOEconomicParams()
    loading = ss.BASELINE_SALT_LOADING if salt_loading is None else float(salt_loading)
    salts = set(df["salt"].unique())
    if len(salts) != 1:
        sys.exit(f"LCOW needs one salt per CSV, found {sorted(salts)}")
    salt = salts.pop()
    thickness_m = float(df["hydrogel_thickness_mm"].iloc[0]) * 1e-3

    lcow = {}
    for name in present:
        lcow[name] = np.array([
            lcow_from_daily_yield(
                float(y), salt_name=salt, salt_loading=loading,
                hydrogel_thickness_m=thickness_m, econ=econ,
            )
            for y in wide[name].to_numpy(float)
        ])
    pooled = np.concatenate([v for v in lcow.values()])
    vmin, vmax = float(pooled.min()), float(pooled.max())
    print(f"\nLCOW: {salt}, {loading:g} g/g, {thickness_m*1e3:g} mm gel, "
          f"BOM-only capex -> ${vmin:.2f}-${vmax:.2f}/m3 (shared log scale)")

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    for name in present:
        # viridis_r keeps the reader's mapping from the yield maps intact: bright = good,
        # which for cost means bright = cheap.
        _one_map(
            lons, lats, lcow[name], step,
            out=out_dir / f"map_lcow_{name}.png",
            title=f"{LABELS.get(name, name)}\n{LCOW_UNITS}, log scale",
            cmap="viridis_r", norm=norm, cbar_label=LCOW_UNITS,
            cbar_ticks=[t for t in (3, 4, 5, 7, 10, 15, 20, 30, 50) if vmin <= t <= vmax],
        )

    band_norm = mcolors.BoundaryNorm(LCOW_BANDS, len(LCOW_BANDS) - 1)
    band_cmap = mcolors.ListedColormap(
        plt.get_cmap("viridis_r")(np.linspace(0.15, 0.9, len(LCOW_BANDS) - 1))
    )
    for name in present:
        _one_map(
            lons, lats, np.clip(lcow[name], LCOW_BANDS[0], LCOW_BANDS[-1] - 1e-9), step,
            out=out_dir / f"map_lcow_bands_{name}.png",
            title=f"{LABELS.get(name, name)}\nLCOW band (USD m$^{{-3}}$)",
            cmap=band_cmap, norm=band_norm, cbar_label=LCOW_UNITS,
        )

    print("  share of land sites under each threshold:")
    header = "    " + "scenario".ljust(36) + "".join(f"<${t:g}".rjust(9) for t in LCOW_BANDS[1:])
    print(header)
    for name in present:
        row = "".join(f"{100.0 * float((lcow[name] < t).mean()):8.1f}%" for t in LCOW_BANDS[1:])
        print("    " + name.ljust(36) + row)


def _one_map(lons, lats, vals, step, *, out: Path, title: str, cmap, norm, cbar_label: str,
             cbar_ticks=None) -> None:
    fig = plt.figure(figsize=(13, 6.5))
    ax = pm.world_ax(fig, 111)
    lon_v, lat_v, grid = pm.interpolate_to_grid(lons, lats, vals, sample_step_deg=step)
    mesh = ax.pcolormesh(lon_v, lat_v, grid, cmap=cmap, norm=norm, shading="auto",
                         transform=pm._ccrs().PlateCarree(), zorder=2)
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.75, pad=0.02)
    if cbar_ticks is not None:
        # A log norm defaults to "4 x 10^1" style labels, which is wrong for money.
        cbar.set_ticks(list(cbar_ticks))
        cbar.set_ticklabels([f"{t:g}" for t in cbar_ticks])
        cbar.minorticks_off()
    cbar.set_label(cbar_label)
    ax.set_title(title, fontsize=11)
    pm._save(fig, out)


def _panel(lons, lats, wide, present, step, vmin, vmax, out: Path) -> None:
    """All scenarios on one figure, one shared colourbar -- the comparison view."""
    n = len(present)
    rows = (n + 1) // 2
    # A PlateCarree world map is 2:1, so the figure has to be sized to the grid's natural
    # aspect or cartopy shrinks each axis to fit and leaves the gaps between them.
    fig = plt.figure(figsize=(14, 3.5 * rows + 1.0))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mesh = None
    for i, name in enumerate(present, start=1):
        ax = pm.world_ax(fig, (rows, 2, i))
        lon_v, lat_v, grid = pm.interpolate_to_grid(
            lons, lats, wide[name].to_numpy(float), sample_step_deg=step
        )
        mesh = ax.pcolormesh(lon_v, lat_v, grid, cmap="viridis", norm=norm, shading="auto",
                             transform=pm._ccrs().PlateCarree(), zorder=2)
        med = float(np.nanmedian(wide[name].to_numpy(float)))
        ax.set_title(f"{LABELS.get(name, name)}  (median {med:.2f})", fontsize=10)
    # One colourbar for the figure: five of them would imply five scales. It gets its own
    # axes rather than ax=fig.axes, which steals a slice from every map and was what
    # shrank the panels and opened the gap between the columns.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.10, wspace=0.06, hspace=0.22)
    cax = fig.add_axes([0.30, 0.045, 0.40, 0.016])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_label(UNITS)
    out.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight" here: it would undo the explicit subplots_adjust geometry
    # the colourbar axes is positioned against.
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
