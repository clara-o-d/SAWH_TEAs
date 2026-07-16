#!/usr/bin/env python3
"""Sample random land locations, run Wilson lumped SAWH simulation per salt, write CSV.

Requires optional deps:  pip install -e ".[maps]"  (Shapely/Cartopy for land sampling)

Each site fetches weather once, simulates each candidate salt (LiCl, NaCl, CaCl2, MgCl2)
with salt-specific DRH / h_des, picks the lowest feasible LCOW, and writes winner + all-salt CSVs.

Plotting: companion ``lcow_plot_maps.py`` reads the winner CSV.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from solar_lumped.economics.params import LCOEconomicParams
from solar_lumped.materials.salt_style import DEFAULT_SALT_MARKER, SALT_MARKERS
from solar_lumped.physics.salt_properties import CANDIDATE_SALTS, get_salt
from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.simulation.site_feasibility import (
    FAIL_LCO,
    passive_gel_temperature_c,
    profile_diagnostics,
    simulate_salt_lcow,
)
from solar_lumped.weather.client import WeatherClient
from solar_lumped.weather.climate import representative_mean_day_df, site_row_from_hourly
from solar_lumped.weather.profiles import profile_from_day_df

_FAIL_LCO = FAIL_LCO


@dataclass(slots=True)
class SiteResult:
    lat: float
    lon: float
    rh_high: float
    rh_low: float
    temp_high_c: float
    temp_low_c: float
    solar_irradiance_w_per_m2: float
    gel_temperature_c: float
    best_salt: str
    best_sl: float
    best_lcow: float
    infeasible: bool
    desorption_aw: float = float("nan")
    daily_yield_m3_per_m2: float = float("nan")
    eta_thermal: float = float("nan")
    backend: str = "solar_lumped"


@dataclass(slots=True)
class SaltAttemptResult:
    lat: float
    lon: float
    salt: str
    feasible: bool
    lcow: float
    yield_kg_m2: float
    eta_thermal: float
    gel_temperature_c: float
    desorption_aw: float
    failure_reason: str = ""


def _try_import_map_stack():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    return plt, LogNorm, ccrs, cfeature


def _prepared_land_union():
    from shapely import geometry as sh_geom
    from shapely.ops import unary_union
    from shapely.prepared import prep

    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(resolution="110m", category="physical", name="land")
    geoms = list(shpreader.Reader(path).geometries())
    u = prep(unary_union(geoms))
    return u, sh_geom


def sample_land_points(
    n: int,
    seed: int,
    *,
    lat_lo: float = -56.0,
    lat_hi: float = 72.0,
    max_tries: int = 200_000,
) -> list[tuple[float, float]]:
    """Rejection sample ``n`` (lat, lon) degrees on land (WGS84)."""
    print(
        "  Loading Natural Earth land polygons (first run may download shapefiles)…",
        flush=True,
    )
    t0 = time.perf_counter()
    land, sh_geom = _prepared_land_union()
    print(
        f"  Land geometry ready in {time.perf_counter() - t0:.2f}s; "
        f"sampling (lat ∈ [{lat_lo}, {lat_hi}])…",
        flush=True,
    )
    rng = np.random.default_rng(seed)
    out: list[tuple[float, float]] = []
    attempts = 0
    for _ in range(max_tries):
        if len(out) >= n:
            break
        attempts += 1
        lat = float(rng.uniform(lat_lo, lat_hi))
        lon = float(rng.uniform(-180.0, 180.0))
        p = sh_geom.Point(lon, lat)
        if land.contains(p):
            out.append((lat, lon))
    if len(out) < n:
        raise RuntimeError(
            f"Only collected {len(out)}/{n} land points after {max_tries} attempts; "
            "try different seed or bounds."
        )
    print(
        f"  Picked {n} land point(s) in {attempts} random draws (seed={seed}, "
        f"acceptance ≈ {100.0 * n / max(attempts, 1):.1f}%).",
        flush=True,
    )
    return out


def parse_salt_names(names: list[str] | None) -> tuple[str, ...]:
    if not names:
        return CANDIDATE_SALTS
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        get_salt(name)
        if name not in seen:
            seen.add(name)
            out.append(name)
    if not out:
        raise ValueError("At least one salt name is required.")
    return tuple(out)


def _build_device_config(
    salt_name: str,
    *,
    salt_loading: float,
    tilt_deg: float,
    fin_area_ratio: float,
    hydrogel_thickness_mm: float,
    vapor_gap_mm: float,
    insulation_gap_mm: float,
) -> DeviceConfig:
    return DeviceConfig(
        salt_name=salt_name,
        salt_to_polymer_ratio=salt_loading,
        hydrogel_thickness_m=hydrogel_thickness_mm * 1e-3,
        vapor_gap_m=vapor_gap_mm * 1e-3,
        insulation_gap_m=insulation_gap_mm * 1e-3,
        tilt_deg=tilt_deg,
        fin_area_ratio=fin_area_ratio,
    )


def _fetch_site_profile(
    lat: float,
    lon: float,
    year: int,
    *,
    cache_dir: str | None,
) -> tuple[object, dict[str, float], dict[str, float]]:
    client = WeatherClient(cache_dir=cache_dir)
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    try:
        _, df = client.get_historical_forecast_site_weather(lat, lon, start, end)
    except Exception:
        df = client.get_historical(lat, lon, start, end)
    row = site_row_from_hourly(df)
    mean_day = representative_mean_day_df(df, reference_day=date(year, 6, 15))
    profile = profile_from_day_df(mean_day)
    diag = profile_diagnostics(profile)
    return profile, row, diag


def run_sites(
    lats: list[float],
    lons: list[float],
    *,
    year: int,
    sleep_s: float,
    cache_dir: str | None,
    econ: LCOEconomicParams | None = None,
    salt_names: tuple[str, ...] | None = None,
    salt_loading: float = 4.0,
    tilt_deg: float = 35.0,
    fin_area_ratio: float = 7.1,
    hydrogel_thickness_mm: float = 4.0,
    vapor_gap_mm: float = 40.0,
    insulation_gap_mm: float = 5.0,
    cyclic_initial: bool = True,
    cyclic_warmup_cycles: int = 1,
    stop_at_first_feasible: bool = False,
) -> tuple[list[SiteResult], list[SaltAttemptResult]]:
    n = len(lats)
    print(
        f"Open-Meteo: historical forecast for {year}, {n} site(s).",
        flush=True,
    )
    if cache_dir is not None:
        print(f"  Weather cache: {cache_dir}", flush=True)
    econ = econ or LCOEconomicParams()
    salts = salt_names or CANDIDATE_SALTS
    results: list[SiteResult] = []
    all_attempts: list[SaltAttemptResult] = []
    t_batch = time.perf_counter()

    for i, (lat, lon) in enumerate(zip(lats, lons, strict=True), start=1):
        t_site = time.perf_counter()
        print(
            f"  [{i}/{n}] ({lat:+.4f}, {lon:+.4f})  fetching weather…",
            end="",
            flush=True,
        )
        try:
            profile, row, diag = _fetch_site_profile(
                lat, lon, year, cache_dir=cache_dir
            )
        except Exception as exc:
            print(f"  → weather failed ({exc})  ({time.perf_counter() - t_site:.1f}s)", flush=True)
            results.append(
                SiteResult(
                    lat=lat,
                    lon=lon,
                    rh_high=float("nan"),
                    rh_low=float("nan"),
                    temp_high_c=float("nan"),
                    temp_low_c=float("nan"),
                    solar_irradiance_w_per_m2=float("nan"),
                    gel_temperature_c=float("nan"),
                    best_salt="none",
                    best_sl=float("nan"),
                    best_lcow=_FAIL_LCO,
                    infeasible=True,
                )
            )
            if sleep_s > 0.0 and i < n:
                time.sleep(sleep_s)
            continue

        rh_high = float(row.get("rh_high_frac", diag["rh_high"]))
        rh_low = float(row.get("rh_low_frac", diag["rh_low"]))
        temp_high = float(row.get("temperature_high_c", diag["temp_high_c"]))
        temp_low = float(row.get("temperature_low_c", diag["temp_low_c"]))
        solar_peak = float(row.get("solar_irradiance_w_per_m2", diag["solar_irradiance_w_per_m2"]))
        ref_config = _build_device_config(
            salts[0],
            salt_loading=salt_loading,
            tilt_deg=tilt_deg,
            fin_area_ratio=fin_area_ratio,
            hydrogel_thickness_mm=hydrogel_thickness_mm,
            vapor_gap_mm=vapor_gap_mm,
            insulation_gap_mm=insulation_gap_mm,
        )
        t_gel_passive = passive_gel_temperature_c(profile, ref_config)
        print(
            f"  RH max={rh_high:.3f}  T max={temp_high:.1f}C  "
            f"I={solar_peak:.0f}W/m²  T_gel(passive)={t_gel_passive:.1f}C",
            flush=True,
        )

        best_name = "none"
        best_lcow = _FAIL_LCO
        best_sl = salt_loading
        best_sim = None
        n_salts = len(salts)

        for j, salt in enumerate(salts, start=1):
            t_salt = time.perf_counter()
            print(f"    salt {j}/{n_salts} {salt}  ", end="", flush=True)
            config = _build_device_config(
                salt,
                salt_loading=salt_loading,
                tilt_deg=tilt_deg,
                fin_area_ratio=fin_area_ratio,
                hydrogel_thickness_mm=hydrogel_thickness_mm,
                vapor_gap_mm=vapor_gap_mm,
                insulation_gap_mm=insulation_gap_mm,
            )
            sim = simulate_salt_lcow(
                profile,
                config,
                econ,
                rh_abs=rh_high,
                cyclic_initial=cyclic_initial,
                cyclic_warmup_cycles=cyclic_warmup_cycles,
            )
            salt_dt = time.perf_counter() - t_salt
            all_attempts.append(
                SaltAttemptResult(
                    lat=lat,
                    lon=lon,
                    salt=salt,
                    feasible=sim.feasible,
                    lcow=sim.lcow,
                    yield_kg_m2=sim.yield_kg_m2,
                    eta_thermal=sim.eta_thermal,
                    gel_temperature_c=sim.gel_temperature_c,
                    desorption_aw=sim.desorption_aw,
                    failure_reason=sim.failure_reason,
                )
            )
            if sim.feasible:
                leader = ""
                if sim.lcow < best_lcow:
                    best_lcow = sim.lcow
                    best_name = salt
                    best_sim = sim
                    leader = "  ★ best so far"
                print(
                    f"LCOW=${sim.lcow:.4f}/m³  yield={sim.yield_kg_m2:.4f} kg/m²  "
                    f"η={sim.eta_thermal:.3f}  T_gel={sim.gel_temperature_c:.1f}C  "
                    f"a_w,des={sim.desorption_aw:.3f}  ({salt_dt:.1f}s){leader}",
                    flush=True,
                )
                if stop_at_first_feasible:
                    break
            else:
                reason = sim.failure_reason or "infeasible"
                print(f"skipped — {reason}  ({salt_dt:.1f}s)", flush=True)

        dt = time.perf_counter() - t_site
        infeasible = best_sim is None or best_lcow >= 0.99 * _FAIL_LCO
        if infeasible:
            print(f"  → site infeasible (no salt passed)  ({dt:.1f}s)", flush=True)
            results.append(
                SiteResult(
                    lat=lat,
                    lon=lon,
                    rh_high=rh_high,
                    rh_low=rh_low,
                    temp_high_c=temp_high,
                    temp_low_c=temp_low,
                    solar_irradiance_w_per_m2=solar_peak,
                    gel_temperature_c=t_gel_passive,
                    best_salt="none",
                    best_sl=salt_loading,
                    best_lcow=_FAIL_LCO,
                    infeasible=True,
                )
            )
        else:
            assert best_sim is not None
            print(
                f"  → winner {best_name}  LCOW=${best_lcow:.4f}/m³  "
                f"yield={best_sim.yield_kg_m2:.4f} kg/m²  ({dt:.1f}s total)",
                flush=True,
            )
            results.append(
                SiteResult(
                    lat=lat,
                    lon=lon,
                    rh_high=rh_high,
                    rh_low=rh_low,
                    temp_high_c=temp_high,
                    temp_low_c=temp_low,
                    solar_irradiance_w_per_m2=solar_peak,
                    gel_temperature_c=best_sim.gel_temperature_c,
                    best_salt=best_name,
                    best_sl=best_sl,
                    best_lcow=best_lcow,
                    infeasible=False,
                    desorption_aw=best_sim.desorption_aw,
                    daily_yield_m3_per_m2=best_sim.yield_kg_m2 / 1000.0,
                    eta_thermal=best_sim.eta_thermal,
                )
            )

        if sleep_s > 0.0 and i < n:
            time.sleep(sleep_s)

    print(
        f"  All sites done in {time.perf_counter() - t_batch:.1f}s "
        f"(avg {(time.perf_counter() - t_batch) / max(n, 1):.1f}s / site, incl. sleep).",
        flush=True,
    )
    return results, all_attempts


def _map_title(year: int, n_sites: int) -> str:
    return (
        "Levelized cost of water — Wilson lumped SAWH (best pure salt)\n"
        f"{n_sites} random land sites  ·  ERA5 {year}  ·  passive solar gel heating"
    )


def plot_map(
    results: list[SiteResult],
    out_path: Path,
    *,
    title: str | None = None,
    year: int = 2023,
    n_sites: int | None = None,
) -> None:
    plt, LogNorm, ccrs, cfeature = _try_import_map_stack()
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    if title is None:
        title = _map_title(year, n_sites if n_sites is not None else len(results))
    lats = np.array([r.lat for r in results])
    lons = np.array([r.lon for r in results])
    lc = np.array([r.best_lcow for r in results])
    salts = [r.best_salt for r in results]
    ok = np.array([not r.infeasible for r in results]) & np.isfinite(lc) & (lc < 0.99 * _FAIL_LCO) & (lc > 0)

    if np.any(ok):
        fvals = np.clip(lc[ok], 1e-9, None)
        vmin = max(float(np.nanmin(fvals) * 0.7), 1e-6)
        vmax = max(float(np.nanmax(fvals) * 1.4), vmin * 10)
    else:
        vmin, vmax = 1e-4, 1.0
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "110m", facecolor="0.88", edgecolor="0.4", linewidth=0.3, zorder=0
        ),
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "ocean", "110m", facecolor="0.92", zorder=0
        ),
    )
    ax.coastlines(resolution="110m", color="0.3", linewidth=0.4, zorder=1)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.35,
        color="0.45",
        alpha=0.45,
        linestyle="--",
        dms=False,
        x_inline=False,
        y_inline=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    sc_last = None
    for salt in sorted(set(salts)):
        idx = np.array([j for j, s in enumerate(salts) if s == salt and ok[j]], dtype=int)
        if idx.size == 0:
            continue
        sc_last = ax.scatter(
            lons[idx],
            lats[idx],
            c=np.clip(lc[idx], 1e-9, None),
            s=64,
            marker=SALT_MARKERS.get(salt, DEFAULT_SALT_MARKER),
            transform=ccrs.PlateCarree(),
            zorder=4,
            cmap="viridis",
            norm=norm,
            edgecolors="0.1",
            linewidths=0.4,
        )
    bad_idx = np.where(~ok)[0]
    if bad_idx.size:
        ax.scatter(
            lons[bad_idx],
            lats[bad_idx],
            c="0.3",
            s=50,
            marker="x",
            transform=ccrs.PlateCarree(),
            zorder=5,
            label="Infeasible or failed",
        )
    mappable = sc_last if sc_last is not None else plt.matplotlib.cm.ScalarMappable(
        norm=norm, cmap="viridis"
    )
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Levelized cost of water (USD per m³ water, log scale)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    leg_items = [
        plt.Line2D([0], [0], marker=m, color="k", linestyle="", label=n, ms=7)
        for n, m in SALT_MARKERS.items()
        if n != "none"
    ]
    ax.legend(
        handles=leg_items,
        loc="lower left",
        title="Best salt",
        framealpha=0.9,
        fontsize=8,
        title_fontsize=9,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _world_ax(fig, pos, *, ccrs, cfeature):
    ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "110m", facecolor="0.88", edgecolor="0.4", linewidth=0.3, zorder=0
        )
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "ocean", "110m", facecolor="0.92", zorder=0
        )
    )
    ax.coastlines(resolution="110m", color="0.35", linewidth=0.4, zorder=1)
    return ax


def _scatter_panel(
    ax,
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    salts: list[str],
    ok: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    label: str,
    ccrs,
    plt,
    bad_idx: np.ndarray | None = None,
    log_scale: bool = False,
    cbar_lines: list[tuple[float, str]] | None = None,
) -> None:
    from matplotlib.colors import LogNorm, Normalize

    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True) if log_scale else Normalize(vmin=vmin, vmax=vmax)
    sc_last = None
    for salt in sorted(set(salts)):
        idx = np.array(
            [
                j
                for j, s in enumerate(salts)
                if s == salt and ok[j] and np.isfinite(values[j])
            ],
            dtype=int,
        )
        if idx.size == 0:
            continue
        sub = values[idx]
        if log_scale:
            sub = np.clip(sub, 1e-9, None)
        sc_last = ax.scatter(
            lons[idx],
            lats[idx],
            c=sub,
            s=50,
            marker=SALT_MARKERS.get(salt, DEFAULT_SALT_MARKER),
            transform=ccrs.PlateCarree(),
            zorder=4,
            cmap=cmap,
            norm=norm,
            edgecolors="0.15",
            linewidths=0.3,
        )
    if bad_idx is not None and bad_idx.size:
        ax.scatter(
            lons[bad_idx],
            lats[bad_idx],
            c="0.4",
            s=35,
            marker="x",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
    mappable = sc_last if sc_last is not None else plt.matplotlib.cm.ScalarMappable(
        norm=norm, cmap=cmap
    )
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.025, pad=0.03, shrink=0.85)
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    if cbar_lines:
        for val, lbl in cbar_lines:
            if not (vmin <= val <= vmax):
                continue
            cbar.ax.axhline(y=val, color="white", linewidth=2.5, zorder=3)
            cbar.ax.axhline(y=val, color="0.1", linewidth=1.0, linestyle="--", zorder=4)
            frac = float(norm(val))
            cbar.ax.text(
                1.08,
                frac,
                lbl,
                transform=cbar.ax.transAxes,
                fontsize=5.5,
                va="center",
                ha="left",
                color="0.15",
            )


def plot_variable_maps(
    results: list[SiteResult],
    out_path: Path,
    *,
    year: int = 2023,
    n_sites: int | None = None,
) -> None:
    plt, LogNorm, ccrs, cfeature = _try_import_map_stack()

    lats = np.array([r.lat for r in results])
    lons = np.array([r.lon for r in results])
    salts = [r.best_salt for r in results]
    ok = np.array([not r.infeasible for r in results]) & np.isfinite(
        np.array([r.best_lcow for r in results])
    )
    bad_idx = np.where(~ok)[0]
    n = n_sites if n_sites is not None else len(results)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        f"Per-site physical variables — Wilson lumped SAWH (best pure salt)\n"
        f"{n} random land sites · ERA5 {year}",
        fontsize=12,
        y=1.01,
    )

    _drh_lines: list[tuple[float, str]] = sorted(
        [
            (get_salt(name).rh_min, f"{name} DRH={get_salt(name).rh_min:.2f}")
            for name in CANDIDATE_SALTS
        ],
        key=lambda t: t[0],
    )

    panels: list[tuple[str, np.ndarray, str, float, float, bool, str, list | None]] = []
    lc = np.array([r.best_lcow for r in results])
    lc_ok = np.where(ok, lc, np.nan)
    finite_lc = lc_ok[np.isfinite(lc_ok) & (lc_ok > 0)]
    lc_vmin = max(float(np.nanmin(finite_lc) * 0.7), 1e-6) if finite_lc.size else 1e-4
    lc_vmax = max(float(np.nanmax(finite_lc) * 1.4), lc_vmin * 10) if finite_lc.size else 1.0
    panels.append(("LCOW (USD/m³, log)", lc_ok, "viridis", lc_vmin, lc_vmax, True, "LCOW (USD m⁻³)", None))

    rh = np.where(ok, np.array([r.rh_high for r in results]), np.nan)
    panels.append(
        (
            "Absorption water activity (= RH high)",
            rh,
            "plasma_r",
            0.0,
            1.0,
            False,
            "a_w absorption",
            _drh_lines,
        )
    )

    a_w_des = np.where(ok, np.array([r.desorption_aw for r in results]), np.nan)
    finite_des = a_w_des[np.isfinite(a_w_des)]
    des_vmin = max(float(np.nanmin(finite_des)) * 0.95, 0.0) if finite_des.size else 0.0
    des_vmax = min(float(np.nanmax(finite_des)) * 1.05, 1.0) if finite_des.size else 1.0
    panels.append(
        (
            "Desorption water activity (simulated T_gel)",
            a_w_des,
            "plasma_r",
            des_vmin,
            des_vmax,
            False,
            "a_w desorption",
            _drh_lines,
        )
    )

    t_gel = np.where(ok, np.array([r.gel_temperature_c for r in results]), np.nan)
    finite_tg = t_gel[np.isfinite(t_gel)]
    tg_vmin = float(np.nanmin(finite_tg)) if finite_tg.size else 20.0
    tg_vmax = float(np.nanmax(finite_tg)) if finite_tg.size else 80.0
    panels.append(("Gel temperature (°C)", t_gel, "inferno", tg_vmin, tg_vmax, False, "T_gel (°C)", None))

    yield_raw = np.array([r.daily_yield_m3_per_m2 for r in results])
    yield_l = np.where(ok & np.isfinite(yield_raw), yield_raw * 1000.0, np.nan)
    finite_yd = yield_l[np.isfinite(yield_l) & (yield_l > 0)]
    yd_vmax = float(np.nanmax(finite_yd)) if finite_yd.size else 1.0
    panels.append(("Daily water yield (L/m²/day)", yield_l, "YlGnBu", 0.0, yd_vmax, False, "Yield (L m⁻² d⁻¹)", None))

    solar = np.where(ok, np.array([r.solar_irradiance_w_per_m2 for r in results]), np.nan)
    finite_sol = solar[np.isfinite(solar)]
    sol_vmin = float(np.nanmin(finite_sol)) if finite_sol.size else 0.0
    sol_vmax = float(np.nanmax(finite_sol)) if finite_sol.size else 1000.0
    panels.append(("Peak solar irradiance (W/m²)", solar, "YlOrRd", sol_vmin, sol_vmax, False, "Solar (W m⁻²)", None))

    for pos, (title, vals, cmap, vmin, vmax, log_sc, cb_lbl, cb_lines) in zip(
        [231, 232, 233, 234, 235, 236], panels, strict=True
    ):
        ax = _world_ax(fig, pos, ccrs=ccrs, cfeature=cfeature)
        ax.set_title(title, fontsize=9, pad=4)
        _scatter_panel(
            ax,
            lons,
            lats,
            vals,
            salts,
            ok,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            label=cb_lbl,
            ccrs=ccrs,
            plt=plt,
            bad_idx=bad_idx,
            log_scale=log_sc,
            cbar_lines=cb_lines,
        )

    leg_items = [
        plt.Line2D([0], [0], marker=m, color="k", linestyle="", label=n, ms=6)
        for n, m in SALT_MARKERS.items()
        if n != "none"
    ]
    fig.legend(
        handles=leg_items,
        loc="lower center",
        ncol=len(leg_items),
        title="Best salt",
        framealpha=0.9,
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20, help="Number of random land points")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep", type=float, default=0.35, help="Seconds between Open-Meteo calls")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=_REPO / "outputs" / "lcow_global" / "lcow_random_sites.csv",
    )
    p.add_argument(
        "--out-all-salts-csv",
        type=Path,
        default=_REPO / "outputs" / "lcow_global" / "lcow_random_sites_all_salts.csv",
    )
    p.add_argument("--cache", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--salts", nargs="+", default=None, help=f"Candidate salts (default: {CANDIDATE_SALTS})")
    p.add_argument("--salt-loading", type=float, default=4.0)
    p.add_argument("--tilt-deg", type=float, default=35.0)
    p.add_argument("--fin-area-ratio", type=float, default=7.1)
    p.add_argument("--hydrogel-thickness-mm", type=float, default=4.0)
    p.add_argument("--vapor-gap-mm", type=float, default=40.0)
    p.add_argument("--insulation-gap-mm", type=float, default=5.0)
    p.add_argument(
        "--no-cyclic",
        action="store_true",
        help="Skip warmup cycles; single-day ODE only (~3× faster, less accurate IC).",
    )
    p.add_argument(
        "--warmup-cycles",
        type=int,
        default=1,
        metavar="N",
        help="Warmup daily cycles before the reporting day when cyclic (default: 1; run_solar_sim uses 2).",
    )
    args = p.parse_args()

    try:
        salt_names = parse_salt_names(args.salts)
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("=== lcow_random_global_map.py (solar_lumped) ===", flush=True)
    print(
        f"  points={args.n}  year={args.year}  seed={args.seed}  "
        f"sleep={args.sleep}s  cache={args.cache}",
        flush=True,
    )
    cyclic_initial = not args.no_cyclic
    if args.warmup_cycles < 0:
        print("--warmup-cycles must be >= 0.", file=sys.stderr)
        return 1
    if args.no_cyclic and args.warmup_cycles != 1:
        print("Note: --warmup-cycles ignored when --no-cyclic is set.", flush=True)

    n_ode_days = 1 if args.no_cyclic else args.warmup_cycles + 1
    est_min_per_site = len(salt_names) * n_ode_days * 0.5
    print(f"  salts={list(salt_names)}  SL={args.salt_loading}", flush=True)
    print(
        f"  cyclic={'off' if args.no_cyclic else f'on ({args.warmup_cycles} warmup + 1 report)'}  "
        f"rough estimate ≈ {est_min_per_site:.0f}–{est_min_per_site * 3:.0f} min/site "
        f"(~{est_min_per_site * args.n / 60:.1f}–{est_min_per_site * 3 * args.n / 60:.1f} hr total)",
        flush=True,
    )
    print(
        "  Tip: run only one instance at a time — duplicate runs compete for CPU and look hung.",
        flush=True,
    )

    t_main = time.perf_counter()
    print(f"--- Step 1: sample {args.n} random land point(s) ---", flush=True)
    try:
        lats, lons = zip(*sample_land_points(args.n, args.seed), strict=True)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    lats, lons = list(lats), list(lons)

    print("--- Step 2: weather + simulate each site ---", flush=True)
    res, attempts = run_sites(
        lats,
        lons,
        year=args.year,
        sleep_s=args.sleep,
        cache_dir=args.cache,
        salt_names=salt_names,
        salt_loading=args.salt_loading,
        tilt_deg=args.tilt_deg,
        fin_area_ratio=args.fin_area_ratio,
        hydrogel_thickness_mm=args.hydrogel_thickness_mm,
        vapor_gap_mm=args.vapor_gap_mm,
        insulation_gap_mm=args.insulation_gap_mm,
        cyclic_initial=cyclic_initial,
        cyclic_warmup_cycles=args.warmup_cycles,
    )

    feas = [r for r in res if not r.infeasible]
    lcs = [r.best_lcow for r in feas if math.isfinite(r.best_lcow) and r.best_lcow < 0.99 * _FAIL_LCO]
    salt_wins = Counter(r.best_salt for r in res)
    print("--- Step 3: summary ---", flush=True)
    print(f"  Feasible LCOW: {len(lcs)}/{len(res)}", flush=True)
    if lcs:
        print(
            f"  LCOW ($/m³) min={min(lcs):.4f}  max={max(lcs):.4f}  "
            f"median={float(np.median(np.asarray(lcs))):.4f}",
            flush=True,
        )
    print(f"  Best salt counts: {dict(salt_wins)}", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in res]).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}", flush=True)
    args.out_all_salts_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(a) for a in attempts]).to_csv(args.out_all_salts_csv, index=False)
    print(f"Wrote {args.out_all_salts_csv}", flush=True)
    print(
        f"  → Run  python scripts/lcow_plot_maps.py --csv {args.out_csv}  to render maps.",
        flush=True,
    )
    print(f"Done in {time.perf_counter() - t_main:.1f}s total.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
