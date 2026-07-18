#!/usr/bin/env python3
"""Run Wilson lumped SAWH LCOW on every land point of a regular lat/lon grid.

Requires optional deps:  pip install -e ".[maps]"  (Shapely/Cartopy for land mask)

Unlike ``lcow_random_global_map.py``, sites are deterministic grid nodes on land
(default 5° spacing). Each site fetches weather once and simulates it with LiCl
(default; pass ``--salts`` for a feasibility-fallback list, tried in order until
one is feasible), writes CSVs, and optionally renders an LCOW color map. Sites
run in parallel worker processes (``--workers``, default cpu_count - 1).

Plot uses uniform markers colored by LCOW (no salt-shape legend).

Examples::

  # Full run (generate CSV + plot):
  python scripts/lcow_full_global_map.py

  # Generate data only:
  python scripts/lcow_full_global_map.py --generate-only

  # Re-plot from existing CSV, cap color scale at 95th percentile, log colors:
  python scripts/lcow_full_global_map.py --plot-only \\
      --csv outputs/lcow_global/lcow_grid_sites.csv \\
      --exclude-top-pct 5 --scale log

  # 1° grid, LiCl only, separate outputs from the 5° run:
  python scripts/lcow_full_global_map.py --generate-only --step 1 --salts LiCl --resume
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_SRC = _REPO / "src"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lcow_random_global_map import (  # noqa: E402
    FAIL_LCO,
    SaltAttemptResult,
    SiteResult,
    parse_salt_names,
    run_sites,
)

_FAIL_LCO = FAIL_LCO
_OUT_DIR = _REPO / "outputs" / "lcow_global"
# Pass --salts to try others as a feasibility fallback (tried in order, first feasible wins).
_DEFAULT_SALTS: tuple[str, ...] = ("LiCl",)


def _grid_tag(step_deg: float) -> str:
    if math.isclose(step_deg, 5.0):
        return "grid"
    step_txt = f"{step_deg:g}".replace(".", "p")
    return f"grid_{step_txt}deg"


def _default_output_paths(step_deg: float) -> tuple[Path, Path, Path]:
    tag = _grid_tag(step_deg)
    return (
        _OUT_DIR / f"lcow_{tag}_sites.csv",
        _OUT_DIR / f"lcow_{tag}_sites_all_salts.csv",
        _OUT_DIR / f"lcow_{tag}_map.png",
    )


def _site_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(float(lat), 6), round(float(lon), 6))


def _write_site_csvs(
    out_csv: Path,
    out_all_salts_csv: Path,
    results: list[SiteResult],
    attempts: list[SaltAttemptResult],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in results]).to_csv(out_csv, index=False)
    out_all_salts_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(a) for a in attempts]).to_csv(out_all_salts_csv, index=False)


def _load_checkpoint(
    out_csv: Path,
    out_all_salts_csv: Path,
) -> tuple[list[SiteResult], list[SaltAttemptResult], set[tuple[float, float]]]:
    if not out_csv.is_file():
        return [], [], set()
    results = load_results(out_csv)
    done = {_site_key(r.lat, r.lon) for r in results}
    attempts: list[SaltAttemptResult] = []
    if out_all_salts_csv.is_file():
        df = pd.read_csv(out_all_salts_csv)
        for _, row in df.iterrows():
            attempts.append(
                SaltAttemptResult(
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    salt=str(row["salt"]),
                    feasible=str(row["feasible"]).strip().lower() == "true",
                    lcow=float(row["lcow"]),
                    yield_kg_m2=float(row["yield_kg_m2"]),
                    eta_thermal=float(row["eta_thermal"]),
                    gel_temperature_c=float(row["gel_temperature_c"]),
                    desorption_aw=float(row.get("desorption_aw", float("nan"))),
                    failure_reason=str(row.get("failure_reason", "")),
                )
            )
    return results, attempts, done


def _run_one_site(lat: float, lon: float, run_kwargs: dict) -> tuple[list[SiteResult], list[SaltAttemptResult]]:
    """Picklable top-level worker: run one site in a subprocess."""
    return run_sites([lat], [lon], **run_kwargs)


def run_sites_checkpointed(
    points: list[tuple[float, float]],
    *,
    done_keys: set[tuple[float, float]],
    existing_results: list[SiteResult],
    existing_attempts: list[SaltAttemptResult],
    out_csv: Path,
    out_all_salts_csv: Path,
    run_kwargs: dict,
    workers: int = 1,
) -> tuple[list[SiteResult], list[SaltAttemptResult]]:
    results = list(existing_results)
    attempts = list(existing_attempts)
    pending = [(lat, lon) for lat, lon in points if _site_key(lat, lon) not in done_keys]
    n_pending = len(pending)
    n_done = len(points) - n_pending
    n_total = len(points)
    if n_done:
        print(f"  Resume: {n_done} site(s) already in {out_csv.name}; {n_pending} remaining.", flush=True)
    if not pending:
        print("  Nothing to run.", flush=True)
        return results, attempts

    if workers <= 1:
        for i, (lat, lon) in enumerate(pending, start=1):
            print(f"  [{n_done + i}/{n_total}] ({lat:+.4f}, {lon:+.4f})", flush=True)
            new_res, new_att = run_sites([lat], [lon], **run_kwargs)
            results.extend(new_res)
            attempts.extend(new_att)
            done_keys.add(_site_key(lat, lon))
            _write_site_csvs(out_csv, out_all_salts_csv, results, attempts)
        return results, attempts

    print(f"  Running {n_pending} pending site(s) across {workers} worker process(es).", flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one_site, lat, lon, run_kwargs): (lat, lon) for lat, lon in pending
        }
        for fut in as_completed(futures):
            lat, lon = futures[fut]
            new_res, new_att = fut.result()
            results.extend(new_res)
            attempts.extend(new_att)
            done_keys.add(_site_key(lat, lon))
            completed += 1
            print(f"  [{n_done + completed}/{n_total}] ({lat:+.4f}, {lon:+.4f}) done", flush=True)
            _write_site_csvs(out_csv, out_all_salts_csv, results, attempts)
    return results, attempts


from solar_lumped.weather.land_grid import (  # noqa: E402
    DEFAULT_EXCLUDE_COUNTRY_ABOVE_LAT,
    grid_land_points,
)

__all__ = [
    "DEFAULT_EXCLUDE_COUNTRY_ABOVE_LAT",
    "grid_land_points",
]


def load_results(csv_path: Path) -> list[SiteResult]:
    df = pd.read_csv(csv_path)

    def _bool(val) -> bool:
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() == "true"

    def _float(val) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    results: list[SiteResult] = []
    for _, row in df.iterrows():
        results.append(
            SiteResult(
                lat=_float(row["lat"]),
                lon=_float(row["lon"]),
                rh_high=_float(row["rh_high"]),
                rh_low=_float(row["rh_low"]),
                temp_high_c=_float(row["temp_high_c"]),
                temp_low_c=_float(row["temp_low_c"]),
                solar_irradiance_w_per_m2=_float(row["solar_irradiance_w_per_m2"]),
                gel_temperature_c=_float(row["gel_temperature_c"]),
                best_salt=str(row["best_salt"]),
                best_sl=_float(row["best_sl"]),
                best_lcow=_float(row["best_lcow"]),
                infeasible=_bool(row["infeasible"]),
                desorption_aw=_float(row.get("desorption_aw", float("nan"))),
                daily_yield_m3_per_m2=_float(row.get("daily_yield_m3_per_m2", float("nan"))),
                eta_thermal=_float(row.get("eta_thermal", float("nan"))),
                backend=str(row.get("backend", "solar_lumped")),
            )
        )
    return results


def _feasible_mask(lc: np.ndarray, infeasible: np.ndarray) -> np.ndarray:
    return (
        ~infeasible
        & np.isfinite(lc)
        & (lc < 0.99 * _FAIL_LCO)
        & (lc > 0)
    )


class LogPowerNorm(mcolors.Normalize):
    """Log-normalize to [0, 1], then apply ``t ** gamma``.

    gamma < 1 spreads out the low-value end of the color scale (more
    contrast among cheap sites) at the cost of compressing the high end;
    gamma > 1 does the opposite. gamma = 1 is a plain log norm.
    """

    def __init__(self, vmin: float, vmax: float, gamma: float = 1.0, clip: bool = True):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.gamma = float(gamma)

    def _log_bounds(self) -> tuple[float, float]:
        vmin = float(self.vmin) if self.vmin is not None else 1e-12
        vmax = float(self.vmax) if self.vmax is not None else vmin * 10
        log_min = math.log(max(vmin, 1e-12))
        log_max = math.log(max(vmax, vmin * 10))
        return log_min, log_max

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        if clip is None:
            clip = self.clip
        log_min, log_max = self._log_bounds()
        result = np.ma.masked_less_equal(result, 0, copy=False)
        t = (np.ma.log(result) - log_min) / (log_max - log_min)
        if clip:
            t = np.ma.clip(t, 0.0, 1.0)
        t = np.ma.power(np.ma.clip(t, 0.0, None), self.gamma)
        return t[0] if is_scalar else t

    def inverse(self, value):
        log_min, log_max = self._log_bounds()
        t = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
        log_v = log_min + np.power(t, 1.0 / self.gamma) * (log_max - log_min)
        return np.exp(log_v)


def _build_regular_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    step_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Snap scattered (lat, lon, value) samples onto a regular step_deg grid.

    Returns (lat_vals, lon_vals, grid) where grid has shape
    (len(lat_vals), len(lon_vals)) and NaN where no sample landed.
    """
    snap = lambda a: np.round(np.round(a / step_deg) * step_deg, 6)  # noqa: E731
    lat_snap = snap(lats)
    lon_snap = snap(lons)
    lat_vals = np.sort(np.unique(lat_snap))
    lon_vals = np.sort(np.unique(lon_snap))
    lat_idx = {v: i for i, v in enumerate(lat_vals)}
    lon_idx = {v: i for i, v in enumerate(lon_vals)}
    grid = np.full((lat_vals.size, lon_vals.size), np.nan)
    for la, lo, val in zip(lat_snap, lon_snap, values):
        grid[lat_idx[la], lon_idx[lo]] = val
    return lat_vals, lon_vals, grid


def plot_lcow_color_map(
    results: list[SiteResult],
    out_path: Path,
    *,
    step_deg: float,
    year: int,
    exclude_top_pct: float = 0.0,
    log_scale: bool = True,
    log_gamma: float = 1.0,
    render: str = "points",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    lats = np.array([r.lat for r in results])
    lons = np.array([r.lon for r in results])
    lc = np.array([r.best_lcow for r in results])
    infeasible = np.array([r.infeasible for r in results])
    ok = _feasible_mask(lc, infeasible)

    plot_ok = ok.copy()
    lc_cap: float | None = None
    if exclude_top_pct > 0.0 and np.any(ok):
        pct = min(max(exclude_top_pct, 0.0), 100.0)
        lc_cap = float(np.percentile(lc[ok], 100.0 - pct))
        plot_ok = ok & (lc <= lc_cap)

    if np.any(plot_ok):
        fvals = lc[plot_ok]
        if log_scale:
            fvals = np.clip(fvals, 1e-9, None)
            vmin = max(float(np.nanmin(fvals) * 0.7), 1e-6)
            vmax = max(float(np.nanmax(fvals) * 1.4), vmin * 10)
        else:
            vmin = max(float(np.nanmin(fvals) * 0.95), 0.0)
            vmax = max(float(np.nanmax(fvals) * 1.05), vmin + 1e-6)
    else:
        vmin, vmax = (1e-4, 1.0) if log_scale else (0.0, 1.0)

    norm: LogPowerNorm | Normalize
    if log_scale:
        norm = LogPowerNorm(vmin=vmin, vmax=vmax, gamma=log_gamma, clip=True)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

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

    marker_size = max(6.0, min(30.0, 150.0 / step_deg))
    sc = None
    if render == "shaded":
        if np.any(plot_ok):
            color_vals_all = np.where(ok, np.clip(lc, 1e-9, None) if log_scale else lc, np.nan)
            lat_vals, lon_vals, grid = _build_regular_grid(lats, lons, color_vals_all, step_deg)
            lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
            sc = ax.pcolormesh(
                lon_grid,
                lat_grid,
                grid,
                shading="gouraud",
                cmap="viridis",
                norm=norm,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
    else:
        if np.any(plot_ok):
            color_vals = lc[plot_ok]
            if log_scale:
                color_vals = np.clip(color_vals, 1e-9, None)
            sc = ax.scatter(
                lons[plot_ok],
                lats[plot_ok],
                c=color_vals,
                s=marker_size,
                marker="o",
                transform=ccrs.PlateCarree(),
                zorder=4,
                cmap="viridis",
                norm=norm,
            )
    bad_idx = np.where(~ok)[0]
    if bad_idx.size:
        ax.scatter(
            lons[bad_idx],
            lats[bad_idx],
            c="0.55",
            s=marker_size * 0.85,
            marker="s",
            transform=ccrs.PlateCarree(),
            zorder=3,
            label="Infeasible",
        )

    mappable = sc if sc is not None else plt.matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.04)
    if log_scale:
        # Tick at each base-e "decade" (e^k) between vmin and vmax, but place the
        # tick mark at norm(value) — under the gamma power-law this spaces ticks
        # unevenly, visually showing where contrast has been stretched or compressed.
        log_min, log_max = norm._log_bounds()
        k_lo, k_hi = math.ceil(log_min), math.floor(log_max)
        if k_hi <= k_lo:
            k_lo, k_hi = math.floor(log_min), math.ceil(log_max)
        tick_vals = np.exp(np.arange(k_lo, k_hi + 1, dtype=float))
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f"{v:,.0f}" if v >= 1 else f"{v:.2g}" for v in tick_vals])
    scale_lbl = f"log scale, base e, gamma={log_gamma:g}" if log_scale else "linear scale"
    cbar.set_label(f"Levelized cost of water (USD per m³ water, {scale_lbl})", fontsize=10)
    n = len(results)
    title_lines = [
        "Levelized cost of water — Wilson lumped SAWH (best pure salt)",
        f"{n} land grid sites ({step_deg:g}°)  ·  ERA5 {year}  ·  passive solar gel heating",
    ]
    if exclude_top_pct > 0.0 and lc_cap is not None:
        title_lines.append(
            f"Color scale capped at top {exclude_top_pct:g}% LCOW (>{lc_cap:.2f} USD/m³); "
            f"all {n} site(s) shown"
        )
    ax.set_title("\n".join(title_lines), fontsize=12, pad=10)
    if bad_idx.size:
        ax.legend(loc="lower left", framealpha=0.9, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _run_plot(
    results: list[SiteResult],
    *,
    out_png: Path,
    step_deg: float,
    year: int,
    exclude_top_pct: float,
    log_scale: bool,
    log_gamma: float = 1.0,
    render: str = "points",
) -> int:
    print("--- Plot: LCOW color map ---", flush=True)
    try:
        plot_lcow_color_map(
            results,
            out_png,
            step_deg=step_deg,
            year=year,
            exclude_top_pct=exclude_top_pct,
            log_scale=log_scale,
            log_gamma=log_gamma,
            render=render,
        )
        print(f"Wrote {out_png}", flush=True)
    except ImportError as exc:
        print(f"Plot failed (missing map deps): {exc}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="Run grid simulation and write CSVs only (no map).",
    )
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="Render map from an existing winner CSV (no simulation).",
    )
    p.add_argument("--step", type=float, default=5.0, help="Grid spacing in degrees (default: 5)")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--sleep", type=float, default=0.35, help="Seconds between Open-Meteo calls")
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Winner CSV for --plot-only (default: --out-csv path)",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Winner CSV (default: outputs/lcow_global/lcow_grid_<step>deg_sites.csv)",
    )
    p.add_argument(
        "--out-all-salts-csv",
        type=Path,
        default=None,
        help="Per-salt attempts CSV (default: paired with --out-csv)",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Map PNG (default: paired with --out-csv)",
    )
    p.add_argument("--cache", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument(
        "--salts",
        nargs="+",
        default=None,
        help=f"Candidate salts, tried in order until one is feasible (default: {_DEFAULT_SALTS})",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        metavar="N",
        help="Parallel worker processes for site simulation (default: cpu_count - 1). "
        "Each site's weather fetch + salt ODE run is independent, so this scales "
        "close to linearly with cores. Use --workers 1 for the old serial behavior; "
        "watch --sleep / Open-Meteo rate limits if you push this high with cache misses.",
    )
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
        help="Warmup daily cycles before the reporting day when cyclic (default: 1).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N grid land points (for testing).",
    )
    p.add_argument(
        "--exclude-top-pct",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Omit the highest PCT%% of feasible LCOW sites from the map and color scale. "
        "Plotting only.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip sites already present in --out-csv; append and checkpoint after each site.",
    )
    p.add_argument(
        "--scale",
        choices=("log", "linear"),
        default="log",
        help="LCOW color scale (default: log).",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        metavar="G",
        help="Power applied to the normalized log color scale (log scale only). "
        "G < 1 spreads out contrast among low-LCOW (cheap) sites at the cost of "
        "the high end; G = 1 is a plain log scale (default: 1.0).",
    )
    p.add_argument(
        "--render",
        choices=("points", "shaded"),
        default="points",
        help="'points': uniform-colored markers per site (default). "
        "'shaded': smoothly interpolated (Gouraud-shaded) surface across grid "
        "points instead of discrete markers.",
    )
    args = p.parse_args()

    if args.step <= 0:
        print("--step must be > 0.", file=sys.stderr)
        return 1
    if args.exclude_top_pct < 0.0 or args.exclude_top_pct >= 100.0:
        print("--exclude-top-pct must be in [0, 100).", file=sys.stderr)
        return 1

    log_scale = args.scale == "log"

    default_csv, default_all_csv, default_png = _default_output_paths(args.step)
    if args.out_csv is None:
        args.out_csv = default_csv
    if args.out_all_salts_csv is None:
        args.out_all_salts_csv = default_all_csv
    if args.out_png is None:
        args.out_png = default_png

    if args.plot_only:
        csv_path = args.csv or args.out_csv
        if not csv_path.is_file():
            print(f"CSV not found: {csv_path}", file=sys.stderr)
            return 1
        print("=== lcow_full_global_map.py (plot only) ===", flush=True)
        print(f"  csv={csv_path}  step={args.step}°  year={args.year}", flush=True)
        if args.exclude_top_pct > 0.0:
            print(f"  exclude_top_pct={args.exclude_top_pct}%  scale={'log' if log_scale else 'linear'}", flush=True)
        results = load_results(csv_path)
        if not results:
            print("No rows in CSV.", file=sys.stderr)
            return 1
        lc_arr = np.array([r.best_lcow for r in results])
        inf_arr = np.array([r.infeasible for r in results])
        feas = int(np.sum(_feasible_mask(lc_arr, inf_arr)))
        print(f"Loaded {len(results)} site(s); {feas} feasible LCOW values.", flush=True)
        return _run_plot(
            results,
            out_png=args.out_png,
            step_deg=args.step,
            year=args.year,
            exclude_top_pct=args.exclude_top_pct,
            log_scale=log_scale,
            log_gamma=args.gamma,
            render=args.render,
        )

    try:
        salt_names = parse_salt_names(args.salts if args.salts is not None else list(_DEFAULT_SALTS))
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.workers <= 0:
        print("--workers must be > 0.", file=sys.stderr)
        return 1

    print("=== lcow_full_global_map.py (solar_lumped) ===", flush=True)
    print(
        f"  step={args.step}°  year={args.year}  sleep={args.sleep}s  cache={args.cache}  "
        f"workers={args.workers}",
        flush=True,
    )
    cyclic_initial = not args.no_cyclic
    if args.warmup_cycles < 0:
        print("--warmup-cycles must be >= 0.", file=sys.stderr)
        return 1

    n_ode_days = 1 if args.no_cyclic else args.warmup_cycles + 1
    # Worst case (all salts infeasible); early-stop on first feasible salt usually finishes sooner.
    est_min_per_site = len(salt_names) * n_ode_days * 0.5

    t_main = time.perf_counter()
    print(f"--- Step 1: build {args.step:g}° land grid ---", flush=True)
    try:
        points = grid_land_points(args.step)
    except Exception as exc:
        print(f"Grid build failed: {exc}", file=sys.stderr)
        return 1

    if args.limit is not None:
        if args.limit <= 0:
            print("--limit must be > 0.", file=sys.stderr)
            return 1
        points = points[: args.limit]
        print(f"  Limited to first {len(points)} point(s) (--limit {args.limit}).", flush=True)

    n_sites = len(points)
    print(f"  salts={list(salt_names)}  SL={args.salt_loading}", flush=True)
    print(
        f"  cyclic={'off' if args.no_cyclic else f'on ({args.warmup_cycles} warmup + 1 report)'}  "
        f"rough estimate ≈ {est_min_per_site:.0f}–{est_min_per_site * 3:.0f} min/site (worst case; "
        f"first-feasible-salt short-circuit usually beats this) "
        f"(~{est_min_per_site * n_sites / args.workers / 60:.1f}"
        f"–{est_min_per_site * 3 * n_sites / args.workers / 60:.1f} hr total "
        f"for {n_sites} sites across {args.workers} worker(s))",
        flush=True,
    )

    run_kwargs = dict(
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
        stop_at_first_feasible=True,
    )

    print("--- Step 2: weather + simulate each site ---", flush=True)
    print(f"  out_csv={args.out_csv}", flush=True)
    print(f"  out_all_salts_csv={args.out_all_salts_csv}", flush=True)
    existing_res, existing_att, done_keys = _load_checkpoint(args.out_csv, args.out_all_salts_csv)
    if existing_res and not args.resume:
        print(
            f"  Note: {args.out_csv.name} already exists; use --resume to continue without redoing sites.",
            flush=True,
        )
        return 1
    res, attempts = run_sites_checkpointed(
        points,
        done_keys=done_keys if args.resume else set(),
        existing_results=existing_res if args.resume else [],
        existing_attempts=existing_att if args.resume else [],
        out_csv=args.out_csv,
        out_all_salts_csv=args.out_all_salts_csv,
        run_kwargs=run_kwargs,
        workers=args.workers,
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

    print(f"Wrote {args.out_csv}", flush=True)
    print(f"Wrote {args.out_all_salts_csv}", flush=True)

    if not args.generate_only:
        rc = _run_plot(
            res,
            out_png=args.out_png,
            step_deg=args.step,
            year=args.year,
            exclude_top_pct=args.exclude_top_pct,
            log_scale=log_scale,
            log_gamma=args.gamma,
            render=args.render,
        )
        if rc != 0:
            return rc

    print(f"Done in {time.perf_counter() - t_main:.1f}s total.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
