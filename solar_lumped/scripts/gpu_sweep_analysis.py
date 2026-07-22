#!/usr/bin/env python3
"""Analysis + plots for the GPU full-grid device-parameter sweep.

Reads a GPU sweep CSV (``gpu_sweep/run_gpu_sweep.py`` output -- 1,405 land sites
x however many device-parameter combos were swept, see
``docs/gpu_sweep_handoff.md``) and produces:

1. Per-parameter yield maps -- for each *swept* device parameter (auto-detected:
   any of hydrogel_thickness_mm/eps_abs/tau_glass/fin_area_ratio that takes more
   than one value in the CSV), one map per value it takes (other swept device
   params held at their median value), all sharing one colorbar scale so the
   maps are directly comparable.
2. An "optimal configuration" LCOW global map -- at every site, the device combo
   with the lowest LCOW, plus maps of which parameter value was chosen at each
   site (for whichever params were actually swept).
3. Tornado plots (LCOW and thermal efficiency) for the swept device parameters,
   using true one-at-a-time sensitivity across the full factorial design.
4. Tornado plots (LCOW and thermal efficiency) for the 3 exogenous weather
   variables (solar, RH, ambient temperature), which are not a designed sweep --
   sensitivity is a linear-regression elasticity across all 1,405 sites at the
   baseline device combo.

Case 2/3 (docs/gpu_sweep_handoff.md's modified radiative-physics cases) only
sweep a subset of the 4 device parameters (Case 3 fixes eps_abs/tau_glass at
their idealized-limit values, sweeping only hydrogel_thickness_mm and
fin_area_ratio) -- this script auto-detects which params were actually swept
per-CSV rather than assuming all 4, so it works unmodified for any case.

Usage::

    python scripts/gpu_sweep_analysis.py                                  # Case 1 (default)
    python scripts/gpu_sweep_analysis.py --csv outputs/gpu_grid_sweep_case2/full_sweep_case2.csv \\
        --out-dir outputs/gpu_grid_sweep_case2/plots --label "Case 2"
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
_SCRIPTS = _REPO / "scripts"
for _p in (_SRC, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from solar_lumped.economics.lcow import lcow_from_daily_yield  # noqa: E402
from solar_lumped.economics.params import LCOEconomicParams  # noqa: E402
from tornado_plot import create_tornado_plot  # noqa: E402

_SALT_LOADING = 4.0  # fixed salt:polymer ratio used by the GPU sweep (docs/gpu_sweep_handoff.md)
_SALT_NAME = "LiCl"

_ALL_DEVICE_PARAMS: tuple[str, ...] = ("hydrogel_thickness_mm", "eps_abs", "tau_glass", "fin_area_ratio")
_EXOGENOUS_PARAMS: tuple[str, ...] = ("mean_solar_w_m2", "mean_rh_frac", "mean_t_amb_c")

_PARAM_LABELS: dict[str, str] = {
    "hydrogel_thickness_mm": "Hydrogel thickness\n(mm)",
    "eps_abs": "Absorber emissivity\n(eps_abs)",
    "tau_glass": "Glass transmittance\n(tau_glass)",
    "fin_area_ratio": "Condenser fin\narea ratio",
    "mean_solar_w_m2": "Mean solar GHI\n(W/m²)",
    "mean_rh_frac": "Mean RH\n(frac)",
    "mean_t_amb_c": "Mean T_amb\n(°C)",
}

_PARAM_TITLES: dict[str, str] = {
    "hydrogel_thickness_mm": "Hydrogel thickness (mm)",
    "eps_abs": "Absorber emissivity (eps_abs)",
    "tau_glass": "Glass transmittance (tau_glass)",
    "fin_area_ratio": "Condenser fin area ratio",
}

_METRIC_TITLES: dict[str, str] = {
    "lcow_usd_per_m3": "LCOW (USD/m³)",
    "mean_eta_thermal": "Thermal efficiency",
}


# --------------------------------------------------------------------------- data loading


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    econ = LCOEconomicParams()
    df["lcow_usd_per_m3"] = [
        lcow_from_daily_yield(
            yield_kg,
            salt_name=_SALT_NAME,
            salt_to_polymer_ratio=_SALT_LOADING,
            hydrogel_thickness_m=thickness_mm / 1000.0,
            econ=econ,
        )
        for yield_kg, thickness_mm in zip(df["mean_yield_kg_m2"], df["hydrogel_thickness_mm"])
    ]
    return df


def swept_device_params(df: pd.DataFrame) -> tuple[str, ...]:
    """Which of the 4 device params actually vary in this CSV (Case 3 only
    sweeps 2 of them -- see module docstring)."""
    return tuple(p for p in _ALL_DEVICE_PARAMS if df[p].nunique() > 1)


def _baseline_levels(df: pd.DataFrame, device_params: tuple[str, ...]) -> dict[str, float]:
    """Median swept value of each device parameter — the hold-fixed value when sweeping the rest."""
    return {p: float(np.median(sorted(df[p].unique()))) for p in device_params}


# --------------------------------------------------------------------------- map plotting helpers


def _world_ax(fig, pos, *, ccrs, cfeature):
    args = pos if isinstance(pos, tuple) else (pos,)
    ax = fig.add_subplot(*args, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "110m", facecolor="0.88", edgecolor="0.4", linewidth=0.3, zorder=0
        )
    )
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="0.92", zorder=0))
    ax.coastlines(resolution="110m", color="0.35", linewidth=0.4, zorder=1)
    return ax


def _import_map_stack():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    return ccrs, cfeature


def _subplot_grid(n: int) -> list[int]:
    """A reasonably square subplot layout (as 3-digit mpl position codes) for n panels."""
    ncols = 2 if n <= 4 else 3
    nrows = -(-n // ncols)
    return [nrows * 100 + ncols * 10 + (i + 1) for i in range(n)]


# --------------------------------------------------------------------------- 1. per-parameter yield maps


def plot_parameter_yield_maps(df: pd.DataFrame, device_params: tuple[str, ...], yield_dir: Path) -> None:
    ccrs, cfeature = _import_map_stack()
    baseline = _baseline_levels(df, device_params)

    vmin = 0.0
    vmax = float(df["mean_yield_kg_m2"].max()) * 1.02

    yield_dir.mkdir(parents=True, exist_ok=True)
    for param in device_params:
        levels = sorted(df[param].unique())
        others = [p for p in device_params if p != param]
        mask = np.ones(len(df), dtype=bool)
        for p in others:
            mask &= np.isclose(df[p].to_numpy(), baseline[p])
        sub = df.loc[mask]

        n = len(levels)
        fig = plt.figure(figsize=(5.2 * n + 1.2, 5.4))
        fixed_txt = ", ".join(f"{_PARAM_TITLES[p]}={baseline[p]:g}" for p in others) or "n/a (only swept param)"
        fig.suptitle(
            f"Mean daily water yield vs. {_PARAM_TITLES[param]}\n"
            f"(all {sub[['lat', 'lon']].drop_duplicates().shape[0]} land sites; other device params fixed at {fixed_txt})",
            fontsize=11,
            y=1.02,
        )

        sc_last = None
        for i, lvl in enumerate(levels):
            ax = _world_ax(fig, (1, n, i + 1), ccrs=ccrs, cfeature=cfeature)
            row = sub.loc[np.isclose(sub[param].to_numpy(), lvl)].sort_values(["lat", "lon"])
            ax.set_title(f"{param} = {lvl:g}", fontsize=10, pad=5)
            sc_last = ax.scatter(
                row["lon"],
                row["lat"],
                c=row["mean_yield_kg_m2"],
                s=12,
                marker="o",
                transform=ccrs.PlateCarree(),
                zorder=4,
                cmap="YlGnBu",
                vmin=vmin,
                vmax=vmax,
            )

        cbar = fig.colorbar(sc_last, ax=fig.axes, fraction=0.02, pad=0.03, shrink=0.85)
        cbar.set_label("Mean daily yield (kg/m²/day)", fontsize=9)

        out_path = yield_dir / f"yield_map_{param}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- 2. optimal-config LCOW map


def build_optimal_config(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["lat", "lon"])["lcow_usd_per_m3"].idxmin()
    return df.loc[idx].reset_index(drop=True)


def plot_optimal_lcow_map(winners: pd.DataFrame, lcow_dir: Path, label: str) -> None:
    ccrs, cfeature = _import_map_stack()

    lc = winners["lcow_usd_per_m3"].to_numpy()
    lc = np.clip(lc, 1e-9, None)
    vmin = max(float(lc.min() * 0.9), 1e-6)
    vmax = float(lc.max() * 1.1)
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "110m", facecolor="0.88", edgecolor="0.4", linewidth=0.3, zorder=0
        )
    )
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="0.92", zorder=0))
    ax.coastlines(resolution="110m", color="0.3", linewidth=0.4, zorder=1)
    # draw_labels=True hits a gridliner rendering bug on this cartopy/shapely/
    # matplotlib version combo ("Points of LinearRing do not form a closed
    # linestring", raised lazily during savefig) -- plain gridlines without
    # tick labels avoid the buggy code path and are still useful visually.
    ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.35, color="0.45",
        alpha=0.45, linestyle="--",
    )

    sc = ax.scatter(
        winners["lon"], winners["lat"], c=lc, s=16, marker="o",
        transform=ccrs.PlateCarree(), zorder=4, cmap="viridis", norm=norm,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Optimal-configuration LCOW (USD per m³ water, log scale)", fontsize=10)
    ax.set_title(
        f"{label}: best achievable LCOW per site — minimum over all swept device-parameter combos\n"
        f"{len(winners)} GPU-sweep land grid sites (3° spacing)",
        fontsize=12,
        pad=10,
    )

    lcow_dir.mkdir(parents=True, exist_ok=True)
    out_path = lcow_dir / "lcow_optimal_map.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def plot_chosen_parameter_maps(
    winners: pd.DataFrame, df: pd.DataFrame, device_params: tuple[str, ...], lcow_dir: Path, label: str
) -> None:
    ccrs, cfeature = _import_map_stack()

    positions = _subplot_grid(len(device_params))
    ncols = positions[0] % 100 // 10
    nrows = positions[0] // 100
    fig = plt.figure(figsize=(6.5 * ncols, 5.0 * nrows))
    fig.suptitle(
        f"{label}: device-parameter value chosen by the optimal (min-LCOW) configuration at each site\n"
        f"{len(winners)} GPU-sweep land grid sites",
        fontsize=12,
        y=1.02,
    )

    for pos, param in zip(positions, device_params, strict=True):
        levels = sorted(df[param].unique())
        cmap = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(levels))))
        boundaries = [levels[0] - (levels[1] - levels[0]) / 2] if len(levels) > 1 else [levels[0] - 0.5]
        boundaries += [(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)]
        boundaries.append(levels[-1] + ((levels[-1] - levels[-2]) / 2 if len(levels) > 1 else 0.5))
        norm = BoundaryNorm(boundaries, cmap.N)

        ax = _world_ax(fig, pos, ccrs=ccrs, cfeature=cfeature)
        ax.set_title(_PARAM_TITLES[param], fontsize=10, pad=5)
        sc = ax.scatter(
            winners["lon"], winners["lat"], c=winners[param], s=14, marker="o",
            transform=ccrs.PlateCarree(), zorder=4, cmap=cmap, norm=norm,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.03, shrink=0.85, ticks=levels)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(_PARAM_LABELS[param].replace("\n", " "), fontsize=8)

    out_path = lcow_dir / "lcow_optimal_chosen_parameters.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- 3. device-parameter tornado


def _device_param_sensitivity(
    df: pd.DataFrame, param: str, metric: str, device_params: tuple[str, ...]
) -> tuple[float, float, int]:
    """True OAT sensitivity of *metric* to *param*, averaged over the full factorial design.

    Every (site, other-swept-device-params) group contains one row per level of
    *param* (the design is a complete factorial), so every pairwise comparison
    within a group is a valid OAT pair — no need for tornado_plot.py's O(n^2)
    row-matching search.
    """
    group_cols = ["lat", "lon"] + [p for p in device_params if p != param]
    pivot = df.pivot_table(index=group_cols, columns=param, values=metric, aggfunc="first")
    levels = sorted(pivot.columns.tolist())

    inc_vals: list[np.ndarray] = []
    dec_vals: list[np.ndarray] = []
    for x_low, x_high in combinations(levels, 2):
        y_low = pivot[x_low].to_numpy(dtype=float)
        y_high = pivot[x_high].to_numpy(dtype=float)
        valid = np.isfinite(y_low) & np.isfinite(y_high) & (y_low != 0) & (y_high != 0)
        y_low, y_high = y_low[valid], y_high[valid]
        if y_low.size == 0:
            continue

        pct_x_inc = (x_high - x_low) / x_low * 100.0 if abs(x_low) > 1e-10 else (x_high - x_low)
        if abs(pct_x_inc) > 0.01:
            sens_inc = ((y_high - y_low) / y_low * 100.0) / pct_x_inc
            sens_inc = sens_inc[np.abs(sens_inc) < 1000]
            if sens_inc.size:
                inc_vals.append(sens_inc)

        pct_x_dec = (x_low - x_high) / x_high * 100.0 if abs(x_high) > 1e-10 else (x_low - x_high)
        if abs(pct_x_dec) > 0.01:
            sens_dec = ((y_low - y_high) / y_high * 100.0) / pct_x_dec
            sens_dec = sens_dec[np.abs(sens_dec) < 1000]
            if sens_dec.size:
                dec_vals.append(sens_dec)

    all_inc = np.concatenate(inc_vals) if inc_vals else np.array([])
    all_dec = np.concatenate(dec_vals) if dec_vals else np.array([])
    avg_inc = float(all_inc.mean()) if all_inc.size else 0.0
    avg_dec = float(all_dec.mean()) if all_dec.size else 0.0
    n_pairs = int(all_inc.size + all_dec.size)
    return avg_inc, avg_dec, n_pairs


def device_param_tornado(df: pd.DataFrame, metric: str, device_params: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for param in device_params:
        avg_inc, avg_dec, n_pairs = _device_param_sensitivity(df, param, metric, device_params)
        rows.append(
            {
                "variable": param,
                "avg_increase_sensitivity": avg_inc,
                "avg_decrease_sensitivity": avg_dec,
                "max_abs_effect": max(abs(avg_inc), abs(avg_dec)),
                "num_point_sensitivities": n_pairs,
            }
        )
        print(
            f"  {param}: increase={avg_inc:.3f}  decrease={avg_dec:.3f}  (n={n_pairs} OAT comparisons)",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values("max_abs_effect", ascending=False)


# --------------------------------------------------------------------------- 4. exogenous tornado


def _exogenous_sensitivity(baseline_df: pd.DataFrame, var: str, metric: str) -> tuple[float, float, int]:
    """Linear-regression elasticity of *metric* to *var* across sites, controlling for the
    other two exogenous variables. Not a designed sweep (real weather covaries across sites),
    so this reports a point elasticity at the sample means rather than a true OAT sensitivity.
    """
    X = baseline_df[list(_EXOGENOUS_PARAMS)].to_numpy(dtype=float)
    y = baseline_df[metric].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    b = beta[1 + _EXOGENOUS_PARAMS.index(var)]
    x_mean = float(X[:, _EXOGENOUS_PARAMS.index(var)].mean())
    y_mean = float(y.mean())
    elasticity = b * x_mean / y_mean if abs(y_mean) > 1e-10 else b * x_mean
    return elasticity, elasticity, len(baseline_df)


def exogenous_tornado(df: pd.DataFrame, metric: str, device_params: tuple[str, ...]) -> pd.DataFrame:
    baseline = _baseline_levels(df, device_params)
    mask = np.ones(len(df), dtype=bool)
    for p in device_params:
        mask &= np.isclose(df[p].to_numpy(), baseline[p])
    baseline_df = df.loc[mask]

    rows = []
    for var in _EXOGENOUS_PARAMS:
        elasticity, _, n = _exogenous_sensitivity(baseline_df, var, metric)
        rows.append(
            {
                "variable": var,
                "avg_increase_sensitivity": elasticity,
                "avg_decrease_sensitivity": elasticity,
                "max_abs_effect": abs(elasticity),
                "num_point_sensitivities": n,
            }
        )
        print(f"  {var}: elasticity={elasticity:.3f} (n={n} sites)", flush=True)
    return pd.DataFrame(rows).sort_values("max_abs_effect", ascending=False)


def _save_tornado(
    sensitivity_df: pd.DataFrame, metric: str, title: str, out_path: Path
) -> None:
    fig, _ = create_tornado_plot(
        sensitivity_df,
        metric,
        title=title,
        param_name_mapping=_PARAM_LABELS,
        metric_label=_METRIC_TITLES.get(metric, metric),
    )
    if fig is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)
    sensitivity_df.to_csv(out_path.with_suffix(".table.csv"), index=False)


# --------------------------------------------------------------------------- main


def run_analysis(csv_path: Path, out_dir: Path, label: str) -> pd.DataFrame:
    """Run the full analysis + plot suite for one sweep CSV. Returns the loaded
    (with lcow_usd_per_m3 added) DataFrame, for reuse by the cross-case comparison script.
    """
    yield_dir = out_dir / "yield_maps"
    lcow_dir = out_dir / "lcow_optimal"
    tornado_dir = out_dir / "tornado"

    print(f"Loading {csv_path} ...", flush=True)
    df = load_data(csv_path)
    device_params = swept_device_params(df)
    print(f"  {len(df)} rows, {df[['lat', 'lon']].drop_duplicates().shape[0]} sites", flush=True)
    print(f"  swept device params: {device_params}", flush=True)

    print("\n--- 1. Per-parameter yield maps ---", flush=True)
    plot_parameter_yield_maps(df, device_params, yield_dir)

    print("\n--- 2. Optimal-configuration LCOW map ---", flush=True)
    winners = build_optimal_config(df)
    plot_optimal_lcow_map(winners, lcow_dir, label)
    plot_chosen_parameter_maps(winners, df, device_params, lcow_dir, label)

    print("\n--- 3. Device-parameter tornado plots ---", flush=True)
    for metric in ("lcow_usd_per_m3", "mean_eta_thermal"):
        print(f"\n  metric={metric}", flush=True)
        sens_df = device_param_tornado(df, metric, device_params)
        _save_tornado(
            sens_df,
            metric,
            title=f"{label}: swept device-parameter sensitivity — {_METRIC_TITLES[metric]}",
            out_path=tornado_dir / f"tornado_device_params_{metric}.png",
        )

    print("\n--- 4. Exogenous-variable tornado plots ---", flush=True)
    for metric in ("lcow_usd_per_m3", "mean_eta_thermal"):
        print(f"\n  metric={metric}", flush=True)
        sens_df = exogenous_tornado(df, metric, device_params)
        _save_tornado(
            sens_df,
            metric,
            title=f"{label}: exogenous weather sensitivity (regression elasticity) — {_METRIC_TITLES[metric]}",
            out_path=tornado_dir / f"tornado_exogenous_{metric}.png",
        )

    print("\nDone.", flush=True)
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--csv", type=Path, default=_REPO / "outputs" / "gpu_grid_sweep" / "full_sweep.csv",
        help="GPU sweep CSV to analyze (default: Case 1's full_sweep.csv)",
    )
    p.add_argument(
        "--out-dir", type=Path, default=_REPO / "outputs" / "gpu_grid_sweep" / "plots",
        help="Directory to write plots/tables into (default: Case 1's plots/ dir)",
    )
    p.add_argument("--label", type=str, default="Case 1", help="Label used in plot titles (e.g. 'Case 2')")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    run_analysis(args.csv, args.out_dir, args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
