#!/usr/bin/env python3
"""Analysis + plots for the GPU BayesOpt full-grid sweep (gpu_sweep/
run_bayesopt_sweep.py output: one row per land site, giving that site's own
optimized 3-variable design -- not a discrete grid like gpu_grid_sweep's).

Reuses gpu_grid_sweep/gpu_sweep_analysis_solar.py's map-plotting primitives
(world axis, land-masked interpolation) rather than reimplementing them.

Produces:
1. A global map of each site's optimized LCOW.
2. Maps of the chosen (continuous) hydrogel_thickness_mm/vapor_gap_mm/
   fin_area_ratio at each site's optimum.
3. A console summary of optimization/diagnostic quality (stopped_reason,
   surrogate-artifact flags, GP calibration, DE convergence).
4. A per-site comparison against the brute-force grid sweep's own
   optimal-config LCOW (analysis/gpu_grid_sweep/full_sweep.csv) at the same
   sites -- BayesOpt searches a continuous box containing the grid's 5x5x5
   discrete combos, so its optimum should very rarely be worse.

Usage::

    python analysis/sawh_bayesopt/bayesopt_sweep_analysis_solar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_GRID_SWEEP_DIR = _HERE.parent / "gpu_grid_sweep"
if str(_GRID_SWEEP_DIR) not in sys.path:
    sys.path.insert(0, str(_GRID_SWEEP_DIR))

from gpu_sweep_analysis_solar import (  # noqa: E402
    _import_map_stack,
    _infer_grid_step,
    _interpolate_to_grid,
    _world_ax,
    build_optimal_config,
    load_data as load_grid_data,
)

sys.path.insert(0, str(_HERE.parent.parent / "solar_lumped" / "src"))  # noqa: E402
from solar_lumped.physics import EPS_ABS_IR_CASE2, EPS_GLASS_IR_CASE2  # noqa: E402

_PARAM_TITLES = {
    "hydrogel_thickness_mm": "Hydrogel thickness (mm)",
    "vapor_gap_mm": "Vapor gap (mm)",
    "fin_area_ratio": "Condenser fin area ratio",
}
_TOTAL_LAND_SITES = 1405  # analysis/gpu_grid_sweep/full_sweep_case2.csv's own site count, --step 3.0


def load_data(csv_path: Path, corrected_csv_path: Path | None = None) -> pd.DataFrame:
    """Load the sweep summary, with LCOW/design overridden by
    recompute_lcow_from_cache.py's corrected values when available (see that
    script's module docstring -- economics.py's LCOW formula had a real bug,
    fixed after this sweep ran; the corrected file re-picks each site's best
    already-cached design under the fixed formula, no GPU re-run needed).
    Diagnostic columns (stopped_reason, cv_rmse, ...) are about the search
    process itself, unaffected by the LCOW-formula fix, so those still come
    from the original summary.
    """
    df = pd.read_csv(csv_path)
    df = df[df["error"].isna()] if "error" in df.columns else df
    if corrected_csv_path is None or not corrected_csv_path.is_file():
        return df
    corrected = pd.read_csv(corrected_csv_path)[
        ["lat", "lon", "corrected_best_combined_lcow_usd_m3", "corrected_hydrogel_thickness_mm",
         "corrected_vapor_gap_mm", "corrected_fin_area_ratio"]
    ].rename(columns={
        "corrected_best_combined_lcow_usd_m3": "best_combined_lcow_usd_m3",
        "corrected_hydrogel_thickness_mm": "hydrogel_thickness_mm",
        "corrected_vapor_gap_mm": "vapor_gap_mm",
        "corrected_fin_area_ratio": "fin_area_ratio",
    })
    df = df.drop(columns=["best_combined_lcow_usd_m3", "hydrogel_thickness_mm", "vapor_gap_mm", "fin_area_ratio"])
    return df.merge(corrected, on=["lat", "lon"], how="inner")


def plot_optimal_lcow_map(df: pd.DataFrame, out_dir: Path, label: str) -> None:
    ccrs, cfeature = _import_map_stack()

    lc = np.clip(df["best_combined_lcow_usd_m3"].to_numpy(), 1e-9, None)
    from matplotlib.colors import LogNorm

    norm = LogNorm(vmin=max(float(lc.min() * 0.9), 1e-6), vmax=float(lc.max() * 1.1), clip=True)

    fig = plt.figure(figsize=(14, 7))
    ax = _world_ax(fig, (1, 1, 1), ccrs=ccrs, cfeature=cfeature)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.35, color="0.45", alpha=0.45, linestyle="--")

    lons, lats = df["lon"].to_numpy(), df["lat"].to_numpy()
    sample_step_deg = max(_infer_grid_step(lons), _infer_grid_step(lats))
    lon_vals, lat_vals, grid = _interpolate_to_grid(lons, lats, lc, sample_step_deg=sample_step_deg)
    sc = ax.pcolormesh(
        lon_vals, lat_vals, grid, shading="gouraud", transform=ccrs.PlateCarree(), zorder=4, cmap="viridis", norm=norm,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("BayesOpt-optimized LCOW (USD per m³ water, log scale)", fontsize=10)
    ax.set_title(
        f"{label}: per-site BayesOpt-optimized LCOW (continuous 3-variable search)\n"
        f"{len(df)}/{_TOTAL_LAND_SITES} land sites completed so far (interpolated)",
        fontsize=12, pad=10,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bayesopt_lcow_map.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def plot_chosen_parameter_maps(df: pd.DataFrame, out_dir: Path, label: str) -> None:
    ccrs, cfeature = _import_map_stack()

    fig = plt.figure(figsize=(6.5 * 3, 5.0))
    fig.suptitle(
        f"{label}: BayesOpt's chosen design value at each site's optimum\n"
        f"{len(df)}/{_TOTAL_LAND_SITES} land sites completed so far",
        fontsize=12, y=1.05,
    )
    for i, param in enumerate(_PARAM_TITLES):
        ax = _world_ax(fig, (1, 3, i + 1), ccrs=ccrs, cfeature=cfeature)
        ax.set_title(_PARAM_TITLES[param], fontsize=10, pad=5)
        vals = df[param].to_numpy()
        sc = ax.scatter(
            df["lon"], df["lat"], c=vals, s=12, marker="o",
            transform=ccrs.PlateCarree(), zorder=4, cmap="viridis", vmin=float(vals.min()), vmax=float(vals.max()),
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.03, shrink=0.85)
        cbar.ax.tick_params(labelsize=7)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bayesopt_chosen_parameters.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def print_diagnostics_summary(df: pd.DataFrame) -> None:
    print(f"\n--- Optimization/diagnostic quality ({len(df)} sites) ---", flush=True)
    print("stopped_reason:\n" + df["stopped_reason"].value_counts().to_string(), flush=True)
    n_artifact = int(df["flagged_as_surrogate_artifact"].astype(str).isin(["True", "true", "1"]).sum())
    print(f"flagged_as_surrogate_artifact: {n_artifact}/{len(df)}", flush=True)
    for col in ("n_evals", "cv_rmse", "standardized_residual_std", "n_hyperparameter_warnings", "frac_de_hit_maxiter"):
        s = pd.to_numeric(df[col], errors="coerce")
        print(f"{col}: median={s.median():.4g}  mean={s.mean():.4g}  max={s.max():.4g}", flush=True)


def plot_diagnostics_histograms(df: pd.DataFrame, out_dir: Path, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(pd.to_numeric(df["cv_rmse"], errors="coerce").dropna(), bins=40, color="C0")
    axes[0].set_xlabel("k-fold CV RMSE (USD/m³)")
    axes[0].set_ylabel("sites")
    axes[0].set_title("GP calibration across sites")

    axes[1].hist(pd.to_numeric(df["frac_de_hit_maxiter"], errors="coerce").dropna(), bins=20, color="C1")
    axes[1].set_xlabel("fraction of EI proposals hitting DE maxiter")
    axes[1].set_title("EI-proposal DE convergence across sites")

    fig.suptitle(f"{label}: BayesOpt diagnostic distributions ({len(df)} sites)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bayesopt_diagnostics_hist.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def _infer_grid_case(grid_df: pd.DataFrame) -> str:
    """CASE_EPS_IR-style label from a grid-sweep CSV's eps_abs_ir/eps_glass_ir
    columns (empty means run_gpu_sweep.py's --eps-abs-ir/--eps-glass-ir
    defaulted to None, i.e. case1's blackbody/cavity approximation)."""
    ir = pd.to_numeric(grid_df["eps_abs_ir"], errors="coerce")
    ig = pd.to_numeric(grid_df["eps_glass_ir"], errors="coerce")
    if ir.isna().all() and ig.isna().all():
        return "case1"
    if np.allclose(ir.dropna().unique(), EPS_ABS_IR_CASE2) and np.allclose(ig.dropna().unique(), EPS_GLASS_IR_CASE2):
        return "case2"
    if np.allclose(ir.dropna().unique(), 0.0) and np.allclose(ig.dropna().unique(), 0.0):
        return "case3"
    return "unknown"


def plot_improvement_map(merged: pd.DataFrame, out_dir: Path, label: str) -> None:
    ccrs, cfeature = _import_map_stack()

    pct = (merged["improvement_frac"] * 100).to_numpy()
    from matplotlib.colors import TwoSlopeNorm

    # Centered at 0% (diverging) rather than min/max -- "no improvement" is a
    # meaningful reference point here, not just an arbitrary low end.
    vmax = max(float(np.abs(pct).max()), 0.1)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig = plt.figure(figsize=(14, 7))
    ax = _world_ax(fig, (1, 1, 1), ccrs=ccrs, cfeature=cfeature)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.35, color="0.45", alpha=0.45, linestyle="--")

    lons, lats = merged["lon"].to_numpy(), merged["lat"].to_numpy()
    sample_step_deg = max(_infer_grid_step(lons), _infer_grid_step(lats))
    lon_vals, lat_vals, grid = _interpolate_to_grid(lons, lats, pct, sample_step_deg=sample_step_deg)
    sc = ax.pcolormesh(
        lon_vals, lat_vals, grid, shading="gouraud", transform=ccrs.PlateCarree(), zorder=4,
        cmap="RdYlGn", norm=norm,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("LCOW improvement of BayesOpt over grid-sweep best combo (%)", fontsize=10)
    ax.set_title(
        f"{label}: BayesOpt vs. brute-force grid-sweep LCOW, {len(merged)} shared sites\n"
        f"green = BayesOpt found a cheaper design; red = grid-sweep combo was cheaper",
        fontsize=12, pad=10,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bayesopt_vs_grid_improvement_map.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def compare_vs_grid_sweep(bo_df: pd.DataFrame, grid_csv: Path, out_dir: Path, label: str) -> None:
    """BayesOpt searches a continuous box that contains the brute-force
    grid's 5x5x5 discrete combos -- so at any site both ran under the *same*
    physics case, BayesOpt's optimum should very rarely be worse than the
    grid's own best combo. Refuses the comparison (rather than silently
    reporting a number) if the two runs used different eps_abs_ir/
    eps_glass_ir cases, or if the grid CSV didn't actually sweep all 3 of
    BayesOpt's optimized variables -- either makes "improvement" meaningless.
    """
    grid_df = load_grid_data(grid_csv)

    bo_cases = bo_df["case"].unique()
    grid_case = _infer_grid_case(grid_df)
    if len(bo_cases) != 1 or bo_cases[0] != grid_case:
        print(
            f"\n--- BayesOpt vs. brute-force grid sweep: SKIPPED ---\n"
            f"BayesOpt case(s)={list(bo_cases)} vs. grid case={grid_case!r} -- different radiative-physics "
            f"cases aren't a valid apples-to-apples LCOW comparison (see solar_lumped/gpu_sweep/"
            f"gpu_sweep_handoff.md's Case table).", flush=True,
        )
        return
    not_swept = [p for p in _PARAM_TITLES if grid_df[p].nunique() <= 1]
    if not_swept:
        print(
            f"\n--- BayesOpt vs. brute-force grid sweep: SKIPPED ---\n"
            f"{grid_csv} didn't sweep {not_swept} (held fixed) -- its 'best combo' isn't optimizing the same "
            f"variables BayesOpt did, so a comparison would be meaningless.", flush=True,
        )
        return

    grid_winners = build_optimal_config(grid_df)[["lat", "lon", "lcow_usd_per_m3"]]

    merged = bo_df.merge(grid_winners, on=["lat", "lon"], how="inner", suffixes=("_bo", "_grid"))
    merged["improvement_frac"] = (
        (merged["lcow_usd_per_m3"] - merged["best_combined_lcow_usd_m3"]) / merged["lcow_usd_per_m3"]
    )

    print(f"\n--- BayesOpt vs. brute-force grid sweep ({len(merged)} sites in both) ---", flush=True)
    print(f"median improvement: {merged['improvement_frac'].median():.2%}", flush=True)
    worse = merged[merged["improvement_frac"] < -0.01]
    print(f"sites where BayesOpt did >1% WORSE than the grid's best combo: {len(worse)}/{len(merged)}", flush=True)
    if len(worse):
        print(worse[["lat", "lon", "best_combined_lcow_usd_m3", "lcow_usd_per_m3", "improvement_frac"]].to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(merged["improvement_frac"] * 100, bins=60, color="C2")
    ax.axvline(0, color="k", linewidth=1)
    ax.set_xlabel("LCOW improvement of BayesOpt over grid-sweep best combo (%)")
    ax.set_ylabel("sites")
    ax.set_title(f"{label}: BayesOpt vs. brute-force grid, {len(merged)} shared sites")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bayesopt_vs_grid_improvement.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)

    plot_improvement_map(merged, out_dir, label)


def run_analysis(csv_path: Path, grid_csv: Path, out_dir: Path, label: str, corrected_csv_path: Path | None = None) -> pd.DataFrame:
    print(f"Loading {csv_path} ...", flush=True)
    df = load_data(csv_path, corrected_csv_path)
    print(f"  {len(df)}/{_TOTAL_LAND_SITES} land sites completed so far", flush=True)

    print("\n--- 1. Optimized LCOW map ---", flush=True)
    plot_optimal_lcow_map(df, out_dir, label)

    print("\n--- 2. Chosen design-parameter maps ---", flush=True)
    plot_chosen_parameter_maps(df, out_dir, label)

    print_diagnostics_summary(df)
    plot_diagnostics_histograms(df, out_dir, label)

    if grid_csv.is_file():
        compare_vs_grid_sweep(df, grid_csv, out_dir, label)
    else:
        print(f"\n(skipping grid-sweep comparison -- {grid_csv} not found)", flush=True)

    print("\nDone.", flush=True)
    return df


def main() -> int:
    run_analysis(
        csv_path=_HERE / "full_sweep_summary.csv",
        corrected_csv_path=_HERE / "full_sweep_summary_corrected.csv",
        grid_csv=_GRID_SWEEP_DIR / "full_sweep_case2.csv",
        out_dir=_HERE / "plots",
        label="BayesOpt sweep (case2, corrected LCOW)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
