"""Every figure, from the run directory: ``python -m sawh_bayesopt.plots [--only NAME ...]``.

Methodology figures read what train/validate saved (coverage curves, surrogate grids and
convergence, BO tuning regret); result figures read the merged global maps. A figure whose
inputs are not on disk yet is reported and skipped, so this can be run after any stage.
Global maps use viridis (purple = low, so purple = cheaper for LCOW), like the repo's
earlier global figures. Figures go to <run-dir>/figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sawh_bayesopt.optimize import ACQUISITIONS, INFO_MODES, KERNELS, SCENARIOS, TILT_MODES  # noqa: E402
from sawh_bayesopt.surrogates import FAMILIES, GRIDS  # noqa: E402
from sawh_bayesopt.train import DEFAULT_RUN_DIR  # noqa: E402

MAP_COLUMNS = (("lcow_usd_m3", "LCOW (USD/m³)"), ("water_kg_m2_day", "yield (kg/m²/day)"),
               ("hydrogel_thickness_mm", "hydrogel thickness (mm)"), ("tilt_mean_deg", "mean tilt (deg)"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--only", nargs="*", help="Figure names to draw (default all).")
    args = ap.parse_args(argv)
    out = args.run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures = {"sampling": fig_sampling, "surrogate_grids": fig_surrogate_grids,
               "surrogate_convergence": fig_surrogate_convergence, "surrogate_comparison": fig_surrogate_comparison,
               "bo_tuning": fig_bo_tuning, "global_maps": fig_global_maps, "scenarios": fig_scenarios,
               "validation": fig_validation}
    for name, fn in figures.items():
        if args.only and name not in args.only:
            continue
        try:
            fn(args.run_dir, out)
            print(f"drew {name}")
        except FileNotFoundError as e:
            print(f"skipped {name}: {e.filename} not found")
    return 0


def fig_sampling(run_dir: Path, out: Path) -> None:
    """Where the training cells are (by sampling stage and split), and how coverage
    converges as alg. 16.8 adds cells: d_max after each pick."""
    sel = pd.read_csv(run_dir / "selection.csv")
    cov = np.load(run_dir / "coverage.npz")
    fig = plt.figure(figsize=(15, 5), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1])
    ax = _map_axes(fig, gs[0])
    for (stage, split), g in sel.groupby(["stage", "split"]):
        ax.scatter(g.lon, g.lat, s=7, marker="o" if stage == "space_filling" else "^",
                   label=f"{stage.replace('_', '-')} / {split} ({len(g)})", transform=_plate())
    ax.legend(loc="lower left", fontsize=7.5, markerscale=1.5)
    ax.set_title("Training locations: space-filling subset + desert oversampling", loc="left")
    ax = fig.add_subplot(gs[1])
    ax.plot(np.arange(1, len(cov["space_filling_d_max"]) + 1), cov["space_filling_d_max"], label="all cells")
    ax.plot(np.arange(1, len(cov["desert_d_max"]) + 1), cov["desert_d_max"], label="desert stage (BWh, BWk)")
    ax.set_xlabel("cells selected")
    ax.set_ylabel("d_max: farthest cell to its nearest pick\n(standardized descriptor units)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Coverage convergence (Kochenderfer alg. 16.8)")
    _save(fig, out / "sampling.png")


def fig_surrogate_grids(run_dir: Path, out: Path) -> None:
    """Validation water RMSE over every family's hyperparameter grid (heat maps; ridge as a
    curve over alpha)."""
    grid = pd.read_csv(run_dir / "surrogate_grid.csv")
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), layout="constrained")
    r = grid[grid.family == "ridge"].sort_values("hyper_alpha")
    axes[0].semilogx(r.hyper_alpha, r.val_water_rmse, marker="o")
    axes[0].set(xlabel="alpha", ylabel="validation water RMSE (kg/m²/day)", title="ridge")
    panels = [("random_forest", "hyper_max_features", "hyper_min_samples_leaf", None),
              ("gradient_boosting", "hyper_learning_rate", "hyper_max_leaf_nodes", None),
              *[("mlp", "hyper_width", "hyper_depth", lr) for lr in sorted({g["lr"] for g in GRIDS["mlp"]})]]
    for ax, (family, rows_col, cols_col, lr) in zip(axes[1:], panels):
        g = grid[grid.family == family]
        if lr is not None:
            g = g[np.isclose(g.hyper_lr, lr)]
        if g.empty:
            ax.set_axis_off()
            continue
        piv = g.pivot(index=rows_col, columns=cols_col, values="val_water_rmse")
        im = ax.imshow(piv.to_numpy(), cmap="viridis", aspect="auto")
        for (i, j), v in np.ndenumerate(piv.to_numpy()):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=8)
        ax.set_xticks(range(piv.shape[1]), piv.columns)
        ax.set_yticks(range(piv.shape[0]), piv.index)
        ax.set(xlabel=cols_col.removeprefix("hyper_"), ylabel=rows_col.removeprefix("hyper_"),
               title=family.replace("_", " ") + (f", lr {lr:g}" if lr is not None else ""))
        fig.colorbar(im, ax=ax, shrink=0.8, label="val water RMSE")
    fig.suptitle("Surrogate hyperparameter grids (validation cells)", x=0.01, ha="left", fontweight="bold")
    _save(fig, out / "surrogate_grids.png")


def fig_surrogate_convergence(run_dir: Path, out: Path) -> None:
    """Validation error against training iterations: MLP loss per step (train and val, every
    config faint, the selected one bold), forest size, boosting rounds."""
    comp = json.loads((run_dir / "surrogate_comparison.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), layout="constrained")
    for ax, family, xlabel in zip(axes, ("mlp", "random_forest", "gradient_boosting"),
                                  ("optimizer step", "trees", "boosting iterations")):
        for d in sorted((run_dir / "surrogates" / family).glob("config_*")):
            c = np.load(d / "convergence.npz")
            chosen = family in comp and comp[family]["config"] == d.name
            kw = dict(linewidth=2.2, alpha=1.0, zorder=3) if chosen else dict(linewidth=0.8, alpha=0.35)
            if family == "mlp":
                ax.plot(c["iteration"], c["val_loss"], color="C0", **kw, label="validation" if chosen else None)
                ax.plot(c["iteration"], c["train_loss"], color="C1", **kw, label="train" if chosen else None)
            else:
                ax.plot(c["iteration"], c["val_water_rmse"], color="C0", **kw,
                        label=f"selected {d.name}" if chosen else None)
        ax.set(xlabel=xlabel, ylabel="loss" if family == "mlp" else "validation water RMSE (kg/m²/day)",
               title=family.replace("_", " "))
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle("Surrogate convergence", x=0.01, ha="left", fontweight="bold")
    _save(fig, out / "surrogate_convergence.png")


def fig_surrogate_comparison(run_dir: Path, out: Path) -> None:
    """Test-cell scores of each family's selected config, and the annual error by Koppen
    group; plus the deployed model's chained annual yield against physics."""
    comp = json.loads((run_dir / "surrogate_comparison.json").read_text())
    fams = [f for f in FAMILIES if f in comp]
    groups = sorted(k for k in comp[fams[0]]["test"] if k != "all")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), layout="constrained")
    x = np.arange(len(fams))
    axes[0].bar(x - 0.2, [comp[f]["test"]["all"]["water_rmse"] for f in fams], 0.4, label="daily water RMSE (kg/m²/day)")
    axes[0].bar(x + 0.2, [comp[f]["test"]["all"]["annual_median_rel_err"] for f in fams], 0.4,
                label="annual yield, median relative error")
    axes[0].set_xticks(x, [f.replace("_", "\n") for f in fams])
    axes[0].set_title(f"Test cells (deployed: {comp['deployed']})")
    axes[0].legend(fontsize=8)
    mat = np.array([[comp[f]["test"][g]["annual_median_rel_err"] or np.nan for g in groups] for f in fams])
    im = axes[1].imshow(100 * mat, cmap="viridis", aspect="auto")
    for (i, j), v in np.ndenumerate(100 * mat):
        axes[1].text(j, i, f"{v:.1f}", ha="center", va="center", color="w", fontsize=8)
    axes[1].set_xticks(range(len(groups)), groups, rotation=30)
    axes[1].set_yticks(range(len(fams)), fams)
    axes[1].set_title("Annual yield median error by Koppen group (%)")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    mat = np.array([[comp[f]["test"][g]["water_rmse"] for g in groups] for f in fams])
    im = axes[2].imshow(mat, cmap="viridis", aspect="auto")
    for (i, j), v in np.ndenumerate(mat):
        axes[2].text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=8)
    axes[2].set_xticks(range(len(groups)), groups, rotation=30)
    axes[2].set_yticks(range(len(fams)), fams)
    axes[2].set_title("Daily water RMSE by Koppen group (kg/m²/day)")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    _save(fig, out / "surrogate_comparison.png")


def fig_bo_tuning(run_dir: Path, out: Path) -> None:
    """Final mean regret over kernel x acquisition; regret curves of the best configs; and
    what BO is searching -- LCOW against thickness at a few tuning cells."""
    tun = np.load(run_dir / "bo_tuning.npz")
    truth = np.load(run_dir / "bo_truth.npz")
    chosen = json.loads((run_dir / "bo_config.json").read_text())
    keys, regret = list(tun["keys"]), tun["regret"]
    final = regret.mean(axis=1)[:, -1]
    mat = np.full((len(KERNELS), len(ACQUISITIONS)), np.nan)
    for k, v in zip(keys, final):
        parts = k.split("|")
        if parts[0] == "sweep1":
            mat[KERNELS.index(parts[1]), ACQUISITIONS.index((parts[2], float(parts[3])))] = v
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.8), layout="constrained")
    # Floored at 0.001%: a config that finds every optimum has zero regret, which a log
    # scale cannot show.
    im = axes[0].imshow(np.maximum(100 * mat, 1e-3), cmap="viridis_r", aspect="auto", norm=matplotlib.colors.LogNorm())
    axes[0].set_xticks(range(len(ACQUISITIONS)), [f"{a}\n{p:g}" for a, p in ACQUISITIONS], fontsize=8)
    axes[0].set_yticks(range(len(KERNELS)), KERNELS)
    axes[0].set_title("Mean relative regret after 15 evaluations (%)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    for i in np.argsort(final)[:6]:
        axes[1].semilogy(np.arange(1, regret.shape[2] + 1), 100 * np.maximum(regret[i].mean(axis=0), 1e-4),
                         label=keys[i].replace("|", " "))
    axes[1].axvline(chosen["budget"], color="0.5", linestyle="--", linewidth=0.8)
    axes[1].set(xlabel="evaluations", ylabel="mean relative regret (%)",
                title=f"Best configurations (chosen budget {chosen['budget']})")
    axes[1].legend(fontsize=7)
    lcow = np.where(truth["feasible"], truth["lcow"], np.nan)
    for i in np.linspace(0, len(lcow) - 1, 6).astype(int):
        axes[2].plot(truth["thickness_m"] * 1e3, lcow[i], label=f"cell {truth['cell'][i]}")
    axes[2].set(xlabel="hydrogel thickness (mm)", ylabel="surrogate LCOW (USD/m³)",
                title="The 1-D objective at six tuning cells (daily tilt, hindsight)")
    axes[2].legend(fontsize=7)
    _save(fig, out / "bo_tuning.png")


def fig_global_maps(run_dir: Path, out: Path) -> None:
    """For each merged scenario: LCOW, yield, thickness and mean tilt at every cell."""
    drew = False
    for tilt_mode, info in SCENARIOS:
        path = run_dir / f"maps_{tilt_mode}_{info}.csv"
        if not path.exists():
            continue
        m = pd.read_csv(path)
        m = m[m.feasible]
        fig = plt.figure(figsize=(20, 9), layout="constrained")
        for k, (col, label) in enumerate(MAP_COLUMNS):
            ax = _map_axes(fig, (2, 2, k + 1))
            hi = m[col].quantile(0.98) if col == "lcow_usd_m3" else m[col].max()
            sc = ax.scatter(m.lon, m.lat, c=m[col], s=4, cmap="viridis", vmin=m[col].min(), vmax=hi,
                            transform=_plate())
            fig.colorbar(sc, ax=ax, shrink=0.7, label=label, extend="max" if col == "lcow_usd_m3" else "neither")
            ax.set_title(f"{label}: median {m[col].median():.2f}", loc="left")
        fig.suptitle(f"Global optimum, tilt {tilt_mode.replace('_', ' ')}, {info} weather ({len(m):,} feasible cells)",
                     x=0.01, ha="left", fontweight="bold")
        _save(fig, out / f"map_{tilt_mode}_{info}.png")
        drew = True
    if not drew:
        raise FileNotFoundError(None, None, str(run_dir / "maps_*.csv"))


def fig_scenarios(run_dir: Path, out: Path) -> None:
    """Median LCOW over tilt mode x information regime (the value of moving the panel more
    often and of knowing more about the weather), and by Koppen group."""
    maps = {(t, i): pd.read_csv(run_dir / f"maps_{t}_{i}.csv") for t, i in SCENARIOS
            if (run_dir / f"maps_{t}_{i}.csv").exists()}
    if not maps:
        raise FileNotFoundError(None, None, str(run_dir / "maps_*.csv"))
    mat = np.full((len(TILT_MODES), len(INFO_MODES)), np.nan)
    for (t, i), m in maps.items():
        mat[TILT_MODES.index(t), INFO_MODES.index(i)] = m.loc[m.feasible, "lcow_usd_m3"].median()
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), layout="constrained")
    im = axes[0].imshow(mat, cmap="viridis", aspect="auto")
    for (r, c), v in np.ndenumerate(mat):
        axes[0].text(c, r, f"{v:.2f}", ha="center", va="center", color="w")
    axes[0].set_xticks(range(len(INFO_MODES)), INFO_MODES)
    axes[0].set_yticks(range(len(TILT_MODES)), [t.replace("_", " ") for t in TILT_MODES])
    axes[0].set_title("Median LCOW (USD/m³) by scenario")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    table = pd.concat([m.loc[m.feasible].assign(scenario=f"{t}/{i}") for (t, i), m in maps.items()])
    piv = table.groupby(["koppen_group", "scenario"]).lcow_usd_m3.median().unstack()
    im = axes[1].imshow(piv.to_numpy(), cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(piv.shape[1]), piv.columns, rotation=60, ha="right", fontsize=8)
    axes[1].set_yticks(range(piv.shape[0]), piv.index)
    axes[1].set_title("Median LCOW (USD/m³) by Koppen group")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    _save(fig, out / "scenarios.png")


def fig_validation(run_dir: Path, out: Path) -> None:
    """Surrogate-assisted against physics-only BO at the test cells: the optima's true
    LCOWs, both searches' convergence, and the surrogate's own predictions against physics
    for every scenario."""
    v = run_dir / "validate"
    table = pd.read_csv(v / "validation.csv")
    pb, sb = np.load(v / "physics_bo.npz"), np.load(v / "surrogate_bo.npz")
    piv = table.pivot(index="cell", columns="source", values="physics_lcow")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
    lim = [np.nanmin(piv[["physics_bo", "surrogate_bo"]].to_numpy()), np.nanmax(piv[["physics_bo", "surrogate_bo"]].to_numpy())]
    axes[0].scatter(piv.physics_bo, piv.surrogate_bo, s=18)
    axes[0].plot(lim, lim, color="0.5", linewidth=0.8)
    axes[0].set(xlabel="physics-only BO optimum, physics LCOW (USD/m³)",
                ylabel="surrogate BO optimum, physics LCOW (USD/m³)", title="Which search finds the better optimum")
    for z, label in ((pb, "physics BO (true LCOW)"), (sb, "surrogate BO (surrogate LCOW)")):
        y = np.where(z["trace_feasible"], z["trace_y"], np.nan)
        best = np.fmin.accumulate(np.nan_to_num(y, nan=np.inf), axis=1) / np.nanmin(y, axis=1, keepdims=True)
        axes[1].plot(np.arange(1, best.shape[1] + 1), np.median(best, axis=0) - 1, label=label)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="evaluations", ylabel="median (best so far / own final best) - 1", title="Convergence")
    axes[1].legend()
    s = table[table.source.ne("physics_bo") & table.physics_lcow.notna()]
    axes[2].scatter(s.physics_lcow, s.predicted_lcow, s=8, c=pd.factorize(s.source)[0], cmap="viridis")
    lim = [s[["physics_lcow", "predicted_lcow"]].min().min(), s[["physics_lcow", "predicted_lcow"]].max().max()]
    axes[2].plot(lim, lim, color="0.5", linewidth=0.8)
    axes[2].set(xlabel="physics LCOW (USD/m³)", ylabel="surrogate-predicted LCOW (USD/m³)",
                title="Surrogate fidelity at its own optima, every scenario")
    _save(fig, out / "validation.png")


def _map_axes(fig, spec):
    import cartopy.feature as cfeature

    ax = fig.add_subplot(*spec, projection=_plate()) if isinstance(spec, tuple) else fig.add_subplot(spec, projection=_plate())
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m", facecolor="0.9", edgecolor="0.4",
                                                linewidth=0.3, zorder=0))
    ax.coastlines(resolution="110m", color="0.35", linewidth=0.4, zorder=1)
    ax.set_extent([-180, 180, -58, 80], crs=_plate())
    return ax


def _plate():
    import cartopy.crs as ccrs

    return ccrs.PlateCarree()


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
