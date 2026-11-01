"""Build the surrogate and tune the optimizer: ``python -m sawh_bayesopt.train <stage>``.

Stages, in order (scripts/submit_train.sh submits them on Sherlock with dependencies):

  features    scan the weather cache -> features.npz, pca.npz                       (CPU)
  select      750 space-filling cells (Kochenderfer alg. 16.8) + 250 desert cells,
              70/15/15 split by cell, 5 stratified thicknesses each -> selection.csv,
              coverage.npz, campaign.npz                                             (CPU)
  simulate    one chunk of 700 physics years under random daily controls
              -> runs/chunk_NNNN.npz                                         (GPU, array)
  fit         one family's hyperparameter grid (or one config of it), trained on the
              train cells, early-stopped and scored on the validation cells
              -> surrogates/<family>/<config>/                     (CPU; GPU for mlp)
  compare     best config per family on validation, its test-cell scores, and the
              deployed (GPU-resident) surrogate -> surrogate_comparison.json, model/ (CPU)
  tune-grid   ground truth for BO tuning: the surrogate-optimal year at every 0.05 mm
              thickness for the tuning cells (validation split), daily tilt + hindsight
              -> bo_truth.npz                                                       (GPU)
  tune-bo     every kernel x acquisition, then GP settings for the winner, scored by
              simple regret against that truth -> bo_tuning.npz, bo_config.json      (CPU)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sawh_bayesopt import climate

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = _REPO / "sawh_bayesopt" / "outputs" / "main"
DEFAULT_CACHE = _REPO / "solar_lumped" / ".weather_cache"
CHUNK_SIZE = 700  # instance-years per physics task: the width at which an A100 saturates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("features")
    p = sub.add_parser("select")
    p.add_argument("--n-space-filling", type=int, default=750)
    p.add_argument("--n-desert", type=int, default=250)
    p.add_argument("--thicknesses-per-cell", type=int, default=5)
    p = sub.add_parser("simulate")
    p.add_argument("--chunk-index", type=int, required=True)
    p = sub.add_parser("fit")
    p.add_argument("--family", choices=("ridge", "random_forest", "gradient_boosting", "mlp"), required=True)
    p.add_argument("--config-index", type=int, help="Fit one config of the grid (array jobs); default all.")
    sub.add_parser("compare")
    p = sub.add_parser("tune-grid")
    p.add_argument("--n-cells", type=int, default=60)
    p.add_argument("--batch", type=int, default=16)
    sub.add_parser("tune-bo")
    args = ap.parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    return {"features": stage_features, "select": stage_select, "simulate": stage_simulate, "fit": stage_fit,
            "compare": stage_compare, "tune-grid": stage_tune_grid, "tune-bo": stage_tune_bo}[args.stage](args)


def stage_features(args) -> int:
    print(climate.featurize_cache(args.cache_dir, args.run_dir / "features.npz", seed=args.seed))
    return 0


def stage_select(args) -> int:
    from sawh_bayesopt.simulate import stratified_thicknesses

    f = climate.load_features(args.run_dir / "features.npz")
    sel, curves = climate.select_cells(f["descriptors"], f["koppen_group"], f["koppen"],
                                       n_space_filling=args.n_space_filling, n_desert=args.n_desert,
                                       split_fracs=(0.70, 0.15, 0.15), seed=args.seed)
    sel["lat"], sel["lon"] = f["lat"][sel["cell"]], f["lon"][sel["cell"]]
    sel.to_csv(args.run_dir / "selection.csv", index=False)
    np.savez(args.run_dir / "coverage.npz", **curves)
    thickness = stratified_thicknesses(np.random.default_rng([args.seed, 1]), len(sel), args.thicknesses_per_cell)
    np.savez(args.run_dir / "campaign.npz", cell=np.repeat(sel["cell"].to_numpy(), args.thicknesses_per_cell),
             thickness_m=thickness.ravel())
    n = len(sel) * args.thicknesses_per_cell
    print(sel.groupby(["koppen_group", "split"]).size().unstack(fill_value=0).to_string())
    print(f"{n} physics years -> sbatch --array=0-{(n - 1) // CHUNK_SIZE} for the simulate stage")
    return 0


def stage_simulate(args) -> int:
    from sawh_bayesopt.simulate import random_controls, simulate_years

    camp = np.load(args.run_dir / "campaign.npz")
    lo = args.chunk_index * CHUNK_SIZE
    cells, thickness = camp["cell"][lo:lo + CHUNK_SIZE], camp["thickness_m"][lo:lo + CHUNK_SIZE]
    out = args.run_dir / "runs" / f"chunk_{args.chunk_index:04d}.npz"
    if not len(cells) or out.exists():
        print("nothing to do" if not len(cells) else f"{out} exists")
        return 0
    f = np.load(args.run_dir / "features.npz")
    frames = {c: climate.frame_from_cache(args.cache_dir, str(f["cache_key"][c])) for c in np.unique(cells)}
    day_index = np.flatnonzero(f["valid"][np.unique(cells)].all(axis=0))
    controls = random_controls(np.random.default_rng([args.seed, args.chunk_index]), len(cells), len(day_index))
    rows = simulate_years([frames[c] for c in cells], thickness, controls, day_index)
    rows["cell"] = cells[rows["instance"]]
    rows["instance"] = rows["instance"] + lo
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **rows)
    print(f"wrote {out}: {len(rows['water'])} rows")
    return 0


def stage_fit(args) -> int:
    from sawh_bayesopt import surrogates

    data = load_splits(args.run_dir)
    grid = surrogates.GRIDS[args.family]
    which = range(len(grid)) if args.config_index is None else [args.config_index]
    for k in which:
        out = args.run_dir / "surrogates" / args.family / f"config_{k:02d}"
        if (out / "val_metrics.json").exists():
            print(f"{out} exists")
            continue
        print(f"{args.family} config {k}: {grid[k]}", flush=True)
        model, convergence = surrogates.fit_family(args.family, grid[k], data, seed=args.seed)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "convergence.npz", **convergence)
        # Test scores are computed now, while the model is in memory (a random forest is
        # GBs on disk), but compare reads them only for the config validation chose.
        (out / "test_metrics.json").write_text(json.dumps(surrogates.evaluate(model, data["test"]), indent=2))
        if args.family in ("ridge", "mlp"):
            surrogates.save(model, out / "model")
        (out / "val_metrics.json").write_text(json.dumps(
            {"hyper": grid[k], "fit_seconds": model["fit_seconds"], **surrogates.evaluate(model, data["val"])}, indent=2))
    return 0


def stage_compare(args) -> int:
    """Per family, the config with the lowest validation water RMSE, and its test scores.
    The deployed surrogate is the better of ridge and the MLP on that same criterion
    (surrogates.deployable says why trees cannot be deployed)."""
    import shutil

    from sawh_bayesopt import surrogates

    rows, best = [], {}
    for family in surrogates.FAMILIES:
        for d in sorted((args.run_dir / "surrogates" / family).glob("config_*")):
            val = json.loads((d / "val_metrics.json").read_text())
            rows.append({"family": family, "config": d.name, **{f"hyper_{k}": v for k, v in val["hyper"].items()},
                         "fit_seconds": val["fit_seconds"],
                         **{f"val_{k}": v for k, v in val["all"].items()}})
            if family not in best or val["all"]["water_rmse"] < best[family][1]["all"]["water_rmse"]:
                best[family] = (d, val)
    table = pd.DataFrame(rows)
    table.to_csv(args.run_dir / "surrogate_grid.csv", index=False)
    summary = {f: {"config": d.name, "hyper": v["hyper"], "val": v, "test": json.loads((d / "test_metrics.json").read_text())}
               for f, (d, v) in best.items()}
    deploy = min((f for f in ("ridge", "mlp") if f in best), key=lambda f: best[f][1]["all"]["water_rmse"])
    summary["deployed"] = deploy
    (args.run_dir / "surrogate_comparison.json").write_text(json.dumps(summary, indent=2))
    if (args.run_dir / "model").exists():
        shutil.rmtree(args.run_dir / "model")
    shutil.copytree(best[deploy][0] / "model", args.run_dir / "model")
    for f, s in summary.items():
        if f != "deployed":
            print(f"{f:18s} {s['config']}  val RMSE {s['val']['all']['water_rmse']:.4f}  "
                  f"test RMSE {s['test']['all']['water_rmse']:.4f}  "
                  f"test annual median err {s['test']['all']['annual_median_rel_err']:.4f}")
    print(f"deployed: {deploy} ({best[deploy][0].name})")
    return 0


def stage_tune_grid(args) -> int:
    """Grid-search ground truth for BO tuning, on the deployed surrogate under daily tilt +
    hindsight: every 0.05 mm thickness at ``--n-cells`` validation cells, drawn stratified
    by Koppen group. Validation cells, not training or test: tuning the optimizer must not
    touch the test cells the validate module scores."""
    from sawh_bayesopt.optimize import THICKNESS_GRID_M

    sel = pd.read_csv(args.run_dir / "selection.csv")
    val = sel[sel["split"] == "val"]
    rng = np.random.default_rng([args.seed, 2])
    share = args.n_cells / len(val)
    cells = np.sort(np.concatenate([rng.choice(g["cell"].to_numpy(), max(1, int(round(share * len(g)))), replace=False)
                                    for _k, g in val.groupby("koppen_group")]))
    ctx = scenario_context(args.run_dir)
    lcow, g, water, capped = evaluate_grid(ctx, cells, THICKNESS_GRID_M, "daily", "hindsight", batch=args.batch)
    np.savez(args.run_dir / "bo_truth.npz", cell=cells, thickness_m=THICKNESS_GRID_M, lcow=lcow, g=g, feasible=g <= 0.0,
             water=water, capped_frac=capped)
    print(f"wrote bo_truth.npz: {len(cells)} cells x {len(THICKNESS_GRID_M)} thicknesses")
    return 0


def stage_tune_bo(args) -> int:
    """Two sweeps scored on the tuning cells' known optima (bo_truth.npz):
      1. every kernel x acquisition at n_init 3, budget 15;
      2. the winner's n_init x GP restarts x white-noise term.
    Score: mean relative simple regret at the end of the budget, ties broken by the area
    under the regret curve. The run budget is the first evaluation count at which the
    final config's mean regret falls to 0.5%, capped at 15."""
    from dataclasses import replace

    from sawh_bayesopt.optimize import ACQUISITIONS, KERNELS, BOConfig, bo_regret, run_bo

    truth = np.load(args.run_dir / "bo_truth.npz")
    lcow = np.where(truth["feasible"], truth["lcow"], np.inf)
    optimum = lcow.min(axis=1)
    keep = np.isfinite(optimum)
    lcow_all, g_all, opt = truth["lcow"][keep], truth["g"][keep], optimum[keep]
    cand = ((truth["thickness_m"] - truth["thickness_m"][0]) / np.ptp(truth["thickness_m"]))[:, None]
    print(f"{int(keep.sum())}/{len(keep)} tuning cells have a feasible thickness")

    def evaluate(requests):
        return [(float(lcow_all[p, i]), float(g_all[p, i]), None) for p, i in requests]

    def score(cfg):
        reg = bo_regret(run_bo(evaluate, len(opt), cand, cfg, seed=args.seed, log_every=0), opt)
        mean = reg.mean(axis=0)
        return reg, (float(mean[-1]), float(mean.mean()))

    results = {}
    for kernel in KERNELS:
        for acq, param in ACQUISITIONS:
            cfg = BOConfig(kernel=kernel, acquisition=acq, acq_param=param, n_init=3, budget=15)
            reg, s = score(cfg)
            results[f"sweep1|{kernel}|{acq}|{param}"] = (cfg, reg, s)
            print(f"  {kernel:22s} {acq:10s} {param:<5} final regret {s[0]:.4%}", flush=True)
    winner = min((v for k, v in results.items() if k.startswith("sweep1")), key=lambda v: v[2])[0]
    for n_init in (2, 3, 4, 6):
        for restarts in (0, 2, 8):
            for white in (True, False):
                cfg = replace(winner, n_init=n_init, n_restarts=restarts, white_noise=white)
                reg, s = score(cfg)
                results[f"sweep2|{n_init}|{restarts}|{white}"] = (cfg, reg, s)
                print(f"  n_init {n_init} restarts {restarts} white {white}: final regret {s[0]:.4%}", flush=True)
    final, reg, _s = min(results.values(), key=lambda v: v[2])
    mean = reg.mean(axis=0)
    hit = np.flatnonzero(mean <= 0.005)
    final = replace(final, budget=int(hit[0] + 1) if len(hit) else final.budget)
    np.savez(args.run_dir / "bo_tuning.npz", keys=np.array(list(results)),
             configs=np.array([json.dumps(v[0].to_dict()) for v in results.values()]),
             regret=np.stack([v[1] for v in results.values()]), optimum=opt)
    (args.run_dir / "bo_config.json").write_text(json.dumps(final.to_dict(), indent=2))
    print(f"chosen: {final}")
    return 0


# --- Shared by the stages above and by run_global / validate ---


def load_splits(run_dir: Path) -> dict:
    """Physics rows, split by cell, as {split: {"X", "Y", "rows"}}.

    Each instance's first walked day starts from the Aitken extrapolation, not from a
    simulated previous day, so its start state is not one the physics produces: it is
    dropped (1 row in 366), and chained evaluations start from a real day-2 state.
    """
    from sawh_bayesopt import surrogates

    files = sorted((run_dir / "runs").glob("chunk_*.npz"))
    if not files:
        raise SystemExit(f"no physics chunks under {run_dir}/runs")
    parts = [dict(np.load(p)) for p in files]
    rows = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort((rows["day"], rows["instance"]))
    first = order[np.r_[True, np.diff(rows["instance"][order]) != 0]]
    keep = np.ones(len(rows["day"]), bool)
    keep[first] = False
    rows = {k: v[keep] for k, v in rows.items()}
    sel = pd.read_csv(run_dir / "selection.csv").set_index("cell")
    rows["split"] = sel.loc[rows["cell"], "split"].to_numpy()
    rows["koppen_group"] = sel.loc[rows["cell"], "koppen_group"].to_numpy()
    f = climate.load_features(run_dir / "features.npz")
    X = surrogates.assemble_inputs(rows["thickness_m"], rows["control"], rows["state"],
                                   f["day_features"][rows["cell"], rows["day"]]).astype(np.float64)
    Y = surrogates.targets(rows)
    print(f"loaded {len(files)} physics chunk(s): {len(Y)} rows", flush=True)
    return {s: {"X": X[m], "Y": Y[m], "rows": {k: v[m] for k, v in rows.items()}}
            for s in ("train", "val", "test") for m in [rows["split"] == s]}


def scenario_context(run_dir: Path) -> dict:
    """What every surrogate-side optimization needs: the deployed surrogate in JAX form, the
    filled day features, the calendar month of each day, and the economics."""
    from sawh_bayesopt import surrogates
    from solar_lumped.economics import LCOEconomicParams

    f = climate.load_features(run_dir / "features.npz")
    net, norm, capped_logit = surrogates.deployable(surrogates.load(run_dir / "model"))
    return {"features": f, "net": net, "norm": norm, "capped_logit": capped_logit,
            "month": f["month"], "econ": LCOEconomicParams(), "year_fns": {}}


def surrogate_years(ctx: dict, cells: np.ndarray, thickness_m: np.ndarray, tilt_mode: str, info: str, *,
                    batch: int) -> dict:
    """The surrogate-optimal year of each (cell, thickness) pair under one scenario:
    water (n, 366), capped (n, 366), controls (n, 366, 5)."""
    import jax.numpy as jnp

    from sawh_bayesopt.optimize import decision_features, initial_states, make_year_fn

    key = (tilt_mode, info)
    if key not in ctx["year_fns"]:
        ctx["year_fns"][key] = make_year_fn(ctx["net"], ctx["norm"], ctx["capped_logit"], tilt_mode, ctx["month"])
    year = ctx["year_fns"][key]
    feats = ctx["features"]["day_features"]
    out = {"water": [], "capped": [], "controls": []}
    for a in range(0, len(cells), batch):
        c, h = np.asarray(cells[a:a + batch]), np.asarray(thickness_m[a:a + batch], float)
        n = len(c)
        if n < batch:  # pad the last batch so every call reuses one compilation
            c, h = np.r_[c, np.repeat(c[-1:], batch - n)], np.r_[h, np.repeat(h[-1:], batch - n)]
        act = feats[c]
        dec = decision_features(act, ctx["month"], info)
        w, cap, u = year(jnp.asarray(h, jnp.float32), jnp.asarray(act), jnp.asarray(dec, jnp.float32),
                         jnp.asarray(initial_states(h), jnp.float32))
        out["water"].append(np.asarray(w)[:n])
        out["capped"].append(np.asarray(cap)[:n])
        out["controls"].append(np.asarray(u)[:n])
    return {k: np.concatenate(v) for k, v in out.items()}


def evaluate_grid(ctx: dict, cells: np.ndarray, thickness_m: np.ndarray, tilt_mode: str, info: str, *, batch: int):
    """LCOW, constraint value g, mean water and capped-day share at every (cell, thickness)."""
    from sawh_bayesopt.optimize import lcow_and_constraint

    cc, hh = np.repeat(cells, len(thickness_m)), np.tile(thickness_m, len(cells))
    yr = surrogate_years(ctx, cc, hh, tilt_mode, info, batch=batch)
    water, capped = yr["water"].mean(axis=1), yr["capped"].mean(axis=1)
    scored = [lcow_and_constraint(w, cf, h, ctx["econ"]) for w, cf, h in zip(water, capped, hh)]
    shape = (len(cells), len(thickness_m))
    return (np.array([s[0] for s in scored]).reshape(shape), np.array([s[1] for s in scored]).reshape(shape),
            water.reshape(shape), capped.reshape(shape))


if __name__ == "__main__":
    raise SystemExit(main())
