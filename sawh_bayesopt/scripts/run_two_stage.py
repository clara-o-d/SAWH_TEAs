#!/usr/bin/env python3
"""Two-stage pipeline driver: climate features -> representative cells -> physics campaign
-> daily surrogate -> per-cell design BO over enumerated daily control -> maps, plus the
two validations (featurization, and optimality gap against per-site true-physics BO).

Stages, in order, all under --run-dir (default outputs/two_stage/<run-id>):

  features           scan the Open-Meteo cache -> features.npz, pca.npz            (CPU, once)
  validate-features  real vs PCA-reconstructed years through physics -> featval.*  (GPU)
  select             cluster + stratify cells -> selection.csv, config.json          (CPU)
  simulate           one chunk of the physics campaign -> runs/chunk_NNNN.npz        (GPU, array)
  fit                surrogate ensemble -> model/, holdout_report.json               (GPU)
  optimize           design BO per cell for one tilt x schedule mode -> opt/<mode>/  (GPU, array)
  active             physics at/around the optima -> active/roundR_chunk_NNNN.npz    (GPU, array)
  maps               merge optimize parts -> maps_<mode>.csv                        (CPU)
  validate-bo        two-stage vs per-site BO on held-out cells -> validate_bo.*     (GPU)

Chunked stages are resumable: a chunk whose output exists is skipped. Every stage reads
config.json for the retained PC count so they cannot silently disagree on it.

Run with solar_lumped/.venv_gpu (jax + diffrax + equinox).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG / "src") not in sys.path:
    sys.path.insert(0, str(_PKG / "src"))

from sawh_bayesopt import climate  # noqa: E402

_DEFAULT_CACHE = _PKG.parent / "solar_lumped" / ".weather_cache"
BASELINE_DESIGN = np.array([0.004, 0.016, 4.0])  # Table S3 thickness/loading, the 16 mm scan optimum
BASELINE_CONTROL = np.array([0.0, 0.0, 30.0])
# Modes validate-bo scores. (fixed, constant) is the per-site BO's own control space.
VALIDATE_MODES = (("fixed", "constant"), ("fixed", "persistence"), ("seasonal", "persistence"),
                  ("daily", "persistence"), ("daily", "hindsight"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE)
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="stage", required=True)

    sub.add_parser("features")

    p = sub.add_parser("validate-features")
    p.add_argument("--n-cells", type=int, default=300)
    p.add_argument("--components", type=lambda s: [int(v) for v in s.split(",")], default=[4, 8, 12, 15, 20])
    p.add_argument("--chunk-size", type=int, default=700)

    p = sub.add_parser("select")
    p.add_argument("--n-components", type=int, required=True, help="Retained PCs, from validate-features.")
    p.add_argument("--n-clusters", type=int, default=800)
    p.add_argument("--quota", type=_quota, default="high_altitude=80,hyper_arid=100,monsoonal=80,coastal_humid=80",
                   help="Minimum cells per tail regime, e.g. hyper_arid=100,monsoonal=80")
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--designs-per-cell", type=int, default=16)

    for name in ("simulate", "active"):
        p = sub.add_parser(name)
        p.add_argument("--chunk-index", type=int, required=True)
        p.add_argument("--chunk-size", type=int, default=700)
        if name == "active":
            _mode_args(p)
            p.add_argument("--round", type=int, required=True)
            p.add_argument("--n-cells", type=int, default=300)
            p.add_argument("--n-perturb", type=int, default=4)

    p = sub.add_parser("fit")
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--steps", type=int, default=20_000)

    p = sub.add_parser("optimize")
    _mode_args(p)
    p.add_argument("--cells", choices=("anchors", "all"), default="all",
                   help="anchors: the training cells only (run first; they warm-start the rest).")
    p.add_argument("--cell-range", type=int, nargs=2, metavar=("START", "END"))
    _bo_args(p)

    p = sub.add_parser("maps")
    _mode_args(p)

    p = sub.add_parser("validate-bo")
    p.add_argument("--emit-sites", action="store_true",
                   help="Write bo_sites.txt (held-out cells) for run_bayesopt_sweep.py and stop.")
    p.add_argument("--bo-summary", type=Path, help="run_bayesopt_sweep.py summary.csv over those sites.")
    p.add_argument("--chunk-size", type=int, default=700)
    _bo_args(p)

    args = ap.parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "features": stage_features, "validate-features": stage_validate_features, "select": stage_select,
        "simulate": stage_simulate, "fit": stage_fit, "optimize": stage_optimize, "active": stage_active,
        "maps": stage_maps, "validate-bo": stage_validate_bo,
    }[args.stage](args)


def _quota(s: str) -> dict[str, int]:
    return {k: int(v) for k, v in (item.split("=") for item in s.split(",") if item)}


def _mode_args(p):
    from sawh_bayesopt.two_stage import SCHEDULE_MODES, TILT_MODES

    p.add_argument("--tilt-mode", choices=TILT_MODES, default="fixed")
    p.add_argument("--schedule-mode", choices=SCHEDULE_MODES, default="persistence")


def _bo_args(p):
    p.add_argument("--n-init", type=int, default=8)
    p.add_argument("--n-total", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--warm-n-total", type=int, default=12, help="Budget for warm-started (non-anchor) cells.")


# --- Stages ---


def stage_features(args) -> int:
    summary = climate.featurize_cache(args.cache_dir, args.run_dir / "features.npz", seed=args.seed)
    print(json.dumps(summary, indent=2, default=int))
    return 0


def stage_validate_features(args) -> int:
    """Real hourly years vs years rebuilt from the leading k PCs, through the same physics
    at a fixed baseline design and schedule. If yields do not track at k, add components."""
    feats = np.load(args.run_dir / "features.npz")
    pca = dict(np.load(args.run_dir / "pca.npz"))
    rng = np.random.default_rng(args.seed)
    regime = feats["regime"]
    # Stratified: equal shares per regime, so the tails are not a rounding error here.
    per = max(1, args.n_cells // len(np.unique(regime)))
    cells = np.concatenate([rng.choice(np.flatnonzero(regime == r), min(per, int((regime == r).sum())), replace=False)
                            for r in np.unique(regime)])
    # 0 = the real year. 72 (full rank) is always run too: it is the resampling floor --
    # the error the hourly round-trip alone costs -- so truncation error at k reads
    # against it rather than against zero.
    variants = [0, *sorted(set(args.components) | {3 * climate.HOURS})]

    # Chunks hold whole cells, so a cell's real and rebuilt years always share a chunk and
    # walk the same days, and each finished chunk is saved at once: resubmitting the same
    # command (a timed-out job) skips them. The directory is keyed by the arguments, so a
    # different sample can never be mixed into these results.
    tag = f"n{args.n_cells}_k{'-'.join(map(str, variants[1:]))}_s{args.seed}"
    chunk_dir = args.run_dir / "featval_chunks" / tag
    chunk_dir.mkdir(parents=True, exist_ok=True)
    per_chunk = max(1, args.chunk_size // len(variants))
    cell_chunks = [cells[i:i + per_chunk] for i in range(0, len(cells), per_chunk)]
    out = []
    for n, chunk_cells in enumerate(cell_chunks):
        path = chunk_dir / f"chunk_{n:03d}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                out += pickle.load(fh)
            print(f"chunk {n + 1}/{len(cell_chunks)}: loaded {path}", flush=True)
            continue
        print(f"chunk {n + 1}/{len(cell_chunks)}: {len(chunk_cells)} cells x {len(variants)} variants", flush=True)
        part = []
        for c in chunk_cells:
            df = climate.frame_from_cache(args.cache_dir, str(feats["cache_key"][c]))
            part += [(int(c), k, df if k == 0 else climate.reconstruct_day_frame(df, pca, k)) for k in variants]
        rows = _simulate([f for _c, _k, f in part], [c for c, _k, _f in part], feats["valid"],
                         np.tile(BASELINE_DESIGN, (len(part), 1)),
                         np.tile(BASELINE_CONTROL, (len(part), climate.N_DAYS, 1)))
        done = []
        for j, (c, k, _f) in enumerate(part):
            water = rows["water"][rows["instance"] == j]
            done.append({"cell": c, "regime": regime[c], "k": k, "annual_water": water.mean(), "daily": water})
        with open(path, "wb") as fh:
            pickle.dump(done, fh)
        out += done
    real = {r["cell"]: r for r in out if r["k"] == 0}
    table = pd.DataFrame([
        {"cell": r["cell"], "regime": r["regime"], "k": r["k"],
         "annual_rel_err": r["annual_water"] / real[r["cell"]]["annual_water"] - 1,
         "daily_rmse_kg_m2": float(np.sqrt(np.mean((r["daily"] - real[r["cell"]]["daily"]) ** 2)))}
        for r in out if r["k"] > 0 and r["cell"] in real and len(r["daily"]) == len(real[r["cell"]]["daily"])
    ])
    table.to_csv(args.run_dir / "featval.csv", index=False)
    summary = table.assign(abs_err=table.annual_rel_err.abs()).groupby(["k", "regime"]).agg(
        median_abs_annual_rel_err=("abs_err", "median"), p90_abs_annual_rel_err=("abs_err", lambda s: s.quantile(0.9)),
        median_daily_rmse=("daily_rmse_kg_m2", "median"), n=("cell", "size"))
    summary.to_csv(args.run_dir / "featval_summary.csv")
    print(summary.to_string())
    return 0


def stage_select(args) -> int:
    from scipy.stats import qmc

    from sawh_bayesopt.daily_surrogate import design_bounds

    f = climate.load_features(args.run_dir / "features.npz", args.n_components)
    desc = climate.cell_descriptors(f["day_features"], f["valid"], f)
    sel = climate.select_representatives(desc, f["regime"], n_clusters=args.n_clusters, regime_quota=args.quota,
                                         test_frac=args.test_frac, seed=args.seed)
    sel.to_csv(args.run_dir / "selection.csv", index=False)
    # The whole physics campaign, fixed here so every array task derives the same list:
    # per cell, a scrambled-Sobol set seeded by the cell (so cells differ) of designs.
    bounds = design_bounds()
    campaign = []
    for c in sel["cell"]:
        u = qmc.Sobol(d=3, scramble=True, seed=int(c) + args.seed).random(args.designs_per_cell)
        campaign += [(int(c), bounds[:, 0] + ui * (bounds[:, 1] - bounds[:, 0])) for ui in u]
    with open(args.run_dir / "campaign.pkl", "wb") as fh:
        pickle.dump(campaign, fh)
    config = {"n_components": args.n_components, "seed": args.seed, "designs_per_cell": args.designs_per_cell,
              "n_instances": len(campaign)}
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(sel.groupby(["regime", "split"]).size().unstack(fill_value=0).to_string())
    print(f"{len(campaign)} physics instance-years; at --chunk-size 700 that is "
          f"{-(-len(campaign) // 700)} simulate chunks")
    return 0


def stage_simulate(args) -> int:
    from sawh_bayesopt.daily_surrogate import random_controls

    with open(args.run_dir / "campaign.pkl", "rb") as fh:
        campaign = pickle.load(fh)
    lo = args.chunk_index * args.chunk_size
    part = campaign[lo:lo + args.chunk_size]
    out = args.run_dir / "runs" / f"chunk_{args.chunk_index:04d}.npz"
    if not part or out.exists():
        print("nothing to do" if not part else f"{out} exists")
        return 0
    rng = np.random.default_rng([args.seed, args.chunk_index])
    controls = random_controls(rng, len(part), climate.N_DAYS)
    _run_and_save([(c, x, u) for (c, x), u in zip(part, controls)], args, out, instance_offset=lo)
    return 0


def stage_active(args) -> int:
    from sawh_bayesopt.two_stage import active_requests

    anchors = _load_anchors(args)
    reqs = active_requests(anchors, n_cells=args.n_cells, n_perturb=args.n_perturb,
                           rng=np.random.default_rng([args.seed, 1000 + args.round]))
    lo = args.chunk_index * args.chunk_size
    part = reqs[lo:lo + args.chunk_size]
    out = args.run_dir / "active" / f"round{args.round}_chunk_{args.chunk_index:04d}.npz"
    if not part or out.exists():
        print(f"{len(reqs)} active requests; " + ("nothing to do" if not part else f"{out} exists"))
        return 0
    _run_and_save(part, args, out, instance_offset=(args.round + 1) * 10_000_000 + lo)
    return 0


def stage_fit(args) -> int:
    from sawh_bayesopt import daily_surrogate as ds

    _cfg, f, sel = _context(args)
    rows = _load_rows(args.run_dir)
    split = dict(zip(sel["cell"], sel["split"]))
    is_test = np.array([split.get(int(c), "train") == "test" for c in rows["cell"]])
    X = ds.assemble_inputs(rows["design"], rows["control"], rows["state"], f["day_features"][rows["cell"], rows["day"]])
    train = ~is_test
    print(f"fitting on {int(train.sum())} rows from {len(np.unique(rows['cell'][train]))} cells; "
          f"{int(is_test.sum())} held-out rows", flush=True)
    model = ds.fit_ensemble(X[train], rows["water"][train], rows["state_end"][train], rows["capped"][train],
                            rows["cell"][train], n_members=args.members, width=args.width, steps=args.steps,
                            seed=args.seed)
    ds.save_surrogate(model, args.run_dir / "model")
    test_rows = {k: v[is_test] for k, v in rows.items()}
    report = {
        "n_train_rows": int(train.sum()), "n_test_rows": int(is_test.sum()),
        "holdout": ds.grouped_holdout_report(model, test_rows, f["day_features"], f["regime"]),
        "monotonicity_heldout": ds.monotonicity_check(model, test_rows, f["day_features"], f["regime"]),
        "expected_signs": ds.EXPECTED_SIGNS,
    }
    (args.run_dir / "holdout_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def stage_optimize(args) -> int:
    from sawh_bayesopt.two_stage import neighbor_warm_starts, optimize_cells

    _cfg, f, sel, model, econ, feats, desc = _optimize_context(args)
    mode_dir = args.run_dir / "opt" / f"{args.tilt_mode}_{args.schedule_mode}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    train_cells = sel.loc[sel.split == "train", "cell"].to_numpy()
    if args.cells == "anchors":
        best = optimize_cells(model, train_cells, feats, f["month"], tilt_mode=args.tilt_mode,
                              schedule_mode=args.schedule_mode, econ=econ, n_init=args.n_init,
                              n_total=args.n_total, batch_size=args.batch_size, seed=args.seed)
        with open(mode_dir / "anchors.pkl", "wb") as fh:
            pickle.dump(best, fh)
        _write_part(best, f, desc, train_cells, mode_dir / "anchors")
        return 0

    anchors = _load_anchors(args)
    n = len(f["lat"])
    start, end = args.cell_range or (0, n)
    targets = np.array([c for c in range(start, min(end, n)) if c not in anchors])
    part = mode_dir / f"part_{start:05d}_{end:05d}"
    if part.with_suffix(".csv").exists():
        print(f"{part}.csv exists")
        return 0
    best = optimize_cells(model, targets, feats, f["month"], tilt_mode=args.tilt_mode,
                          schedule_mode=args.schedule_mode, econ=econ, n_init=max(args.n_init // 2, 2),
                          n_total=args.warm_n_total, batch_size=args.batch_size, seed=args.seed,
                          warm_starts=neighbor_warm_starts(desc, anchors, targets))
    best.update({c: anchors[c] for c in range(start, min(end, n)) if c in anchors})
    _write_part(best, f, desc, train_cells, part)
    return 0


def stage_maps(args) -> int:
    mode_dir = args.run_dir / "opt" / f"{args.tilt_mode}_{args.schedule_mode}"
    parts = sorted(mode_dir.glob("part_*.csv"))
    if not parts:
        print(f"no parts under {mode_dir}")
        return 1
    maps = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates("cell").sort_values("cell")
    out = args.run_dir / f"maps_{args.tilt_mode}_{args.schedule_mode}.csv"
    maps.to_csv(out, index=False)
    print(f"wrote {out}: {len(maps)} cells, {int(maps.ood.sum())} flagged out-of-distribution, "
          f"{int((~maps.feasible).sum())} infeasible")
    return 0


def stage_validate_bo(args) -> int:
    """Optimality gap in $/m3 on held-out cells: two-stage design + schedule scored by TRUE
    physics, against per-site true-physics BO (run_bayesopt_sweep.py) at the same cells.

    (fixed, constant) is like-for-like with that BO -- same control space -- so its gap is
    the surrogate's optimality gap. The daily modes' gaps are that plus the value of daily
    control under the given information regime."""
    from sawh_bayesopt.two_stage import lcow_or_penalty, neighbor_warm_starts, optimize_cells

    if args.emit_sites:
        _cfg, f, sel = _context(args)
        test_cells = sel.loc[sel.split == "test", "cell"].to_numpy()
        path = args.run_dir / "bo_sites.txt"
        path.write_text("".join(f"{f['lat'][c]:.6f} {f['lon'][c]:.6f}\n" for c in test_cells))
        print(f"wrote {len(test_cells)} sites to {path}")
        return 0
    if args.bo_summary is None:
        raise SystemExit("validate-bo needs --bo-summary (or --emit-sites first)")

    _cfg, f, sel, model, econ, feats, desc = _optimize_context(args)
    test_cells = sel.loc[sel.split == "test", "cell"].to_numpy()
    bo = pd.read_csv(args.bo_summary)
    rows = []
    for tilt_mode, schedule_mode in VALIDATE_MODES:
        args.tilt_mode, args.schedule_mode = tilt_mode, schedule_mode
        anchors_path = args.run_dir / "opt" / f"{tilt_mode}_{schedule_mode}" / "anchors.pkl"
        warm = neighbor_warm_starts(desc, _load_anchors(args), test_cells) if anchors_path.exists() else None
        best = optimize_cells(model, test_cells, feats, f["month"], tilt_mode=tilt_mode, schedule_mode=schedule_mode,
                              econ=econ, n_init=args.n_init, n_total=args.n_total, batch_size=args.batch_size,
                              warm_starts=warm, seed=args.seed)
        reqs = [(c, best[c]["design"], best[c]["controls"]) for c in test_cells]
        true = []
        for i in range(0, len(reqs), args.chunk_size):
            part = reqs[i:i + args.chunk_size]
            r = _physics(part, args, f)
            true += [(r["water"][r["instance"] == j].mean(), r["capped"][r["instance"] == j].mean())
                     for j in range(len(part))]
        for (c, x, _u), (water, capped) in zip(reqs, true):
            lcow_true, feasible_true = lcow_or_penalty(water, capped, x, econ)
            d = np.hypot(bo["lat"] - f["lat"][c], bo["lon"] - f["lon"][c])
            lcow_bo = float(bo.loc[d.idxmin(), "best_combined_lcow_usd_m3"]) if d.min() < 0.5 else np.nan
            rows.append({"cell": int(c), "regime": f["regime"][c], "tilt_mode": tilt_mode,
                         "schedule_mode": schedule_mode, "lcow_surrogate": best[c]["lcow"],
                         "lcow_true": lcow_true, "feasible_true": feasible_true, "lcow_bo_true": lcow_bo,
                         "gap_usd_m3": lcow_true - lcow_bo})
    table = pd.DataFrame(rows)
    table.to_csv(args.run_dir / "validate_bo.csv", index=False)
    summary = table.assign(sur_err=(table.lcow_surrogate / table.lcow_true - 1).abs()).groupby(
        ["tilt_mode", "schedule_mode", "regime"]).agg(
        median_gap_usd_m3=("gap_usd_m3", "median"), p90_gap_usd_m3=("gap_usd_m3", lambda s: s.quantile(0.9)),
        median_surrogate_lcow_err=("sur_err", "median"), n=("cell", "size"))
    summary.to_csv(args.run_dir / "validate_bo_summary.csv")
    print(summary.to_string())
    return 0


# --- Shared plumbing ---


def _context(args):
    cfg = json.loads((args.run_dir / "config.json").read_text())
    f = climate.load_features(args.run_dir / "features.npz", cfg["n_components"])
    return cfg, f, pd.read_csv(args.run_dir / "selection.csv")


def _optimize_context(args):
    from sawh_bayesopt.daily_surrogate import load_surrogate
    from sawh_bayesopt.two_stage import fill_days
    from solar_lumped.economics import LCOEconomicParams

    cfg, f, sel = _context(args)
    desc = climate.cell_descriptors(f["day_features"], f["valid"], f)
    return (cfg, f, sel, load_surrogate(args.run_dir / "model"), LCOEconomicParams(),
            fill_days(f["day_features"], f["valid"]), desc)


def _load_anchors(args) -> dict:
    path = args.run_dir / "opt" / f"{args.tilt_mode}_{args.schedule_mode}" / "anchors.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _write_part(best: dict, f: dict, desc: np.ndarray, train_cells: np.ndarray, stem: Path) -> None:
    """One CSV row per cell (the map), plus the chosen daily schedules as npz."""
    d2, ood, thr = climate.mahalanobis_ood(desc[train_cells], desc)
    cells = np.array(sorted(best))
    rows = []
    for c in cells:
        b, u = best[c], np.asarray(best[c]["controls"])
        rows.append({
            "cell": int(c), "lat": f["lat"][c], "lon": f["lon"][c], "elevation_m": f["elevation_m"][c],
            "regime": f["regime"][c], "mahalanobis_d2": d2[c], "ood": bool(ood[c]),
            "hydrogel_thickness_mm": b["design"][0] * 1000.0, "vapor_gap_mm": b["design"][1] * 1000.0,
            "salt_loading": b["design"][2], "water_kg_m2_day": b["water"],
            "water_rel_std": b["water_std"] / max(b["water"], 1e-9), "capped_frac": b["capped_frac"],
            "lcow_usd_m3": b["lcow"], "feasible": b["feasible"], "n_evals": b["n_evals"],
            "tilt_mean_deg": u[:, 2].mean(), "tilt_std_deg": u[:, 2].std(),
            "seal_mean_h": u[:, 0].mean(), "open_mean_h": u[:, 1].mean(),
        })
    pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False)
    np.savez(stem.with_suffix(".npz"), cell=cells, controls=np.stack([best[c]["controls"] for c in cells]))
    print(f"wrote {stem}.csv ({len(rows)} cells; OOD threshold d2 > {thr:.1f})")


def _run_and_save(requests, args, out: Path, *, instance_offset: int) -> None:
    _cfg, f, _sel = _context(args)
    rows = _physics(requests, args, f)
    rows["instance"] = rows["instance"] + instance_offset
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **rows)
    print(f"wrote {out}: {len(rows['water'])} rows")


def _physics(requests, args, f) -> dict:
    """True physics for (cell, design, controls-over-all-366-days) requests, sharing each
    cell's frame across its instances."""
    frames = {c: climate.frame_from_cache(args.cache_dir, str(f["cache_key"][c])) for c in {c for c, _x, _u in requests}}
    return _simulate([frames[c] for c, _x, _u in requests], [c for c, _x, _u in requests], f["valid"],
                     np.array([x for _c, x, _u in requests]), np.array([u for _c, _x, u in requests]))


def _simulate(frames, cells, valid, designs, controls) -> dict:
    """simulate_years over the calendar days every involved cell has, with ``cell`` joined."""
    from sawh_bayesopt.daily_surrogate import simulate_years

    cells = np.asarray(cells)
    day_index = np.flatnonzero(valid[np.unique(cells)].all(axis=0))
    rows = simulate_years(frames, designs, controls[:, day_index], day_index)
    rows["cell"] = cells[rows["instance"]]
    return rows


def _load_rows(run_dir: Path) -> dict:
    files = sorted((run_dir / "runs").glob("chunk_*.npz")) + sorted((run_dir / "active").glob("round*_chunk_*.npz"))
    if not files:
        raise SystemExit(f"no physics chunks under {run_dir}/runs")
    parts = [dict(np.load(p)) for p in files]
    rows = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # Each instance's first walked day starts from the warm-up's Aitken extrapolation, not
    # from a simulated previous day, so its start state is not one the physics produces --
    # chunks run before the per-component guard (jax_daily_cycle) carry h up to 310 m
    # there. Drop it: 1 row in 366, and the chained holdout then starts every year from
    # a real day-2 state.
    order = np.lexsort((rows["day"], rows["instance"]))
    first = order[np.r_[True, np.diff(rows["instance"][order]) != 0]]
    keep = np.ones(len(rows["day"]), bool)
    keep[first] = False
    print(f"loaded {len(files)} physics chunk(s); dropped {len(first)} first-day rows", flush=True)
    return {k: v[keep] for k, v in rows.items()}


if __name__ == "__main__":
    raise SystemExit(main())
