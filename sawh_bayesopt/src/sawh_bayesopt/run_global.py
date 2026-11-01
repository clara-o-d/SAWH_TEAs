"""Optimize every cell on Earth: ``python -m sawh_bayesopt.run_global <stage>``.

  run    one array task = one (scenario, block of cells): the tuned BO (bo_config.json)
         over thickness, each evaluation the surrogate-optimal year at that thickness
         -> global/<tilt>_<info>/part_NNNN.npz                                (GPU, array)
  merge  every part of every scenario -> maps_<tilt>_<info>.csv (one row per cell: LCOW,
         yield, thickness, gap, tilt and schedule summaries, Koppen, OOD flag) and
         schedules_<tilt>_<info>.npz (the 366 x 5 daily controls of each cell's optimum,
         plus every cell's BO trace)                                                (CPU)

The array index enumerates scenarios (optimize.SCENARIOS, 12) x ``--n-chunks`` cell blocks;
scripts/submit_global.sh sizes the array. Parts whose output exists are skipped, so
resubmitting the same array resumes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sawh_bayesopt import climate
from sawh_bayesopt.optimize import SCENARIOS, THICKNESS_GRID_M, BOConfig, lcow_and_constraint, run_bo
from sawh_bayesopt.simulate import GAP_HEADROOM_M
from sawh_bayesopt.train import DEFAULT_RUN_DIR, scenario_context, surrogate_years

# Cells per batched surrogate call. Three-per-day tilt scores 136k candidates a day, ten
# times the daily set, so it takes a tenth of the batch.
BATCH = {"three_per_day": 2, "daily": 16, "seasonal": 8}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("run")
    p.add_argument("--task-index", type=int, required=True)
    p.add_argument("--n-chunks", type=int, required=True)
    p.add_argument("--cells", type=int, nargs="*", help="Explicit cells (smoke tests); overrides the block.")
    sub.add_parser("merge")
    args = ap.parse_args(argv)
    return {"run": stage_run, "merge": stage_merge}[args.stage](args)


def stage_run(args) -> int:
    tilt_mode, info = SCENARIOS[args.task_index // args.n_chunks]
    chunk = args.task_index % args.n_chunks
    out = args.run_dir / "global" / f"{tilt_mode}_{info}" / f"part_{chunk:04d}.npz"
    if out.exists():
        print(f"{out} exists")
        return 0
    ctx = scenario_context(args.run_dir)
    n_cells = len(ctx["features"]["lat"])
    cells = np.array(args.cells) if args.cells else np.array_split(np.arange(n_cells), args.n_chunks)[chunk]
    cfg = BOConfig(**json.loads((args.run_dir / "bo_config.json").read_text()))
    print(f"{tilt_mode}/{info}: {len(cells)} cells, {cfg}", flush=True)
    cand = ((THICKNESS_GRID_M - THICKNESS_GRID_M[0]) / np.ptp(THICKNESS_GRID_M))[:, None]

    def evaluate(requests):
        cc = np.array([cells[p] for p, _i in requests])
        hh = THICKNESS_GRID_M[[i for _p, i in requests]]
        yr = surrogate_years(ctx, cc, hh, tilt_mode, info, batch=BATCH[tilt_mode])
        res = []
        for k, h in enumerate(hh):
            water, capped = float(yr["water"][k].mean()), float(yr["capped"][k].mean())
            lcow, g = lcow_and_constraint(water, capped, h, ctx["econ"])
            res.append((lcow, g, (water, capped, yr["controls"][k].astype(np.float32))))
        return res

    hist = run_bo(evaluate, len(cells), cand, cfg, seed=args.seed)
    best = [int(np.argmin(np.where(h["feasible"], h["y"], np.inf))) if any(h["feasible"]) else int(np.argmin(h["y"]))
            for h in hist]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, cell=cells,
             thickness_m=np.array([THICKNESS_GRID_M[h["idx"][b]] for h, b in zip(hist, best)]),
             lcow=np.array([h["y"][b] for h, b in zip(hist, best)]),
             feasible=np.array([h["feasible"][b] for h, b in zip(hist, best)]),
             water=np.array([h["extra"][b][0] for h, b in zip(hist, best)]),
             capped_frac=np.array([h["extra"][b][1] for h, b in zip(hist, best)]),
             controls=np.stack([h["extra"][b][2] for h, b in zip(hist, best)]),
             trace_thickness_m=np.array([THICKNESS_GRID_M[h["idx"]] for h in hist]),
             trace_lcow=np.array([h["y"] for h in hist]),
             trace_feasible=np.array([h["feasible"] for h in hist]))
    print(f"wrote {out}")
    return 0


def stage_merge(args) -> int:
    f = climate.load_features(args.run_dir / "features.npz")
    sel = pd.read_csv(args.run_dir / "selection.csv")
    d2, ood, threshold = climate.mahalanobis_ood(f["descriptors"][sel.loc[sel["split"] == "train", "cell"]],
                                                 f["descriptors"])
    for tilt_mode, info in SCENARIOS:
        parts = sorted((args.run_dir / "global" / f"{tilt_mode}_{info}").glob("part_*.npz"))
        if not parts:
            print(f"{tilt_mode}/{info}: no parts yet")
            continue
        z = {k: np.concatenate([np.load(p)[k] for p in parts]) for k in np.load(parts[0]).files}
        c, u = z["cell"], z["controls"]
        tilt = u[..., 2:5]
        table = pd.DataFrame({
            "cell": c, "lat": f["lat"][c], "lon": f["lon"][c], "elevation_m": f["elevation_m"][c],
            "koppen": f["koppen"][c], "koppen_group": f["koppen_group"][c],
            "mahalanobis_d2": d2[c], "ood": ood[c],
            "lcow_usd_m3": z["lcow"], "feasible": z["feasible"], "water_kg_m2_day": z["water"],
            "capped_frac": z["capped_frac"], "hydrogel_thickness_mm": z["thickness_m"] * 1e3,
            "vapor_gap_mm": (z["thickness_m"] + GAP_HEADROOM_M) * 1e3,
            "tilt_mean_deg": tilt.mean(axis=(1, 2)), "tilt_std_deg": tilt.std(axis=(1, 2)),
            "tilt_intraday_range_deg": (tilt.max(axis=2) - tilt.min(axis=2)).mean(axis=1),
            "seal_offset_mean_h": u[..., 0].mean(axis=1), "open_offset_mean_h": u[..., 1].mean(axis=1),
            "n_evals": z["trace_lcow"].shape[1],
        }).sort_values("cell")
        stem = f"{tilt_mode}_{info}"
        table.to_csv(args.run_dir / f"maps_{stem}.csv", index=False)
        np.savez(args.run_dir / f"schedules_{stem}.npz", **z)
        print(f"{stem}: {len(table)} cells, median LCOW {table.loc[table.feasible, 'lcow_usd_m3'].median():.2f} "
              f"USD/m3, {int((~table.feasible).sum())} infeasible, OOD threshold d2 {threshold:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
