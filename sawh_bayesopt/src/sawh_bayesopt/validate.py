"""Bayesian optimization with and without the physics surrogate, on the same problem at the
same held-out cells: ``python -m sawh_bayesopt.validate <stage>``.

The shared problem is the one true physics can afford: thickness plus one annual schedule
(seal offset, open offset, tilt -- held constant all year), four dimensions. Daily control
is out of reach for a physics optimizer (14k candidates x 366 days per evaluation), so it is
compared separately: the global run's daily designs and schedules are re-scored with physics.

  cells          test-split cells, stratified by Koppen group -> validate/cells.csv  (CPU)
  surrogate-bo   the tuned BO on the surrogate's annual-schedule year
                 -> validate/surrogate_bo.npz                                         (GPU)
  physics-bo     the same BO, same budget and batch, on true physics. One physics year per
                 round whatever the batch width (~2 h on an A100), so batches of 8 per
                 round; every evaluation is cached, and a resubmitted job replays the
                 deterministic search through the cache -> validate/physics_evals.npz  (GPU)
  score          physics years for the surrogate-BO optima and for every global scenario's
                 optimum at these cells -> validate/physics_scores.npz                (GPU)
  report         the comparison tables -> validate/validation.csv, validation_summary.json
                                                                                     (CPU)

Test cells are untouched by training, early stopping, model selection and BO tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sawh_bayesopt import climate
from sawh_bayesopt.optimize import SCENARIOS, BOConfig, lcow_and_constraint, run_bo
from sawh_bayesopt.simulate import OFFSET_BOUNDS_H, THICKNESS_BOUNDS_M, TILT_BOUNDS_DEG
from sawh_bayesopt.train import DEFAULT_CACHE, DEFAULT_RUN_DIR

BUDGET, N_INIT, BATCH_SIZE = 40, 8, 8
N_CANDIDATES = 2 ** 14


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("cells")
    p.add_argument("--n-cells", type=int, default=40)
    for name in ("surrogate-bo", "physics-bo", "score", "report"):
        sub.add_parser(name)
    args = ap.parse_args(argv)
    (args.run_dir / "validate").mkdir(parents=True, exist_ok=True)
    return {"cells": stage_cells, "surrogate-bo": stage_surrogate_bo, "physics-bo": stage_physics_bo,
            "score": stage_score, "report": stage_report}[args.stage](args)


def stage_cells(args) -> int:
    sel = pd.read_csv(args.run_dir / "selection.csv")
    test = sel[sel["split"] == "test"]
    rng = np.random.default_rng([args.seed, 3])
    share = args.n_cells / len(test)
    picked = pd.concat([g.loc[rng.choice(g.index, max(1, int(round(share * len(g)))), replace=False)]
                        for _k, g in test.groupby("koppen_group")]).sort_values("cell")
    picked.to_csv(args.run_dir / "validate" / "cells.csv", index=False)
    print(picked.groupby("koppen_group").size().to_string())
    return 0


def stage_surrogate_bo(args) -> int:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from sawh_bayesopt.optimize import initial_states
    from sawh_bayesopt.surrogates import assemble_inputs
    from sawh_bayesopt.train import scenario_context

    cells = pd.read_csv(args.run_dir / "validate" / "cells.csv")["cell"].to_numpy()
    cand_unit, cand = candidate_set(args.seed)
    ctx = scenario_context(args.run_dir)
    net, norm, capped_logit = ctx["net"], ctx["norm"], ctx["capped_logit"]

    def year(thk, control, feats, guess):
        """One annual schedule, every day: the surrogate year with no control search."""
        def step(state, a):
            x = assemble_inputs(thk, control[None], state, a)
            out = jax.vmap(net)((x - norm["x_mean"]) / norm["x_std"])
            y = out[:, :3] * norm["y_std"] + norm["y_mean"]
            p = jax.nn.sigmoid(out[:, 3]) if capped_logit else jnp.clip(out[:, 3], 0.0, 1.0)
            return jnp.clip(y[0, 1:3], norm["state_lo"], norm["state_hi"]), (jnp.maximum(jnp.expm1(y[0, 0]), 0.0), p[0] > 0.5)
        state0 = jax.lax.fori_loop(0, 8, lambda _i, s: step(s, feats[0])[0], guess)
        return jax.lax.scan(step, state0, feats)[1]

    years = eqx.filter_jit(jax.vmap(year))
    feats = ctx["features"]["day_features"]

    def evaluate(requests):
        x = cand[[i for _p, i in requests]]
        c = cells[[p for p, _i in requests]]
        water, capped = years(jnp.asarray(x[:, 0], jnp.float32), jnp.asarray(controls_of(x), jnp.float32),
                              jnp.asarray(feats[c]), jnp.asarray(initial_states(x[:, 0]), jnp.float32))
        water, capped = np.asarray(water).mean(axis=1), np.asarray(capped).mean(axis=1)
        return [(*lcow_and_constraint(float(w), float(cf), float(h), ctx["econ"]), (float(w), float(cf)))
                for w, cf, h in zip(water, capped, x[:, 0])]

    hist = run_bo(evaluate, len(cells), cand_unit, bo_config(args.run_dir), seed=args.seed, batch_size=BATCH_SIZE)
    np.savez(args.run_dir / "validate" / "surrogate_bo.npz", cell=cells, **history_arrays(hist, cand))
    return 0


def stage_physics_bo(args) -> int:
    from solar_lumped.economics import LCOEconomicParams

    path = args.run_dir / "validate" / "physics_evals.npz"
    cells = pd.read_csv(args.run_dir / "validate" / "cells.csv")["cell"].to_numpy()
    cand_unit, cand = candidate_set(args.seed)
    econ = LCOEconomicParams()
    cache: dict[tuple[int, int], tuple] = {}
    if path.exists():
        z = np.load(path)
        cache = {(int(p), int(i)): (float(w), float(cf)) for p, i, w, cf in zip(z["problem"], z["idx"], z["water"], z["capped"])}
        print(f"resuming with {len(cache)} cached physics evaluations", flush=True)

    def evaluate(requests):
        todo = [r for r in requests if r not in cache]
        if todo:
            x = cand[[i for _p, i in todo]]
            water, capped = physics_annual(args, cells[[p for p, _i in todo]], x[:, 0], controls_of(x))
            cache.update({r: (float(w), float(cf)) for r, w, cf in zip(todo, water, capped)})
            keys = list(cache)
            np.savez(path, problem=[k[0] for k in keys], idx=[k[1] for k in keys],
                     water=[cache[k][0] for k in keys], capped=[cache[k][1] for k in keys])
        return [(*lcow_and_constraint(cache[r][0], cache[r][1], float(cand[r[1], 0]), econ), cache[r]) for r in requests]

    hist = run_bo(evaluate, len(cells), cand_unit, bo_config(args.run_dir), seed=args.seed, batch_size=BATCH_SIZE)
    np.savez(args.run_dir / "validate" / "physics_bo.npz", cell=cells, **history_arrays(hist, cand))
    return 0


def stage_score(args) -> int:
    """True physics for what the surrogate side chose: the surrogate-BO optimum (one annual
    schedule) and each global scenario's design and daily schedule, at every validation
    cell, in one batched physics year."""
    cells = pd.read_csv(args.run_dir / "validate" / "cells.csv")["cell"].to_numpy()
    sb = np.load(args.run_dir / "validate" / "surrogate_bo.npz")
    labels, cc, hh, uu = [], [], [], []
    for k, c in enumerate(cells):
        labels.append(("surrogate_bo", int(c)))
        cc.append(c), hh.append(sb["best_x"][k, 0]), uu.append(np.broadcast_to(controls_of(sb["best_x"][k:k + 1])[0], (366, 5)))
    for tilt_mode, info in SCENARIOS:
        path = args.run_dir / f"schedules_{tilt_mode}_{info}.npz"
        if not path.exists():
            print(f"skipping {tilt_mode}/{info}: no merged global run")
            continue
        z = np.load(path)
        where = {int(c): j for j, c in enumerate(z["cell"])}
        for c in cells:
            j = where[int(c)]
            labels.append((f"{tilt_mode}_{info}", int(c)))
            cc.append(c), hh.append(z["thickness_m"][j]), uu.append(z["controls"][j])
    water, capped = physics_annual(args, np.array(cc), np.array(hh), np.stack(uu))
    np.savez(args.run_dir / "validate" / "physics_scores.npz", source=[s for s, _c in labels],
             cell=[c for _s, c in labels], thickness_m=hh, water=water, capped=capped)
    return 0


def stage_report(args) -> int:
    from solar_lumped.economics import LCOEconomicParams

    econ = LCOEconomicParams()
    v = args.run_dir / "validate"
    sb, pb, ps = np.load(v / "surrogate_bo.npz"), np.load(v / "physics_bo.npz"), np.load(v / "physics_scores.npz")
    meta = pd.read_csv(v / "cells.csv").set_index("cell")
    rows = []
    for k, c in enumerate(sb["cell"]):
        rows.append({"cell": int(c), "source": "surrogate_bo", "predicted_lcow": float(sb["best_y"][k]),
                     "thickness_mm": float(sb["best_x"][k, 0] * 1e3)})
        rows.append({"cell": int(c), "source": "physics_bo", "predicted_lcow": np.nan,
                     "physics_lcow": float(pb["best_y"][k]), "physics_feasible": bool(pb["best_feasible"][k]),
                     "thickness_mm": float(pb["best_x"][k, 0] * 1e3)})
    for tilt_mode, info in SCENARIOS:
        path = args.run_dir / f"maps_{tilt_mode}_{info}.csv"
        if path.exists():
            m = pd.read_csv(path).set_index("cell")
            for c in sb["cell"]:
                rows.append({"cell": int(c), "source": f"{tilt_mode}_{info}",
                             "predicted_lcow": float(m.loc[c, "lcow_usd_m3"]),
                             "thickness_mm": float(m.loc[c, "hydrogel_thickness_mm"])})
    table = pd.DataFrame(rows)
    scored = {(s, int(c)): (w, cf, h) for s, c, w, cf, h in zip(ps["source"], ps["cell"], ps["water"], ps["capped"],
                                                                 ps["thickness_m"])}
    for r in table.index:
        key = (table.at[r, "source"], int(table.at[r, "cell"]))
        if key in scored:
            w, cf, h = scored[key]
            lcow, g = lcow_and_constraint(float(w), float(cf), float(h), econ)
            table.at[r, "physics_lcow"], table.at[r, "physics_feasible"] = lcow, g <= 0.0
    table["koppen_group"] = meta.loc[table["cell"], "koppen_group"].to_numpy()
    table.to_csv(v / "validation.csv", index=False)

    pivot = table.pivot(index="cell", columns="source", values="physics_lcow")
    gap = pivot["surrogate_bo"] - pivot["physics_bo"]
    own = table[table.source == "surrogate_bo"]
    summary = {
        "n_cells": int(len(pivot)),
        "surrogate_bo_minus_physics_bo_usd_m3": {"median": float(gap.median()), "p10": float(gap.quantile(0.1)),
                                                 "p90": float(gap.quantile(0.9)),
                                                 "surrogate_bo_better_or_equal_frac": float((gap <= 0).mean())},
        # Positive mean bias here is the optimizer's winner's curse: it picks what the
        # surrogate overrates.
        "surrogate_lcow_rel_err_at_own_optimum_median": float(np.median(np.abs(own.predicted_lcow / own.physics_lcow - 1))),
        "surrogate_lcow_rel_bias_at_own_optimum_mean": float(np.mean(own.physics_lcow / own.predicted_lcow - 1)),
        "scenarios_physics_lcow_median": {s: float(pivot[s].median()) for s in pivot.columns},
        "scenarios_minus_physics_bo_median": {s: float((pivot[s] - pivot["physics_bo"]).median()) for s in pivot.columns},
    }
    (v / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


# --- Shared by the stages above ---


def candidate_set(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The 4-D search set: a scrambled Sobol cloud (unit cube) and its physical values --
    thickness snapped to the 0.05 mm grid, offsets to the 15-minute grid, tilt to 1 degree."""
    from scipy.stats import qmc

    u = qmc.Sobol(d=4, scramble=True, seed=seed + 11).random(N_CANDIDATES)
    lo = np.array([THICKNESS_BOUNDS_M[0], OFFSET_BOUNDS_H[0], OFFSET_BOUNDS_H[0], TILT_BOUNDS_DEG[0]])
    hi = np.array([THICKNESS_BOUNDS_M[1], OFFSET_BOUNDS_H[1], OFFSET_BOUNDS_H[1], TILT_BOUNDS_DEG[1]])
    x = lo + u * (hi - lo)
    x[:, 0] = np.round(x[:, 0] / 5e-5) * 5e-5
    x[:, 1:3] = np.round(x[:, 1:3] / 0.25) * 0.25
    x[:, 3] = np.round(x[:, 3])
    return (x - lo) / (hi - lo), x


def controls_of(x: np.ndarray) -> np.ndarray:
    """(n, 5) daily controls of 4-D points: seal, open, and one tilt for all three segments."""
    return np.column_stack([x[:, 1], x[:, 2], x[:, 3], x[:, 3], x[:, 3]])


def bo_config(run_dir: Path) -> BOConfig:
    """The tuned kernel and acquisition, at this problem's budget: 4-D needs more points than
    the 1-D thickness search the tuning was run on."""
    cfg = json.loads((run_dir / "bo_config.json").read_text())
    return BOConfig(**{**cfg, "n_init": N_INIT, "budget": BUDGET})


def physics_annual(args, cells: np.ndarray, thickness_m: np.ndarray, controls: np.ndarray):
    """Annual mean water and capped-day share per request, true physics. ``controls`` is
    (n, 5) for an annual schedule or (n, 366, 5) for a daily one. An instance the solver
    drops (step cap) scores as no water."""
    from sawh_bayesopt.simulate import simulate_years

    f = np.load(args.run_dir / "features.npz")
    frames = {c: climate.frame_from_cache(args.cache_dir, str(f["cache_key"][c])) for c in np.unique(cells)}
    day_index = np.flatnonzero(f["valid"][np.unique(cells)].all(axis=0))
    u = controls if controls.ndim == 3 else np.broadcast_to(controls[:, None], (len(cells), climate.N_DAYS, 5))
    rows = simulate_years([frames[c] for c in cells], thickness_m, np.ascontiguousarray(u[:, day_index]), day_index)
    water, capped = np.zeros(len(cells)), np.ones(len(cells))
    for i in np.unique(rows["instance"]):
        m = rows["instance"] == i
        water[i], capped[i] = rows["water"][m].mean(), rows["capped"][m].mean()
    return water, capped


def history_arrays(hist: list[dict], cand: np.ndarray) -> dict:
    best = [int(np.argmin(np.where(h["feasible"], h["y"], np.inf))) if any(h["feasible"]) else int(np.argmin(h["y"]))
            for h in hist]
    return {"trace_x": np.array([cand[h["idx"]] for h in hist]), "trace_y": np.array([h["y"] for h in hist]),
            "trace_feasible": np.array([h["feasible"] for h in hist]),
            "best_x": np.array([cand[h["idx"][b]] for h, b in zip(hist, best)]),
            "best_y": np.array([h["y"][b] for h, b in zip(hist, best)]),
            "best_feasible": np.array([h["feasible"][b] for h, b in zip(hist, best)])}


if __name__ == "__main__":
    raise SystemExit(main())
