"""Two-stage optimization on the daily surrogate: design once per cell, control every day.

Inner loop (control): exact enumeration, not gradient descent. A day's candidates are the
33 x 33 seal/open schedules on the 15-minute grid, times 13 tilt levels when tilt is a
daily control -- ~10^3-10^4 candidates, one batched surrogate pass and an argmax. No
local-minimum risk, and a discrete schedule is what a real controller runs anyway. The
year is walked day by day, each day's predicted end state starting the next.

Outer loop (design): per-cell GP + constrained EI over (thickness, vapor gap, salt
loading), reusing the true-physics BO's own surrogate.py / acquisition.py machinery, with
cells advanced in lockstep so every round's designs across all cells go through one
vmapped inner sweep. Warm-started from the optima of already-solved climate neighbours.

Tilt at three levels -- fixed (the best of the 13 levels for the whole year, a design
choice), seasonal (re-chosen each quarter) and daily -- so the yield gain of each step up
is itself a result. Schedules are scored under four information regimes: hindsight (the
day's own weather), persistence (yesterday's weather: the day-ahead-forecast stand-in),
climatological (the month's mean day), and constant (one annual schedule -- the per-site
true-physics BO's own control space, so constant vs BO is the like-for-like optimality
gap). A schedule is always *chosen* on the decision features and *scored* on the real day.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from sawh_bayesopt.daily_surrogate import (
    DESIGN_VARS, OPEN_GRID, SEAL_GRID, TILT_GRID, DailySurrogate, assemble_inputs, design_bounds, predict,
)

TILT_MODES: tuple[str, ...] = ("fixed", "seasonal", "daily")
SCHEDULE_MODES: tuple[str, ...] = ("hindsight", "persistence", "climatological", "constant")

# A design whose gel sits on the swelling ceiling (the gel/condenser clearance) on more
# than this share of days is infeasible: past it, uptake is set by the clearance constant
# rather than the isotherm, and in the real device that is brine reaching the condenser.
# This is the climate-dependent deliquescence/over-swelling constraint -- high RH at high
# salt loading is what drives the gel there. ponytail: a blunt share-of-days threshold;
# a leakage model would replace it if one existed.
CAPPED_DAY_FRACTION_MAX = 0.05

# Surrogate warm-up rounds on day 1 standing in for the physics' Aitken cyclic-state
# search, so the year does not start from an arbitrary loading.
WARMUP_ROUNDS = 8

_PAIRS = np.array([(s, o) for s in SEAL_GRID for o in OPEN_GRID])


def candidates(tilts) -> np.ndarray:
    """(len(_PAIRS) * len(tilts), 3) seal/open/tilt candidates, tilt-major."""
    return np.array([(s, o, t) for t in tilts for s, o in _PAIRS])


def decision_features(features: np.ndarray, month: np.ndarray, schedule_mode: str) -> np.ndarray:
    """What the controller knows when it picks day d's schedule. Persistence rolls by one
    day, so day 1 sees Dec 31 of the same year -- a same-season stand-in for the unmodelled
    previous year, and one day out of 366."""
    if schedule_mode in ("hindsight", "constant"):
        return features
    if schedule_mode == "persistence":
        return np.roll(features, 1, axis=-2)
    if schedule_mode == "climatological":
        out = np.empty_like(features)
        for m in np.unique(month):
            sel = month == m
            out[..., sel, :] = features[..., sel, :].mean(axis=-2, keepdims=True)
        return out
    raise ValueError(f"unknown schedule_mode {schedule_mode!r}")


def evaluate_designs(model: DailySurrogate, requests: list[tuple[int, np.ndarray]], day_features: np.ndarray,
                     month: np.ndarray, *, tilt_mode: str, schedule_mode: str, econ, chunk: int = 16) -> list[dict]:
    """Surrogate-optimal year for each (cell, design) request, and its LCOW.

    ``day_features`` is (n_cells, n_days, F) with every day filled (see fill_days).
    Each result: water (annual mean kg/m2/day), water_std (ensemble spread, the mean of the
    daily spreads -- i.e. assuming members' errors are correlated across days, the
    conservative reading), capped_frac, controls (n_days, 3), lcow, feasible.
    """
    if schedule_mode == "constant" and tilt_mode != "fixed":
        raise ValueError("a constant annual schedule only pairs with fixed tilt")
    dec_all = decision_features(day_features, month, schedule_mode)
    quarters = _segments(month, tilt_mode)
    cell_year = eqx.filter_jit(jax.vmap(
        lambda mdl, d, f, df, s: _cell_year(mdl, d, f, df, s, tilt_mode=tilt_mode,
                                            schedule_mode=schedule_mode, segments=quarters),
        in_axes=(None, 0, 0, 0, 0)))

    out: list[dict] = []
    # A round of a few thousand design-years is one silent loop otherwise, and an hour of
    # silence reads exactly like a hung job. ~10 lines per call.
    n_calls = -(-len(requests) // chunk)
    every = max(1, n_calls // 10)
    t0 = time.perf_counter()
    for n, i in enumerate(range(0, len(requests), chunk), start=1):
        if n % every == 0 or n == n_calls:
            per = (time.perf_counter() - t0) / max(n - 1, 1)
            print(f"    designs {i}/{len(requests)} ({tilt_mode}/{schedule_mode})  "
                  f"~{per * (n_calls - n + 1) / 60:.0f} min left in this round", flush=True)
        part = requests[i:i + chunk]
        cells = np.array([c for c, _x in part])
        designs = np.array([x for _c, x in part], dtype=float)
        # Default float dtype, not a hard float32: jax_physics enables x64 at import, and
        # in a process that also ran physics a float32 scan carry would meet float64
        # predictions. Following the default keeps every carry one dtype either way.
        res = cell_year(model, jnp.asarray(designs), jnp.asarray(day_features[cells]),
                        jnp.asarray(dec_all[cells]), jnp.asarray(initial_states(designs)))
        res = {k: np.asarray(v) for k, v in res.items()}
        for j, x in enumerate(designs):
            water, capped = float(res["water"][j].mean()), float(res["capped"][j].mean())
            lcow, feasible = lcow_or_penalty(water, capped, x, econ)
            out.append({"water": water, "water_std": float(res["water_std"][j].mean()),
                        "capped_frac": capped, "controls": res["controls"][j], "lcow": lcow,
                        "feasible": feasible})
    return out


def _segments(month: np.ndarray, tilt_mode: str) -> tuple[tuple[int, int], ...]:
    """Static day ranges within which tilt is held: the whole year for fixed, quarters for
    seasonal. Daily tilt rides in the candidate set instead, so it needs no segments."""
    if tilt_mode == "seasonal":
        q = (np.asarray(month) - 1) // 3
        edges = np.flatnonzero(np.r_[True, np.diff(q) != 0, True])
        return tuple((int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]))
    return ((0, len(month)),)


def _cell_year(model, design, feats, dec, state_guess, *, tilt_mode, schedule_mode, segments):
    """One (design, cell) year on the surrogate. Python-level branching on the modes is
    static under jit; everything else traces."""
    all_cands = jnp.asarray(candidates(TILT_GRID), dtype=jnp.float32)
    per_tilt = all_cands.reshape(len(TILT_GRID), len(_PAIRS), 3)

    # Cyclic start: repeat day 1 under its own greedy choice until the state settles.
    warm_cands = all_cands if tilt_mode == "daily" else per_tilt[len(TILT_GRID) // 2]
    state0 = jax.lax.fori_loop(
        0, WARMUP_ROUNDS,
        lambda _i, s: _greedy(model, design, s, dec[:1], feats[:1], warm_cands)[1], state_guess)

    if schedule_mode == "constant":
        # One schedule for the whole year: every candidate is its own chained year. The
        # search carries only a running total -- per-day outputs for ~14k candidate years
        # would be a GB per chunk -- and the winner is re-walked for its days.
        def total(control):
            def body(carry, a):
                state, acc = carry
                p = predict(model, assemble_inputs(design, control[None], state, a))
                return (jnp.clip(p["state_mean"][0], model.state_lo, model.state_hi), acc + p["water_mean"][0]), None
            return jax.lax.scan(body, (state0, jnp.zeros((), state0.dtype)), feats)[0][1]

        best = all_cands[jnp.argmax(jax.vmap(total)(all_cands))]
        water, std, capped = _fixed(model, design, state0, feats, best)
        return {"water": water, "water_std": std, "capped": capped,
                "controls": jnp.broadcast_to(best, (feats.shape[0], 3))}

    if tilt_mode == "daily":
        (water, std, capped, k), _ = _greedy(model, design, state0, dec, feats, all_cands)
        return {"water": water, "water_std": std, "capped": capped, "controls": all_cands[k]}

    parts, state = [], state0
    for a, b in segments:  # fixed: one segment; seasonal: four, chained
        (w, s, c, k), ends = jax.vmap(lambda cands: _greedy(model, design, state, dec[a:b], feats[a:b], cands))(per_tilt)
        t = jnp.argmax(w.sum(axis=1))
        parts.append((w[t], s[t], c[t], per_tilt[t][k[t]]))
        state = ends[t]
    return {key: jnp.concatenate([p[i] for p in parts]) for i, key in
            enumerate(("water", "water_std", "capped", "controls"))}


def _greedy(model, design, state0, dec, act, cands):
    """Walk the days: argmax predicted water over ``cands`` on the decision features, score
    the chosen schedule on the actual day, carry the predicted end state.

    ponytail: greedy per day, so it ignores what today's choice does to tomorrow's starting
    state. validate-bo measures what that costs; DP over a discretized state is the
    upgrade if the gap says it matters."""
    def body(state, inp):
        d, a = inp
        choice = jnp.argmax(predict(model, assemble_inputs(design, cands, state, d))["water_mean"])
        p = predict(model, assemble_inputs(design, cands[choice][None], state, a))
        nxt = jnp.clip(p["state_mean"][0], model.state_lo, model.state_hi)
        return nxt, (p["water_mean"][0], p["water_std"][0], p["p_capped"][0] > 0.5, choice)

    end, out = jax.lax.scan(body, state0, (dec, act))
    return out, end


def _fixed(model, design, state0, act, control):
    """One control every day: (water, water_std, capped) per day."""
    def body(state, a):
        p = predict(model, assemble_inputs(design, control[None], state, a))
        nxt = jnp.clip(p["state_mean"][0], model.state_lo, model.state_hi)
        return nxt, (p["water_mean"][0], p["water_std"][0], p["p_capped"][0] > 0.5)

    return jax.lax.scan(body, state0, act)[1]


def initial_states(designs: np.ndarray) -> np.ndarray:
    """(n, 2) start-of-year state guess per design -- the same (initial_loading, H0) the
    physics warm-up starts from. [c_w mol/m3, h mm]."""
    from sawh_bayesopt import design_space
    from solar_lumped.physics import initial_loading
    from solar_lumped.simulation import SystemConfig

    out = []
    for th, gap, sl in designs:
        cfg = SystemConfig(**design_space.to_system_config_kwargs(np.array([th, gap, sl, 30.0, 0.0, 0.0])))
        out.append((initial_loading(cfg), th * 1000.0))
    return np.array(out)


def lcow_or_penalty(water: float, capped_frac: float, design: np.ndarray, econ) -> tuple[float, bool]:
    """LCOW (USD/m3) of a year's mean daily yield, or the true-physics BO's own finite
    penalty if the design is infeasible. Salt loading enters twice: through the yield (the
    surrogate) and through the sorbent cost (economics prices salt x loading x dry mass
    every gel lifetime), which is what makes it a real trade rather than "more is better"."""
    from sawh_bayesopt.evaluator import PENALTY_LCOW_USD_PER_M3
    from solar_lumped.economics import FAIL_LCO, lcow_from_daily_yield

    if capped_frac > CAPPED_DAY_FRACTION_MAX or not water > 0.0:
        return PENALTY_LCOW_USD_PER_M3, False
    lcow = lcow_from_daily_yield(water, salt_name="LiCl", salt_loading=float(design[2]),
                                 hydrogel_thickness_m=float(design[0]), econ=econ)
    if not np.isfinite(lcow) or lcow >= 0.99 * FAIL_LCO:
        return PENALTY_LCOW_USD_PER_M3, False
    return float(lcow), True


# --- Outer loop: design BO per cell, lockstep across cells ---


@dataclass(frozen=True)
class DesignBox:
    """DesignBounds' two methods surrogate.py / acquisition.py read, over DESIGN_VARS only.
    DesignBounds itself always carries simple mode's 6 dims."""
    lo_hi: np.ndarray

    def names(self) -> tuple[str, ...]:
        return DESIGN_VARS

    def as_array(self) -> np.ndarray:
        return self.lo_hi


def optimize_cells(model: DailySurrogate, cells: np.ndarray, day_features: np.ndarray, month: np.ndarray, *,
                   tilt_mode: str, schedule_mode: str, econ, n_init: int = 8, n_total: int = 24,
                   batch_size: int = 4, warm_starts: dict[int, list] | None = None, seed: int = 0,
                   de_maxiter: int = 100, de_popsize: int = 15, n_jobs: int = -1) -> dict[int, dict]:
    """Best design per cell: its evaluate_designs record plus ``design`` and ``n_evals``.

    Every cell starts from the same scrambled-Sobol set plus its own warm starts, then runs
    Kriging-Believer EI batches until ``n_total``. Proposals are CPU work per cell (a GP and
    a small DE in 3-D), parallelized across cells with joblib; evaluations are one vmapped
    surrogate sweep per round across every cell.
    """
    from joblib import Parallel, delayed
    from scipy.stats import qmc

    from sawh_bayesopt.surrogate import SurrogateState, append_observations, build_gp

    box = DesignBox(design_bounds())
    lo, hi = box.lo_hi[:, 0], box.lo_hi[:, 1]
    sobol = lo + qmc.Sobol(d=3, scramble=True, seed=seed).random(n_init) * (hi - lo)
    warm_starts = warm_starts or {}
    states = {int(c): SurrogateState(gp=build_gp(n_dims=3, seed=seed), bounds=box, X_raw=np.zeros((0, 3)))
              for c in cells}
    records: dict[int, list] = {int(c): [] for c in cells}
    pending = {int(c): [*[np.asarray(x, float) for x in warm_starts.get(int(c), [])], *sobol] for c in cells}

    while pending:
        requests = [(c, x) for c, xs in pending.items() for x in xs]
        results = evaluate_designs(model, requests, day_features, month, tilt_mode=tilt_mode,
                                   schedule_mode=schedule_mode, econ=econ)
        for (c, x), r in zip(requests, results):
            records[c].append({**r, "design": x})
        for c, xs in pending.items():
            new = records[c][-len(xs):]
            states[c] = append_observations(states[c], np.array([r["design"] for r in new]),
                                            np.array([r["lcow"] for r in new]),
                                            np.array([r["feasible"] for r in new]))
        todo = [c for c in pending if len(records[c]) < n_total]
        proposals = Parallel(n_jobs=n_jobs)(
            delayed(_propose)(states[c], min(batch_size, n_total - len(records[c])), seed + len(records[c]),
                              de_maxiter, de_popsize, lo, hi)
            for c in todo)
        pending = dict(zip(todo, proposals))
        print(f"  outer round: {len(todo)} cell(s) still searching", flush=True)

    best = {}
    for c, recs in records.items():
        i = int(np.argmin([r["lcow"] for r in recs]))
        best[c] = {**recs[i], "n_evals": len(recs)}
    return best


def _propose(state, n: int, seed: int, de_maxiter: int, de_popsize: int, lo, hi) -> list[np.ndarray]:
    """EI batch from a fitted GP, or random designs while too few are feasible to fit one."""
    from sawh_bayesopt.acquisition import propose_batch
    from sawh_bayesopt.bayesopt import _try_fit

    state, fitted = _try_fit(state, seed=seed)
    if not fitted:
        return list(lo + np.random.default_rng(seed).random((n, 3)) * (hi - lo))
    return propose_batch(state, batch_size=n, seed=seed, maxiter=de_maxiter, popsize=de_popsize)


def neighbor_warm_starts(descriptors: np.ndarray, solved: dict[int, dict], targets: np.ndarray, *,
                         k: int = 3) -> dict[int, list]:
    """For each target cell, the optimal designs of its ``k`` nearest solved cells in
    standardized climate-descriptor space."""
    from scipy.spatial import cKDTree

    z = (descriptors - descriptors.mean(0)) / (descriptors.std(0) + 1e-12)
    solved_ids = np.array(sorted(solved))
    tree = cKDTree(z[solved_ids])
    _d, idx = tree.query(z[targets], k=min(k, len(solved_ids)))
    idx = np.atleast_2d(idx).reshape(len(targets), -1)
    return {int(t): [solved[int(solved_ids[j])]["design"] for j in row] for t, row in zip(targets, idx)}


def fill_days(day_features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Invalid (missing) days take the nearest earlier valid day's features, then the
    nearest later one: the inner loop walks all 366 calendar days for every cell."""
    import pandas as pd

    out = np.where(valid[..., None], day_features, np.nan)
    n, d, f = out.shape
    flat = pd.DataFrame(out.transpose(1, 0, 2).reshape(d, n * f)).ffill().bfill()
    return flat.to_numpy().reshape(d, n, f).transpose(1, 0, 2)


# --- Active round ---


def active_requests(best: dict[int, dict], *, n_cells: int, n_perturb: int, rng: np.random.Generator,
                    perturb_frac: float = 0.10, random_day_frac: float = 0.3) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(cell, design, controls) physics requests at and around the surrogate optima.

    Cells are drawn with probability proportional to the relative ensemble spread of
    their optimal yield -- accuracy is needed near the optimum and about its value, where
    the ensemble is least sure. Each gets its optimum plus ``n_perturb`` designs within
    +-``perturb_frac`` of each span, all run under the surrogate's own chosen schedule with
    jitter (one grid step per control, or a fully random control on ``random_day_frac`` of
    days) so the new rows cover the neighbourhood of the operating point, not just the point.
    """
    from sawh_bayesopt.daily_surrogate import random_controls

    bounds = design_bounds()
    span = bounds[:, 1] - bounds[:, 0]
    ids = np.array(sorted(best))
    rel = np.array([best[c]["water_std"] / max(best[c]["water"], 1e-9) for c in ids])
    p = rel / rel.sum() if rel.sum() > 0 else None
    chosen = rng.choice(ids, size=min(n_cells, len(ids)), replace=False, p=p)
    grids = (SEAL_GRID, OPEN_GRID, TILT_GRID)
    out = []
    for c in chosen:
        x0 = np.asarray(best[c]["design"], float)
        designs = [x0, *[np.clip(x0 + rng.uniform(-perturb_frac, perturb_frac, 3) * span, bounds[:, 0], bounds[:, 1])
                         for _ in range(n_perturb)]]
        for x in designs:
            ctrl = np.array(best[c]["controls"], float)
            for j, g in enumerate(grids):
                step = g[1] - g[0]
                ctrl[:, j] = np.clip(ctrl[:, j] + rng.integers(-1, 2, len(ctrl)) * step, g[0], g[-1])
            rand = rng.random(len(ctrl)) < random_day_frac
            ctrl[rand] = random_controls(rng, 1, int(rand.sum()))[0]
            out.append((int(c), x, ctrl))
    return out
