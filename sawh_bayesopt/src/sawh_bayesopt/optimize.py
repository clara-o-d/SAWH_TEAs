"""Two-stage optimization on the surrogate: hydrogel thickness once per cell (outer Bayesian
optimization), operation every day (inner brute-force enumeration).

Inner level. A day's candidates are every seal/open schedule on the 15-minute grid crossed
with the tilt options of the scenario's tilt mode; one batched surrogate pass and an argmax
picks the day's control, and the year is walked day by day, each day's predicted end state
starting the next. A control is always *chosen* on the decision features (what the
controller knows) and *scored* on the real day. The walk is greedy -- it ignores what
today's choice does to tomorrow's start state; dynamic programming over a discretized state
would be the exact version.

Tilt modes:
  three_per_day  the sealed window in thirds, each at its own 15-degree level (125 combos);
  daily          one 5-degree level for the whole day (13);
  seasonal       one 5-degree level per calendar quarter, chosen at the quarter's start by
                 walking the quarter on the decision features under each level.
Information regimes (the decision features):
  hindsight      the day's own weather (perfect foresight; an upper bound);
  persistence    yesterday's weather, the day-ahead-forecast stand-in (day 1 sees Dec 31);
  monthly        the mean day of the day's calendar month;
  yearly         the mean day of the year.
Monthly and yearly means are taken over the same 2024 year -- a single-year stand-in for a
multi-year climatology.

Outer level. Constrained Bayesian optimization (Jones et al. 1998's EGO loop with
Gardner et al. 2014's constraint model): a Gaussian process on feasible LCOWs, a second GP
on the continuous constraint value for P(feasible), and an acquisition function maximized
over a finite candidate
set -- the 0.05 mm thickness grid in 1-D, a scrambled Sobol cloud in the 4-D validation
problem. Kernels, acquisitions and GP settings are chosen by train.stage_tune_bo against
grid-search ground truth. Cells advance in lockstep, so each round is one batched
surrogate evaluation across every cell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sawh_bayesopt.simulate import OPEN_GRID, SEAL_GRID, THICKNESS_BOUNDS_M, TILT3_GRID, TILT_GRID

TILT_MODES: tuple[str, ...] = ("three_per_day", "daily", "seasonal")
INFO_MODES: tuple[str, ...] = ("hindsight", "persistence", "monthly", "yearly")
SCENARIOS: tuple[tuple[str, str], ...] = tuple((t, i) for t in TILT_MODES for i in INFO_MODES)

# A design whose gel sits on the swelling ceiling (the gel/condenser clearance) on more
# than this share of days is infeasible: past it, uptake is set by the clearance rather
# than the isotherm, and in the real device that is brine reaching the condenser.
# ponytail: a blunt share-of-days threshold; a leakage model would replace it if one existed.
CAPPED_DAY_FRACTION_MAX = 0.05
PENALTY_LCOW_USD_PER_M3 = 1e4
# Day-1 surrogate rounds standing in for the physics' Aitken cyclic-state search.
WARMUP_ROUNDS = 8
# The outer search space: thickness on a 0.05 mm grid (fabrication resolution), so BO,
# grid-search ground truth and the global run all evaluate the same finite set.
THICKNESS_GRID_M = np.round(np.arange(THICKNESS_BOUNDS_M[0], THICKNESS_BOUNDS_M[1] + 1e-9, 5e-5), 7)

_PAIRS = np.array([(s, o) for s in SEAL_GRID for o in OPEN_GRID])


def candidates(tilt_mode: str) -> np.ndarray:
    """(n_cand, 5) controls a day chooses among, tilt-major. Seasonal returns the daily set;
    its quarter-level tilt choice reshapes it to (13, 1089, 5)."""
    if tilt_mode == "three_per_day":
        tilts = [(a, b, c) for a in TILT3_GRID for b in TILT3_GRID for c in TILT3_GRID]
    else:
        tilts = [(t, t, t) for t in TILT_GRID]
    return np.array([(s, o, *t) for t in tilts for s, o in _PAIRS])


def decision_features(features: np.ndarray, month: np.ndarray, info: str) -> np.ndarray:
    """What the controller knows when it picks day d's control, for (..., 366, F) features."""
    if info == "hindsight":
        return features
    if info == "persistence":
        return np.roll(features, 1, axis=-2)
    if info == "monthly":
        out = np.empty_like(features)
        for m in np.unique(month):
            sel = month == m
            out[..., sel, :] = features[..., sel, :].mean(axis=-2, keepdims=True)
        return out
    if info == "yearly":
        return np.broadcast_to(features.mean(axis=-2, keepdims=True), features.shape).copy()
    raise ValueError(f"unknown information regime {info!r}")


def make_year_fn(net, norm, capped_logit: bool, tilt_mode: str, month: np.ndarray):
    """Jitted (thickness (B,), feats (B, D, F), dec (B, D, F), state_guess (B, 2)) ->
    water (B, D), capped (B, D), controls (B, D, 5): the surrogate-optimal year of each
    (cell, thickness) under one scenario. ``net, norm, capped_logit`` from
    surrogates.deployable."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from sawh_bayesopt.surrogates import assemble_inputs

    lo, hi = norm["state_lo"], norm["state_hi"]
    cands = jnp.asarray(candidates(tilt_mode), jnp.float32)
    daily = jnp.asarray(candidates("daily"), jnp.float32)
    per_tilt = daily.reshape(len(TILT_GRID), len(_PAIRS), 5)
    q = (np.asarray(month) - 1) // 3
    edges = np.flatnonzero(np.r_[True, np.diff(q) != 0, True])
    quarters = tuple((int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]))

    def predict(thk, control, state, feats):
        x = assemble_inputs(thk, control, state, feats)
        out = jax.vmap(net)((x - norm["x_mean"]) / norm["x_std"])
        y = out[:, :3] * norm["y_std"] + norm["y_mean"]
        p = jax.nn.sigmoid(out[:, 3]) if capped_logit else jnp.clip(out[:, 3], 0.0, 1.0)
        return jnp.maximum(jnp.expm1(y[:, 0]), 0.0), y[:, 1:3], p

    def walk(thk, state0, dec, act, cset):
        """Greedy days: argmax over cset on dec, score the choice on act, carry the state."""
        def body(state, inp):
            d, a = inp
            k = jnp.argmax(predict(thk, cset, state, d)[0])
            w, s, p = predict(thk, cset[k][None], state, a)
            return jnp.clip(s[0], lo, hi), (w[0], p[0] > 0.5, cset[k])
        return jax.lax.scan(body, state0, (dec, act))

    def year(thk, feats, dec, guess):
        warm_set = daily if tilt_mode == "seasonal" else cands
        state0 = jax.lax.fori_loop(0, WARMUP_ROUNDS, lambda _i, s: walk(thk, s, dec[:1], feats[:1], warm_set)[0], guess)
        if tilt_mode != "seasonal":
            _end, (water, capped, controls) = walk(thk, state0, dec, feats, cands)
            return water, capped, controls
        parts, state = [], state0
        for a, b in quarters:
            # Plan the quarter under every level on the decision features alone, keep the
            # level that plans best, then live the quarter under it.
            planned = jax.vmap(lambda cs: walk(thk, state, dec[a:b], dec[a:b], cs)[1][0].sum())(per_tilt)
            state, out = walk(thk, state, dec[a:b], feats[a:b], per_tilt[jnp.argmax(planned)])
            parts.append(out)
        return tuple(jnp.concatenate([p[i] for p in parts]) for i in range(3))

    return eqx.filter_jit(jax.vmap(year))


def initial_states(thickness_m: np.ndarray) -> np.ndarray:
    """(n, 2) start-of-year state guess per thickness -- the (initial_loading, H0) the
    physics warm-up starts from. [c_w mol/m3, h mm]."""
    from sawh_bayesopt.simulate import system_config
    from solar_lumped.physics import initial_loading

    return np.array([(initial_loading(system_config(float(h), elevation_m=0.0)), float(h) * 1000.0)
                     for h in thickness_m])


def lcow_and_constraint(water: float, capped_frac: float, thickness_m: float, econ) -> tuple[float, float]:
    """(LCOW in USD/m3, constraint value g); feasible means g <= 0.

    g is the capped-day share over CAPPED_DAY_FRACTION_MAX -- a continuous constraint the
    outer BO models with its own GP -- or 1.0 (plainly infeasible) when the year has no
    finite positive yield. Infeasible designs score the penalty LCOW. Thickness enters
    LCOW twice -- through the yield and through the sorbent cost, which economics prices
    per gel lifetime -- which is the trade BO resolves."""
    from sawh_bayesopt.simulate import SALT_LOADING
    from solar_lumped.economics import FAIL_LCO, lcow_from_daily_yield

    lcow = lcow_from_daily_yield(water, salt_name="LiCl", salt_loading=SALT_LOADING,
                                 hydrogel_thickness_m=float(thickness_m), econ=econ) if water > 0.0 else FAIL_LCO
    if not np.isfinite(lcow) or lcow >= 0.99 * FAIL_LCO:
        return PENALTY_LCOW_USD_PER_M3, 1.0
    g = float(capped_frac) - CAPPED_DAY_FRACTION_MAX
    return (float(lcow) if g <= 0.0 else PENALTY_LCOW_USD_PER_M3), g


# --- Outer level: constrained BO, lockstep across problems ---

KERNELS: tuple[str, ...] = ("rbf", "matern12", "matern32", "matern52", "rational_quadratic", "matern52_plus_linear")
# (name, parameter): xi for EI/PI, kappa for LCB. "prediction" (exploit the mean) and
# "error" (explore the variance) are the two ends of the trade, as baselines
# (Kochenderfer & Wheeler, ch. 19 of the 2nd edition: prediction-based, error-based,
# lower confidence bound, probability of improvement, expected improvement).
ACQUISITIONS: tuple[tuple[str, float], ...] = (
    ("ei", 0.0), ("ei", 0.01), ("ei", 0.1), ("pi", 0.01), ("pi", 0.1),
    ("lcb", 1.0), ("lcb", 2.0), ("lcb", 3.0), ("prediction", 0.0), ("error", 0.0),
)


@dataclass(frozen=True)
class BOConfig:
    kernel: str = "matern52"
    acquisition: str = "ei"
    acq_param: float = 0.01
    n_init: int = 3
    budget: int = 12
    n_restarts: int = 2
    white_noise: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def run_bo(evaluate, n_problems: int, cand_unit: np.ndarray, cfg: BOConfig, *, seed: int, batch_size: int = 1,
           n_jobs: int = -1, log_every: int = 1) -> list[dict]:
    """Constrained BO on ``n_problems`` independent problems sharing one candidate set.

    ``cand_unit`` (m, dim) is the finite search set on the unit cube. ``evaluate`` takes a
    list of (problem, candidate index) and returns one (lcow, g, extra) per request, g the
    constraint value of lcow_and_constraint; every round's requests go in one call, which
    is where the batching lives. Initial points are a Latin hypercube (Kochenderfer &
    Wheeler sec. 16.3) snapped to the candidates. ``batch_size`` > 1 proposes that many per
    round by Kriging-Believer (Ginsbourger et al. 2010): fantasize the GP mean at each pick
    and refit before the next.

    Returns per problem: idx, y, g, feasible, extra (lists in evaluation order).
    """
    from joblib import Parallel, delayed
    from scipy.stats import qmc

    dim = cand_unit.shape[1]
    lhs = qmc.LatinHypercube(d=dim, seed=seed).random(cfg.n_init)
    init = [int(np.argmin(((cand_unit - u) ** 2).sum(axis=1))) for u in lhs]
    init = list(dict.fromkeys(init))  # a coarse 1-D grid can snap two points together
    hist = [{"idx": [], "y": [], "g": [], "feasible": [], "extra": []} for _ in range(n_problems)]
    pending = {p: init for p in range(n_problems)}
    rnd = 0
    while pending:
        requests = [(p, i) for p, idx in pending.items() for i in idx]
        for (p, i), (y, g, extra) in zip(requests, evaluate(requests)):
            h = hist[p]
            h["idx"].append(i), h["y"].append(y), h["g"].append(g), h["feasible"].append(g <= 0.0)
            h["extra"].append(extra)
        todo = [p for p in range(n_problems) if len(hist[p]["idx"]) < cfg.budget]
        props = Parallel(n_jobs=n_jobs)(
            delayed(propose)(cand_unit, hist[p]["idx"], hist[p]["y"], hist[p]["g"], cfg,
                             n=min(batch_size, cfg.budget - len(hist[p]["idx"])), seed=seed + 7919 * p + rnd)
            for p in todo)
        pending = dict(zip(todo, props))
        rnd += 1
        if log_every and rnd % log_every == 0:
            print(f"  BO round {rnd}: {len(todo)} problem(s) still searching", flush=True)
    return hist


def propose(cand_unit: np.ndarray, idx: list, y: list, g: list, cfg: BOConfig, *, n: int, seed: int) -> list[int]:
    """The next ``n`` candidate indices for one problem (see run_bo).

    P(feasible) is P(g <= 0) under a GP fitted to every evaluated constraint value
    (Gardner et al. 2014), not a classifier on the feasible/infeasible label: a Laplace GP
    classifier's probabilities stay soft (~0.2-0.3) even inside a region ringed by
    infeasible samples, and EI's large extrapolated variance there then outweighs them.

    Runs in a joblib worker: BLAS is held to one thread and the GP's hyperparameter-at-bound
    warnings are dropped (thousands per round)."""
    import warnings

    from scipy.stats import norm as normal
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel, DotProduct, Matern, RationalQuadratic, RBF, WhiteKernel,
    )
    from threadpoolctl import threadpool_limits

    dim = cand_unit.shape[1]
    idx, y, g = list(idx), np.asarray(y, float), np.asarray(g, float)
    feasible = g <= 0.0
    out: list[int] = []
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        p_feas = np.ones(len(cand_unit))
        if len(idx) >= 2 and not feasible.all():
            gp_g = GaussianProcessRegressor(ConstantKernel(1.0, (1e-3, 1e3)) * Matern([1.0] * dim, (1e-2, 1e2), nu=2.5)
                                            + WhiteKernel(1e-6, (1e-10, 1e-1)), normalize_y=True,
                                            n_restarts_optimizer=cfg.n_restarts, random_state=seed).fit(cand_unit[idx], g)
            mu_g, sd_g = gp_g.predict(cand_unit, return_std=True)
            p_feas = normal.cdf(-mu_g / np.maximum(sd_g, 1e-12))
        X_fit, y_fit = list(cand_unit[np.asarray(idx)[feasible]]), list(y[feasible])
        for _ in range(n):
            free = np.ones(len(cand_unit), bool)
            free[idx + out] = False
            if len(y_fit) < 2:
                # Nothing to regress yet: search for feasibility (Gelbart et al. 2014) -- the
                # candidate most likely feasible, the one farthest from everything tried
                # breaking ties (all of them, before any constraint value is infeasible).
                gap = ((cand_unit[:, None, :] - cand_unit[idx + out][None]) ** 2).sum(axis=2).min(axis=1)
                score = np.where(free, p_feas + 1e-6 * gap / (gap.max() + 1e-12), -np.inf)
                out.append(int(np.argmax(score)))
                continue
            amp = ConstantKernel(1.0, (1e-3, 1e3))
            base = {
                "rbf": lambda: RBF([1.0] * dim, (1e-2, 1e2)),
                "matern12": lambda: Matern([1.0] * dim, (1e-2, 1e2), nu=0.5),
                "matern32": lambda: Matern([1.0] * dim, (1e-2, 1e2), nu=1.5),
                "matern52": lambda: Matern([1.0] * dim, (1e-2, 1e2), nu=2.5),
                "rational_quadratic": lambda: RationalQuadratic(1.0, 1.0, (1e-2, 1e2), (1e-2, 1e2)),
                "matern52_plus_linear": lambda: Matern([1.0] * dim, (1e-2, 1e2), nu=2.5),
            }[cfg.kernel]()
            kernel = amp * base
            if cfg.kernel == "matern52_plus_linear":
                kernel = kernel + ConstantKernel(0.1, (1e-4, 1e2)) * DotProduct(1.0, (1e-3, 1e2))
            if cfg.white_noise:
                kernel = kernel + WhiteKernel(1e-6, (1e-10, 1e-1))
            gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=cfg.n_restarts,
                                          random_state=seed).fit(np.array(X_fit), np.array(y_fit))
            mu, sd = gp.predict(cand_unit, return_std=True)
            sd = np.maximum(sd, 1e-12)
            y_best = float(np.min(y_fit))
            gain = y_best - mu - cfg.acq_param
            if cfg.acquisition in ("ei", "pi", "error"):
                utility = {"ei": lambda: gain * normal.cdf(gain / sd) + sd * normal.pdf(gain / sd),
                           "pi": lambda: normal.cdf(gain / sd),
                           "error": lambda: sd}[cfg.acquisition]() * p_feas
            else:
                # Signed scores cannot be weighted by a probability; restrict to the region
                # the constraint model calls feasible instead (or its most-feasible point).
                score = -(mu - cfg.acq_param * sd) if cfg.acquisition == "lcb" else -mu
                ok = p_feas >= 0.5 if (p_feas >= 0.5).any() else p_feas == p_feas.max()
                utility = np.where(ok, score, -np.inf)
            utility = np.where(free, utility, -np.inf)
            k = int(np.argmax(utility))
            out.append(k)
            X_fit.append(cand_unit[k])
            y_fit.append(float(mu[k]))  # Kriging-Believer fantasy; only matters when n > 1
    return out


def bo_regret(hist: list[dict], optimum: np.ndarray) -> np.ndarray:
    """(n_problems, n_evals) relative simple regret: best feasible LCOW found after each
    evaluation over the known optimum, minus one. Infeasible-so-far reads as the penalty."""
    curves = []
    for h, opt in zip(hist, optimum):
        y = np.where(h["feasible"], h["y"], PENALTY_LCOW_USD_PER_M3)
        curves.append(np.minimum.accumulate(y) / opt - 1.0)
    width = max(len(c) for c in curves)
    return np.array([np.pad(c, (0, width - len(c)), mode="edge") for c in curves])
