"""The daily physics surrogate: g(design, control, start-of-day state, day features) ->
(water that day, end-of-day state, P(swelling cap)), trained on JAX physics years.

Physics outputs, not dollars: LCOW is applied afterwards (two_stage), so the economics
stay auditable and every map regenerates when cost assumptions change without touching
the surrogate.

Start-of-day state is an input because the physics chains it -- the gel starts each day
where yesterday left it -- and whether that memory matters (does absorption saturate
overnight?) is climate-dependent. Training years are chained under a *random control
every day*, so each simulated year is ~366 realistic training rows spanning the control
space, and the state distribution the surrogate sees is the one real operation produces.

Tilt is a daily control here, not a design constant: it moves solar gain (POA
transposition, in the profile) and gap convection (``tilt_deg`` in the system arrays), and
the second one is why the step function takes the system arrays as an argument
(jax_daily_cycle.make_day_step_fn) -- the tilt array is swapped per day without recompiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# The two-stage design block, chosen once per cell. Bounds are design_space's (the
# parameters.xlsx sweep ranges), so the surrogate and the true-physics reference BO span
# the same box.
DESIGN_VARS: tuple[str, ...] = ("hydrogel_thickness_m", "vapor_gap_m", "salt_loading")
CONTROL_VARS: tuple[str, ...] = ("seal_offset_h", "open_offset_h", "tilt_deg")
STATE_VARS: tuple[str, ...] = ("c_w_mol_m3", "h_mm")

# Control grids. 0.25 h is the 15-minute weather step (design_space.VAR_GRID says why a
# finer offset is invisible to the physics); 33 x 33 schedules per tilt level.
SEAL_GRID = np.arange(-4.0, 4.0 + 1e-9, 0.25)
OPEN_GRID = np.arange(-4.0, 4.0 + 1e-9, 0.25)
TILT_GRID = np.arange(0.0, 60.0 + 1e-9, 5.0)

def design_bounds() -> np.ndarray:
    """(3, 2) low/high of DESIGN_VARS, from design_space.DesignBounds."""
    from sawh_bayesopt.design_space import DesignBounds

    b = DesignBounds()
    return np.array([getattr(b, name) for name in DESIGN_VARS], dtype=float)


def random_controls(rng: np.random.Generator, n_inst: int, n_days: int) -> np.ndarray:
    """(n_inst, n_days, 3) controls drawn uniformly from the grids, independently per day."""
    return np.stack([rng.choice(SEAL_GRID, (n_inst, n_days)),
                     rng.choice(OPEN_GRID, (n_inst, n_days)),
                     rng.choice(TILT_GRID, (n_inst, n_days))], axis=-1)


def simulate_years(frames: list, designs: np.ndarray, controls: np.ndarray, day_index: np.ndarray,
                   *, case: str = "case2", progress_every: int = 30) -> dict:
    """True JAX physics for one chained year per instance, under per-day controls.

    ``frames[i]`` is instance i's weather frame (instances sharing a cell share the object),
    ``designs`` (n_inst, 3) raw DESIGN_VARS, ``controls`` (n_inst, n_days, 3) CONTROL_VARS,
    and ``day_index`` the (n_days,) day-of-year indices every frame covers -- the walk.

    Returns flat per-(instance, day) rows: instance, day, design, control, state (start of
    day), water, state_end, capped. Instances whose solver hit its step cap on any day are
    dropped whole, since their later states descend from a truncated day.

    Profiles are built lazily, one day at a time: a year of them for ~700 instances is tens
    of GB of Python floats. ponytail: that build is serial CPU pandas work interleaved with
    the GPU step; overlap it with a thread or process pool if it dominates the day time.
    """
    import dataclasses as dc

    from sawh_bayesopt import design_space
    from sawh_bayesopt.evaluator import _load_jax_daily_cycle
    from solar_lumped.physics import initial_loading
    from solar_lumped.simulation import SystemConfig
    from solar_lumped.weather import PHASE_DT_S, _steps_for, profile_from_day_df, site_elevation_m

    jdc = _load_jax_daily_cycle()
    n_inst, n_days = len(frames), len(day_index)
    by_frame: dict[int, dict] = {}
    for df in frames:
        if id(df) not in by_frame:
            by_frame[id(df)] = {d.timetuple().tm_yday - 1: g for d, g in df.groupby(df.index.date)}
    groups = [by_frame[id(df)] for df in frames]

    # Either phase is a subset of its day's rows, so the day's own step count bounds both
    # exactly -- and _pad_to truncates silently, so an estimate that ran short would be a
    # wrong answer rather than an error.
    n_max = 0
    for g in by_frame.values():
        for doy in day_index:
            day = g[int(doy)]
            native_dt = float(day.index.to_series().diff().dropna().dt.total_seconds().median())
            n_max = max(n_max, _steps_for(len(day), native_dt))

    configs = []
    for df, (thickness, gap, loading) in zip(frames, designs):
        x = np.array([thickness, gap, loading, TILT_GRID[0], 0.0, 0.0])
        cfg = SystemConfig(**design_space.to_system_config_kwargs(x, case=case))
        configs.append(dc.replace(cfg, site_elevation_m=site_elevation_m(df)))
    system = jdc.build_system_arrays(configs)
    day_step = jdc.make_day_step_fn(len(system), PHASE_DT_S, n_max, n_max)
    system_vals = tuple(system.values())
    tilt_at = list(system).index("tilt_deg")

    def step_fn(c_w, h, weather):
        vals = tuple(weather[-1] if k == tilt_at else v for k, v in enumerate(system_vals))
        return day_step(c_w, h, weather[:-1], vals)

    class _LazyDays:
        """The year's per-day weather tuples, built on access (see docstring)."""

        def __len__(self):
            return n_days

        def __getitem__(self, d):
            if d >= n_days:
                raise IndexError(d)
            profiles = [
                profile_from_day_df(groups[i][int(day_index[d])], seal_offset_h=float(c[0]),
                                    open_offset_h=float(c[1]), poa_tilt_deg=float(c[2]))
                for i, c in enumerate(controls[:, d])
            ]
            return jdc.build_day_weather(profiles, n_max, n_max) + (jnp.asarray(controls[:, d, 2]),)

    days = jdc.run_year_batched(
        step_fn, _LazyDays(),
        c_w_initial=np.array([initial_loading(c) for c in configs]),
        h_initial=np.array([c.hydrogel_thickness_m for c in configs]),
        progress_every=progress_every,
    )
    keep = days["ok"].all(axis=0)
    if not keep.all():
        print(f"    dropping {int((~keep).sum())}/{n_inst} instance(s) that hit the step cap", flush=True)
    inst = np.flatnonzero(keep)
    grid_inst, grid_day = np.meshgrid(inst, np.arange(n_days), indexing="ij")
    grid_inst, grid_day = grid_inst.ravel(), grid_day.ravel()

    def per_day(key, scale=1.0):
        return days[key][grid_day, grid_inst] * scale

    return {
        "instance": grid_inst,
        "day": np.asarray(day_index)[grid_day],
        "design": designs[grid_inst],
        "control": controls[grid_inst, grid_day],
        "state": np.column_stack([per_day("c_w_start"), per_day("h_start", 1000.0)]),
        "water": per_day("water"),
        "state_end": np.column_stack([per_day("c_w_end"), per_day("h_end", 1000.0)]),
        "capped": per_day("capped").astype(bool),
    }


# --- The surrogate ---


def assemble_inputs(design, control, state, features):
    """Model input row(s): design in mm/mm/- units, control, state, day features. Works on
    numpy or jax arrays, broadcasting leading axes."""
    xp = jnp if any(isinstance(a, jax.Array) for a in (design, control, state, features)) else np
    design = design * xp.asarray([1000.0, 1000.0, 1.0], dtype=design.dtype)
    shape = xp.broadcast_shapes(design.shape[:-1], control.shape[:-1], state.shape[:-1], features.shape[:-1])
    parts = [xp.broadcast_to(a, shape + a.shape[-1:]) for a in (design, control, state, features)]
    return xp.concatenate(parts, axis=-1)


class DailyNet(eqx.Module):
    """Shared trunk, separate heads: log1p(water), end state (2), swelling-cap logit."""
    trunk: eqx.nn.MLP
    water: eqx.nn.MLP
    state: eqx.nn.MLP
    capped: eqx.nn.MLP

    def __init__(self, n_in: int, width: int, key):
        k = jr.split(key, 4)
        act = jax.nn.silu
        self.trunk = eqx.nn.MLP(n_in, width, width, depth=2, activation=act, final_activation=act, key=k[0])
        self.water = eqx.nn.MLP(width, 1, width // 2, depth=1, activation=act, key=k[1])
        self.state = eqx.nn.MLP(width, 2, width // 2, depth=1, activation=act, key=k[2])
        self.capped = eqx.nn.MLP(width, 1, width // 2, depth=1, activation=act, key=k[3])

    def __call__(self, z):
        t = self.trunk(z)
        return self.water(t)[0], self.state(t), self.capped(t)[0]


class DailySurrogate(eqx.Module):
    """An ensemble of DailyNets plus the normalization it was trained under. A pytree, so
    it goes into jitted functions as an argument rather than as baked-in constants.

    ``members`` is one DailyNet whose array leaves carry a leading ensemble axis. The
    ensemble spread is the predictive uncertainty the active round samples on."""
    members: DailyNet
    norm: dict
    hyper: tuple = eqx.field(static=True)  # sorted (key, value) pairs; static must hash

    @property
    def state_lo(self):
        return self.norm["state_lo"]

    @property
    def state_hi(self):
        return self.norm["state_hi"]


def predict(model: DailySurrogate, x):
    """Mean/std across members for (N, n_in) raw inputs. Pure and jit-able in ``x``.

    water_mean / water_std (kg/m2/day), state_mean (N, 2), p_capped (N,).
    """
    n = model.norm
    z = (x - n["x_mean"]) / n["x_std"]
    w, s, c = eqx.filter_vmap(lambda m: jax.vmap(m)(z))(model.members)
    water = jnp.maximum(jnp.expm1(w * n["water_std"] + n["water_mean"]), 0.0)
    state = s * n["state_std"] + n["state_mean"]
    return {"water_mean": water.mean(0), "water_std": water.std(0),
            "state_mean": state.mean(0), "p_capped": jax.nn.sigmoid(c).mean(0)}


def fit_ensemble(X: np.ndarray, water: np.ndarray, state_end: np.ndarray, capped: np.ndarray,
                 cell: np.ndarray, *, n_members: int = 5, width: int = 128, steps: int = 20_000,
                 batch: int = 4096, lr: float = 2e-3, seed: int = 0) -> DailySurrogate:
    """Fit the ensemble. Each member trains on a bootstrap over *cells* (Poisson(1) weight
    per cell), not rows: resampling rows would let every member see every climate and the
    spread would stop meaning "unsure about climates like this one".

    Adam with cosine decay, written out (a dozen lines) rather than adding optax. Loss is
    MSE on standardized log1p(water) and end state plus BCE on the cap flag, all weighted
    equally.
    """
    rng = np.random.default_rng(seed)
    y_water = np.log1p(np.maximum(water, 0.0))
    norm = {
        "x_mean": X.mean(0), "x_std": X.std(0) + 1e-6,
        "water_mean": y_water.mean(), "water_std": y_water.std() + 1e-6,
        "state_mean": state_end.mean(0), "state_std": state_end.std(0) + 1e-6,
        # Chained predictions are clipped to the state range training actually covered.
        "state_lo": np.minimum(state_end.min(0), X[:, 6:8].min(0)),
        "state_hi": np.maximum(state_end.max(0), X[:, 6:8].max(0)),
    }
    norm = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in norm.items()}
    Xz = jnp.asarray((X - np.asarray(norm["x_mean"])) / np.asarray(norm["x_std"]), dtype=jnp.float32)
    Yw = jnp.asarray((y_water - float(norm["water_mean"])) / float(norm["water_std"]), dtype=jnp.float32)
    Ys = jnp.asarray((state_end - np.asarray(norm["state_mean"])) / np.asarray(norm["state_std"]),
                     dtype=jnp.float32)
    Yc = jnp.asarray(capped, dtype=jnp.float32)
    cells, cell_idx = np.unique(cell, return_inverse=True)
    cell_w = jnp.asarray(rng.poisson(1.0, size=(n_members, len(cells))), dtype=jnp.float32)
    cell_idx = jnp.asarray(cell_idx)

    members = eqx.filter_vmap(lambda k: DailyNet(X.shape[1], width, k))(jr.split(jr.PRNGKey(seed), n_members))
    params, static = eqx.partition(members, eqx.is_array)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-8

    def member_loss(p, w, xb, wb, sb, cb):
        net = eqx.combine(p, static)
        pw, ps, pc = jax.vmap(net)(xb)
        per_row = ((pw - wb) ** 2 + ((ps - sb) ** 2).mean(-1)
                   + jnp.maximum(pc, 0) - pc * cb + jnp.log1p(jnp.exp(-jnp.abs(pc))))
        return (w * per_row).sum() / (w.sum() + 1e-6)

    # Data goes in as arguments: closed over, a multi-million-row array is baked into the
    # compiled program as a constant.
    @jax.jit
    def train_step(params, m, v, t, key, data):
        Xz, Yw, Ys, Yc, cell_w, cell_idx = data
        idx = jr.randint(key, (batch,), 0, Xz.shape[0])
        w = cell_w[:, cell_idx[idx]]
        loss, g = jax.vmap(jax.value_and_grad(member_loss), in_axes=(0, 0, None, None, None, None))(
            params, w, Xz[idx], Yw[idx], Ys[idx], Yc[idx])
        lr_t = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * t / steps))
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
        params = jax.tree_util.tree_map(
            lambda p, a, b: p - lr_t * (a / (1 - b1 ** t)) / (jnp.sqrt(b / (1 - b2 ** t)) + eps),
            params, m, v)
        return params, m, v, loss.mean()

    data = (Xz, Yw, Ys, Yc, cell_w, cell_idx)
    key = jr.PRNGKey(seed + 1)
    for t in range(1, steps + 1):
        key, sub = jr.split(key)
        # float32 step count: a Python int would be int64 under x64 (jax_physics turns it
        # on at import), and b1 ** t would then silently promote the params to float64.
        params, m, v, loss = train_step(params, m, v, jnp.float32(t), sub, data)
        if t % max(steps // 10, 1) == 0:
            print(f"    step {t}/{steps}  loss {float(loss):.4f}", flush=True)
    hyper = {"n_in": int(X.shape[1]), "width": width, "n_members": n_members, "seed": seed}
    return DailySurrogate(eqx.combine(params, static), norm, tuple(sorted(hyper.items())))


def save_surrogate(model: DailySurrogate, directory: str | Path) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(d / "members.eqx", model.members)
    np.savez(d / "norm.npz", **{k: np.asarray(v) for k, v in model.norm.items()})
    (d / "hyper.json").write_text(json.dumps(dict(model.hyper), indent=2))


def load_surrogate(directory: str | Path) -> DailySurrogate:
    d = Path(directory)
    hyper = json.loads((d / "hyper.json").read_text())
    skeleton = eqx.filter_vmap(lambda k: DailyNet(hyper["n_in"], hyper["width"], k))(
        jr.split(jr.PRNGKey(0), hyper["n_members"]))
    members = eqx.tree_deserialise_leaves(d / "members.eqx", skeleton)
    norm = {k: jnp.asarray(v) for k, v in np.load(d / "norm.npz").items()}
    return DailySurrogate(members, norm, tuple(sorted(hyper.items())))


# --- Checks ---


def chain_year(model: DailySurrogate, design, controls, features, state0):
    """Surrogate-only year for one instance under a *given* control sequence: each day's
    predicted end state feeds the next day, exactly as two_stage's inner loop runs it.
    Returns per-day predicted water. vmap over instances for many."""
    def body(state, inp):
        control, feat = inp
        p = predict(model, assemble_inputs(design, control, state, feat)[None])
        nxt = jnp.clip(p["state_mean"][0], model.state_lo, model.state_hi)
        return nxt, p["water_mean"][0]

    _, water = jax.lax.scan(body, state0, (controls, features))
    return water


def grouped_holdout_report(model: DailySurrogate, rows: dict, day_features: np.ndarray,
                           regime_by_cell: np.ndarray, *, chunk: int = 200_000) -> dict:
    """Error on climates never seen in training, broken out by regime.

    ``rows`` must already be restricted to held-out cells. Per regime:
      * daily: RMSE and median relative error of one-day water given the *true* start state;
      * annual: the surrogate chained through the whole year under the physics' own control
        sequence from the physics' day-1 state -- the number that says whether state error
        compounds, and the one that matters for LCOW;
      * design_ranking: within each cell, the rank correlation between surrogate and physics
        annual yield across that cell's designs, and whether the two agree on the sign of
        thickness's effect with vapor gap and salt loading held fixed (multiple regression).
        This is what the outer design search relies on;
      * direction: within each cell, the slope of daily water against thickness and against
        the sealed (desorption) window, fitted to physics and to the surrogate on the very
        same rows, and whether they agree -- plus which way the physics goes. Physics is
        the reference; no direction is assumed. Any confounding in a one-variable slope
        hits both sides identically, so agreement is a fair test of the response.

    ``rows["instance"]`` must be unique across the whole campaign (the driver offsets it).
    """
    X = assemble_inputs(rows["design"], rows["control"], rows["state"],
                        day_features[rows["cell"], rows["day"]])
    water_of = eqx.filter_jit(lambda mdl, x: predict(mdl, x)["water_mean"])
    pred_daily = np.concatenate([np.asarray(water_of(model, jnp.asarray(X[i:i + chunk], dtype=jnp.float32)))
                                 for i in range(0, len(X), chunk)])
    regime = regime_by_cell[rows["cell"]]
    report: dict = {"daily": {}, "annual": {}}
    for name in [*np.unique(regime), "all"]:
        sel = np.ones(len(regime), bool) if name == "all" else regime == name
        wet = sel & (rows["water"] > 0.05)
        report["daily"][name] = {
            "n_rows": int(sel.sum()),
            "rmse_kg_m2": float(np.sqrt(np.mean((pred_daily[sel] - rows["water"][sel]) ** 2))),
            "median_rel_err": float(np.median(np.abs(pred_daily[wet] / rows["water"][wet] - 1))) if wet.any() else None,
        }

    # Annual chained error, one instance at a time grouped by year length.
    uid = rows["instance"]
    order = np.lexsort((rows["day"], uid))
    starts = np.flatnonzero(np.r_[True, np.diff(uid[order]) != 0])
    by_len: dict[int, list] = {}
    for a, b in zip(starts, np.r_[starts[1:], len(order)]):
        by_len.setdefault(b - a, []).append(order[a:b])
    chained = eqx.filter_jit(jax.vmap(chain_year, in_axes=(None, 0, 0, 0, 0)))
    annual = []
    for groups in by_len.values():
        idx = np.stack(groups)
        feats = day_features[rows["cell"][idx], rows["day"][idx]]
        water = np.asarray(chained(model, jnp.asarray(rows["design"][idx[:, 0]]), jnp.asarray(rows["control"][idx]),
                                   jnp.asarray(feats), jnp.asarray(rows["state"][idx[:, 0]])))
        for k, g in enumerate(idx):
            c = int(rows["cell"][g[0]])
            annual.append((regime_by_cell[c], water[k].mean(), rows["water"][g].mean(), c, rows["design"][g[0]]))
    for name in sorted({a[0] for a in annual}) + ["all"]:
        errs = np.array([abs(p / t - 1) for r, p, t, _c, _x in annual if (name == "all" or r == name) and t > 0])
        report["annual"][name] = {"n_instances": int(len(errs)),
                                  "median_rel_err": float(np.median(errs)) if len(errs) else None,
                                  "p90_rel_err": float(np.quantile(errs, 0.9)) if len(errs) else None}

    from scipy.stats import spearmanr

    per_cell: dict[int, list] = {}
    for r, p, t, c, x in annual:
        per_cell.setdefault(c, []).append((p, t, x))
    ranking = []  # (regime, spearman, thickness-coefficient signs agree, physics coefficient > 0)
    for c, items in per_cell.items():
        if len(items) < 5:  # 4 regressors need at least 5 designs
            continue
        pred, true = np.array([i[0] for i in items]), np.array([i[1] for i in items])
        A = np.column_stack([np.ones(len(items)), np.array([i[2] for i in items])])
        b_true, b_pred = np.linalg.lstsq(A, true, rcond=None)[0][1], np.linalg.lstsq(A, pred, rcond=None)[0][1]
        ranking.append((regime_by_cell[c], spearmanr(pred, true)[0], np.sign(b_true) == np.sign(b_pred), b_true > 0))
    report["design_ranking"] = _by_regime(ranking, {"median_spearman": lambda v: float(np.median([x[1] for x in v])),
                                                    "thickness_sign_agreement": lambda v: float(np.mean([x[2] for x in v])),
                                                    "physics_thickness_positive": lambda v: float(np.mean([x[3] for x in v]))})

    duration = rows["control"][:, 1] - rows["control"][:, 0]
    direction = []  # (regime, (agree, physics>0) for thickness, same for duration)
    for c in np.unique(rows["cell"]):
        m = rows["cell"] == c
        out = []
        for x in (rows["design"][m, 0], duration[m]):
            s_true, s_pred = np.polyfit(x, rows["water"][m], 1)[0], np.polyfit(x, pred_daily[m], 1)[0]
            out.append((np.sign(s_true) == np.sign(s_pred), s_true > 0))
        direction.append((regime_by_cell[c], *out))
    report["direction"] = _by_regime(direction, {
        "thickness_agreement": lambda v: float(np.mean([x[1][0] for x in v])),
        "physics_thickness_positive": lambda v: float(np.mean([x[1][1] for x in v])),
        "sealed_window_agreement": lambda v: float(np.mean([x[2][0] for x in v])),
        "physics_sealed_window_positive": lambda v: float(np.mean([x[2][1] for x in v])),
    })
    return report


def _by_regime(items: list, stats: dict) -> dict:
    """Apply each named statistic to the items of each regime (item[0]) and to all of them."""
    out = {}
    for name in sorted({i[0] for i in items}) + ["all"]:
        v = [i for i in items if name == "all" or i[0] == name]
        out[name] = {"n_cells": len(v), **{k: f(v) for k, f in stats.items()}} if v else {"n_cells": 0}
    return out
