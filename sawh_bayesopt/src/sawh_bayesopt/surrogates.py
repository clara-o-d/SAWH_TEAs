"""Surrogates of the daily physics map

    g(thickness, control, start-of-day state, day features) -> (water, end state, P(capped))

and the protocol they are compared under. The map is fitted on physics outputs, not
dollars: LCOW is applied afterwards (optimize.lcow_and_constraint), so every map regenerates
under new cost assumptions without retraining.

Four families, each over a hyperparameter grid (GRIDS):
  * ridge regression (Hoerl & Kennard 1970) on standardized inputs -- the linear baseline;
  * random forest (Breiman 2001), multi-output, grown in steps to record convergence;
  * histogram gradient boosting (Friedman 2001; Ke et al. 2017's binned trees), one model
    per target, grown in steps with early stopping on the validation cells;
  * a multilayer perceptron with SiLU activations, Adam (Kingma & Ba 2015) with cosine
    learning-rate decay, early stopping on the validation cells.

Protocol. Cells are split 70/15/15 into train/validation/test (climate.select_cells),
never rows: a cell's rows are thousands of correlated days of one climate. Every fit sees
the training cells only; hyperparameters and early stopping are chosen on the validation
cells; the test cells are scored once, by train.stage_compare, for the reported numbers.

Targets are log1p(water), the end-of-day state (c_w, h) and the swelling-cap flag, all
standardized except the flag. Every family predicts all four. The MLP fits the flag with a
logistic loss; the tree and linear families regress it, so their flag output is a
probability-like score in [0, 1] (Brier loss), thresholded at 0.5 like the MLP's.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

FAMILIES: tuple[str, ...] = ("ridge", "random_forest", "gradient_boosting", "mlp")
GRIDS: dict[str, list[dict]] = {
    "ridge": [{"alpha": float(a)} for a in 10.0 ** np.arange(-4, 5)],
    "random_forest": [{"max_features": mf, "min_samples_leaf": leaf}
                      for mf in (0.33, 0.66, 1.0) for leaf in (1, 5, 20)],
    "gradient_boosting": [{"learning_rate": lr, "max_leaf_nodes": leaves}
                          for lr in (0.03, 0.1, 0.3) for leaves in (31, 127, 511)],
    "mlp": [{"width": w, "depth": d, "lr": lr}
            for w in (64, 128, 256, 512) for d in (2, 3, 4) for lr in (1e-3, 3e-3)],
}
# A wet day, for relative errors: below this a 20% miss is a few grams.
WET_DAY_KG_M2 = 0.05


def fit_family(family: str, hyper: dict, data: dict, *, seed: int) -> tuple[dict, dict]:
    """Fit one configuration on data["train"], early-stopping on data["val"] where the family
    iterates. Returns (surrogate, convergence). ``data[split]`` holds X (raw inputs) and Y
    (raw targets, see targets()); a surrogate is the dict predict() and save() understand."""
    norm = normalization(data["train"]["X"], data["train"]["Y"])
    Xtr, Ytr = standardize(norm, data["train"]["X"], data["train"]["Y"])
    Xva, Yva = standardize(norm, data["val"]["X"], data["val"]["Y"])
    t0 = time.perf_counter()

    if family == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=hyper["alpha"]).fit(Xtr, Ytr)
        surrogate = {"family": family, "hyper": hyper, "norm": norm, "model": model}
        convergence = {}

    elif family == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        # Each tree bags a 10% row sample: a full bootstrap of ~2M rows per tree buys no
        # accuracy over 200k at this row count, and costs 10x the memory.
        model = RandomForestRegressor(n_estimators=0, max_features=hyper["max_features"],
                                      min_samples_leaf=hyper["min_samples_leaf"], max_samples=0.1,
                                      warm_start=True, n_jobs=-1, random_state=seed)
        trees, val_rmse = [], []
        for n_trees in range(25, 301, 25):
            model.set_params(n_estimators=n_trees).fit(Xtr, Ytr)
            surrogate = {"family": family, "hyper": hyper, "norm": norm, "model": model}
            trees.append(n_trees)
            val_rmse.append(_water_rmse(surrogate, data["val"]))
        convergence = {"iteration": np.array(trees), "val_water_rmse": np.array(val_rmse)}

    elif family == "gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        models = [HistGradientBoostingRegressor(learning_rate=hyper["learning_rate"],
                                                max_leaf_nodes=hyper["max_leaf_nodes"], max_iter=0,
                                                early_stopping=False, warm_start=True, random_state=seed)
                  for _ in range(Ytr.shape[1])]
        surrogate = {"family": family, "hyper": hyper, "norm": norm, "model": models}
        iters, val_rmse, best, since_best, step, patience = [], [], np.inf, 0, 50, 4
        for n_iter in range(step, 3001, step):
            for k, m in enumerate(models):
                m.set_params(max_iter=n_iter).fit(Xtr, Ytr[:, k])
            iters.append(n_iter)
            val_rmse.append(_water_rmse(surrogate, data["val"]))
            best, since_best = (val_rmse[-1], 0) if val_rmse[-1] < best else (best, since_best + 1)
            if since_best >= patience:
                break
        # Keep the best iteration count, not the last: refit to it from scratch (warm start
        # cannot remove trees).
        n_best = iters[int(np.argmin(val_rmse))]
        for k, m in enumerate(models):
            m.set_params(max_iter=n_best, warm_start=False).fit(Xtr, Ytr[:, k])
        convergence = {"iteration": np.array(iters), "val_water_rmse": np.array(val_rmse)}

    elif family == "mlp":
        surrogate, convergence = _fit_mlp(hyper, norm, Xtr, Ytr, Xva, Yva, data["val"], seed=seed)

    else:
        raise ValueError(f"unknown family {family!r}")

    surrogate["fit_seconds"] = time.perf_counter() - t0
    return surrogate, convergence


def _fit_mlp(hyper, norm, Xtr, Ytr, Xva, Yva, val_raw, *, seed, steps=20_000, batch=4096, eval_every=500):
    """Adam with cosine decay, written out (a dozen lines) rather than adding optax. Loss is
    MSE on the three standardized regression targets plus binary cross-entropy on the cap
    flag, weighted equally. The returned network is the validation-best checkpoint."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    net = eqx.nn.MLP(Xtr.shape[1], 4, hyper["width"], hyper["depth"], activation=jax.nn.silu, key=jr.PRNGKey(seed))
    params, static = eqx.partition(net, eqx.is_array)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, hyper["lr"]

    def loss_fn(p, xb, yb):
        out = jax.vmap(eqx.combine(p, static))(xb)
        logit = out[:, 3]
        bce = jnp.maximum(logit, 0) - logit * yb[:, 3] + jnp.log1p(jnp.exp(-jnp.abs(logit)))
        return (((out[:, :3] - yb[:, :3]) ** 2).mean(axis=1) + bce).mean()

    # Data goes in as arguments: closed over, a multi-million-row array is baked into the
    # compiled program as a constant.
    @jax.jit
    def train_step(p, m, v, t, key, X, Y):
        idx = jr.randint(key, (batch,), 0, X.shape[0])
        loss, g = jax.value_and_grad(loss_fn)(p, X[idx], Y[idx])
        lr_t = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * t / steps))
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
        p = jax.tree_util.tree_map(lambda q, a, b: q - lr_t * (a / (1 - b1 ** t)) / (jnp.sqrt(b / (1 - b2 ** t)) + eps),
                                   p, m, v)
        return p, m, v, loss

    val_loss = eqx.filter_jit(lambda p, X, Y: jnp.mean(jnp.stack(
        [loss_fn(p, X[i:i + 65536], Y[i:i + 65536]) for i in range(0, X.shape[0], 65536)])))
    X, Y = jnp.asarray(Xtr, jnp.float32), jnp.asarray(Ytr, jnp.float32)
    Xv, Yv = jnp.asarray(Xva, jnp.float32), jnp.asarray(Yva, jnp.float32)
    key = jr.PRNGKey(seed + 1)
    log = {"iteration": [], "train_loss": [], "val_loss": [], "val_water_rmse": []}
    best, best_params, running = np.inf, params, []
    for t in range(1, steps + 1):
        key, sub = jr.split(key)
        # float32 step count: a Python int would be int64 under x64 (jax_physics turns it
        # on at import), and b1 ** t would then silently promote the params to float64.
        params, m, v, loss = train_step(params, m, v, jnp.float32(t), sub, X, Y)
        running.append(loss)
        if t % eval_every == 0:
            vl = float(val_loss(params, Xv, Yv))
            surrogate = {"family": "mlp", "hyper": hyper, "norm": norm, "model": eqx.combine(params, static)}
            log["iteration"].append(t)
            log["train_loss"].append(float(np.mean(np.asarray(jnp.stack(running)))))
            log["val_loss"].append(vl)
            log["val_water_rmse"].append(_water_rmse(surrogate, val_raw))
            running = []
            if vl < best:
                best, best_params = vl, params
            print(f"    step {t}/{steps}  train {log['train_loss'][-1]:.4f}  val {vl:.4f}", flush=True)
    surrogate = {"family": "mlp", "hyper": hyper, "norm": norm, "model": eqx.combine(best_params, static)}
    return surrogate, {k: np.array(v) for k, v in log.items()}


def predict(surrogate: dict, X: np.ndarray, *, chunk: int = 262_144) -> dict:
    """Raw inputs -> water (kg/m2/day), state (n, 2), p_capped (n,). Numpy in, numpy out."""
    norm = surrogate["norm"]
    out = []
    for a in range(0, len(X), chunk):
        Xz = (X[a:a + chunk] - norm["x_mean"]) / norm["x_std"]
        if surrogate["family"] == "mlp":
            out.append(np.asarray(_mlp_forward(surrogate["model"], Xz.astype(np.float32))))
        elif surrogate["family"] == "gradient_boosting":
            out.append(np.column_stack([m.predict(Xz) for m in surrogate["model"]]))
        else:
            out.append(surrogate["model"].predict(Xz))
    z = np.concatenate(out)
    capped = 1.0 / (1.0 + np.exp(-z[:, 3])) if surrogate["family"] == "mlp" else np.clip(z[:, 3], 0.0, 1.0)
    y = z[:, :3] * norm["y_std"] + norm["y_mean"]
    return {"water": np.maximum(np.expm1(y[:, 0]), 0.0), "state": y[:, 1:3], "p_capped": capped}


def _mlp_forward(model, Xz):
    import equinox as eqx
    import jax

    return eqx.filter_jit(lambda mdl, x: jax.vmap(mdl)(x))(model, Xz)


def evaluate(surrogate: dict, split: dict) -> dict:
    """Daily and chained-annual errors on one split's rows, overall and by Koppen group.

    daily: water RMSE / MAE / R^2 and median relative error on wet days, given the *true*
    start-of-day state; state RMSE; cap-flag accuracy.
    annual: each instance's year chained through the surrogate under the physics' own
    controls from the physics' first kept state -- the number that says whether state error
    compounds, and the one LCOW depends on. Relative error of the annual mean yield.
    """
    pred = predict(surrogate, split["X"])
    rows = split["rows"]
    water = rows["water"]
    annual = chained_annual(surrogate, split)
    out = {}
    for group in [*np.unique(rows["koppen_group"]), "all"]:
        sel = np.ones(len(water), bool) if group == "all" else rows["koppen_group"] == group
        wet = sel & (water > WET_DAY_KG_M2)
        err = pred["water"][sel] - water[sel]
        a = annual if group == "all" else {k: v[annual["koppen_group"] == group] for k, v in annual.items()}
        rel = np.abs(a["pred"] / a["true"] - 1.0)[a["true"] > 0]
        out[str(group)] = {
            "n_rows": int(sel.sum()),
            "water_rmse": float(np.sqrt(np.mean(err ** 2))),
            "water_mae": float(np.mean(np.abs(err))),
            "water_r2": float(1.0 - np.sum(err ** 2) / np.sum((water[sel] - water[sel].mean()) ** 2)),
            "water_median_rel_err_wet": float(np.median(np.abs(pred["water"][wet] / water[wet] - 1))) if wet.any() else None,
            "state_rmse_c_w": float(np.sqrt(np.mean((pred["state"][sel, 0] - rows["state_end"][sel, 0]) ** 2))),
            "state_rmse_h_mm": float(np.sqrt(np.mean((pred["state"][sel, 1] - rows["state_end"][sel, 1]) ** 2))),
            "capped_accuracy": float(np.mean((pred["p_capped"][sel] > 0.5) == rows["capped"][sel])),
            "n_instances": int(len(rel)),
            "annual_median_rel_err": float(np.median(rel)) if len(rel) else None,
            "annual_p90_rel_err": float(np.quantile(rel, 0.9)) if len(rel) else None,
        }
    return out


def chained_annual(surrogate: dict, split: dict) -> dict:
    """Per instance: surrogate annual mean yield chained under the physics' controls, the
    physics annual mean, and the instance's Koppen group. Instances of equal length are
    chained together, one batched predict per day."""
    rows, X = split["rows"], split["X"]
    lo, hi = surrogate["norm"]["state_lo"], surrogate["norm"]["state_hi"]
    order = np.lexsort((rows["day"], rows["instance"]))
    starts = np.flatnonzero(np.r_[True, np.diff(rows["instance"][order]) != 0])
    by_len: dict[int, list] = {}
    for a, b in zip(starts, np.r_[starts[1:], len(order)]):
        by_len.setdefault(b - a, []).append(order[a:b])
    pred, true, group = [], [], []
    for groups in by_len.values():
        idx = np.stack(groups)  # (n_inst, n_days) row indices, day-sorted
        state = rows["state"][idx[:, 0]]
        water = np.zeros(idx.shape)
        for d in range(idx.shape[1]):
            x = X[idx[:, d]].copy()
            x[:, 6:8] = state
            p = predict(surrogate, x)
            water[:, d] = p["water"]
            state = np.clip(p["state"], lo, hi)
        pred.append(water.mean(axis=1))
        true.append(rows["water"][idx].mean(axis=1))
        group.append(rows["koppen_group"][idx[:, 0]])
    return {"pred": np.concatenate(pred), "true": np.concatenate(true), "koppen_group": np.concatenate(group)}


def _water_rmse(surrogate: dict, split: dict) -> float:
    return float(np.sqrt(np.mean((predict(surrogate, split["X"])["water"] - split["rows"]["water"]) ** 2)))


def assemble_inputs(thickness_m, control, state, features):
    """Model input row(s): thickness (mm), the 5 controls, the start-of-day state (c_w, h
    in mm), and the day features. Columns 6:8 are the state, which chaining overwrites.
    Works on numpy or jax arrays, broadcasting leading axes."""
    import jax
    import jax.numpy as jnp

    xp = jnp if any(isinstance(a, jax.Array) for a in (thickness_m, control, state, features)) else np
    thickness = xp.asarray(thickness_m)[..., None] * 1000.0
    shape = xp.broadcast_shapes(thickness.shape[:-1], control.shape[:-1], state.shape[:-1], features.shape[:-1])
    parts = [xp.broadcast_to(a, shape + a.shape[-1:]) for a in (thickness, control, state, features)]
    return xp.concatenate(parts, axis=-1)


def targets(rows: dict) -> np.ndarray:
    """(n, 4) raw targets: log1p(water), c_w_end, h_end (mm), capped."""
    return np.column_stack([np.log1p(np.maximum(rows["water"], 0.0)), rows["state_end"], rows["capped"].astype(float)])


def normalization(X: np.ndarray, Y: np.ndarray) -> dict:
    """Training-split statistics. The flag column is not standardized. state_lo/hi are the
    state range training covered, which chained predictions are clipped to."""
    return {"x_mean": X.mean(0), "x_std": X.std(0) + 1e-6,
            "y_mean": Y[:, :3].mean(0), "y_std": Y[:, :3].std(0) + 1e-6,
            "state_lo": np.minimum(Y[:, 1:3].min(0), X[:, 6:8].min(0)),
            "state_hi": np.maximum(Y[:, 1:3].max(0), X[:, 6:8].max(0))}


def standardize(norm: dict, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Yz = Y.copy()
    Yz[:, :3] = (Y[:, :3] - norm["y_mean"]) / norm["y_std"]
    return ((X - norm["x_mean"]) / norm["x_std"]).astype(np.float32), Yz.astype(np.float32)


def deployable(surrogate: dict):
    """(network, norm) for the inner loop's JAX predictor (optimize.predict_jax). A ridge
    model becomes a depth-0 MLP -- one Linear layer carrying its coefficients -- so ridge and
    the MLP share one compiled code path; the flag output stays a regressed score for ridge
    (``capped_logit`` False). Trees have no JAX form: at ~10^12 predictions per scenario the
    inner loop needs a GPU-resident model, so train.stage_compare deploys the best of these
    two families and reports the tree families' test scores alongside."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    norm = {k: jnp.asarray(v, jnp.float32) for k, v in surrogate["norm"].items()}
    if surrogate["family"] == "mlp":
        return surrogate["model"], norm, True
    if surrogate["family"] != "ridge":
        raise ValueError(f"{surrogate['family']} has no JAX form; deploy ridge or mlp")
    model = surrogate["model"]
    net = eqx.nn.MLP(model.coef_.shape[1], 4, 1, 0, activation=jax.nn.silu, key=jr.PRNGKey(0))
    net = eqx.tree_at(lambda n: (n.layers[0].weight, n.layers[0].bias), net,
                      (jnp.asarray(model.coef_, jnp.float32), jnp.asarray(model.intercept_, jnp.float32)))
    return net, norm, False


def save(surrogate: dict, directory: Path) -> None:
    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    meta = {"family": surrogate["family"], "hyper": surrogate["hyper"], "fit_seconds": surrogate["fit_seconds"]}
    np.savez(directory / "norm.npz", **surrogate["norm"])
    if surrogate["family"] == "mlp":
        import equinox as eqx

        meta["n_in"] = int(len(surrogate["norm"]["x_mean"]))
        eqx.tree_serialise_leaves(directory / "model.eqx", surrogate["model"])
    else:
        joblib.dump(surrogate["model"], directory / "model.joblib")
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))


def load(directory: Path) -> dict:
    import joblib

    meta = json.loads((directory / "meta.json").read_text())
    norm = dict(np.load(directory / "norm.npz"))
    if meta["family"] == "mlp":
        import equinox as eqx
        import jax
        import jax.random as jr

        h = meta["hyper"]
        skeleton = eqx.nn.MLP(meta["n_in"], 4, h["width"], h["depth"], activation=jax.nn.silu, key=jr.PRNGKey(0))
        model = eqx.tree_deserialise_leaves(directory / "model.eqx", skeleton)
    else:
        model = joblib.load(directory / "model.joblib")
    return {**meta, "norm": norm, "model": model}
