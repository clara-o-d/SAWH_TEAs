"""Gaussian-process surrogate for combined_lcow(design_vector).

scikit-learn's GaussianProcessRegressor, not BoTorch/GPyTorch: the problem is
6-D with a ~50-80 point evaluation budget, no GPU/multi-fidelity need, and
scikit-learn+scipy are already shared, lightweight dependencies across every
sibling solar_lumped/waste-heat_lumped package. BoTorch is the natural
upgrade for principled batch qEI or a joint multi-output (both-sites) model
if a future version needs it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from sawh_bayesopt.design_space import DesignBounds, VAR_ORDER, to_unit_cube


def build_gp(
    *,
    n_dims: int = len(VAR_ORDER),
    n_restarts_optimizer: int = 10,
    seed: int = 0,
) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * n_dims,
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(1e-3, (1e-8, 1e-1))
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=seed,
    )


@dataclass
class SurrogateState:
    gp: GaussianProcessRegressor
    bounds: DesignBounds
    X_raw: np.ndarray = field(default_factory=lambda: np.zeros((0, len(VAR_ORDER))))
    y: np.ndarray = field(default_factory=lambda: np.zeros((0,)))

    @property
    def y_best(self) -> float:
        if self.y.size == 0:
            return float("inf")
        return float(np.min(self.y))

    @property
    def x_best(self) -> np.ndarray:
        return self.X_raw[int(np.argmin(self.y))]


def append_observations(state: SurrogateState, X_new: np.ndarray, y_new: np.ndarray) -> SurrogateState:
    X_new = np.asarray(X_new, dtype=float).reshape(-1, len(VAR_ORDER))
    y_new = np.asarray(y_new, dtype=float).reshape(-1)
    X_all = np.vstack([state.X_raw, X_new]) if state.X_raw.size else X_new
    y_all = np.concatenate([state.y, y_new]) if state.y.size else y_new
    return SurrogateState(gp=state.gp, bounds=state.bounds, X_raw=X_all, y=y_all)


def fit(state: SurrogateState) -> SurrogateState:
    if state.X_raw.shape[0] < 2:
        raise ValueError("Need at least 2 observations to fit a GP.")
    u = to_unit_cube(state.X_raw, state.bounds)
    state.gp.fit(u, state.y)
    return state


def predict(state: SurrogateState, x: np.ndarray) -> tuple[float, float]:
    """(mu, sigma) of combined_lcow at raw (un-normalized) design vector x."""
    u = to_unit_cube(np.asarray(x, dtype=float).reshape(1, -1), state.bounds)
    mu, sigma = state.gp.predict(u, return_std=True)
    return float(mu[0]), float(sigma[0])


def predict_batch(state: SurrogateState, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = to_unit_cube(np.asarray(X, dtype=float).reshape(-1, len(VAR_ORDER)), state.bounds)
    mu, sigma = state.gp.predict(u, return_std=True)
    return mu, sigma


def save_state(state: SurrogateState, path: str | Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state.gp, path)
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "bounds": {name: list(getattr(state.bounds, name)) for name in VAR_ORDER},
                "X_raw": state.X_raw.tolist(),
                "y": state.y.tolist(),
                "kernel": str(state.gp.kernel_) if hasattr(state.gp, "kernel_") else str(state.gp.kernel),
            },
            indent=2,
        )
    )


def load_state(path: str | Path) -> SurrogateState:
    import joblib

    path = Path(path)
    gp = joblib.load(path)
    sidecar = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    bounds = DesignBounds(**{name: tuple(v) for name, v in sidecar["bounds"].items()})
    return SurrogateState(
        gp=gp,
        bounds=bounds,
        X_raw=np.array(sidecar["X_raw"], dtype=float),
        y=np.array(sidecar["y"], dtype=float),
    )
