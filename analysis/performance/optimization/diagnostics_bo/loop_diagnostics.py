#!/usr/bin/env python3
"""Optimization-loop diagnostics for a completed sawh_bayesopt run.

Answers: "is the search itself behaving sensibly, or is it stuck/thrashing?"
Replays a run's history.csv in the exact order it was evaluated (n_init
points, then batch_size-sized rounds -- read from config.json, written by
run_bayesopt.py) and, at each round boundary, refits a fresh GP on
everything observed so far. This is a *replay*, not a re-run of the search
itself -- it reconstructs what the GP looked like at each point in time
without needing the live loop to have been instrumented.

Produces, in <run-dir>/diagnostics/:
  - loop_best_so_far.png: incumbent best combined_lcow vs. evaluation index.
    Should drop steeply early, then flatten -- a curve that never flattens
    means the budget was too small or the space wasn't adequately searched.
  - loop_hyperparameters.png: GP kernel length-scales / signal variance /
    noise vs. round index. Should stabilize, not oscillate between bounds.
  - loop_acquisition_and_exploration.png: two panels --
      (a) the achieved EI of each round's proposed batch (computed from the
          GP fit on everything observed *before* that round), which should
          trend toward 0 as the search converges;
      (b) each round's within-batch spread (mean pairwise unit-cube
          distance) and distance from the round's centroid to the
          then-current incumbent best, which should trend from
          "spread out, far from incumbent" (exploration) toward "tight,
          close to incumbent" (exploitation).

Usage:
    python3 scripts/diagnostics/loop_diagnostics.py --run-dir outputs/runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sawh_bayesopt.acquisition import expected_improvement  # noqa: E402
from sawh_bayesopt.design_space import DesignBounds, VAR_ORDER, to_unit_cube  # noqa: E402
from sawh_bayesopt.surrogate import SurrogateState, build_gp, fit, predict_batch  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    return p.parse_args(argv)


def _load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(
            f"{config_path} not found -- loop_diagnostics.py needs n_init/batch_size/seed/ei_xi "
            "from a run's config.json (written by run_bayesopt.py). Re-run the optimization with "
            "the current scripts/run_bayesopt.py, or hand-write a config.json for an older run."
        )
    return json.loads(config_path.read_text())


def _load_history_in_order(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(run_dir / "history.csv").sort_values("index")
    X = df[list(VAR_ORDER)].to_numpy(dtype=float)
    y = df["combined_lcow"].to_numpy(dtype=float)
    return X, y


def _rounds(n_total: int, n_init: int, batch_size: int) -> list[tuple[int, int]]:
    """[(start, end), ...) index ranges into the evaluation-ordered history,
    matching bayesopt.py::run_bayesopt's own loop structure exactly."""
    rounds = [(0, n_init)]
    start = n_init
    while start < n_total:
        end = min(start + batch_size, n_total)
        rounds.append((start, end))
        start = end
    return rounds


def replay(X: np.ndarray, y: np.ndarray, bounds: DesignBounds, *, n_init: int, batch_size: int, seed: int, ei_xi: float) -> dict:
    n = len(y)
    rounds = _rounds(n, n_init, batch_size)

    best_so_far = []
    round_end_index = []
    hyperparam_trace = []
    acquisition_trace = []
    exploration_trace = []

    state = SurrogateState(gp=build_gp(seed=seed), bounds=bounds)
    for r, (start, end) in enumerate(rounds):
        if r > 0:
            # EI of this round's actually-proposed batch, under the GP fit on
            # everything observed strictly *before* this round.
            mu, sigma = predict_batch(state, X[start:end])
            ei = expected_improvement(mu, sigma, state.y_best, xi=ei_xi)
            acquisition_trace.append({"round": r, "max_ei_in_batch": float(np.max(ei)), "mean_ei_in_batch": float(np.mean(ei))})

            u_batch = to_unit_cube(X[start:end], bounds)
            u_incumbent = to_unit_cube(X[int(np.argmin(y[:start]))].reshape(1, -1), bounds)[0]
            pairwise = [
                float(np.linalg.norm(u_batch[i] - u_batch[j]))
                for i in range(len(u_batch)) for j in range(i + 1, len(u_batch))
            ]
            exploration_trace.append({
                "round": r,
                "mean_pairwise_unit_cube_dist": float(np.mean(pairwise)) if pairwise else 0.0,
                "mean_dist_to_incumbent": float(np.mean([np.linalg.norm(u - u_incumbent) for u in u_batch])),
            })

        state = fit(SurrogateState(gp=build_gp(seed=seed), bounds=bounds, X_raw=X[:end], y=y[:end]))
        kernel = state.gp.kernel_
        k1, white = kernel.k1, kernel.k2
        hyperparam_trace.append({
            "round": r,
            "n_observed": end,
            "signal_variance": float(k1.k1.constant_value),
            "length_scales": {name: float(ls) for name, ls in zip(VAR_ORDER, np.atleast_1d(k1.k2.length_scale))},
            "noise_level": float(white.noise_level),
        })

        for i in range(start, end):
            best_so_far.append(float(np.min(y[: i + 1])))
            round_end_index.append(i)

    return {
        "best_so_far": best_so_far,
        "round_end_index": round_end_index,
        "hyperparameter_trace": hyperparam_trace,
        "acquisition_trace": acquisition_trace,
        "exploration_trace": exploration_trace,
        "rounds": rounds,
    }


def plot_best_so_far(replay_result: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(replay_result["round_end_index"], replay_result["best_so_far"], "-o", markersize=3)
    ax.set_xlabel("evaluation index")
    ax.set_ylabel("incumbent best combined LCOW (USD/m^3)")
    ax.set_title("Best-so-far trace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hyperparameters(replay_result: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace = replay_result["hyperparameter_trace"]
    rounds = [t["round"] for t in trace]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    axes[0].plot(rounds, [t["signal_variance"] for t in trace], "-o", markersize=3)
    axes[0].set_title("signal variance (constant kernel)")
    axes[0].set_xlabel("round")

    for name in VAR_ORDER:
        axes[1].plot(rounds, [t["length_scales"][name] for t in trace], "-o", markersize=3, label=name)
    axes[1].set_title("Matern length scales (unit cube)")
    axes[1].set_xlabel("round")
    axes[1].legend(fontsize=6)
    axes[1].set_yscale("log")

    axes[2].plot(rounds, [t["noise_level"] for t in trace], "-o", markersize=3)
    axes[2].set_title("WhiteKernel noise level")
    axes[2].set_xlabel("round")
    axes[2].set_yscale("log")

    fig.suptitle("Hyperparameter convergence (should stabilize, not keep oscillating)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_acquisition_and_exploration(replay_result: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acq = replay_result["acquisition_trace"]
    expl = replay_result["exploration_trace"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if acq:
        rounds = [a["round"] for a in acq]
        axes[0].plot(rounds, [a["max_ei_in_batch"] for a in acq], "-o", markersize=3, label="max EI in batch")
        axes[0].plot(rounds, [a["mean_ei_in_batch"] for a in acq], "-o", markersize=3, label="mean EI in batch")
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("round")
    axes[0].set_title("Achieved EI of proposed batch\n(should trend toward 0 as search converges)")

    if expl:
        rounds = [e["round"] for e in expl]
        axes[1].plot(rounds, [e["mean_pairwise_unit_cube_dist"] for e in expl], "-o", markersize=3, label="within-batch spread")
        axes[1].plot(rounds, [e["mean_dist_to_incumbent"] for e in expl], "-o", markersize=3, label="dist. to incumbent")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("round")
    axes[1].set_title("Exploration -> exploitation\n(both should shrink over rounds)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir
    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(run_dir)
    bounds = DesignBounds(**{name: tuple(v) for name, v in config["bounds"].items()})
    X, y = _load_history_in_order(run_dir)
    print(f"Loaded {len(y)} evaluations from {run_dir}/history.csv, replaying in original order.", flush=True)

    replay_result = replay(
        X, y, bounds,
        n_init=config["n_init"], batch_size=config["batch_size"], seed=config["seed"], ei_xi=config["ei_xi"],
    )

    plot_best_so_far(replay_result, out_dir / "loop_best_so_far.png")
    plot_hyperparameters(replay_result, out_dir / "loop_hyperparameters.png")
    plot_acquisition_and_exploration(replay_result, out_dir / "loop_acquisition_and_exploration.png")

    report_path = out_dir / "loop_diagnostics_report.json"
    report_path.write_text(json.dumps({
        "n_rounds": len(replay_result["rounds"]),
        "final_best_so_far": replay_result["best_so_far"][-1],
        "hyperparameter_trace": replay_result["hyperparameter_trace"],
        "acquisition_trace": replay_result["acquisition_trace"],
        "exploration_trace": replay_result["exploration_trace"],
    }, indent=2))

    print(f"Wrote loop_best_so_far.png, loop_hyperparameters.png, loop_acquisition_and_exploration.png, "
          f"and {report_path.name} to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
