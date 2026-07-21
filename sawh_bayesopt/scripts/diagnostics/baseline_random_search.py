#!/usr/bin/env python3
"""Sanity baseline: does the Bayesian optimizer actually beat plain random
search under the same evaluation budget?

Runs *pure* IID-uniform random search (deliberately not the Latin-hypercube
sampler bayesopt.py uses for its own init -- LHS is already a mild
space-filling improvement over IID random, and the point of this baseline is
the crudest reasonable comparison) against the same true model
(solar_lumped/gpu_sweep's JAX fast path, via evaluator.evaluate_batch) and
the same weather-cached sites, for the same total number of evaluations a
BayesOpt run already used. A working BayesOpt setup should reach a lower (or
equal, this is stochastic) best LCOW using the same budget, and should also
reach any given LCOW threshold using fewer evaluations -- if random search
matches or beats it, something about the surrogate/acquisition loop isn't
adding value.

Usage:
    python3 scripts/diagnostics/baseline_random_search.py \\
        --bayesopt-run-dir outputs/runs/<run_id> --run-id random_baseline
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

from sawh_bayesopt.design_space import DesignBounds  # noqa: E402
from sawh_bayesopt.evaluator import EvalCache, evaluate_batch  # noqa: E402
from sawh_bayesopt.sites import ATACAMA, CAMBRIDGE, DEFAULT_SITES, fetch_monthly_profiles  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bayesopt-run-dir", type=Path, required=True, help="Completed run to compare against and copy config from.")
    p.add_argument("--run-id", type=str, default="random_baseline")
    p.add_argument("--eval-batch-size", type=int, default=6, help="How many random points to evaluate per evaluate_batch() call (parallelism knob only, doesn't affect the search itself).")
    p.add_argument("--seed", type=int, default=12345, help="Distinct from the BayesOpt run's own seed on purpose.")
    p.add_argument("--weather-cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    return p.parse_args(argv)


def random_uniform_design(n: int, bounds: DesignBounds, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bounds_arr = bounds.as_array()
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
    return rng.uniform(lo, hi, size=(n, len(lo)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bo_config_path = args.bayesopt_run_dir / "config.json"
    if not bo_config_path.is_file():
        raise SystemExit(f"{bo_config_path} not found -- need a completed BayesOpt run's config.json for a fair n_total/sites match.")
    bo_config = json.loads(bo_config_path.read_text())

    bounds = DesignBounds(**{name: tuple(v) for name, v in bo_config["bounds"].items()})
    site_by_name = {s.name: s for s in DEFAULT_SITES + (CAMBRIDGE, ATACAMA)}
    sites = tuple(site_by_name[name] for name in bo_config["sites"])
    n_total = bo_config["n_total"]

    run_dir = _REPO / "outputs" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**bo_config, "seed": args.seed, "search": "random_uniform"}, indent=2))

    site_profiles = {s.name: fetch_monthly_profiles(s, cache_dir=args.weather_cache_dir) for s in sites}
    cache = EvalCache(run_dir / "cache.jsonl")

    from solar_lumped.economics.params import LCOEconomicParams

    econ = LCOEconomicParams()

    print(f"Random search: n_total={n_total} sites={[s.name for s in sites]} seed={args.seed}", flush=True)
    X = random_uniform_design(n_total, bounds, seed=args.seed)

    history = []
    for start in range(0, n_total, args.eval_batch_size):
        batch = list(X[start : start + args.eval_batch_size])
        results = evaluate_batch(
            batch, cache=cache, sites=sites, site_profiles=site_profiles, econ=econ,
            combine_rule=bo_config["combine_rule"], resolution=bo_config["resolution"],
        )
        history.extend(results)
        print(f"  {len(history)}/{n_total} evaluated, best so far: {min(r.combined_lcow for r in history):.4f}", flush=True)

    lcows = [r.combined_lcow for r in history]
    best_so_far_random = list(np.minimum.accumulate(lcows))

    bo_history_csv = args.bayesopt_run_dir / "history.csv"
    comparison = {"random_search": {"n_total": n_total, "best_so_far": best_so_far_random, "final_best": best_so_far_random[-1]}}
    if bo_history_csv.is_file():
        import pandas as pd

        bo_lcows = pd.read_csv(bo_history_csv).sort_values("index")["combined_lcow"].to_numpy()
        best_so_far_bo = list(np.minimum.accumulate(bo_lcows))
        comparison["bayesopt"] = {"n_total": len(bo_lcows), "best_so_far": best_so_far_bo, "final_best": best_so_far_bo[-1]}
        comparison["bayesopt_beats_random"] = bool(best_so_far_bo[-1] <= best_so_far_random[-1])

        def _evals_to_reach(best_so_far: list[float], threshold: float) -> int | None:
            for i, v in enumerate(best_so_far):
                if v <= threshold:
                    return i + 1
            return None

        threshold = best_so_far_bo[-1]  # BayesOpt's own final best -- "how fast could random reach what BO reached"
        comparison["evals_for_bayesopt_to_reach_its_own_final_best"] = _evals_to_reach(best_so_far_bo, threshold)
        comparison["evals_for_random_to_reach_bayesopts_final_best"] = _evals_to_reach(best_so_far_random, threshold)

        plot_comparison(best_so_far_bo, best_so_far_random, run_dir / "diagnostics" / "bayesopt_vs_random.png")

    report_path = run_dir / "random_search_report.json"
    report_path.write_text(json.dumps(comparison, indent=2))
    print(f"Report written to {report_path}", flush=True)
    if "bayesopt_beats_random" in comparison:
        verdict = "BEATS" if comparison["bayesopt_beats_random"] else "DOES NOT BEAT"
        print(f"Verdict: BayesOpt {verdict} random search under the same budget.", flush=True)
    return 0


def plot_comparison(best_so_far_bo: list[float], best_so_far_random: list[float], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(best_so_far_bo) + 1), best_so_far_bo, "-o", markersize=3, label="Bayesian optimization")
    ax.plot(range(1, len(best_so_far_random) + 1), best_so_far_random, "-o", markersize=3, label="random search")
    ax.set_xlabel("evaluation count")
    ax.set_ylabel("incumbent best combined LCOW (USD/m^3)")
    ax.set_title("BayesOpt vs. random search under the same budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
