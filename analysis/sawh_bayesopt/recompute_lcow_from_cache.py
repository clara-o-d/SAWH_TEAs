#!/usr/bin/env python3
"""Recompute each BayesOpt site's best design under the CURRENT
economics.py, from its cache.jsonl -- no GPU/Sherlock work needed, since
cache.jsonl already has every evaluated design's yield_kg_m2 (unaffected by
an LCOW-formula fix) alongside the *stale* lcow computed at sweep time.

For each site: re-run lcow_from_daily_yield over every cached design's yield
with today's economics.py, re-pick whichever cached design is now cheapest,
and compare it against the design the original (possibly stale-objective)
EGO search actually reported as best. A site where these differ is a sign
the true optimum may have moved outside what was already explored -- worth
a real re-run; most won't, since a search that already covered the box
reasonably densely usually still has the new optimum among its samples.

Usage::

    python analysis/sawh_bayesopt/recompute_lcow_from_cache.py
"""

from __future__ import annotations

import json
import sys
from collections import namedtuple
from pathlib import Path

import pandas as pd

_FeasibleLcow = namedtuple("_FeasibleLcow", ["feasible", "lcow"])

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "solar_lumped" / "src"))
sys.path.insert(0, str(_HERE.parent.parent / "sawh_bayesopt" / "src"))

from sawh_bayesopt.evaluator import PENALTY_LCOW_USD_PER_M3, combine_site_lcows  # noqa: E402
from solar_lumped.economics import FAIL_LCO, LCOEconomicParams, lcow_from_daily_yield  # noqa: E402

_ECON = LCOEconomicParams()


def _recompute_site_lcow(site_result: dict) -> float:
    if not site_result["feasible"]:
        return FAIL_LCO
    return lcow_from_daily_yield(
        site_result["yield_kg_m2"],
        salt_name="LiCl",  # run_bayesopt_sweep.py's fixed --salt default; not in the design vector
        salt_to_polymer_ratio=site_result.get("salt_to_polymer_ratio"),
        hydrogel_thickness_m=site_result["hydrogel_thickness_m"],
        econ=_ECON,
    )


def recompute_site(cache_path: Path) -> dict | None:
    best = None  # (corrected_lcow, design_vector)
    original_best = None  # (original stale combined_lcow, design_vector) -- what the sweep actually reported
    with cache_path.open() as f:
        for line in f:
            rec = json.loads(line)
            result = rec["result"]
            dv = result["design_vector"]
            hydrogel_thickness_m, _, _, _, _, salt_to_polymer_ratio = dv
            site_results = [
                {**sr, "hydrogel_thickness_m": hydrogel_thickness_m, "salt_to_polymer_ratio": salt_to_polymer_ratio}
                for sr in result["site_results"]
            ]
            corrected_per_site = [_recompute_site_lcow(sr) for sr in site_results]
            corrected_combined = combine_site_lcows(
                tuple(_FeasibleLcow(sr["feasible"], lc) for sr, lc in zip(site_results, corrected_per_site)),
                combine_rule="mean",
                penalty=PENALTY_LCOW_USD_PER_M3,
            )
            if best is None or corrected_combined < best[0]:
                best = (corrected_combined, dv)
            if original_best is None or result["combined_lcow"] < original_best[0]:
                original_best = (result["combined_lcow"], dv, corrected_combined)

    if best is None:
        return None
    corrected_lcow, corrected_dv = best
    original_lcow, original_dv, original_dv_corrected_lcow = original_best
    # "regret": how much worse the sweep's *originally reported* design looks under the
    # corrected formula, vs. the best already-cached candidate under that same corrected
    # formula -- distinguishes "ranking flipped among near-tied candidates" (~0% regret)
    # from "the true optimum needs a real re-run" (large regret).
    regret_frac = (original_dv_corrected_lcow - corrected_lcow) / corrected_lcow
    return {
        "corrected_best_combined_lcow_usd_m3": corrected_lcow,
        "corrected_hydrogel_thickness_mm": corrected_dv[0] * 1000.0,
        "corrected_vapor_gap_mm": corrected_dv[1] * 1000.0,
        "corrected_fin_area_ratio": corrected_dv[3],
        "original_best_combined_lcow_usd_m3": original_lcow,
        "original_design_corrected_lcow_usd_m3": original_dv_corrected_lcow,
        "regret_frac": regret_frac,
        "design_changed": tuple(round(v, 6) for v in corrected_dv) != tuple(round(v, 6) for v in original_dv),
    }


def main() -> int:
    summary = pd.read_csv(_HERE / "full_sweep_summary.csv")
    rows = []
    for _, row in summary.iterrows():
        site_name = f"{row['lat']:+.4f}_{row['lon']:+.4f}"
        cache_paths = list((_HERE / "outputs").glob(f"task_*/{site_name}/cache.jsonl"))
        if not cache_paths:
            continue
        recomputed = recompute_site(cache_paths[0])
        if recomputed is None:
            continue
        rows.append({"lat": row["lat"], "lon": row["lon"], **recomputed})

    out = pd.DataFrame(rows)
    out_path = _HERE / "full_sweep_summary_corrected.csv"
    out.to_csv(out_path, index=False)

    frac_change = (out["original_best_combined_lcow_usd_m3"] - out["corrected_best_combined_lcow_usd_m3"]) / out["original_best_combined_lcow_usd_m3"]
    print(f"{len(out)} sites recomputed -> {out_path}")
    print(f"median LCOW reduction from the fix: {frac_change.median():.2%}")
    print(f"sites where the argmin design changed: {int(out['design_changed'].sum())}/{len(out)}")
    print(f"regret (originally-reported design vs. best-in-cache, both under corrected formula):")
    print(f"  median={out['regret_frac'].median():.2%}  mean={out['regret_frac'].mean():.2%}  "
          f"max={out['regret_frac'].max():.2%}  p90={out['regret_frac'].quantile(0.9):.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
