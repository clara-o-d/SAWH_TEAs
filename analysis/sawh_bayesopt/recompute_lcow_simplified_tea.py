#!/usr/bin/env python3
"""Recompute each BayesOpt site's best design's LCOW under a "simplified" TEA
(economics.py's ``simplified=True``: drops the DI-water and non-acrylamide
polymer (APS/MBA/TEMED) hydrogel cost terms), from cache.jsonl -- no
GPU/Sherlock work needed, since cache.jsonl already has every evaluated
design's yield_kg_m2.

For each site: re-run lcow_from_daily_yield over every cached design's yield
under both the standard and simplified cost model, pick whichever cached
design is cheapest under each independently (the argmin can shift once
sorbent cost matters less relative to CAPEX), and compare the two optima.

Usage::

    python analysis/sawh_bayesopt/recompute_lcow_simplified_tea.py
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


def _recompute_site_lcow(site_result: dict, *, simplified: bool) -> float:
    if not site_result["feasible"]:
        return FAIL_LCO
    return lcow_from_daily_yield(
        site_result["yield_kg_m2"],
        salt_name="LiCl",  # run_bayesopt_sweep.py's fixed --salt default; not in the design vector
        salt_to_polymer_ratio=site_result.get("salt_to_polymer_ratio"),
        hydrogel_thickness_m=site_result["hydrogel_thickness_m"],
        econ=_ECON,
        simplified=simplified,
    )


def _best_combined_lcow(cache_path: Path, *, simplified: bool) -> tuple[float, tuple] | None:
    best = None  # (combined_lcow, design_vector)
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
            per_site = [_recompute_site_lcow(sr, simplified=simplified) for sr in site_results]
            combined = combine_site_lcows(
                tuple(_FeasibleLcow(sr["feasible"], lc) for sr, lc in zip(site_results, per_site)),
                combine_rule="mean",
                penalty=PENALTY_LCOW_USD_PER_M3,
            )
            if best is None or combined < best[0]:
                best = (combined, dv)
    return best


def recompute_site(cache_path: Path) -> dict | None:
    standard = _best_combined_lcow(cache_path, simplified=False)
    simplified = _best_combined_lcow(cache_path, simplified=True)
    if standard is None or simplified is None:
        return None
    standard_lcow, standard_dv = standard
    simplified_lcow, simplified_dv = simplified
    return {
        "standard_best_combined_lcow_usd_m3": standard_lcow,
        "standard_hydrogel_thickness_mm": standard_dv[0] * 1000.0,
        "simplified_best_combined_lcow_usd_m3": simplified_lcow,
        "simplified_hydrogel_thickness_mm": simplified_dv[0] * 1000.0,
        "lcow_reduction_frac": (standard_lcow - simplified_lcow) / standard_lcow,
        "optimal_design_changed": tuple(round(v, 6) for v in standard_dv)
        != tuple(round(v, 6) for v in simplified_dv),
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
    out_path = _HERE / "full_sweep_summary_simplified_tea.csv"
    out.to_csv(out_path, index=False)

    print(f"{len(out)} sites recomputed -> {out_path}")
    print(f"median LCOW reduction (simplified vs. standard TEA): {out['lcow_reduction_frac'].median():.2%}")
    print(f"mean LCOW reduction: {out['lcow_reduction_frac'].mean():.2%}")
    print(f"sites where the optimal cached design changed: "
          f"{int(out['optimal_design_changed'].sum())}/{len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
