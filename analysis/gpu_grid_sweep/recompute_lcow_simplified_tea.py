#!/usr/bin/env python3
"""Recompute the GPU grid-sweep's per-site optimal LCOW under a "simplified"
TEA (economics.py's ``simplified=True``: drops the DI-water and non-acrylamide
polymer (APS/MBA/TEMED) hydrogel cost terms) and compares it against the
standard TEA, using the same cached ``mean_yield_kg_m2`` for every already-run
device combo -- no GPU/Sherlock work needed.

For each site (lat, lon): compute LCOW for every cached device combo under
both cost models, take the argmin under each independently (the cheapest
combo can shift once sorbent cost matters less), and report the standard vs.
simplified optimum.

Usage::

    python analysis/gpu_grid_sweep/recompute_lcow_simplified_tea.py \\
        --csv full_sweep.csv --out full_sweep_simplified_tea.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "solar_lumped" / "src"))

from solar_lumped.economics import LCOEconomicParams, lcow_from_daily_yield  # noqa: E402

_SALT_LOADING = 4.0  # fixed salt:polymer ratio used by the GPU sweep
_SALT_NAME = "LiCl"
_ECON = LCOEconomicParams()


def _lcow_column(df: pd.DataFrame, *, simplified: bool) -> pd.Series:
    return pd.Series(
        [
            lcow_from_daily_yield(
                yield_kg,
                salt_name=_SALT_NAME,
                salt_to_polymer_ratio=_SALT_LOADING,
                hydrogel_thickness_m=thickness_mm / 1000.0,
                econ=_ECON,
                simplified=simplified,
            )
            for yield_kg, thickness_mm in zip(df["mean_yield_kg_m2"], df["hydrogel_thickness_mm"])
        ],
        index=df.index,
    )


def recompute(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["lcow_standard_usd_m3"] = _lcow_column(df, simplified=False)
    df["lcow_simplified_usd_m3"] = _lcow_column(df, simplified=True)

    best_standard = df.loc[df.groupby(["lat", "lon"])["lcow_standard_usd_m3"].idxmin()]
    best_simplified = df.loc[df.groupby(["lat", "lon"])["lcow_simplified_usd_m3"].idxmin()]

    out = best_standard[["lat", "lon", "hydrogel_thickness_mm", "lcow_standard_usd_m3"]].merge(
        best_simplified[["lat", "lon", "hydrogel_thickness_mm", "lcow_simplified_usd_m3"]],
        on=["lat", "lon"],
        suffixes=("_standard", "_simplified"),
    )
    out["lcow_reduction_frac"] = (
        out["lcow_standard_usd_m3"] - out["lcow_simplified_usd_m3"]
    ) / out["lcow_standard_usd_m3"]
    out["optimal_thickness_changed"] = ~pd.Series(out["hydrogel_thickness_mm_standard"]).eq(
        out["hydrogel_thickness_mm_simplified"]
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=_HERE / "full_sweep.csv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = recompute(args.csv)
    out_path = args.out or args.csv.with_name(args.csv.stem + "_simplified_tea.csv")
    out.to_csv(out_path, index=False)

    print(f"{len(out)} sites -> {out_path}")
    print(f"median LCOW reduction (simplified vs. standard TEA): {out['lcow_reduction_frac'].median():.2%}")
    print(f"mean LCOW reduction: {out['lcow_reduction_frac'].mean():.2%}")
    print(f"sites where the optimal hydrogel thickness changed: "
          f"{int(out['optimal_thickness_changed'].sum())}/{len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
