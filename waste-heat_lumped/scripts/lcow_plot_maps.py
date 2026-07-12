#!/usr/bin/env python3
"""Render LCOW global maps from a CSV produced by lcow_random_global_map.py.

Requires optional deps:  pip install -e ".[maps]"

Reads the per-site winner CSV and produces:

* ``--out-png``           — LCOW scatter map (log-scale colorbar, salt marker shapes)
* ``--out-variables-png`` — 2×3 panel: LCOW, absorption a_w, desorption a_w, T_gel,
                            yield, solar irradiance
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lcow_random_global_map import (  # noqa: E402
    SiteResult,
    _FAIL_LCO,
    plot_map,
    plot_variable_maps,
)


def load_results(csv_path: Path) -> list[SiteResult]:
    df = pd.read_csv(csv_path)

    def _bool(val) -> bool:
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() == "true"

    def _float(val) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    results: list[SiteResult] = []
    for _, row in df.iterrows():
        results.append(
            SiteResult(
                lat=_float(row["lat"]),
                lon=_float(row["lon"]),
                rh_high=_float(row["rh_high"]),
                rh_low=_float(row["rh_low"]),
                temp_high_c=_float(row["temp_high_c"]),
                temp_low_c=_float(row["temp_low_c"]),
                solar_irradiance_w_per_m2=_float(row["solar_irradiance_w_per_m2"]),
                gel_temperature_c=_float(row["gel_temperature_c"]),
                best_salt=str(row["best_salt"]),
                best_sl=_float(row["best_sl"]),
                best_lcow=_float(row["best_lcow"]),
                infeasible=_bool(row["infeasible"]),
                desorption_aw=_float(row.get("desorption_aw", float("nan"))),
                daily_yield_m3_per_m2=_float(row.get("daily_yield_m3_per_m2", float("nan"))),
                eta_thermal=_float(row.get("eta_thermal", float("nan"))),
                backend=str(row.get("backend", "waste_heat_lumped")),
            )
        )
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True, help="Winner CSV from lcow_random_global_map.py")
    p.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="LCOW map PNG (default: <csv_stem>.png next to CSV)",
    )
    p.add_argument(
        "--out-variables-png",
        type=Path,
        default=None,
        help="Variable panel PNG (default: <csv_stem>_variables.png)",
    )
    p.add_argument("--year", type=int, default=None, help="ERA5 year for map title (optional)")
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    results = load_results(args.csv)
    if not results:
        print("No rows in CSV.", file=sys.stderr)
        return 1

    stem = args.csv.with_suffix("")
    out_png = args.out_png or stem.with_name(stem.name + ".png")
    out_vars = args.out_variables_png or stem.with_name(stem.name + "_variables.png")
    year = args.year if args.year is not None else 2024

    feas = sum(
        1
        for r in results
        if not r.infeasible
        and math.isfinite(r.best_lcow)
        and r.best_lcow < 0.99 * _FAIL_LCO
    )
    print(f"Loaded {len(results)} site(s); {feas} feasible LCOW values.", flush=True)

    plot_map(results, out_png, year=year, n_sites=len(results))
    print(f"Wrote {out_png}", flush=True)
    plot_variable_maps(results, out_vars, year=year, n_sites=len(results))
    print(f"Wrote {out_vars}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
