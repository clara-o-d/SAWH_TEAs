#!/usr/bin/env python3
"""One-at-a-time (star) sweep across every material/heat-transfer/external/
financial parameter registered in ``parameter_sweep.py::make_sweep_params``.

Unlike ``parameter_sweep.py --params a b c``, which builds a *full factorial*
grid (``n_levels ** len(params)`` runs -- infeasible once more than a
handful of parameters are combined), this script varies exactly ONE
parameter at a time while holding every other parameter at its baseline
value, and writes every row with the FULL set of parameter columns (baseline
value filled in for whichever parameters aren't varying in that row). This
keeps ``tornado_plot.py``'s OAT pair-matching (which requires all *other*
input columns to match exactly) working correctly across the combined
dataset, at a cost of only ``n_params * n_levels`` simulation runs instead of
an exponential blow-up.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_SRC = _REPO / "src"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from parameter_sweep import (  # noqa: E402
    _apply_combo,
    _BASELINE_WATER_PRICE_USD_PER_M3,
    _metrics_from_result,
    _sweep_grid,
    make_sweep_params,
)
from run_solar_sim import (  # noqa: E402
    register_cyclic_warmup_arguments,
    register_solar_sim_arguments,
    resolve_solar_sim_arguments,
)
from solar_lumped.economics.params import LCOEconomicParams  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    register_solar_sim_arguments(ap)
    register_cyclic_warmup_arguments(ap)
    ap.set_defaults(weather_mode="baseline")
    ap.add_argument(
        "--n-levels",
        type=int,
        default=7,
        help="Points per parameter (odd numbers include the baseline exactly)",
    )
    ap.add_argument(
        "--params",
        nargs="*",
        default=None,
        help="Subset of parameter keys to sweep (default: all registered parameters)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO / "outputs" / "parameter_sweeps" / "full_oat_sweep.csv",
    )
    args = ap.parse_args()
    resolve_solar_sim_arguments(args, ap)

    base_args = copy.copy(args)
    base_econ = LCOEconomicParams()
    all_params = make_sweep_params(base_args, base_econ)
    if args.params:
        known = {p.key for p in all_params}
        unknown = [k for k in args.params if k not in known]
        if unknown:
            ap.error(f"Unknown sweep parameter(s): {', '.join(unknown)}")
        swept_params = [p for p in all_params if p.key in set(args.params)]
    else:
        swept_params = all_params

    baseline_row = {p.key: p.baseline for p in all_params}
    param_keys = [p.key for p in all_params]

    print(f"OAT star sweep: {len(swept_params)} parameters x up to {args.n_levels} levels each")

    rows: list[dict] = []

    # Shared baseline row (all parameters at their default value).
    water_price = baseline_row.get("water_price_usd_per_m3", _BASELINE_WATER_PRICE_USD_PER_M3)
    base_result = _apply_combo({}, base_args, base_econ)
    rows.append({**baseline_row, "swept_param": "baseline", **_metrics_from_result(base_result, water_price)})

    run_i = 0
    for sp in swept_params:
        values = _sweep_grid(sp, args.n_levels)
        for value in values:
            if abs(value - sp.baseline) < 1e-12:
                continue  # baseline row already recorded once, above
            run_i += 1
            row_params = dict(baseline_row)
            row_params[sp.key] = value
            water_price = row_params.get("water_price_usd_per_m3", _BASELINE_WATER_PRICE_USD_PER_M3)
            result = _apply_combo({sp.key: value}, base_args, base_econ)
            rows.append({**row_params, "swept_param": sp.key, **_metrics_from_result(result, water_price)})
        print(f"  [{sp.key}] done ({run_i} runs so far)", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        *param_keys,
        "swept_param",
        "daily_yield_kg_m2",
        "thermal_efficiency",
        "lcow_usd_per_m3",
        "capex_usd_per_m3",
        "opex_usd_per_m3",
        "npv_usd_per_m2",
        "payback_years_simple",
        "payback_years_discounted",
    ]
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
