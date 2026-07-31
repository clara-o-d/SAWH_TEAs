#!/usr/bin/env python3
"""Economic-parameter tornado sensitivity of LCOW at a single fixed site/yield.

No re-simulation: yield is supplied directly (e.g. a GPU-sweep site's optimal
``mean_yield_kg_m2``), so the LCOW equation's financial knobs -- discount
rate, device/hydrogel lifetime, investment/maintenance factors, utilization,
and the salt/polymer/water unit prices -- can be perturbed one-at-a-time and
re-costed directly via ``lcow_from_daily_yield``. None of them affect yield,
so this needs zero physics simulation.

Usage::

    python analysis/sensitivity/econ_tornado_site.py \\
        --site-label "Cambridge, MA" --yield-kg-m2 1.7145 --hydrogel-thickness-mm 3.25
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SENSITIVITY = Path(__file__).resolve().parent
_ANALYSIS = _SENSITIVITY.parent
_REPO_ROOT = _ANALYSIS.parent
for _p in (_SENSITIVITY, _ANALYSIS, _REPO_ROOT / "solar_lumped" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd

from comparison.lib.adapters import _replace_econ  # noqa: E402
from solar_lumped.economics import (  # noqa: E402
    LCOEconomicParams,
    get_salt_price_usd_per_kg,
    lcow_from_daily_yield,
)
from tornado_plot_solar import create_tornado_plot  # noqa: E402

_POLYMER_FIELDS: tuple[str, ...] = (
    "c_am_usd_per_kg", "c_aps_usd_per_kg", "c_mba_usd_per_kg", "c_temed_usd_per_kg",
)

_PARAM_LABELS: dict[str, str] = {
    "discount_rate": "Discount rate\n(i)",
    "device_lifetime_years": "Device lifetime\n(L, yr)",
    "total_investment_factor": "Total investment\nfactor (f_inv)",
    "maintenance_cost_fraction": "Maintenance cost\nfraction (f_maint)",
    "utilization_factor": "Utilization factor\n(f_util)",
    "hydrogel_lifetime_years": "Hydrogel lifetime\n(L_gel, yr)",
    "salt_price_usd_per_kg": "Salt price\n(c_salt, USD/kg)",
    "polymer_price_multiplier": "Polymer sub-mix price\n(c_k, x baseline)",
    "c_water_gel_usd_per_kg": "DI water price\n(c_DI, USD/kg)",
}


def calculate_sensitivity(x1: float, y1: float, x2: float, y2: float) -> float:
    """% change in LCOW per % change in the parameter (0 if the param barely moved)."""
    pct_x = (x2 - x1) / x1 * 100.0 if abs(x1) > 1e-10 else (x2 - x1)
    pct_y = (y2 - y1) / y1 * 100.0 if abs(y1) > 1e-10 else (y2 - y1)
    return pct_y / pct_x if abs(pct_x) > 0.01 else 0.0


def compute_tornado(
    *, yield_kg_m2: float, hydrogel_thickness_mm: float, salt_name: str, salt_loading: float,
) -> pd.DataFrame:
    econ = LCOEconomicParams()
    thickness_m = hydrogel_thickness_mm / 1000.0
    baseline_salt_price = get_salt_price_usd_per_kg(salt_name)

    def lcow_at(econ_override: LCOEconomicParams, salt_price_override: float | None = None) -> float:
        return lcow_from_daily_yield(
            yield_kg_m2, salt_name=salt_name, salt_to_polymer_ratio=salt_loading,
            hydrogel_thickness_m=thickness_m, econ=econ_override, salt_price_usd_per_kg=salt_price_override,
        )

    baseline_lcow = lcow_at(econ)
    rows: list[dict] = []

    def _add_row(variable: str, baseline_x: float, low_x: float, high_x: float, lcow_low: float, lcow_high: float) -> None:
        inc = calculate_sensitivity(baseline_x, baseline_lcow, high_x, lcow_high)
        dec = calculate_sensitivity(baseline_x, baseline_lcow, low_x, lcow_low)
        rows.append({
            "variable": variable, "avg_increase_sensitivity": inc, "avg_decrease_sensitivity": dec,
            "max_abs_effect": max(abs(inc), abs(dec)), "num_point_sensitivities": 2,
        })

    # LCOEconomicParams fields -- perturbed +/-50% of the current baseline, except
    # the two bounded by convention (device/hydrogel lifetime, utilization, total
    # investment factor) which keep their established sensitivity-analysis ranges.
    field_ranges = (
        ("discount_rate", econ.discount_rate * 0.5, econ.discount_rate * 1.5),
        ("device_lifetime_years", 10, 30),
        ("total_investment_factor", 0.5, 2.0),
        ("maintenance_cost_fraction", econ.maintenance_cost_fraction * 0.5, econ.maintenance_cost_fraction * 1.5),
        ("utilization_factor", 0.7, 1.0),
        ("hydrogel_lifetime_years", 0.5, 2.0),
    )
    for field, low, high in field_ranges:
        baseline_x = getattr(econ, field)
        lcow_low = lcow_at(_replace_econ(econ, **{field: low}))
        lcow_high = lcow_at(_replace_econ(econ, **{field: high}))
        _add_row(field, baseline_x, low, high, lcow_low, lcow_high)

    # Salt price -- not an LCOEconomicParams field, passed as an explicit override.
    low, high = baseline_salt_price * 0.5, baseline_salt_price * 1.5
    _add_row(
        "salt_price_usd_per_kg", baseline_salt_price, low, high,
        lcow_at(econ, low), lcow_at(econ, high),
    )

    # Polymer sub-mix (AM/APS/MBA/TEMED) -- scaled together by one multiplier,
    # matching how the blended polymer price enters the hydrogel cost as one line item.
    econ_poly_low = _replace_econ(econ, **{f: getattr(econ, f) * 0.5 for f in _POLYMER_FIELDS})
    econ_poly_high = _replace_econ(econ, **{f: getattr(econ, f) * 1.5 for f in _POLYMER_FIELDS})
    _add_row("polymer_price_multiplier", 1.0, 0.5, 1.5, lcow_at(econ_poly_low), lcow_at(econ_poly_high))

    # DI water price (hydrogel manufacturing).
    low, high = econ.c_water_gel_usd_per_kg * 0.5, econ.c_water_gel_usd_per_kg * 1.5
    _add_row(
        "c_water_gel_usd_per_kg", econ.c_water_gel_usd_per_kg, low, high,
        lcow_at(_replace_econ(econ, c_water_gel_usd_per_kg=low)),
        lcow_at(_replace_econ(econ, c_water_gel_usd_per_kg=high)),
    )

    df = pd.DataFrame(rows).sort_values("max_abs_effect", ascending=False)
    print(
        f"Baseline LCOW: ${baseline_lcow:.3f}/m3  "
        f"(yield={yield_kg_m2:.4f} kg/m2/d, thickness={hydrogel_thickness_mm:.2f} mm, "
        f"salt={salt_name}, S/L={salt_loading:g})"
    )
    for _, r in df.iterrows():
        print(
            f"  {r['variable']:28s} increase={r['avg_increase_sensitivity']:+.3f}  "
            f"decrease={r['avg_decrease_sensitivity']:+.3f}"
        )
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site-label", default="site")
    ap.add_argument("--yield-kg-m2", type=float, required=True)
    ap.add_argument("--hydrogel-thickness-mm", type=float, required=True)
    ap.add_argument("--salt", default="LiCl")
    ap.add_argument("--salt-loading", type=float, default=4.0)
    ap.add_argument("--out-dir", type=Path, default=_SENSITIVITY / "outputs" / "econ_tornado")
    args = ap.parse_args()

    df = compute_tornado(
        yield_kg_m2=args.yield_kg_m2, hydrogel_thickness_mm=args.hydrogel_thickness_mm,
        salt_name=args.salt, salt_loading=args.salt_loading,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^a-z0-9]+", "_", args.site_label.lower()).strip("_")
    table_csv = args.out_dir / f"econ_tornado_{tag}.table.csv"
    out_png = args.out_dir / f"econ_tornado_{tag}.png"
    df.to_csv(table_csv, index=False)

    fig, _ = create_tornado_plot(
        df, "lcow_usd_per_m3",
        title=f"{args.site_label}: LCOW sensitivity to economic parameters",
        param_name_mapping=_PARAM_LABELS, metric_label="LCOW (USD/m³)",
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Wrote {table_csv}")
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
