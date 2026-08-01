#!/usr/bin/env python3
"""Plot the stacked LCOW cost breakdown from a black-box TEA workbook.

Usage: python plot_lcow_breakdown.py --case {solar,waste_heat}
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tea_workbook_plots import plot_lcow_breakdown_from_workbook

_TEA_ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=["solar", "waste_heat"], default="solar")
    ap.add_argument("--workbook", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--table-csv", type=Path, default=None)
    args = ap.parse_args()

    case_dir = _TEA_ROOT / args.case
    workbook = args.workbook or case_dir / f"{args.case}_sawh_tea.xlsx"
    table_csv = args.table_csv or case_dir / "outputs" / "lcow_breakdown.csv"
    if not workbook.is_file():
        sys.exit(f"Missing workbook: {workbook}. Run scripts/build_tea_workbook.py first.")

    breakdown, png_path = plot_lcow_breakdown_from_workbook(
        workbook, output_path=args.output or case_dir / "outputs" / "lcow_breakdown.png"
    )

    table_csv.parent.mkdir(parents=True, exist_ok=True)
    with table_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "annual_usd_per_m2", "lcow_usd_per_m3"])
        w.writerows((label, f"{annual:.6f}", f"{per_m3:.6f}") for label, annual, per_m3 in breakdown.segments)

    print(f"LCOW = ${breakdown.lcow_usd_per_m3:.2f}/m³ ({len(breakdown.segments)} segments)")
    print(f"Wrote {png_path}")
    print(f"Wrote {table_csv}")


if __name__ == "__main__":
    main()
