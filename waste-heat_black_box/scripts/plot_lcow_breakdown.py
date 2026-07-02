#!/usr/bin/env python3
"""Plot stacked LCOW cost breakdown from waste_heat_sawh_tea.xlsx."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_TEA_ROOT = _REPO.parent
if str(_TEA_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEA_ROOT))

from tea_workbook_plots import plot_lcow_breakdown_from_workbook  # noqa: E402

_DEFAULT_WORKBOOK = _REPO / "waste_heat_sawh_tea.xlsx"
_DEFAULT_OUTPUT = _REPO / "outputs" / "lcow_breakdown.png"
_DEFAULT_TABLE = _REPO / "outputs" / "lcow_breakdown.csv"


def _write_table_csv(path: Path, breakdown) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "annual_usd_per_m2", "lcow_usd_per_m3"])
        for label, annual, usd_per_m3 in breakdown.segments:
            w.writerow([label, f"{annual:.6f}", f"{usd_per_m3:.6f}"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot LCOW cost breakdown from waste-heat black-box TEA")
    ap.add_argument("--workbook", type=Path, default=_DEFAULT_WORKBOOK)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    ap.add_argument("--table-csv", type=Path, default=_DEFAULT_TABLE)
    args = ap.parse_args()

    if not args.workbook.is_file():
        sys.exit(f"Missing workbook: {args.workbook}. Run scripts/build_tea_workbook.py first.")

    breakdown, png_path = plot_lcow_breakdown_from_workbook(
        args.workbook,
        output_path=args.output,
    )
    _write_table_csv(args.table_csv, breakdown)
    print(f"LCOW = ${breakdown.lcow_usd_per_m3:.2f}/m³ ({len(breakdown.segments)} segments)")
    print(f"Wrote {png_path}")
    print(f"Wrote {args.table_csv}")


if __name__ == "__main__":
    main()
