"""Loads the Physics and Economics parameter tables directly from
``docs/parameters.xlsx`` -- the single source of truth for device physics
constants and LCOW/NPV economics inputs. Replaces the former
``lcow_economic_params.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl

_XLSX_PATH = Path(__file__).resolve().parents[2] / "docs" / "parameters.xlsx"


def _load_sheet(sheet_name: str) -> dict[str, dict[str, Any]]:
    wb = openpyxl.load_workbook(_XLSX_PATH, data_only=True, read_only=True)
    ws = wb[sheet_name]
    rows = ws.iter_rows(min_row=2, values_only=True)
    return {
        str(name).strip(): {"value": value, "lower": lower, "upper": upper, "source": source}
        for name, _equation, value, _units, lower, upper, source in rows
        if name is not None
    }


PHYSICS: dict[str, dict[str, Any]] = _load_sheet("Physics")
ECONOMICS: dict[str, dict[str, Any]] = _load_sheet("Economics")


def physics_value(name: str, *, mm_to_m: bool = False) -> float:
    value = float(PHYSICS[name]["value"])
    return value / 1000.0 if mm_to_m else value


def economics_value(name: str) -> Any:
    return ECONOMICS[name]["value"]
