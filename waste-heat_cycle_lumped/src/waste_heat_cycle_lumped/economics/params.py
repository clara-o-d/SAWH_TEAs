"""Levelized cost of water (LCOW) economics — verbatim from electrolyte_optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

_COL_PARAMETER = "parameter"
_COL_VALUE = "value"
_DEVICE_BOM_PREFIX = "device_bom_"
_PHYSICAL_SCALAR_PARAMS: tuple[str, ...] = (
    "hydrogel_thickness_m",
    "hydrogel_thickness_min_m",
    "hydrogel_thickness_max_m",
    "hydrogel_density_kg_m3",
    "water_density_kg_per_l",
    "l_per_m3",
    "mass_transfer_convection_coefficient_m_s",
)


def lcow_economic_params_csv_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "data"
        / "economics"
        / "lcow_economic_params.csv"
    )


def _coerce_lcow_param(name: str, raw: Any) -> Any:
    if name == "device_lifetime_years":
        return int(round(float(raw)))
    if name == "include_desorption_enthalpy":
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        return s in {"1", "true", "yes", "y", "t"}
    return float(raw)


def _load_economic_data(
    csv_path: Path | str | None = None,
) -> tuple[dict[str, Any], tuple[tuple[str, float], ...]]:
    path = Path(csv_path) if csv_path is not None else lcow_economic_params_csv_path()
    if not path.is_file():
        raise FileNotFoundError(f"LCOW economic params not found at {path}")
    import pandas as pd

    df = pd.read_csv(path)
    if _COL_PARAMETER not in df.columns or _COL_VALUE not in df.columns:
        raise ValueError(f"Expected parameter/value columns in {path}.")

    scalars: dict[str, Any] = {}
    bom_rows: list[tuple[str, float]] = []
    for _, row in df.iterrows():
        name = str(row[_COL_PARAMETER]).strip()
        if not name:
            continue
        raw_val = row[_COL_VALUE]
        value = _coerce_lcow_param(name, raw_val)
        if name.startswith(_DEVICE_BOM_PREFIX):
            notes = row.get("notes")
            label = str(notes).strip() if notes == notes and str(notes).strip() else name
            bom_rows.append((label, float(value)))
        else:
            scalars[name] = value

    lcow_field_names = {f.name for f in fields(LCOEconomicParams)}
    missing = [
        name
        for name in (*_PHYSICAL_SCALAR_PARAMS, *sorted(lcow_field_names))
        if name not in scalars
    ]
    if not bom_rows:
        missing.append("device_bom_*")
    if missing:
        raise ValueError(f"Missing required parameters in {path}: {', '.join(missing)}")
    return scalars, tuple(bom_rows)


@dataclass(frozen=True, slots=True, init=False)
class LCOEconomicParams:
    """LCOW = annual_cost / (utilization_factor * gross_annual_water_m3)."""

    discount_rate: float
    device_lifetime_years: int
    total_investment_factor: float
    maintenance_cost_fraction: float
    utilization_factor: float
    hydrogel_lifetime_years: float
    energy_cost_usd_per_year: float
    energy_cost_usd_per_extra_half_cycle_per_day: float
    c_acrylamide_usd_per_kg: float
    c_additives_usd_per_kg_composite: float
    electricity_price_usd_per_kwh: float
    desorption_hours_per_day: float
    max_electric_heat_w_per_m2: float
    include_desorption_enthalpy: bool

    def __init__(self, **kwargs: Any) -> None:
        defaults = _LCOW_DEFAULTS
        for f in fields(self):
            value = kwargs[f.name] if f.name in kwargs else defaults[f.name]
            object.__setattr__(self, f.name, value)

    def annual_extra_cycle_energy_cost_usd(self, cycles_per_day: float) -> float:
        extra = max(0.0, float(cycles_per_day) - 1.0)
        return extra * 365.0 * float(self.energy_cost_usd_per_extra_half_cycle_per_day)

    def capital_recovery_factor(self) -> float:
        i = self.discount_rate
        L = self.device_lifetime_years
        if i <= 0.0 or L < 1:
            raise ValueError("discount_rate must be > 0 and device_lifetime_years >= 1")
        return (i * (1.0 + i) ** L) / ((1.0 + i) ** L - 1.0)


_SCALARS, _DEVICE_BOM_ROWS = _load_economic_data()

HYDROGEL_THICKNESS_M: float = float(_SCALARS["hydrogel_thickness_m"])
HYDROGEL_THICKNESS_MIN_M: float = float(_SCALARS["hydrogel_thickness_min_m"])
HYDROGEL_THICKNESS_MAX_M: float = float(_SCALARS["hydrogel_thickness_max_m"])
HYDROGEL_DENSITY_KG_M3: float = float(_SCALARS["hydrogel_density_kg_m3"])
MASS_TRANSFER_CONVECTION_COEFFICIENT_M_S: float = float(
    _SCALARS["mass_transfer_convection_coefficient_m_s"]
)
WATER_DENSITY_KG_PER_L: float = float(_SCALARS["water_density_kg_per_l"])
L_PER_M3: float = float(_SCALARS["l_per_m3"])
KG_WATER_PER_M3: float = WATER_DENSITY_KG_PER_L * L_PER_M3
DEVICE_BOM_USD_PER_M2: tuple[tuple[str, float], ...] = _DEVICE_BOM_ROWS
C_DEVICE_USD: float = sum(cost for _, cost in DEVICE_BOM_USD_PER_M2)
_LCOW_DEFAULTS: dict[str, Any] = {f.name: _SCALARS[f.name] for f in fields(LCOEconomicParams)}


def dry_composite_mass_kg(hydrogel_thickness_m: float) -> float:
    return float(hydrogel_thickness_m) * HYDROGEL_DENSITY_KG_M3
