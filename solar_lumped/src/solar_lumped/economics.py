"""Levelized cost of water (LCOW), NPV, and payback economics for the solar SAWH device.

Consolidated from the former economics/{params, lcow, npv}.py. Section headers below
mark each former module's boundary for traceability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from solar_lumped.physics import get_salt_price_usd_per_kg


# =============================================================================
# Levelized cost of water (LCOW) economics -- verbatim from electrolyte_optimization
# =============================================================================

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
        Path(__file__).resolve().parent
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

# =============================================================================
# LCOW from Wilson simulation daily yield -- same equation as electrolyte_optimization lcow_zsr_at_sl
# =============================================================================

FAIL_LCO: float = 1e30


@dataclass(frozen=True, slots=True)
class LcowCostBreakdown:
    items: tuple[tuple[str, float], ...]

    @property
    def total_usd_per_m3(self) -> float:
        return float(sum(v for _, v in self.items))


def _hydrogel_cost_per_kg(
    salt_price_usd_per_kg: float,
    salt_to_polymer_ratio: float,
    econ: LCOEconomicParams,
) -> float:
    sl = salt_to_polymer_ratio
    return (
        (salt_price_usd_per_kg * sl + econ.c_acrylamide_usd_per_kg) / (1.0 + sl)
        + econ.c_additives_usd_per_kg_composite
    )


def _sorbent_replacement_annual_usd(
    *,
    sorbent: str,
    salt_name: str,
    salt_to_polymer_ratio: float,
    hydrogel_thickness_m: float,
    mof_mass_kg_m2: float,
    mof_price_usd_per_kg: float,
    econ: LCOEconomicParams,
    salt_price_usd_per_kg: float | None = None,
) -> float:
    gel_lifetime = econ.hydrogel_lifetime_years
    if sorbent == "mof":
        return mof_mass_kg_m2 * mof_price_usd_per_kg / gel_lifetime
    sl = salt_to_polymer_ratio
    dry_mass = dry_composite_mass_kg(hydrogel_thickness_m)
    salt_price = (
        salt_price_usd_per_kg
        if salt_price_usd_per_kg is not None
        else get_salt_price_usd_per_kg(salt_name)
    )
    hydrogel_cost_per_kg = _hydrogel_cost_per_kg(salt_price, sl, econ)
    return hydrogel_cost_per_kg * dry_mass / gel_lifetime


def lcow_from_daily_yield(
    daily_yield_kg_per_m2: float,
    *,
    salt_name: str,
    salt_to_polymer_ratio: float,
    hydrogel_thickness_m: float,
    econ: LCOEconomicParams,
    cycles_per_day: float = 1.0,
    electric_heat_w_per_m2: float = 0.0,
    salt_price_usd_per_kg: float | None = None,
    sorbent: str = "hydrogel",
    mof_mass_kg_m2: float = 0.0,
    mof_price_usd_per_kg: float = 0.0,
) -> float:
    """Scalar LCOW (USD/m³) — identical structure to lcow_zsr_at_sl."""
    if daily_yield_kg_per_m2 <= 0.0 or not math.isfinite(daily_yield_kg_per_m2):
        return FAIL_LCO
    if sorbent == "hydrogel" and (
        salt_to_polymer_ratio <= 0.0 or not math.isfinite(salt_to_polymer_ratio)
    ):
        return FAIL_LCO

    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(daily_yield_kg_per_m2)
    sorbent_replacement = _sorbent_replacement_annual_usd(
        sorbent=sorbent,
        salt_name=salt_name,
        salt_to_polymer_ratio=salt_to_polymer_ratio,
        hydrogel_thickness_m=hydrogel_thickness_m,
        mof_mass_kg_m2=mof_mass_kg_m2,
        mof_price_usd_per_kg=mof_price_usd_per_kg,
        econ=econ,
        salt_price_usd_per_kg=salt_price_usd_per_kg,
    )

    annual_electricity_cost = (
        econ.electricity_price_usd_per_kwh
        * float(electric_heat_w_per_m2)
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
    )
    annual_extra_cycle_energy = econ.annual_extra_cycle_energy_cost_usd(cycles_per_day)

    annual_cost_usd = (
        econ.capital_recovery_factor() * econ.total_investment_factor * C_DEVICE_USD
        + sorbent_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * C_DEVICE_USD
        + econ.energy_cost_usd_per_year
        + annual_electricity_cost
        + annual_extra_cycle_energy
    )
    if not math.isfinite(annual_cost_usd):
        return FAIL_LCO
    return float(
        annual_cost_usd
        / (econ.utilization_factor * (annual_water_yield_kg / KG_WATER_PER_M3 + 1e-9))
    )


def lcow_cost_breakdown_from_daily_yield(
    daily_yield_kg_per_m2: float,
    *,
    salt_name: str,
    salt_to_polymer_ratio: float,
    hydrogel_thickness_m: float,
    econ: LCOEconomicParams,
    cycles_per_day: float = 1.0,
    electric_heat_w_per_m2: float = 0.0,
    salt_price_usd_per_kg: float | None = None,
    sorbent: str = "hydrogel",
    mof_mass_kg_m2: float = 0.0,
    mof_price_usd_per_kg: float = 0.0,
) -> LcowCostBreakdown | None:
    """Per-term LCOW breakdown — same segments as lcow_cost_breakdown_usd_per_m3."""
    lcow = lcow_from_daily_yield(
        daily_yield_kg_per_m2,
        salt_name=salt_name,
        salt_to_polymer_ratio=salt_to_polymer_ratio,
        hydrogel_thickness_m=hydrogel_thickness_m,
        econ=econ,
        cycles_per_day=cycles_per_day,
        electric_heat_w_per_m2=electric_heat_w_per_m2,
        salt_price_usd_per_kg=salt_price_usd_per_kg,
        sorbent=sorbent,
        mof_mass_kg_m2=mof_mass_kg_m2,
        mof_price_usd_per_kg=mof_price_usd_per_kg,
    )
    if not math.isfinite(lcow) or lcow >= 0.99 * FAIL_LCO:
        return None

    sl = salt_to_polymer_ratio
    dry_mass = dry_composite_mass_kg(hydrogel_thickness_m)
    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(daily_yield_kg_per_m2)
    if annual_water_yield_kg <= 0.0:
        return None

    denom = econ.utilization_factor * (annual_water_yield_kg / KG_WATER_PER_M3 + 1e-9)
    inv = econ.total_investment_factor
    crf = econ.capital_recovery_factor()
    maint_frac = econ.maintenance_cost_fraction
    gel_lifetime = econ.hydrogel_lifetime_years

    def _lcow_seg(annual_usd: float) -> float:
        return float(annual_usd / denom)

    segments: list[tuple[str, float]] = []
    maintenance_annual = 0.0
    for name, line_cost in DEVICE_BOM_USD_PER_M2:
        scaled = inv * line_cost
        segments.append((f"CAPEX: {name}", _lcow_seg(crf * scaled)))
        maintenance_annual += maint_frac * scaled
    segments.append(("Maintenance", _lcow_seg(maintenance_annual)))

    if sorbent == "mof":
        mof_annual = mof_mass_kg_m2 * mof_price_usd_per_kg / gel_lifetime
        segments.append(("MOF sorbent", _lcow_seg(mof_annual)))
    else:
        salt_price = (
            salt_price_usd_per_kg
            if salt_price_usd_per_kg is not None
            else get_salt_price_usd_per_kg(salt_name)
        )
        salt_annual = salt_price * sl / (1.0 + sl) * dry_mass / gel_lifetime
        acrylamide_annual = econ.c_acrylamide_usd_per_kg / (1.0 + sl) * dry_mass / gel_lifetime
        additives_annual = econ.c_additives_usd_per_kg_composite * dry_mass / gel_lifetime
        segments.append(("Hydrogel: salt", _lcow_seg(salt_annual)))
        segments.append(("Hydrogel: acrylamide", _lcow_seg(acrylamide_annual)))
        segments.append(("Hydrogel: additives", _lcow_seg(additives_annual)))

    annual_electricity_cost = (
        econ.electricity_price_usd_per_kwh
        * float(electric_heat_w_per_m2)
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
    )
    annual_extra = econ.annual_extra_cycle_energy_cost_usd(cycles_per_day)
    segments.append(("Fixed energy", _lcow_seg(econ.energy_cost_usd_per_year)))
    segments.append(("Electricity (active heat)", _lcow_seg(annual_electricity_cost)))
    segments.append(("Extra cycling energy", _lcow_seg(annual_extra)))

    return LcowCostBreakdown(items=tuple(segments))

# =============================================================================
# Net present value (NPV) and payback period for the Wilson lumped SAWH device
# =============================================================================

@dataclass(frozen=True, slots=True)
class NpvResult:
    capex_usd_per_m2: float
    annual_revenue_usd_per_m2: float
    annual_opex_usd_per_m2: float
    annual_net_cash_flow_usd_per_m2: float
    npv_usd_per_m2: float
    payback_years_simple: float
    payback_years_discounted: float


def _present_value_annuity_factor(discount_rate: float, years: float) -> float:
    i = discount_rate
    if i <= 0.0:
        return years
    return (1.0 - (1.0 + i) ** (-years)) / i


def npv_from_daily_yield(
    daily_yield_kg_per_m2: float,
    water_price_usd_per_m3: float,
    *,
    salt_name: str,
    salt_to_polymer_ratio: float,
    hydrogel_thickness_m: float,
    econ: LCOEconomicParams,
    cycles_per_day: float = 1.0,
    electric_heat_w_per_m2: float = 0.0,
    salt_price_usd_per_kg: float | None = None,
    sorbent: str = "hydrogel",
    mof_mass_kg_m2: float = 0.0,
    mof_price_usd_per_kg: float = 0.0,
) -> NpvResult | None:
    """NPV and payback period (USD/m2 of device footprint) for one site."""
    if daily_yield_kg_per_m2 <= 0.0 or not math.isfinite(daily_yield_kg_per_m2):
        return None
    if sorbent == "hydrogel" and (
        salt_to_polymer_ratio <= 0.0 or not math.isfinite(salt_to_polymer_ratio)
    ):
        return None

    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(daily_yield_kg_per_m2)
    gross_annual_water_m3 = econ.utilization_factor * (annual_water_yield_kg / KG_WATER_PER_M3)

    sorbent_replacement = _sorbent_replacement_annual_usd(
        sorbent=sorbent,
        salt_name=salt_name,
        salt_to_polymer_ratio=salt_to_polymer_ratio,
        hydrogel_thickness_m=hydrogel_thickness_m,
        mof_mass_kg_m2=mof_mass_kg_m2,
        mof_price_usd_per_kg=mof_price_usd_per_kg,
        econ=econ,
        salt_price_usd_per_kg=salt_price_usd_per_kg,
    )
    annual_electricity_cost = (
        econ.electricity_price_usd_per_kwh
        * float(electric_heat_w_per_m2)
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
    )
    annual_extra_cycle_energy = econ.annual_extra_cycle_energy_cost_usd(cycles_per_day)

    capex = econ.total_investment_factor * C_DEVICE_USD
    annual_opex = (
        sorbent_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * C_DEVICE_USD
        + econ.energy_cost_usd_per_year
        + annual_electricity_cost
        + annual_extra_cycle_energy
    )
    annual_revenue = gross_annual_water_m3 * float(water_price_usd_per_m3)
    annual_net_cash_flow = annual_revenue - annual_opex

    if not math.isfinite(annual_net_cash_flow):
        return None

    lifetime_years = float(econ.device_lifetime_years)
    i = econ.discount_rate
    pvaf = _present_value_annuity_factor(i, lifetime_years)
    npv = -capex + annual_net_cash_flow * pvaf

    payback_simple = capex / annual_net_cash_flow if annual_net_cash_flow > 0.0 else float("inf")

    if annual_net_cash_flow <= 0.0:
        payback_discounted = float("inf")
    elif i <= 0.0:
        payback_discounted = payback_simple
    else:
        ratio = 1.0 - i * capex / annual_net_cash_flow
        payback_discounted = -math.log(ratio) / math.log(1.0 + i) if ratio > 0.0 else float("inf")

    return NpvResult(
        capex_usd_per_m2=capex,
        annual_revenue_usd_per_m2=annual_revenue,
        annual_opex_usd_per_m2=annual_opex,
        annual_net_cash_flow_usd_per_m2=annual_net_cash_flow,
        npv_usd_per_m2=npv,
        payback_years_simple=payback_simple,
        payback_years_discounted=payback_discounted,
    )
