"""Levelized cost of water (LCOW), NPV, payback, patent bill-of-materials, parasitic
electricity, and specific-energy economics for the two-bed waste-heat SAWH device
(no HTF loop -- direct waste-heat coupling).

Consolidated from the former economics/{params, bom, parasitic, specific_energy, lcow,
npv}.py. Section headers below mark each former module's boundary for traceability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from waste_heat_cycle_lumped_no_loop.physics import (
    DRY_COMPOSITE_DENSITY_KG_M3,
    H_FG_J_PER_KG,
    get_salt_price_usd_per_kg,
)

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


def _load_economic_data(
    csv_path: Path | str | None = None,
) -> tuple[dict[str, Any], tuple[tuple[str, float], ...]]:
    def _coerce(name: str, raw: Any) -> Any:
        if name == "device_lifetime_years":
            return int(round(float(raw)))
        if name == "include_desorption_enthalpy":
            if isinstance(raw, bool):
                return raw
            s = str(raw).strip().lower()
            return s in {"1", "true", "yes", "y", "t"}
        return float(raw)

    path = (
        Path(csv_path)
        if csv_path is not None
        else Path(__file__).resolve().parent / "data" / "economics" / "lcow_economic_params.csv"
    )
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
        value = _coerce(name, raw_val)
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
    """Dry (solids-only) composite mass per m^2 at the given thickness.

    Uses the DVS-derived dry-basis density, not the raw Table S3 rho_gel (that's
    measured at 20% RH and already carries ~126% equilibrium water by mass --
    see physics.py::DRY_COMPOSITE_DENSITY_KG_M3), since the recipe-based hydrogel
    cost below is priced per kg of dry (post-synthesis) solids.
    """
    return float(hydrogel_thickness_m) * DRY_COMPOSITE_DENSITY_KG_M3
PATENT_BOM_USD_PER_M2: tuple[tuple[str, float], ...] = (
    ("Vacuum pump (28)", 3500.0),
    ("Chambers (22A, 22B) with door assemblies (38)", 1050.0),
    ("Three-way valve (32) + check valve (30)", 275.0),
    ("Condenser (24)", 850.0),
    ("Coolant source (26)", 325.0),
    ("Water pump (34)", 165.0),
    ("Controller (16) + sensors (36)", 800.0),
    ("Purge pump (234)", 400.0),
    ("Structural housing, manifolds, plumbing, fasteners", 1350.0),
)

C_PATENT_BOM_USD: float = sum(cost for _, cost in PATENT_BOM_USD_PER_M2)
@dataclass(frozen=True, slots=True)
class ElectricalLoadSpec:
    """One electrical component's parasitic load per m² footprint."""

    name: str
    shaft_power_w_per_m2: float
    motor_efficiency: float
    operating_hours_per_day: float
    notes: str = ""

    @property
    def grid_power_w_per_m2(self) -> float:
        eta = float(self.motor_efficiency)
        if eta <= 0.0:
            return 0.0
        return float(self.shaft_power_w_per_m2) / eta

    def annual_kwh_per_m2(self) -> float:
        return self.grid_power_w_per_m2 * float(self.operating_hours_per_day) * 365.0 / 1000.0

    def annual_cost_usd_per_m2(self, electricity_price_usd_per_kwh: float) -> float:
        return float(electricity_price_usd_per_kwh) * self.annual_kwh_per_m2()


def default_electrical_loads(
    *,
    vacuum_operating_hours_per_day: float = 12.0,
) -> tuple[ElectricalLoadSpec, ...]:
    """Default parasitic loads tied to the data-center baseline device.

    The pumped HTF loop (and its transfer pump) has been removed; the
    desorbing contactor now couples directly to the waste-heat stream.
    """
    return (
        ElectricalLoadSpec(
            name="Vacuum pump (28)",
            shaft_power_w_per_m2=45.0,
            motor_efficiency=0.35,
            operating_hours_per_day=vacuum_operating_hours_per_day,
            notes="Roughing pump during desorption half-cycles",
        ),
        ElectricalLoadSpec(
            name="Water pump (34)",
            shaft_power_w_per_m2=3.0,
            motor_efficiency=0.50,
            operating_hours_per_day=2.0,
            notes="Product-water transfer",
        ),
        ElectricalLoadSpec(
            name="Purge pump (234)",
            shaft_power_w_per_m2=8.0,
            motor_efficiency=0.45,
            operating_hours_per_day=1.0,
            notes="Manifold / valve purge",
        ),
        ElectricalLoadSpec(
            name="Controller (16) + sensors (36)",
            shaft_power_w_per_m2=2.5,
            motor_efficiency=0.85,
            operating_hours_per_day=24.0,
            notes="Controls and instrumentation",
        ),
    )


def total_parasitic_electricity_annual_usd_per_m2(
    loads: tuple[ElectricalLoadSpec, ...],
    electricity_price_usd_per_kwh: float,
) -> float:
    return sum(
        load.annual_cost_usd_per_m2(electricity_price_usd_per_kwh) for load in loads
    )


_JOULES_PER_KWH = 3.6e6
_FAIL_SPECIFIC_ENERGY = float("inf")


def waste_heat_specific_energy_kwh_per_l(
    *,
    thermal_efficiency: float,
    h_fg_j_per_kg: float = H_FG_J_PER_KG,
) -> float:
    """Waste-heat input per liter water produced (kWh/L).

    Derived from η = m_water h_fg / Q_wh ⇒ Q_wh / m_water = h_fg / η.
    """
    eta = float(thermal_efficiency)
    if eta <= 0.0 or not math.isfinite(eta):
        return _FAIL_SPECIFIC_ENERGY
    j_per_l = float(h_fg_j_per_kg) / eta
    if not math.isfinite(j_per_l):
        return _FAIL_SPECIFIC_ENERGY
    return j_per_l / _JOULES_PER_KWH


def parasitic_specific_energy_kwh_per_l(
    daily_yield_kg_per_m2: float,
    *,
    cycles_per_day: float = 1.0,
    loads: tuple[ElectricalLoadSpec, ...] | None = None,
) -> float:
    """Grid electricity for pumps and controls, amortized per liter water (kWh/L)."""
    yield_kg = float(daily_yield_kg_per_m2)
    if yield_kg <= 0.0 or not math.isfinite(yield_kg):
        return _FAIL_SPECIFIC_ENERGY
    load_specs = loads if loads is not None else default_electrical_loads()
    kwh_per_m2_yr = sum(load.annual_kwh_per_m2() for load in load_specs)
    water_l_per_m2_yr = float(cycles_per_day) * 365.0 * yield_kg
    return kwh_per_m2_yr / water_l_per_m2_yr


def supplemental_heat_specific_energy_kwh_per_l(
    *,
    electric_heat_w_per_m2: float,
    econ: LCOEconomicParams,
    daily_yield_kg_per_m2: float,
    cycles_per_day: float = 1.0,
) -> float:
    """Optional supplemental electric desorption heat per liter water (kWh/L)."""
    yield_kg = float(daily_yield_kg_per_m2)
    if yield_kg <= 0.0 or not math.isfinite(yield_kg):
        return _FAIL_SPECIFIC_ENERGY
    kwh_per_m2_yr = (
        float(electric_heat_w_per_m2)
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
    )
    water_l_per_m2_yr = float(cycles_per_day) * 365.0 * yield_kg
    return kwh_per_m2_yr / water_l_per_m2_yr


def total_specific_energy_kwh_per_l(
    daily_yield_kg_per_m2: float,
    *,
    thermal_efficiency: float,
    h_fg_j_per_kg: float = H_FG_J_PER_KG,
    econ: LCOEconomicParams | None = None,
    cycles_per_day: float = 1.0,
    electric_heat_w_per_m2: float = 0.0,
    loads: tuple[ElectricalLoadSpec, ...] | None = None,
) -> float:
    """Waste heat + parasitic grid + supplemental electric heat per liter (kWh/L)."""
    wh = waste_heat_specific_energy_kwh_per_l(
        thermal_efficiency=thermal_efficiency,
        h_fg_j_per_kg=h_fg_j_per_kg,
    )
    parasitic = parasitic_specific_energy_kwh_per_l(
        daily_yield_kg_per_m2,
        cycles_per_day=cycles_per_day,
        loads=loads,
    )
    supplemental = 0.0
    if electric_heat_w_per_m2 > 0.0 and econ is not None:
        supplemental = supplemental_heat_specific_energy_kwh_per_l(
            electric_heat_w_per_m2=electric_heat_w_per_m2,
            econ=econ,
            daily_yield_kg_per_m2=daily_yield_kg_per_m2,
            cycles_per_day=cycles_per_day,
        )
    total = wh + parasitic + supplemental
    if not math.isfinite(total):
        return _FAIL_SPECIFIC_ENERGY
    return total
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
) -> float:
    """Scalar LCOW (USD/m³) — identical structure to lcow_zsr_at_sl."""
    if daily_yield_kg_per_m2 <= 0.0 or not math.isfinite(daily_yield_kg_per_m2):
        return FAIL_LCO
    if salt_to_polymer_ratio <= 0.0 or not math.isfinite(salt_to_polymer_ratio):
        return FAIL_LCO

    sl = salt_to_polymer_ratio
    dry_mass = dry_composite_mass_kg(hydrogel_thickness_m)
    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(daily_yield_kg_per_m2)

    salt_price = (
        salt_price_usd_per_kg
        if salt_price_usd_per_kg is not None
        else get_salt_price_usd_per_kg(salt_name)
    )
    hydrogel_cost_per_kg = _hydrogel_cost_per_kg(salt_price, sl, econ)
    hydrogel_replacement = hydrogel_cost_per_kg * dry_mass / econ.hydrogel_lifetime_years

    annual_electricity_cost = (
        econ.electricity_price_usd_per_kwh
        * float(electric_heat_w_per_m2)
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
    )
    annual_parasitic_electricity = total_parasitic_electricity_annual_usd_per_m2(
        default_electrical_loads(),
        econ.electricity_price_usd_per_kwh,
    )
    annual_extra_cycle_energy = econ.annual_extra_cycle_energy_cost_usd(cycles_per_day)

    annual_cost_usd = (
        econ.capital_recovery_factor() * econ.total_investment_factor * C_DEVICE_USD
        + hydrogel_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * C_DEVICE_USD
        + econ.energy_cost_usd_per_year
        + annual_electricity_cost
        + annual_parasitic_electricity
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
    segments.append(("Electricity (supplemental heat)", _lcow_seg(annual_electricity_cost)))
    for load in default_electrical_loads():
        annual_usd = load.annual_cost_usd_per_m2(econ.electricity_price_usd_per_kwh)
        segments.append((f"Electricity: {load.name}", _lcow_seg(annual_usd)))
    segments.append(("Extra cycling energy", _lcow_seg(annual_extra)))

    return LcowCostBreakdown(items=tuple(segments))
@dataclass(frozen=True, slots=True)
class NpvResult:
    capex_usd_per_m2: float
    annual_revenue_usd_per_m2: float
    annual_opex_usd_per_m2: float
    annual_net_cash_flow_usd_per_m2: float
    npv_usd_per_m2: float
    payback_years_simple: float
    payback_years_discounted: float


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
) -> NpvResult | None:
    """NPV and payback period (USD/m2 of device footprint) for one site."""
    if daily_yield_kg_per_m2 <= 0.0 or not math.isfinite(daily_yield_kg_per_m2):
        return None
    if salt_to_polymer_ratio <= 0.0 or not math.isfinite(salt_to_polymer_ratio):
        return None

    sl = salt_to_polymer_ratio
    dry_mass = dry_composite_mass_kg(hydrogel_thickness_m)
    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(daily_yield_kg_per_m2)
    gross_annual_water_m3 = econ.utilization_factor * (annual_water_yield_kg / KG_WATER_PER_M3)

    salt_price = (
        salt_price_usd_per_kg
        if salt_price_usd_per_kg is not None
        else get_salt_price_usd_per_kg(salt_name)
    )
    hydrogel_cost_per_kg = _hydrogel_cost_per_kg(salt_price, sl, econ)
    hydrogel_replacement = hydrogel_cost_per_kg * dry_mass / econ.hydrogel_lifetime_years

    annual_electricity_cost = (
        econ.electricity_price_usd_per_kwh
        * float(electric_heat_w_per_m2)
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
    )
    annual_parasitic_electricity = total_parasitic_electricity_annual_usd_per_m2(
        default_electrical_loads(),
        econ.electricity_price_usd_per_kwh,
    )
    annual_extra_cycle_energy = econ.annual_extra_cycle_energy_cost_usd(cycles_per_day)

    capex = econ.total_investment_factor * C_DEVICE_USD
    annual_opex = (
        hydrogel_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * C_DEVICE_USD
        + econ.energy_cost_usd_per_year
        + annual_electricity_cost
        + annual_parasitic_electricity
        + annual_extra_cycle_energy
    )
    annual_revenue = gross_annual_water_m3 * float(water_price_usd_per_m3)
    annual_net_cash_flow = annual_revenue - annual_opex

    if not math.isfinite(annual_net_cash_flow):
        return None

    lifetime_years = float(econ.device_lifetime_years)
    i = econ.discount_rate
    pvaf = lifetime_years if i <= 0.0 else (1.0 - (1.0 + i) ** (-lifetime_years)) / i
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
