"""Levelized cost of water (LCOW), NPV, payback, patent bill-of-materials, parasitic
electricity, and specific-energy economics for the two-bed waste-heat SAWH system with
direct waste-heat coupling (no HTF loop)."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

from solar_lumped._parameters_xlsx import ECONOMICS as _ECON_XLSX
from solar_lumped._parameters_xlsx import PHYSICS as _PHYS_XLSX
from solar_lumped._parameters_xlsx import economics_value as _ev
from solar_lumped._parameters_xlsx import physics_value as _pv
from waste_heat.physics import (
    DRY_COMPOSITE_DENSITY_KG_M3,
    H_FG_J_PER_KG,
    get_salt_price_usd_per_kg,
)

# --- Economic parameter loading and defaults ---
# Read directly from solar_lumped/docs/parameters.xlsx (Economics + Physics sheets), the
# repo-wide single source of truth; no separate CSV. Rows prefixed "Waste-heat" are this
# system's own, the rest are shared with the solar device.

_BOM_PREFIX = "Waste-heat BOM: "

# LCOEconomicParams field name -> parameters.xlsx Economics-sheet row name.
_ECON_FIELD_ROWS: dict[str, str] = {
    "discount_rate": "Discount rate (i)",
    "system_lifetime_years": "Device lifetime (L)",
    "total_investment_factor": "Total investment factor (f_inv)",
    "maintenance_cost_fraction": "Maintenance cost fraction (f_maint)",
    "utilization_factor": "Utilization factor (f_util)",
    "hydrogel_lifetime_years": "Hydrogel lifetime (L_gel)",
    "energy_cost_usd_per_year": "Waste-heat: fixed annual energy cost",
    "energy_cost_usd_per_extra_half_cycle_per_day": (
        "Waste-heat: energy cost per extra half-cycle per day"
    ),
    "c_acrylamide_usd_per_kg": "Acrylamide price, AM (c_AM)",
    "c_additives_usd_per_kg_composite": (
        "Hydrogel additives cost per kg composite (APS + MBA + TEMED)"
    ),
    "electricity_price_usd_per_kwh": "Electricity price (p_elec)",
    "desorption_hours_per_day": "Desorption hours per day (t_des)",
    "max_electric_heat_w_per_m2": "Max electric heat, optimizer bound (Q_elec,max)",
    "include_desorption_enthalpy": "Include desorption enthalpy in T_g solve (flag)",
}


def _coerce_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "t"}


def _load_economic_data() -> tuple[dict[str, Any], tuple[tuple[str, float], ...]]:
    scalars: dict[str, Any] = {}
    for field_name, row_name in _ECON_FIELD_ROWS.items():
        raw = _ev(row_name)
        if field_name == "system_lifetime_years":
            scalars[field_name] = int(round(float(raw)))
        elif field_name == "include_desorption_enthalpy":
            scalars[field_name] = _coerce_bool(raw)
        else:
            scalars[field_name] = float(raw)

    # Physical (not economic) quantities from the Physics sheet, needed here for the
    # sorbent-replacement cost calculation.
    h0_row = _PHYS_XLSX["Hydrogel reference thickness (H0)"]
    scalars["hydrogel_thickness_m"] = float(h0_row["value"]) / 1000.0
    scalars["hydrogel_thickness_min_m"] = float(h0_row["lower"]) / 1000.0
    scalars["hydrogel_thickness_max_m"] = float(h0_row["upper"]) / 1000.0
    scalars["hydrogel_density_kg_m3"] = _pv("Composite (hydrogel) density at 20% RH (rho_gel)")
    scalars["mass_transfer_convection_coefficient_m_s"] = _pv(
        "Chamber convection coefficient, absorption (g_chamber)"
    )

    bom_rows = tuple(
        (name[len(_BOM_PREFIX) :], float(row["value"]))
        for name, row in _ECON_XLSX.items()
        if name.startswith(_BOM_PREFIX)
    )

    lcow_field_names = {f.name for f in fields(LCOEconomicParams)}
    physical_scalar_params = (
        "hydrogel_thickness_m",
        "hydrogel_thickness_min_m",
        "hydrogel_thickness_max_m",
        "hydrogel_density_kg_m3",
        "mass_transfer_convection_coefficient_m_s",
    )
    missing = [
        name
        for name in (*physical_scalar_params, *sorted(lcow_field_names))
        if name not in scalars
    ]
    if not bom_rows:
        missing.append("Waste-heat BOM: *")
    if missing:
        raise ValueError(f"Missing required parameters in parameters.xlsx: {', '.join(missing)}")
    return scalars, bom_rows


@dataclass(frozen=True, slots=True, init=False)
class LCOEconomicParams:
    """LCOW = annual_cost / (utilization_factor * gross_annual_water_m3)."""

    discount_rate: float
    system_lifetime_years: int
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
        L = self.system_lifetime_years
        if i <= 0.0 or L < 1:
            raise ValueError("discount_rate must be > 0 and system_lifetime_years >= 1")
        return (i * (1.0 + i) ** L) / ((1.0 + i) ** L - 1.0)


_SCALARS, _SYSTEM_BOM_ROWS = _load_economic_data()

HYDROGEL_THICKNESS_M: float = float(_SCALARS["hydrogel_thickness_m"])
HYDROGEL_THICKNESS_MIN_M: float = float(_SCALARS["hydrogel_thickness_min_m"])
HYDROGEL_THICKNESS_MAX_M: float = float(_SCALARS["hydrogel_thickness_max_m"])
HYDROGEL_DENSITY_KG_M3: float = float(_SCALARS["hydrogel_density_kg_m3"])
MASS_TRANSFER_CONVECTION_COEFFICIENT_M_S: float = float(
    _SCALARS["mass_transfer_convection_coefficient_m_s"]
)
KG_WATER_PER_M3: float = _ev("Water density (rho_w)")
SYSTEM_BOM_USD_PER_M2: tuple[tuple[str, float], ...] = _SYSTEM_BOM_ROWS
C_SYSTEM_USD: float = sum(cost for _, cost in SYSTEM_BOM_USD_PER_M2)
_LCOW_DEFAULTS: dict[str, Any] = {f.name: _SCALARS[f.name] for f in fields(LCOEconomicParams)}


def dry_composite_mass_kg(hydrogel_thickness_m: float) -> float:
    """Dry (solids-only) composite mass per m² at the given thickness, using the
    dry-basis density -- Table S3's rho_gel is measured at 20% RH and already carries
    ~126% water, while the hydrogel cost below is priced per kg of dry solids."""
    return float(hydrogel_thickness_m) * DRY_COMPOSITE_DENSITY_KG_M3
# The patent BOM *is* the system BOM here -- both names used to hold the same nine line
# items, one from the CSV and one hardcoded. Kept as aliases so callers of either survive.
PATENT_BOM_USD_PER_M2: tuple[tuple[str, float], ...] = SYSTEM_BOM_USD_PER_M2
C_PATENT_BOM_USD: float = C_SYSTEM_USD
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


_DEFAULT_VACUUM_HOURS_PER_DAY: float = _ev("Waste-heat parasitic: vacuum pump operating hours")


def default_electrical_loads(
    *,
    vacuum_operating_hours_per_day: float = _DEFAULT_VACUUM_HOURS_PER_DAY,
) -> tuple[ElectricalLoadSpec, ...]:
    """Default parasitic loads for the data-center baseline system: no pumped HTF loop or
    transfer pump, since the desorbing contactor couples directly to the waste-heat stream."""
    return (
        ElectricalLoadSpec(
            name="Vacuum pump (28)",
            shaft_power_w_per_m2=_ev("Waste-heat parasitic: vacuum pump shaft power"),
            motor_efficiency=_ev("Waste-heat parasitic: vacuum pump motor efficiency"),
            operating_hours_per_day=vacuum_operating_hours_per_day,
            notes="Roughing pump during desorption half-cycles",
        ),
        ElectricalLoadSpec(
            name="Water pump (34)",
            shaft_power_w_per_m2=_ev("Waste-heat parasitic: water pump shaft power"),
            motor_efficiency=_ev("Waste-heat parasitic: water pump motor efficiency"),
            operating_hours_per_day=_ev("Waste-heat parasitic: water pump operating hours"),
            notes="Product-water transfer",
        ),
        ElectricalLoadSpec(
            name="Purge pump (234)",
            shaft_power_w_per_m2=_ev("Waste-heat parasitic: purge pump shaft power"),
            motor_efficiency=_ev("Waste-heat parasitic: purge pump motor efficiency"),
            operating_hours_per_day=_ev("Waste-heat parasitic: purge pump operating hours"),
            notes="Manifold / valve purge",
        ),
        ElectricalLoadSpec(
            name="Controller (16) + sensors (36)",
            shaft_power_w_per_m2=_ev("Waste-heat parasitic: controller and sensors shaft power"),
            motor_efficiency=_ev("Waste-heat parasitic: controller and sensors efficiency"),
            operating_hours_per_day=_ev(
                "Waste-heat parasitic: controller and sensors operating hours"
            ),
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
    yield_per_cycle_kg_per_m2: float,
    *,
    cycles_per_day: float = 1.0,
    loads: tuple[ElectricalLoadSpec, ...] | None = None,
) -> float:
    """Grid electricity for pumps and controls, amortized per liter water (kWh/L)."""
    yield_kg = float(yield_per_cycle_kg_per_m2)
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
    yield_per_cycle_kg_per_m2: float,
    cycles_per_day: float = 1.0,
) -> float:
    """Optional supplemental electric desorption heat per liter water (kWh/L)."""
    yield_kg = float(yield_per_cycle_kg_per_m2)
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
    yield_per_cycle_kg_per_m2: float,
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
        yield_per_cycle_kg_per_m2,
        cycles_per_day=cycles_per_day,
        loads=loads,
    )
    supplemental = 0.0
    if electric_heat_w_per_m2 > 0.0 and econ is not None:
        supplemental = supplemental_heat_specific_energy_kwh_per_l(
            electric_heat_w_per_m2=electric_heat_w_per_m2,
            econ=econ,
            yield_per_cycle_kg_per_m2=yield_per_cycle_kg_per_m2,
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
    salt_loading: float,
    econ: LCOEconomicParams,
) -> float:
    sl = salt_loading
    return (
        (salt_price_usd_per_kg * sl + econ.c_acrylamide_usd_per_kg) / (1.0 + sl)
        + econ.c_additives_usd_per_kg_composite
    )


def lcow_from_daily_yield(
    yield_per_cycle_kg_per_m2: float,
    *,
    salt_name: str,
    salt_loading: float,
    hydrogel_thickness_m: float,
    econ: LCOEconomicParams,
    cycles_per_day: float = 1.0,
    electric_heat_w_per_m2: float = 0.0,
    salt_price_usd_per_kg: float | None = None,
) -> float:
    """Scalar LCOW (USD/m³) — identical structure to lcow_zsr_at_sl.

    Takes yield **per cycle**, not per day: annual water is
    ``cycles_per_day * 365 * yield_per_cycle_kg_per_m2``, and ``cycles_per_day`` also
    drives the per-cycle energy term, so it has to be the true cycle count. With the
    default ``cycles_per_day=1.0`` a whole day's yield is the correct thing to pass.
    Feeding a multi-cycle daily total *and* the real cycle count double-counts the water
    by that count -- which is exactly what analysis/sensitivity/parameter_sweep.py's
    waste-heat model used to do.
    """
    if yield_per_cycle_kg_per_m2 <= 0.0 or not math.isfinite(yield_per_cycle_kg_per_m2):
        return FAIL_LCO
    if salt_loading <= 0.0 or not math.isfinite(salt_loading):
        return FAIL_LCO

    sl = salt_loading
    dry_mass = dry_composite_mass_kg(hydrogel_thickness_m)
    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(yield_per_cycle_kg_per_m2)

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
        econ.capital_recovery_factor() * econ.total_investment_factor * C_SYSTEM_USD
        + hydrogel_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * C_SYSTEM_USD
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
    yield_per_cycle_kg_per_m2: float,
    *,
    salt_name: str,
    salt_loading: float,
    hydrogel_thickness_m: float,
    econ: LCOEconomicParams,
    cycles_per_day: float = 1.0,
    electric_heat_w_per_m2: float = 0.0,
    salt_price_usd_per_kg: float | None = None,
) -> LcowCostBreakdown | None:
    """Per-term LCOW breakdown — same segments as lcow_cost_breakdown_usd_per_m3."""
    lcow = lcow_from_daily_yield(
        yield_per_cycle_kg_per_m2,
        salt_name=salt_name,
        salt_loading=salt_loading,
        hydrogel_thickness_m=hydrogel_thickness_m,
        econ=econ,
        cycles_per_day=cycles_per_day,
        electric_heat_w_per_m2=electric_heat_w_per_m2,
        salt_price_usd_per_kg=salt_price_usd_per_kg,
    )
    if not math.isfinite(lcow) or lcow >= 0.99 * FAIL_LCO:
        return None

    sl = salt_loading
    dry_mass = dry_composite_mass_kg(hydrogel_thickness_m)
    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(yield_per_cycle_kg_per_m2)
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
    for name, line_cost in SYSTEM_BOM_USD_PER_M2:
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


WATER_PRICE_USD_PER_M3: float = _ev("Water price, NPV/payback scenario input (p_water)")


def npv_from_daily_yield(
    yield_per_cycle_kg_per_m2: float,
    water_price_usd_per_m3: float = WATER_PRICE_USD_PER_M3,
    *,
    salt_name: str,
    salt_loading: float,
    hydrogel_thickness_m: float,
    econ: LCOEconomicParams,
    cycles_per_day: float = 1.0,
    electric_heat_w_per_m2: float = 0.0,
    salt_price_usd_per_kg: float | None = None,
) -> NpvResult | None:
    """NPV and payback period (USD/m2 of system footprint) for one site."""
    if yield_per_cycle_kg_per_m2 <= 0.0 or not math.isfinite(yield_per_cycle_kg_per_m2):
        return None
    if salt_loading <= 0.0 or not math.isfinite(salt_loading):
        return None

    sl = salt_loading
    dry_mass = dry_composite_mass_kg(hydrogel_thickness_m)
    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(yield_per_cycle_kg_per_m2)
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

    capex = econ.total_investment_factor * C_SYSTEM_USD
    annual_opex = (
        hydrogel_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * C_SYSTEM_USD
        + econ.energy_cost_usd_per_year
        + annual_electricity_cost
        + annual_parasitic_electricity
        + annual_extra_cycle_energy
    )
    annual_revenue = gross_annual_water_m3 * float(water_price_usd_per_m3)
    annual_net_cash_flow = annual_revenue - annual_opex

    if not math.isfinite(annual_net_cash_flow):
        return None

    lifetime_years = float(econ.system_lifetime_years)
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
