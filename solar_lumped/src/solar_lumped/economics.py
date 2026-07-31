"""Levelized cost of water (LCOW), NPV, and payback economics for the solar SAWH device."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

from solar_lumped._parameters_xlsx import ECONOMICS as _ECON_XLSX
from solar_lumped._parameters_xlsx import PHYSICS as _PHYS_XLSX
from solar_lumped._parameters_xlsx import economics_value as _ev
from solar_lumped._parameters_xlsx import physics_value as _pv
from solar_lumped.physics import DRY_COMPOSITE_DENSITY_KG_M3, get_salt_price_usd_per_kg

# =============================================================================
# Economic parameter loading and defaults -- read directly from
# docs/parameters.xlsx (Economics + Physics sheets); no separate CSV.
# =============================================================================

_BOM_PREFIX = "BOM: "

# LCOEconomicParams field name -> parameters.xlsx Economics-sheet row name.
_ECON_FIELD_ROWS: dict[str, str] = {
    "discount_rate": "Discount rate (i)",
    "device_lifetime_years": "Device lifetime (L)",
    "total_investment_factor": "Total investment factor (f_inv)",
    "maintenance_cost_fraction": "Maintenance cost fraction (f_maint)",
    "utilization_factor": "Utilization factor (f_util)",
    "hydrogel_lifetime_years": "Hydrogel lifetime (L_gel)",
    "c_am_usd_per_kg": "Acrylamide price, AM (c_AM)",
    "c_aps_usd_per_kg": "Ammonium persulfate price, APS (c_APS)",
    "c_mba_usd_per_kg": "N,N'-methylenebisacrylamide price, MBA (c_MBA)",
    "c_temed_usd_per_kg": "Tetramethylethylenediamine price, TEMED (c_TEMED)",
    "c_water_gel_usd_per_kg": "De-ionized water price, hydrogel manufacturing (c_water_gel)",
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
        if field_name == "device_lifetime_years":
            scalars[field_name] = int(round(float(raw)))
        elif field_name == "include_desorption_enthalpy":
            scalars[field_name] = _coerce_bool(raw)
        else:
            scalars[field_name] = float(raw)

    # Hydrogel geometry/density and mass-transfer coefficient live on the
    # Physics sheet (they're physical, not economic, quantities) but are
    # needed here for the sorbent-replacement cost calculation.
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
        missing.append("BOM: *")
    if missing:
        raise ValueError(f"Missing required parameters in parameters.xlsx: {', '.join(missing)}")
    return scalars, bom_rows


@dataclass(frozen=True, slots=True, init=False)
class LCOEconomicParams:
    """LCOW = annual_cost / (utilization_factor * gross_annual_water_m3)."""

    discount_rate: float
    device_lifetime_years: int
    total_investment_factor: float
    maintenance_cost_fraction: float
    utilization_factor: float
    hydrogel_lifetime_years: float
    c_am_usd_per_kg: float
    c_aps_usd_per_kg: float
    c_mba_usd_per_kg: float
    c_temed_usd_per_kg: float
    c_water_gel_usd_per_kg: float
    electricity_price_usd_per_kwh: float
    desorption_hours_per_day: float
    max_electric_heat_w_per_m2: float
    include_desorption_enthalpy: bool

    def __init__(self, **kwargs: Any) -> None:
        defaults = _LCOW_DEFAULTS
        for f in fields(self):
            value = kwargs[f.name] if f.name in kwargs else defaults[f.name]
            object.__setattr__(self, f.name, value)

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
KG_WATER_PER_M3: float = _ev("Water density (rho_w)")
DEVICE_BOM_USD_PER_M2: tuple[tuple[str, float], ...] = _DEVICE_BOM_ROWS
C_DEVICE_USD: float = sum(cost for _, cost in DEVICE_BOM_USD_PER_M2)
_LCOW_DEFAULTS: dict[str, Any] = {f.name: _SCALARS[f.name] for f in fields(LCOEconomicParams)}

# Polymer sub-mix (AM/APS/MBA/TEMED) blended price and water-to-dry-composite
# ratio, derived from Table S1's whole-batch mass fractions (each tracked as
# its own parameters.xlsx Economics-sheet row). Renormalized because those
# four items' fractions don't sum to 1 on their own (the batch also contains
# salt and water).
_POLYMER_ITEMS: tuple[tuple[str, str], ...] = (
    ("Acrylamide price, AM (c_AM)", "Acrylamide mass fraction, AM (Table S1 recipe)"),
    ("Ammonium persulfate price, APS (c_APS)", "Ammonium persulfate mass fraction, APS (Table S1 recipe)"),
    (
        "N,N'-methylenebisacrylamide price, MBA (c_MBA)",
        "N,N'-methylenebisacrylamide mass fraction, MBA (Table S1 recipe)",
    ),
    (
        "Tetramethylethylenediamine price, TEMED (c_TEMED)",
        "Tetramethylethylenediamine mass fraction, TEMED (Table S1 recipe)",
    ),
)
_polymer_fractions = tuple(float(_ev(frac_row)) for _price_row, frac_row in _POLYMER_ITEMS)
_polymer_fraction_sum = sum(_polymer_fractions)
POLYMER_BLENDED_PRICE_USD_PER_KG: float = sum(
    frac * float(_ev(price_row))
    for (price_row, _frac_row), frac in zip(_POLYMER_ITEMS, _polymer_fractions)
) / _polymer_fraction_sum

# Simplified-TEA polymer price: only the acrylamide (AM) share of the renormalized
# Table S1 polymer sub-mix is costed; APS/MBA/TEMED are treated as free.
_AM_FRACTION_SHARE: float = _polymer_fractions[0] / _polymer_fraction_sum
POLYMER_AM_ONLY_PRICE_USD_PER_KG: float = _AM_FRACTION_SHARE * float(_ev(_POLYMER_ITEMS[0][0]))

_water_gel_fraction = float(_ev("De-ionized water mass fraction (Table S1 recipe)"))
WATER_RATIO_PER_KG_DRY_COMPOSITE: float = _water_gel_fraction / (1.0 - _water_gel_fraction)

# =============================================================================
# Levelized cost of water (LCOW)
# =============================================================================

FAIL_LCO: float = 1e30


def _annual_electricity_cost_usd(econ: LCOEconomicParams, electric_heat_w_per_m2: float) -> float:
    return (
        econ.electricity_price_usd_per_kwh
        * float(electric_heat_w_per_m2)
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
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
    sorbent: str = "hydrogel",
    mof_mass_kg_m2: float = 0.0,
    mof_price_usd_per_kg: float = 0.0,
    simplified: bool = False,
) -> float:
    """Scalar LCOW (USD/m³) — identical structure to lcow_zsr_at_sl.

    ``simplified=True`` drops the DI-water and non-acrylamide polymer (APS/MBA/
    TEMED) cost terms from the hydrogel line item -- see
    ``_sorbent_replacement_annual_usd``.
    """
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
        simplified=simplified,
    )

    annual_electricity_cost = _annual_electricity_cost_usd(econ, electric_heat_w_per_m2)
    annual_cost_usd = (
        econ.capital_recovery_factor() * econ.total_investment_factor * C_DEVICE_USD
        + sorbent_replacement
        + econ.maintenance_cost_fraction * C_DEVICE_USD
        + annual_electricity_cost
    )
    if not math.isfinite(annual_cost_usd):
        return FAIL_LCO
    return float(
        annual_cost_usd
        / (econ.utilization_factor * (annual_water_yield_kg / KG_WATER_PER_M3 + 1e-9))
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
    simplified: bool = False,
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
    if simplified:
        # No DI-water cost, and only the acrylamide share of the polymer sub-mix
        # is costed (APS/MBA/TEMED treated as free).
        hydrogel_cost_per_kg = (salt_price * sl + POLYMER_AM_ONLY_PRICE_USD_PER_KG) / (1.0 + sl)
    else:
        hydrogel_cost_per_kg = (
            (salt_price * sl + POLYMER_BLENDED_PRICE_USD_PER_KG) / (1.0 + sl)
            + WATER_RATIO_PER_KG_DRY_COMPOSITE * econ.c_water_gel_usd_per_kg
        )
    return hydrogel_cost_per_kg * dry_mass / gel_lifetime


def dry_composite_mass_kg(hydrogel_thickness_m: float) -> float:
    """Dry (solids-only) composite mass per m^2 at the given thickness.

    Uses the DVS-derived dry-basis density, not the raw Table S3 rho_gel (that's
    measured at 20% RH and already carries ~126% equilibrium water by mass --
    see physics.py::DRY_COMPOSITE_DENSITY_KG_M3), since the recipe-based hydrogel
    cost below is priced per kg of dry (post-synthesis) solids.
    """
    return float(hydrogel_thickness_m) * DRY_COMPOSITE_DENSITY_KG_M3


@dataclass(frozen=True, slots=True)
class LcowCostBreakdown:
    items: tuple[tuple[str, float], ...]

    @property
    def total_usd_per_m3(self) -> float:
        return float(sum(v for _, v in self.items))


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
    simplified: bool = False,
) -> LcowCostBreakdown | None:
    """Per-term LCOW breakdown — same segments as lcow_cost_breakdown_usd_per_m3.

    ``simplified=True`` matches ``lcow_from_daily_yield(simplified=True)``: the
    water line item is dropped and only the acrylamide (AM) share of the
    polymer sub-mix is billed.
    """
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
        simplified=simplified,
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
        segments.append((f"CAPEX: {name}", _lcow_seg(crf * inv * line_cost)))
        maintenance_annual += maint_frac * line_cost
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
        segments.append(("Hydrogel: salt", _lcow_seg(salt_annual)))
        # Split the polymer sub-mix's blended cost back out into its ingredients
        # using each one's share of the renormalized Table S1 mix. Simplified
        # mode only bills acrylamide (APS/MBA/TEMED/water are dropped).
        ingredients = (
            (_polymer_fractions[0], econ.c_am_usd_per_kg, "Hydrogel: acrylamide (AM)"),
        ) if simplified else (
            (_polymer_fractions[0], econ.c_am_usd_per_kg, "Hydrogel: acrylamide (AM)"),
            (_polymer_fractions[1], econ.c_aps_usd_per_kg, "Hydrogel: APS"),
            (_polymer_fractions[2], econ.c_mba_usd_per_kg, "Hydrogel: MBA"),
            (_polymer_fractions[3], econ.c_temed_usd_per_kg, "Hydrogel: TEMED"),
        )
        for frac, econ_price, label in ingredients:
            ingredient_annual = (
                (frac / _polymer_fraction_sum) * econ_price / (1.0 + sl) * dry_mass / gel_lifetime
            )
            segments.append((label, _lcow_seg(ingredient_annual)))
        if not simplified:
            water_annual = (
                WATER_RATIO_PER_KG_DRY_COMPOSITE * econ.c_water_gel_usd_per_kg * dry_mass / gel_lifetime
            )
            segments.append(("Hydrogel: water", _lcow_seg(water_annual)))

    annual_electricity_cost = _annual_electricity_cost_usd(econ, electric_heat_w_per_m2)
    segments.append(("Electricity (active heat)", _lcow_seg(annual_electricity_cost)))

    return LcowCostBreakdown(items=tuple(segments))

# =============================================================================
# Net present value (NPV) and payback period
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
    annual_electricity_cost = _annual_electricity_cost_usd(econ, electric_heat_w_per_m2)
    capex = econ.total_investment_factor * C_DEVICE_USD
    annual_opex = (
        sorbent_replacement
        + econ.maintenance_cost_fraction * C_DEVICE_USD
        + annual_electricity_cost
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
