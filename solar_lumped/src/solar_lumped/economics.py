"""Levelized cost of water (LCOW), NPV, and payback economics for the solar SAWH system."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

from solar_lumped._parameters_xlsx import ECONOMICS as _ECON_XLSX
from solar_lumped._parameters_xlsx import PHYSICS as _PHYS_XLSX
from solar_lumped._parameters_xlsx import economics_value as _ev
from solar_lumped._parameters_xlsx import physics_value as _pv
from solar_lumped.physics import DRY_COMPOSITE_DENSITY_KG_M3, get_salt_price_usd_per_kg

# --- Economic parameter loading and defaults ---
# Read directly from docs/parameters.xlsx (Economics + Physics sheets); no separate CSV.

_BOM_PREFIX = "BOM: "

# LCOEconomicParams field name -> parameters.xlsx Economics-sheet row name.
_ECON_FIELD_ROWS: dict[str, str] = {
    "discount_rate": "Discount rate (i)",
    "system_lifetime_years": "Device lifetime (L)",
    "total_investment_factor": "Total investment factor (f_inv)",
    "maintenance_cost_fraction": "Maintenance cost fraction (f_maint)",
    "utilization_factor": "Utilization factor (f_util)",
    "hydrogel_lifetime_years": "Hydrogel lifetime (L_gel)",
    "c_am_usd_per_kg": "Acrylamide price, AM (c_AM)",
    "c_aps_usd_per_kg": "Ammonium persulfate price, APS (c_APS)",
    "c_mba_usd_per_kg": "N,N'-methylenebisacrylamide price, MBA (c_MBA)",
    "c_temed_usd_per_kg": "Tetramethylethylenediamine price, TEMED (c_TEMED)",
    "c_water_gel_usd_per_kg": "De-ionized water price, hydrogel manufacturing (c_water_gel)",
    "include_desorption_enthalpy": "Include desorption enthalpy in T_g solve (flag)",
}

# No electricity / electric-heat rows: solar_lumped is the *passive* solar system, so
# there is no purchased-energy term at all (Wilson's Note S4 TEA likewise has none).
# The parameters.xlsx Economics sheet still carries "Electricity price (p_elec)",
# "Desorption hours per day (t_des)" and "Max electric heat, optimizer bound
# (Q_elec,max)" rows -- deliberately left unread here, since the waste-heat packages
# (which do have parasitic/supplemental electric loads) read the same workbook.


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
        missing.append("BOM: *")
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
    c_am_usd_per_kg: float
    c_aps_usd_per_kg: float
    c_mba_usd_per_kg: float
    c_temed_usd_per_kg: float
    c_water_gel_usd_per_kg: float
    include_desorption_enthalpy: bool

    def __init__(self, **kwargs: Any) -> None:
        defaults = _LCOW_DEFAULTS
        for f in fields(self):
            value = kwargs[f.name] if f.name in kwargs else defaults[f.name]
            object.__setattr__(self, f.name, value)

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

# Polymer sub-mix (AM/APS/MBA/TEMED) blended price and water ratio from Table S1 batch
# mass fractions, renormalized since those four don't sum to 1 (the batch has salt+water).
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

# Closure check on the Table S1 recipe: salt + the four polymer components + water are
# the whole batch. If they stop summing to 1 the renormalization above is dividing by a
# denominator that no longer means "the polymer share", so fail loudly at import.
_TABLE_S1_TOTAL = (
    float(_ev("LiCl mass fraction, salt (Table S1 recipe)"))
    + _polymer_fraction_sum
    + _water_gel_fraction
)
if abs(_TABLE_S1_TOTAL - 1.0) > 1e-9:
    raise ValueError(
        f"Table S1 mass fractions in parameters.xlsx sum to {_TABLE_S1_TOTAL!r}, not 1.0"
    )

# --- Levelized cost of water (LCOW) ---

FAIL_LCO: float = 1e30


def complex_system_cost_usd(complex_options, *, fin_area_ratio: float) -> float:
    """System BOM under complex mode: the flat Wilson BOM plus the priced deltas.

    ``complex_options`` of ``None`` returns the flat simple-model BOM unchanged, so
    every existing caller is unaffected. Otherwise B1 (coating), B2 (glazing), B3
    (fin aluminum), and B4 (fans + PV) each add a signed delta against Wilson's own
    build -- see ``complex_model.complex_system_capex_usd_per_m2``.
    """
    if complex_options is None:
        return C_SYSTEM_USD
    from solar_lumped.complex_model import complex_system_capex_usd_per_m2

    return float(
        C_SYSTEM_USD
        + complex_system_capex_usd_per_m2(complex_options, fin_area_ratio=fin_area_ratio)
    )


def lcow_from_daily_yield(
    yield_per_cycle_kg_per_m2: float,
    *,
    salt_name: str,
    salt_loading: float,
    hydrogel_thickness_m: float,
    econ: LCOEconomicParams,
    cycles_per_day: float = 1.0,
    salt_price_usd_per_kg: float | None = None,
    simplified: bool = False,
    complex_options=None,
    fin_area_ratio: float | None = None,
) -> float:
    """Scalar LCOW (USD/m³), same structure as lcow_zsr_at_sl. ``simplified=True`` drops the
    DI-water and non-acrylamide polymer (APS/MBA/TEMED) terms from the hydrogel line.

    ``complex_options`` (a ``ComplexOptions``) switches on the complex-fidelity cost
    model: a design-dependent system BOM, a ZSR-blended salt price, and the annual
    fan-replacement stream. It requires ``fin_area_ratio``, since B3 prices fin
    aluminum per unit of added area.

    Takes yield **per cycle**, not per day: annual water is
    ``cycles_per_day * 365 * yield_per_cycle_kg_per_m2``, and ``cycles_per_day`` also
    drives the per-cycle energy term, so it has to be the true cycle count. This device
    runs one cycle per day, so every caller here leaves ``cycles_per_day`` at 1.0 and
    passes the day's yield. Feeding a multi-cycle daily total *and* the real cycle count
    double-counts the water by that count -- see the same note on waste_heat's copy."""
    if yield_per_cycle_kg_per_m2 <= 0.0 or not math.isfinite(yield_per_cycle_kg_per_m2):
        return FAIL_LCO
    if (
        salt_loading <= 0.0 or not math.isfinite(salt_loading)
    ):
        return FAIL_LCO
    if complex_options is not None and fin_area_ratio is None:
        raise ValueError("complex_options requires fin_area_ratio (B3 prices fin aluminum)")

    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(yield_per_cycle_kg_per_m2)
    if complex_options is not None and salt_price_usd_per_kg is None:
        from solar_lumped.complex_model import zsr_blend_price_usd_per_kg
        from solar_lumped.physics import FABRICATION_EQUILIBRIUM_RH

        salt_price_usd_per_kg = zsr_blend_price_usd_per_kg(
            complex_options.blend_weights, reference_rh=FABRICATION_EQUILIBRIUM_RH
        )
        if not math.isfinite(salt_price_usd_per_kg):
            return FAIL_LCO
    sorbent_replacement = _sorbent_replacement_annual_usd(
        salt_name=salt_name,
        salt_loading=salt_loading,
        hydrogel_thickness_m=hydrogel_thickness_m,
        econ=econ,
        salt_price_usd_per_kg=salt_price_usd_per_kg,
        simplified=simplified,
    )

    c_system = complex_system_cost_usd(complex_options, fin_area_ratio=fin_area_ratio or 0.0)
    annual_fan_replacement = 0.0
    if complex_options is not None:
        from solar_lumped.complex_model import forced_cooling_annual_replacement_usd_per_m2

        annual_fan_replacement = forced_cooling_annual_replacement_usd_per_m2(
            complex_options.condenser_air_speed_m_s
        )

    annual_cost_usd = (
        econ.capital_recovery_factor() * econ.total_investment_factor * c_system
        + sorbent_replacement
        + econ.maintenance_cost_fraction * c_system
        + annual_fan_replacement
    )
    if not math.isfinite(annual_cost_usd):
        return FAIL_LCO
    return float(
        annual_cost_usd
        / (econ.utilization_factor * (annual_water_yield_kg / KG_WATER_PER_M3 + 1e-9))
    )


def _sorbent_replacement_annual_usd(
    *,
    salt_name: str,
    salt_loading: float,
    hydrogel_thickness_m: float,
    econ: LCOEconomicParams,
    salt_price_usd_per_kg: float | None = None,
    simplified: bool = False,
) -> float:
    gel_lifetime = econ.hydrogel_lifetime_years
    sl = salt_loading
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
    """Dry (solids-only) composite mass per m² at the given thickness, using the DVS
    dry-basis density -- Table S3's rho_gel is measured at 20% RH and already carries
    ~126% water, while the hydrogel cost below is priced per kg of dry solids."""
    return float(hydrogel_thickness_m) * DRY_COMPOSITE_DENSITY_KG_M3


@dataclass(frozen=True, slots=True)
class LcowCostBreakdown:
    items: tuple[tuple[str, float], ...]

    @property
    def total_usd_per_m3(self) -> float:
        return float(sum(v for _, v in self.items))


def lcow_cost_breakdown_from_daily_yield(
    yield_per_cycle_kg_per_m2: float,
    *,
    salt_name: str,
    salt_loading: float,
    hydrogel_thickness_m: float,
    econ: LCOEconomicParams,
    cycles_per_day: float = 1.0,
    salt_price_usd_per_kg: float | None = None,
    simplified: bool = False,
) -> LcowCostBreakdown | None:
    """Per-term LCOW breakdown, same segments as lcow_cost_breakdown_usd_per_m3.
    ``simplified=True`` drops the water line and bills only the acrylamide share."""
    lcow = lcow_from_daily_yield(
        yield_per_cycle_kg_per_m2,
        salt_name=salt_name,
        salt_loading=salt_loading,
        hydrogel_thickness_m=hydrogel_thickness_m,
        econ=econ,
        cycles_per_day=cycles_per_day,
        salt_price_usd_per_kg=salt_price_usd_per_kg,
        simplified=simplified,
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
        segments.append((f"CAPEX: {name}", _lcow_seg(crf * inv * line_cost)))
        maintenance_annual += maint_frac * line_cost
    segments.append(("Maintenance", _lcow_seg(maintenance_annual)))

    salt_price = (
        salt_price_usd_per_kg
        if salt_price_usd_per_kg is not None
        else get_salt_price_usd_per_kg(salt_name)
    )
    salt_annual = salt_price * sl / (1.0 + sl) * dry_mass / gel_lifetime
    segments.append(("Hydrogel: salt", _lcow_seg(salt_annual)))
    # Split the blended polymer cost back into ingredients by renormalized Table S1
    # share; simplified mode bills acrylamide only.
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

    return LcowCostBreakdown(items=tuple(segments))

# --- Net present value (NPV) and payback period ---


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
    salt_price_usd_per_kg: float | None = None,
) -> NpvResult | None:
    """NPV and payback period (USD/m2 of system footprint) for one site."""
    if yield_per_cycle_kg_per_m2 <= 0.0 or not math.isfinite(yield_per_cycle_kg_per_m2):
        return None
    if (
        salt_loading <= 0.0 or not math.isfinite(salt_loading)
    ):
        return None

    annual_water_yield_kg = float(cycles_per_day) * 365.0 * float(yield_per_cycle_kg_per_m2)
    gross_annual_water_m3 = econ.utilization_factor * (annual_water_yield_kg / KG_WATER_PER_M3)

    sorbent_replacement = _sorbent_replacement_annual_usd(
        salt_name=salt_name,
        salt_loading=salt_loading,
        hydrogel_thickness_m=hydrogel_thickness_m,
        econ=econ,
        salt_price_usd_per_kg=salt_price_usd_per_kg,
    )
    capex = econ.total_investment_factor * C_SYSTEM_USD
    annual_opex = sorbent_replacement + econ.maintenance_cost_fraction * C_SYSTEM_USD
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
