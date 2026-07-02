"""Shared TEA metric computation for SAWH black-box Excel workbooks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComponentElectricity:
    name: str
    shaft_power_w_per_m2: float
    motor_efficiency: float
    operating_hours_per_day: float
    grid_power_w_per_m2: float
    annual_kwh_per_m2: float
    annual_cost_usd_per_m2: float


@dataclass(frozen=True, slots=True)
class TeaMetrics:
    dry_composite_mass_kg_m2: float
    crf: float
    gross_annual_water_m3: float
    net_annual_water_m3: float
    unit_capex: tuple[float, ...]
    installed_capex: tuple[float, ...]
    annualized_capex: tuple[float, ...]
    installed_capex_total: float
    annualized_capex_total: float
    maintenance_annual: float
    hydrogel_annual: float
    hydrogel_salt_annual: float
    hydrogel_acrylamide_annual: float
    hydrogel_additives_annual: float
    fixed_energy_annual: float
    electricity_annual: float
    parasitic_electricity_annual: float
    component_electricity: tuple[ComponentElectricity, ...]
    extra_cycle_energy_annual: float
    fixed_opex_total: float
    variable_opex_total: float
    total_opex: float
    total_annual_cost: float
    lcow_usd_per_m3: float
    lcow_breakdown_annual: tuple[tuple[str, float], ...]
    lcow_breakdown_usd_per_m3: tuple[tuple[str, float], ...]


def capital_recovery_factor(discount_rate: float, device_lifetime_years: int) -> float:
    i = float(discount_rate)
    life = int(device_lifetime_years)
    return (i * (1.0 + i) ** life) / ((1.0 + i) ** life - 1.0)


def component_electricity_metrics(
    loads: tuple[tuple[str, float, float, float], ...],
    electricity_price_usd_per_kwh: float,
) -> tuple[ComponentElectricity, ...]:
    """Build per-component parasitic electricity rows from (name, W/m², η, h/d) tuples."""
    out: list[ComponentElectricity] = []
    price = float(electricity_price_usd_per_kwh)
    for name, shaft_w, eta, hours in loads:
        eta_f = float(eta)
        grid_w = float(shaft_w) / eta_f if eta_f > 0.0 else 0.0
        annual_kwh = grid_w * float(hours) * 365.0 / 1000.0
        out.append(
            ComponentElectricity(
                name=name,
                shaft_power_w_per_m2=float(shaft_w),
                motor_efficiency=eta_f,
                operating_hours_per_day=float(hours),
                grid_power_w_per_m2=grid_w,
                annual_kwh_per_m2=annual_kwh,
                annual_cost_usd_per_m2=price * annual_kwh,
            )
        )
    return tuple(out)


def compute_tea_metrics(
    defaults: dict[str, float | str],
    bom: tuple[tuple[str, float], ...],
    *,
    electricity_label: str = "Electricity (active heat)",
    parasitic_loads: tuple[tuple[str, float, float, float], ...] = (),
) -> TeaMetrics:
    sl = float(defaults["salt_to_polymer_ratio"])
    dry_mass = float(defaults["hydrogel_thickness_m"]) * float(defaults["hydrogel_density_kg_m3"])
    crf = capital_recovery_factor(
        float(defaults["discount_rate"]),
        int(defaults["device_lifetime_years"]),
    )
    inv = float(defaults["total_investment_factor"])
    cycles = float(defaults["cycles_per_day"])
    daily_yield = float(defaults["daily_yield_kg_per_m2"])
    util = float(defaults["utilization_factor"])
    gel_life = float(defaults["hydrogel_lifetime_years"])
    salt_p = float(defaults["salt_price_usd_per_kg"])
    acry = float(defaults["c_acrylamide_usd_per_kg"])
    add = float(defaults["c_additives_usd_per_kg_composite"])

    unit_capex = tuple(cost for _, cost in bom)
    installed_capex = tuple(inv * cost for cost in unit_capex)
    annualized_capex = tuple(crf * cost for cost in installed_capex)
    installed_total = sum(installed_capex)
    annualized_total = sum(annualized_capex)

    maintenance = float(defaults["maintenance_cost_fraction"]) * installed_total
    hydrogel_salt = salt_p * sl / (1.0 + sl) * dry_mass / gel_life
    hydrogel_acrylamide = acry / (1.0 + sl) * dry_mass / gel_life
    hydrogel_additives = add * dry_mass / gel_life
    hydrogel = hydrogel_salt + hydrogel_acrylamide + hydrogel_additives
    fixed_energy = float(defaults["energy_cost_usd_per_year"])
    elec_price = float(defaults["electricity_price_usd_per_kwh"])
    electricity = (
        elec_price
        * float(defaults["electric_heat_w_per_m2"])
        * float(defaults["desorption_hours_per_day"])
        * 365.0
        / 1000.0
    )
    component_electricity = component_electricity_metrics(parasitic_loads, elec_price)
    parasitic_electricity = sum(row.annual_cost_usd_per_m2 for row in component_electricity)
    extra_cycle = (
        max(0.0, cycles - 1.0)
        * 365.0
        * float(defaults["energy_cost_usd_per_extra_half_cycle_per_day"])
    )
    fixed_opex = maintenance + hydrogel + fixed_energy
    variable_opex = electricity + parasitic_electricity + extra_cycle
    total_opex = fixed_opex + variable_opex

    gross_water = cycles * 365.0 * daily_yield / 1000.0
    net_water = util * gross_water
    total_annual = annualized_total + total_opex
    lcow = total_annual / net_water if net_water > 0.0 else 0.0

    breakdown_annual: list[tuple[str, float]] = []
    breakdown_usd: list[tuple[str, float]] = []
    for (name, _), ann in zip(bom, annualized_capex):
        label = f"CAPEX: {name}"
        breakdown_annual.append((label, ann))
        breakdown_usd.append((label, ann / net_water if net_water > 0 else 0.0))
    breakdown_annual.append(("Maintenance", maintenance))
    breakdown_usd.append(("Maintenance", maintenance / net_water if net_water > 0 else 0.0))
    for label, val in (
        ("Hydrogel: salt", hydrogel_salt),
        ("Hydrogel: acrylamide", hydrogel_acrylamide),
        ("Hydrogel: additives", hydrogel_additives),
        ("Fixed energy", fixed_energy),
        (electricity_label, electricity),
    ):
        breakdown_annual.append((label, val))
        breakdown_usd.append((label, val / net_water if net_water > 0 else 0.0))
    for row in component_electricity:
        label = f"Electricity: {row.name}"
        breakdown_annual.append((label, row.annual_cost_usd_per_m2))
        breakdown_usd.append(
            (label, row.annual_cost_usd_per_m2 / net_water if net_water > 0 else 0.0)
        )
    breakdown_annual.append(("Extra cycling energy", extra_cycle))
    breakdown_usd.append(
        ("Extra cycling energy", extra_cycle / net_water if net_water > 0 else 0.0)
    )

    return TeaMetrics(
        dry_composite_mass_kg_m2=dry_mass,
        crf=crf,
        gross_annual_water_m3=gross_water,
        net_annual_water_m3=net_water,
        unit_capex=unit_capex,
        installed_capex=installed_capex,
        annualized_capex=annualized_capex,
        installed_capex_total=installed_total,
        annualized_capex_total=annualized_total,
        maintenance_annual=maintenance,
        hydrogel_annual=hydrogel,
        hydrogel_salt_annual=hydrogel_salt,
        hydrogel_acrylamide_annual=hydrogel_acrylamide,
        hydrogel_additives_annual=hydrogel_additives,
        fixed_energy_annual=fixed_energy,
        electricity_annual=electricity,
        parasitic_electricity_annual=parasitic_electricity,
        component_electricity=component_electricity,
        extra_cycle_energy_annual=extra_cycle,
        fixed_opex_total=fixed_opex,
        variable_opex_total=variable_opex,
        total_opex=total_opex,
        total_annual_cost=total_annual,
        lcow_usd_per_m3=lcow,
        lcow_breakdown_annual=breakdown_annual,
        lcow_breakdown_usd_per_m3=breakdown_usd,
    )
