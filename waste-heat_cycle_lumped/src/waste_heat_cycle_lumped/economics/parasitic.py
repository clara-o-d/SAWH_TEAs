"""Parasitic grid electricity for waste-heat SAWH electrical components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from waste_heat_cycle_lumped.physics import device_defaults as dd
from waste_heat_cycle_lumped.simulation.ode_system import CycleResult
from waste_heat_cycle_lumped.simulation.operation_hours import (
    DailyOperationHours,
    daily_operating_hours_from_results,
)

_GRAVITY_M_S2 = 9.80665

LoadCategory = Literal[
    "htf_pump",
    "vacuum",
    "uptake_fan",
    "condenser_fan",
    "condenser_active",
    "aux",
]


@dataclass(frozen=True, slots=True)
class ElectricalLoadSpec:
    """One electrical component's parasitic load per m² footprint."""

    name: str
    shaft_power_w_per_m2: float
    motor_efficiency: float
    operating_hours_per_day: float
    category: LoadCategory = "aux"
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


@dataclass(frozen=True, slots=True)
class ParasiticLoadOptions:
    """Configuration for parasitic load models."""

    htf_head_m: float = 8.0
    use_simulation_hours: bool = True
    htf_operating_hours_per_day: float = 24.0
    vacuum_operating_hours_per_day: float = 12.0
    water_pump_hours_per_day: float = 2.0
    purge_pump_hours_per_day: float = 1.0
    controller_hours_per_day: float = 24.0
    include_uptake_fans: bool = False
    include_condenser_fans: bool = False
    uptake_fan_shaft_power_w_per_m2: float = 3.0
    condenser_fan_shaft_power_w_per_m2: float = 5.0
    fan_motor_efficiency: float = 0.85
    include_active_condenser_cooling: bool = False
    active_condenser_cooling_w_per_m2: float = 20.0
    active_condenser_cooling_efficiency: float = 1.0


def htf_pump_shaft_power_w_per_m2(
    *,
    m_dot_kg_s_m2: float,
    head_m: float,
    rho_kg_m3: float = dd.FLUID_RHO_KG_M3,
) -> float:
    """Hydraulic shaft power for the HTF transfer pump (W/m² footprint)."""
    q_m3_s_m2 = float(m_dot_kg_s_m2) / float(rho_kg_m3)
    return float(rho_kg_m3) * _GRAVITY_M_S2 * float(head_m) * q_m3_s_m2


def _resolve_operation_hours(
    results: list[CycleResult] | None,
    options: ParasiticLoadOptions,
) -> DailyOperationHours | None:
    if results is None or not options.use_simulation_hours:
        return None
    return daily_operating_hours_from_results(results)


def electrical_loads_for_operation(
    results: list[CycleResult] | None = None,
    *,
    options: ParasiticLoadOptions | None = None,
) -> tuple[ElectricalLoadSpec, ...]:
    """Build parasitic loads, optionally coupling operating hours to simulation."""
    opts = options or ParasiticLoadOptions()
    hours = _resolve_operation_hours(results, opts)

    if hours is not None:
        htf_hours = hours.operating_hours_per_day
        vacuum_hours = hours.desorption_hours_per_day
        uptake_hours = hours.absorption_hours_per_day
        condenser_fan_hours = hours.desorption_hours_per_day
        active_condenser_hours = hours.desorption_hours_per_day
    else:
        htf_hours = opts.htf_operating_hours_per_day
        vacuum_hours = opts.vacuum_operating_hours_per_day
        uptake_hours = opts.vacuum_operating_hours_per_day
        condenser_fan_hours = opts.vacuum_operating_hours_per_day
        active_condenser_hours = opts.vacuum_operating_hours_per_day

    htf_shaft = htf_pump_shaft_power_w_per_m2(
        m_dot_kg_s_m2=dd.M_F_BASE_KG_S_M2,
        head_m=opts.htf_head_m,
    )
    loads: list[ElectricalLoadSpec] = [
        ElectricalLoadSpec(
            name="Transfer pump (18)",
            shaft_power_w_per_m2=htf_shaft,
            motor_efficiency=0.55,
            operating_hours_per_day=htf_hours,
            category="htf_pump",
            notes=f"HTF loop: ρgHQ at ṁ={dd.M_F_BASE_KG_S_M2:.2f} kg/s/m², H={opts.htf_head_m:.0f} m",
        ),
        ElectricalLoadSpec(
            name="Vacuum pump (28)",
            shaft_power_w_per_m2=45.0,
            motor_efficiency=0.35,
            operating_hours_per_day=vacuum_hours,
            category="vacuum",
            notes="Roughing pump during desorption half-cycles",
        ),
        ElectricalLoadSpec(
            name="Water pump (34)",
            shaft_power_w_per_m2=3.0,
            motor_efficiency=0.50,
            operating_hours_per_day=opts.water_pump_hours_per_day,
            category="aux",
            notes="Product-water transfer",
        ),
        ElectricalLoadSpec(
            name="Purge pump (234)",
            shaft_power_w_per_m2=8.0,
            motor_efficiency=0.45,
            operating_hours_per_day=opts.purge_pump_hours_per_day,
            category="aux",
            notes="Manifold / valve purge",
        ),
        ElectricalLoadSpec(
            name="Controller (16) + sensors (36)",
            shaft_power_w_per_m2=2.5,
            motor_efficiency=0.85,
            operating_hours_per_day=opts.controller_hours_per_day,
            category="aux",
            notes="Controls and instrumentation",
        ),
    ]

    if opts.include_uptake_fans:
        loads.append(
            ElectricalLoadSpec(
                name="Uptake fan",
                shaft_power_w_per_m2=opts.uptake_fan_shaft_power_w_per_m2,
                motor_efficiency=opts.fan_motor_efficiency,
                operating_hours_per_day=uptake_hours,
                category="uptake_fan",
                notes="Forced process air over adsorbing contactor",
            )
        )
    if opts.include_condenser_fans:
        loads.append(
            ElectricalLoadSpec(
                name="Condenser fan",
                shaft_power_w_per_m2=opts.condenser_fan_shaft_power_w_per_m2,
                motor_efficiency=opts.fan_motor_efficiency,
                operating_hours_per_day=condenser_fan_hours,
                category="condenser_fan",
                notes="Forced air over finned condenser during desorption",
            )
        )
    if opts.include_active_condenser_cooling:
        loads.append(
            ElectricalLoadSpec(
                name="Active condenser cooling",
                shaft_power_w_per_m2=opts.active_condenser_cooling_w_per_m2,
                motor_efficiency=opts.active_condenser_cooling_efficiency,
                operating_hours_per_day=active_condenser_hours,
                category="condenser_active",
                notes="Chiller or active heat rejection for condenser",
            )
        )
    return tuple(loads)


def default_electrical_loads(
    *,
    htf_head_m: float = 8.0,
    htf_operating_hours_per_day: float = 24.0,
    vacuum_operating_hours_per_day: float = 12.0,
) -> tuple[ElectricalLoadSpec, ...]:
    """Default parasitic loads with fixed operating hours (LCOW / black-box TEA)."""
    return electrical_loads_for_operation(
        None,
        options=ParasiticLoadOptions(
            htf_head_m=htf_head_m,
            use_simulation_hours=False,
            htf_operating_hours_per_day=htf_operating_hours_per_day,
            vacuum_operating_hours_per_day=vacuum_operating_hours_per_day,
        ),
    )


def total_parasitic_electricity_annual_usd_per_m2(
    loads: tuple[ElectricalLoadSpec, ...],
    electricity_price_usd_per_kwh: float,
) -> float:
    return sum(
        load.annual_cost_usd_per_m2(electricity_price_usd_per_kwh) for load in loads
    )


def parasitic_electricity_breakdown(
    loads: tuple[ElectricalLoadSpec, ...],
    electricity_price_usd_per_kwh: float,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (
            f"Electricity: {load.name}",
            load.annual_cost_usd_per_m2(electricity_price_usd_per_kwh),
        )
        for load in loads
    )
