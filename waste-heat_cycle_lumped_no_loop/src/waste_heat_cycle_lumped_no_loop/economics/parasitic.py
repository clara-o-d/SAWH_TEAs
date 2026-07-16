"""Parasitic grid electricity for waste-heat SAWH electrical components."""

from __future__ import annotations

from dataclasses import dataclass


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
