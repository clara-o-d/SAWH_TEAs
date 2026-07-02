"""Parasitic grid electricity for waste-heat SAWH electrical components."""

from __future__ import annotations

from dataclasses import dataclass

from waste_heat_lumped.physics import device_defaults as dd

_GRAVITY_M_S2 = 9.80665


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


def htf_pump_shaft_power_w_per_m2(
    *,
    m_dot_kg_s_m2: float,
    head_m: float,
    rho_kg_m3: float = dd.FLUID_RHO_KG_M3,
) -> float:
    """Hydraulic shaft power for the HTF transfer pump (W/m² footprint)."""
    q_m3_s_m2 = float(m_dot_kg_s_m2) / float(rho_kg_m3)
    return float(rho_kg_m3) * _GRAVITY_M_S2 * float(head_m) * q_m3_s_m2


def default_electrical_loads(
    *,
    htf_head_m: float = 8.0,
    htf_operating_hours_per_day: float = 24.0,
    vacuum_operating_hours_per_day: float = 12.0,
) -> tuple[ElectricalLoadSpec, ...]:
    """Default parasitic loads tied to the data-center baseline device."""
    htf_shaft = htf_pump_shaft_power_w_per_m2(
        m_dot_kg_s_m2=dd.M_F_BASE_KG_S_M2,
        head_m=htf_head_m,
    )
    return (
        ElectricalLoadSpec(
            name="Transfer pump (18)",
            shaft_power_w_per_m2=htf_shaft,
            motor_efficiency=0.55,
            operating_hours_per_day=htf_operating_hours_per_day,
            notes=f"HTF loop: ρgHQ at ṁ={dd.M_F_BASE_KG_S_M2:.2f} kg/s/m², H={htf_head_m:.0f} m",
        ),
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
