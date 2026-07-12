"""Annual yield aggregation (optional multi-day operation)."""

from __future__ import annotations

from dataclasses import dataclass

from waste_heat_cycle_lumped.simulation.device_config import DeviceConfig
from waste_heat_cycle_lumped.simulation.ode_system import run_daily_operation
from waste_heat_cycle_lumped.weather.profiles import HalfCycleProfile


@dataclass(frozen=True, slots=True)
class SimulationResult:
    mean_daily_yield_kg_m2: float
    mean_daily_yield_l_m2: float
    mean_thermal_efficiency: float
    n_days: int


def simulate_daily(
    profile: HalfCycleProfile,
    config: DeviceConfig,
    *,
    n_cycles: int | None = None,
) -> SimulationResult:
    yield_kg, eta, _ = run_daily_operation(profile, config, n_cycles=n_cycles)
    return SimulationResult(
        mean_daily_yield_kg_m2=yield_kg,
        mean_daily_yield_l_m2=yield_kg * 1000.0,
        mean_thermal_efficiency=eta,
        n_days=1,
    )
