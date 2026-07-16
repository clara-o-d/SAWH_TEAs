"""Annual yield aggregation (optional multi-day operation)."""

from __future__ import annotations

from dataclasses import dataclass

from waste_heat_cycle_lumped.economics.parasitic import ParasiticLoadOptions
from waste_heat_cycle_lumped.economics.specific_energy import (
    SpecificEnergyBreakdown,
    specific_energy_breakdown_from_daily_operation,
)
from waste_heat_cycle_lumped.simulation.device_config import DeviceConfig
from waste_heat_cycle_lumped.simulation.ode_system import run_daily_operation
from waste_heat_cycle_lumped.weather.profiles import HalfCycleProfile


@dataclass(frozen=True, slots=True)
class SimulationResult:
    mean_daily_yield_kg_m2: float
    mean_daily_yield_l_m2: float
    mean_thermal_efficiency: float
    specific_energy: SpecificEnergyBreakdown
    n_days: int

    @property
    def specific_energy_wh_kwh_per_l(self) -> float:
        return self.specific_energy.wh_kwh_per_l

    @property
    def specific_energy_parasitic_kwh_per_l(self) -> float:
        return self.specific_energy.parasitic_kwh_per_l

    @property
    def specific_energy_total_kwh_per_l(self) -> float:
        return self.specific_energy.total_kwh_per_l

    @property
    def n_cycles_per_day(self) -> int:
        return self.specific_energy.n_cycles_per_day


def simulate_daily(
    profile: HalfCycleProfile,
    config: DeviceConfig,
    *,
    n_cycles: int | None = None,
    parasitic_options: ParasiticLoadOptions | None = None,
    electric_heat_w_per_m2: float = 0.0,
) -> SimulationResult:
    yield_kg, eta, results = run_daily_operation(profile, config, n_cycles=n_cycles)
    h_fg = config.thermal_params().h_fg_j_per_kg
    energy = specific_energy_breakdown_from_daily_operation(
        yield_kg,
        thermal_efficiency=eta,
        cycle_results=results,
        h_fg_j_per_kg=h_fg,
        parasitic_options=parasitic_options,
        electric_heat_w_per_m2=electric_heat_w_per_m2,
    )
    return SimulationResult(
        mean_daily_yield_kg_m2=yield_kg,
        mean_daily_yield_l_m2=yield_kg * 1000.0,
        mean_thermal_efficiency=eta,
        specific_energy=energy,
        n_days=1,
    )
