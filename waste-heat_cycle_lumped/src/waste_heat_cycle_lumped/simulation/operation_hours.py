"""Operating hours derived from daily cycle simulation results."""

from __future__ import annotations

from dataclasses import dataclass

from waste_heat_cycle_lumped.simulation.ode_system import CycleResult


_DAY_HOURS = 24.0


@dataclass(frozen=True, slots=True)
class DailyOperationHours:
    """Per-day operating hours per m² footprint."""

    n_cycles: int
    desorption_hours_per_day: float
    absorption_hours_per_day: float
    operating_hours_per_day: float


def daily_operating_hours_from_results(results: list[CycleResult]) -> DailyOperationHours:
    """Hours when beds are actively cycling (one adsorbing, one desorbing at all times).

    Desorption and absorption each span the full cycle duration because the two
    beds alternate roles every half-cycle.
    """
    operating_s = 0.0
    for cyc in results:
        operating_s += float(cyc.half_a.time_s[-1]) + float(cyc.half_b.time_s[-1])
    operating_h = min(operating_s / 3600.0, _DAY_HOURS)
    return DailyOperationHours(
        n_cycles=len(results),
        desorption_hours_per_day=operating_h,
        absorption_hours_per_day=operating_h,
        operating_hours_per_day=operating_h,
    )
