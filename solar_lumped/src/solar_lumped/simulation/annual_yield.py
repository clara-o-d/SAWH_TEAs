"""Annual yield aggregation over real weather days."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.simulation.ode_system import cycle_end_state, run_daily_cycle
from solar_lumped.weather.profiles import DailyWeatherProfile


@dataclass(frozen=True, slots=True)
class SimulationResult:
    mean_daily_yield_kg_m2: float
    mean_daily_yield_l_m2: float
    mean_thermal_efficiency: float
    n_days: int
    daily_yields_kg_m2: tuple[float, ...]


def simulate_single_day(
    profile: DailyWeatherProfile,
    config: DeviceConfig,
    *,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
) -> tuple[float, float, tuple[float, float]]:
    """Run one day; return (yield, eta, (c_w, H) after desorption)."""
    yield_kg, eta, _, des_res = run_daily_cycle(
        profile,
        config,
        c_w_initial=c_w_initial,
        h_initial=h_initial,
    )
    return yield_kg, eta, cycle_end_state(des_res)


def aggregate_yields(
    day_profiles: list[tuple[date, DailyWeatherProfile]] | list[DailyWeatherProfile],
    config: DeviceConfig,
    *,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
    warmup: bool = False,
) -> SimulationResult:
    yields: list[float] = []
    etas: list[float] = []
    cw, h = c_w_initial, h_initial
    for i, item in enumerate(day_profiles):
        prof = item[1] if isinstance(item, tuple) else item
        y, eta, (cw, h) = simulate_single_day(
            prof, config, c_w_initial=cw, h_initial=h
        )
        if warmup and i == 0:
            continue
        if y >= 0.0:
            yields.append(y)
            etas.append(eta)
    if not yields:
        return SimulationResult(0.0, 0.0, 0.0, 0, tuple())
    mean_y = sum(yields) / len(yields)
    mean_eta = sum(etas) / len(etas)
    return SimulationResult(
        mean_daily_yield_kg_m2=mean_y,
        mean_daily_yield_l_m2=mean_y,
        mean_thermal_efficiency=mean_eta,
        n_days=len(yields),
        daily_yields_kg_m2=tuple(yields),
    )
