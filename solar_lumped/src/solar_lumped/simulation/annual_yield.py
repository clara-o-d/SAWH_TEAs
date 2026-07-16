"""Annual yield aggregation over real weather days."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from solar_lumped.simulation.detailed_plots import (
    DetailedSeries,
    detailed_series,
    write_detailed_csv,
)
from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.simulation.ode_system import cycle_end_state, run_daily_cycle
from solar_lumped.simulation.water_inventory import (
    WaterInventorySeries,
    water_inventory_series,
    write_water_inventory_csv,
)
from solar_lumped.weather.climate import day_weather_stats
from solar_lumped.weather.profiles import DailyWeatherProfile


@dataclass(frozen=True, slots=True)
class SimulationResult:
    mean_daily_yield_kg_m2: float
    mean_daily_yield_l_m2: float
    mean_thermal_efficiency: float
    n_days: int
    daily_yields_kg_m2: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DailySimulationRecord:
    date: date
    day_of_year: int
    rh_avg_frac: float
    rh_peak_frac: float
    temp_avg_c: float
    temp_peak_c: float
    solar_avg_w_m2: float
    solar_peak_w_m2: float
    daily_yield_kg_m2: float
    daily_yield_l_m2: float
    eta_thermal: float
    water_uptake_l_m2: float
    water_release_l_m2: float
    t_abs_peak_c: float
    t_glass_peak_c: float
    t_cond_peak_c: float
    t_gel_peak_c: float


DAILY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "date",
    "day_of_year",
    "rh_avg_frac",
    "rh_peak_frac",
    "temp_avg_c",
    "temp_peak_c",
    "solar_avg_w_m2",
    "solar_peak_w_m2",
    "daily_yield_kg_m2",
    "daily_yield_l_m2",
    "eta_thermal",
    "water_uptake_l_m2",
    "water_release_l_m2",
    "t_abs_peak_c",
    "t_glass_peak_c",
    "t_cond_peak_c",
    "t_gel_peak_c",
)


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


def _water_uptake_l_m2(
    inventory: WaterInventorySeries,
    *,
    absorption_end_s: float,
) -> float:
    abs_mask = inventory.time_s <= absorption_end_s + 1e-9
    water_abs = inventory.water_l_m2[abs_mask]
    if len(water_abs) == 0:
        return 0.0
    start = float(water_abs[0])
    return max(0.0, float(water_abs.max()) - start)


def _record_from_day(
    day_key: date,
    profile: DailyWeatherProfile,
    day_df: pd.DataFrame,
    config: DeviceConfig,
    *,
    c_w_initial: float | None,
    h_initial: float | None,
) -> tuple[DailySimulationRecord, tuple[float, float], DetailedSeries, WaterInventorySeries]:
    yield_kg, eta, abs_res, des_res = run_daily_cycle(
        profile,
        config,
        c_w_initial=c_w_initial,
        h_initial=h_initial,
    )
    detailed = detailed_series(profile, abs_res, des_res, config)
    inventory = water_inventory_series(abs_res, des_res, config=config)
    weather = day_weather_stats(day_df)

    record = DailySimulationRecord(
        date=day_key,
        day_of_year=day_key.timetuple().tm_yday,
        rh_avg_frac=weather.get("rh_avg_frac", 0.0),
        rh_peak_frac=weather.get("rh_peak_frac", 0.0),
        temp_avg_c=weather.get("temp_avg_c", 0.0),
        temp_peak_c=weather.get("temp_peak_c", 0.0),
        solar_avg_w_m2=weather.get("solar_avg_w_m2", 0.0),
        solar_peak_w_m2=weather.get("solar_peak_w_m2", 0.0),
        daily_yield_kg_m2=float(yield_kg),
        daily_yield_l_m2=float(yield_kg),
        eta_thermal=float(eta),
        water_uptake_l_m2=_water_uptake_l_m2(
            inventory,
            absorption_end_s=inventory.absorption_end_s,
        ),
        water_release_l_m2=float(yield_kg),
        t_abs_peak_c=float(detailed.t_abs_c.max()),
        t_glass_peak_c=float(detailed.t_glass_c.max()),
        t_cond_peak_c=float(detailed.t_cond_c.max()),
        t_gel_peak_c=float(detailed.t_gel_c.max()),
    )
    return record, cycle_end_state(des_res), detailed, inventory


def warmup_on_profile(
    profile: DailyWeatherProfile,
    config: DeviceConfig,
    *,
    n_cycles: int = 2,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
) -> tuple[float, float]:
    """Run repeated daily cycles on one profile; return post-desorption (c_w, H)."""
    cw, h = c_w_initial, h_initial
    for _ in range(max(0, n_cycles)):
        _, _, _, des_res = run_daily_cycle(
            profile,
            config,
            c_w_initial=cw,
            h_initial=h,
        )
        cw, h = cycle_end_state(des_res)
    return cw, h


def simulate_annual_year(
    day_items: list[tuple[date, DailyWeatherProfile, pd.DataFrame]],
    config: DeviceConfig,
    *,
    warmup_cycles: int = 2,
    save_daily_timeseries: bool = False,
    timeseries_dir: Path | None = None,
    progress_callback: Callable[[int, int, date], None] | None = None,
) -> list[DailySimulationRecord]:
    """Simulate a sequential year, warming up on Jan 1 weather before recording."""
    if not day_items:
        return []

    jan1 = date(day_items[0][0].year, 1, 1)
    warmup_profile = next(
        (prof for day_key, prof, _ in day_items if day_key == jan1),
        day_items[0][1],
    )
    cw, h = warmup_on_profile(
        warmup_profile,
        config,
        n_cycles=warmup_cycles,
    )

    records: list[DailySimulationRecord] = []
    n_days = len(day_items)
    for i, (day_key, profile, day_df) in enumerate(day_items):
        record, (cw, h), detailed, inventory = _record_from_day(
            day_key,
            profile,
            day_df,
            config,
            c_w_initial=cw,
            h_initial=h,
        )
        records.append(record)

        if save_daily_timeseries and timeseries_dir is not None:
            day_tag = day_key.isoformat()
            write_detailed_csv(
                timeseries_dir / f"{day_tag}_diagnostics.csv",
                detailed,
            )
            write_water_inventory_csv(
                timeseries_dir / f"{day_tag}_water_inventory.csv",
                inventory,
            )

        if progress_callback is not None:
            progress_callback(i + 1, n_days, day_key)

    return records


def write_daily_summary_csv(
    path: Path,
    records: list[DailySimulationRecord],
) -> None:
    """Write one row per simulated day."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DAILY_SUMMARY_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "date": rec.date.isoformat(),
                    "day_of_year": rec.day_of_year,
                    "rh_avg_frac": f"{rec.rh_avg_frac:.6f}",
                    "rh_peak_frac": f"{rec.rh_peak_frac:.6f}",
                    "temp_avg_c": f"{rec.temp_avg_c:.4f}",
                    "temp_peak_c": f"{rec.temp_peak_c:.4f}",
                    "solar_avg_w_m2": f"{rec.solar_avg_w_m2:.2f}",
                    "solar_peak_w_m2": f"{rec.solar_peak_w_m2:.2f}",
                    "daily_yield_kg_m2": f"{rec.daily_yield_kg_m2:.6f}",
                    "daily_yield_l_m2": f"{rec.daily_yield_l_m2:.6f}",
                    "eta_thermal": f"{rec.eta_thermal:.6f}",
                    "water_uptake_l_m2": f"{rec.water_uptake_l_m2:.6f}",
                    "water_release_l_m2": f"{rec.water_release_l_m2:.6f}",
                    "t_abs_peak_c": f"{rec.t_abs_peak_c:.4f}",
                    "t_glass_peak_c": f"{rec.t_glass_peak_c:.4f}",
                    "t_cond_peak_c": f"{rec.t_cond_peak_c:.4f}",
                    "t_gel_peak_c": f"{rec.t_gel_peak_c:.4f}",
                }
            )


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
