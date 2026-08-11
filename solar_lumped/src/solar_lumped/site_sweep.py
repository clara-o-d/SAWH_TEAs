"""Shared definition of the full-factorial system-parameter sweep: the combo grid, the
per-combo SystemConfig, and the output CSV schema.

There is no CPU sweep driver -- sweeping runs on GPU only (gpu_sweep/run_gpu_sweep.py,
all 365 real days per combo), and imports this module so the grid, config, and CSV
schema are defined once rather than mirrored. For a single-day CPU run of one design,
use ``system.py --weather-mode real --day YYYY-MM-DD``."""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

from solar_lumped.physics import (
    EPS_ABS_IR_CASE2,
    EPS_GLASS_IR_CASE2,
    SystemThermalParams,
    # Re-exported as gps.TILT_DEG by gpu_sweep's CLI defaults -- not unused.
    TILT_DEG,  # noqa: F401
)
from solar_lumped.simulation import SystemConfig
from solar_lumped.weather import DailyWeatherProfile

# Baselines: table_s3.H0_M=4mm, L_G_M=40mm, EPS_ABS=0.95, TAU_GLASS=0.9, FIN_AREA_RATIO=7.1.
DEFAULT_HYDROGEL_THICKNESS_MM: tuple[float, ...] = (1.0, 3.25, 5.5, 7.75, 10.0)
DEFAULT_FIN_AREA_RATIO: tuple[float, ...] = (3.0, 5.275, 7.55, 9.825, 12.0)
DEFAULT_VAPOR_GAP_MM: tuple[float, ...] = (20.0, 30.0, 40.0, 50.0, 60.0)
# eps_abs and tau_glass are fixed constants per case (not swept) -- see --eps-abs/--tau-glass.
DEFAULT_EPS_ABS: float = 0.95
DEFAULT_TAU_GLASS: float = 0.90


@dataclass(frozen=True, slots=True)
class Combo:
    hydrogel_thickness_mm: float
    fin_area_ratio: float
    vapor_gap_mm: float


def combo_grid(
    *,
    hydrogel_thickness_mm: list[float],
    fin_area_ratio: list[float],
    vapor_gap_mm: list[float],
) -> list[Combo]:
    return [
        Combo(*vals)
        for vals in itertools.product(hydrogel_thickness_mm, fin_area_ratio, vapor_gap_mm)
    ]


def build_system_config(
    combo: Combo,
    *,
    salt: str,
    salt_loading: float,
    insulation_gap_mm: float,
    tilt_deg: float,
    eps_abs: float,
    tau_glass: float,
    eps_abs_ir: float | None = EPS_ABS_IR_CASE2,
    eps_glass_ir: float | None = EPS_GLASS_IR_CASE2,
) -> SystemConfig:
    thermal = SystemThermalParams(
        insulation_gap_m=insulation_gap_mm * 1e-3,
        vapor_gap_m=combo.vapor_gap_mm * 1e-3,
        eps_abs=eps_abs,
        tau_glass=tau_glass,
        tilt_deg=tilt_deg,
        eps_abs_ir=eps_abs_ir,
        eps_glass_ir=eps_glass_ir,
    )
    return SystemConfig(
        salt_name=salt,
        salt_to_polymer_ratio=salt_loading,
        hydrogel_thickness_m=combo.hydrogel_thickness_mm * 1e-3,
        vapor_gap_m=combo.vapor_gap_mm * 1e-3,
        insulation_gap_m=insulation_gap_mm * 1e-3,
        fin_area_ratio=combo.fin_area_ratio,
        tilt_deg=tilt_deg,
        thermal=thermal,
    )


def mean_weather_stats(
    profiles: list[DailyWeatherProfile],
) -> tuple[float, float, float]:
    """Mean RH (fraction), ambient temperature (C), and daylight solar irradiance (W/m²)
    across *profiles* -- a site property, computed once and reused across combos. Solar
    averages the desorption phase only, since absorption is by definition the low-solar
    half of the day."""
    rh_means: list[float] = []
    t_means: list[float] = []
    solar_means: list[float] = []
    for profile in profiles:
        rh = list(profile.absorption.relative_humidity) + list(profile.desorption.relative_humidity)
        t = list(profile.absorption.temperature_c) + list(profile.desorption.temperature_c)
        solar = profile.desorption.solar_w_m2
        rh_means.append(sum(rh) / len(rh))
        t_means.append(sum(t) / len(t))
        solar_means.append(sum(solar) / len(solar))
    n = len(profiles)
    return (sum(rh_means) / n, sum(t_means) / n, sum(solar_means) / n)


_CSV_COLUMNS: tuple[str, ...] = (
    "lat",
    "lon",
    "mean_rh_frac",
    "mean_t_amb_c",
    "mean_solar_w_m2",
    "salt",
    "hydrogel_thickness_mm",
    "eps_abs",
    "tau_glass",
    "eps_abs_ir",
    "eps_glass_ir",
    "fin_area_ratio",
    "vapor_gap_mm",
    "warmup_method",
    "resolution",
    "mean_yield_kg_m2",
    "mean_eta_thermal",
    "n_periods",
    # Complex-fidelity settings (solar_lumped.complex_model). Always written, empty
    # in simple mode, so simple and complex sweeps share one schema and their rows
    # can be concatenated without silently losing which fidelity produced them.
    "fidelity",
    "cx_eps_abs_ir",
    "cx_glazing_panes",
    "cx_evacuated_gap",
    "cx_condenser_air_speed_m_s",
    "cx_blend_weights",
    # "ode" (default, Eq. 2) or "ambient" (T_cond == T_amb, infinite-cooling limit).
    "condenser_mode",
    # "hydrate" (default, n*c_s) or "drh" (equilibrium c_w at the deliquescence RH) --
    # which physical limit stopped desorption. See SystemConfig.c_w_floor_mode.
    "c_w_floor_mode",
)


def _existing_combo_keys(path: Path, lat: float, lon: float) -> set[tuple]:
    if not path.is_file():
        return set()
    import pandas as pd

    df = pd.read_csv(path)
    df = df[(df["lat"] == lat) & (df["lon"] == lon)]
    keys = set()
    for _, row in df.iterrows():
        keys.add(
            (
                round(float(row["hydrogel_thickness_mm"]), 6),
                round(float(row["fin_area_ratio"]), 6),
                round(float(row["vapor_gap_mm"]), 6),
            )
        )
    return keys


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
