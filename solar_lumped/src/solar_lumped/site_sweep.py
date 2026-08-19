"""Shared definition of the global sweep: the scenario list, the baseline design every
scenario runs at, the per-instance SystemConfig, and the output CSV schema.

The global sweep varies **site x scenario only** -- geometry is fixed at the
parameters.xlsx baseline (``BASELINE_COMBO``). ``Combo``/``combo_grid`` and the
``DEFAULT_*`` grids remain for the callers that still sweep geometry (the Bayes-opt
driver and the tolerance validator).

There is no CPU sweep driver -- sweeping runs on GPU only (gpu_sweep/run_gpu_sweep.py,
all 365 real days per combo), and imports this module so the grid, config, and CSV
schema are defined once rather than mirrored. For a single-day CPU run of one design,
use ``system.py --weather-mode real --day YYYY-MM-DD``."""

from __future__ import annotations

import csv
import dataclasses
import itertools
from dataclasses import dataclass
from pathlib import Path

from solar_lumped.physics import (
    EPS_ABS,
    EPS_ABS_IR_CASE2,
    EPS_GLASS_IR_CASE2,
    FIN_AREA_RATIO,
    H0_M,
    L_G_M,
    L_INS_M,
    SALT_LOADING_DEFAULT,
    SystemThermalParams,
    TAU_GLASS,
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
DEFAULT_EPS_ABS: float = EPS_ABS
DEFAULT_TAU_GLASS: float = TAU_GLASS


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


# --- The scenario axis of the global sweep -------------------------------------------
# Every scenario runs the same parameters.xlsx baseline design; the sweep is site x
# scenario and nothing else.
BASELINE_COMBO = Combo(
    hydrogel_thickness_mm=H0_M * 1e3,
    fin_area_ratio=FIN_AREA_RATIO,
    vapor_gap_mm=L_G_M * 1e3,
)
BASELINE_INSULATION_GAP_MM: float = L_INS_M * 1e3
BASELINE_SALT_LOADING: float = SALT_LOADING_DEFAULT


@dataclass(frozen=True, slots=True)
class Scenario:
    """Absorber/glass optics, plus which physical limits are relaxed.

    ``instant_equilibrium`` (g -> infinity) and ``condenser_ambient`` (T_cond == T_amb)
    each select a *code path* in the JAX step, not a per-instance number, so they are
    uniform across a compiled batch -- which is why the driver runs the scenarios in
    groups keyed on this pair rather than one flat batch.
    """

    eps_abs: float
    tau_glass: float
    eps_abs_ir: float
    eps_glass_ir: float
    instant_equilibrium: bool = False
    condenser_ambient: bool = False


# Wilson & Diaz-Marin's original blackbody/cavity radiative exchange (eps_IR = 1).
_WILSON = Scenario(eps_abs=EPS_ABS, tau_glass=TAU_GLASS, eps_abs_ir=1.0, eps_glass_ir=1.0)
# "Reasonable improvements": a real selective absorber behind ordinary glass.
_IMPROVED = Scenario(
    eps_abs=EPS_ABS, tau_glass=TAU_GLASS,
    eps_abs_ir=EPS_ABS_IR_CASE2, eps_glass_ir=EPS_GLASS_IR_CASE2,
)
# "Optical material limits": perfect absorber, perfect glass, no IR loss.
_LIMITS = Scenario(eps_abs=1.0, tau_glass=1.0, eps_abs_ir=0.0, eps_glass_ir=0.0)

SCENARIOS: dict[str, Scenario] = {
    "wilson": _WILSON,
    "improved": _IMPROVED,
    "optical_limits": _LIMITS,
    # Instantaneous sorption (g -> infinity): desorption becomes energy-limited rather
    # than transport-limited. Only meaningful on top of the two improved optics cases.
    "improved_instant_g": dataclasses.replace(_IMPROVED, instant_equilibrium=True),
    "optical_limits_instant_g": dataclasses.replace(_LIMITS, instant_equilibrium=True),
    # Perfect condenser (T_cond == T_amb): infinite cooling capacity, so the condenser
    # never warms under its own latent load.
    "improved_perfect_cond": dataclasses.replace(_IMPROVED, condenser_ambient=True),
    "optical_limits_perfect_cond": dataclasses.replace(_LIMITS, condenser_ambient=True),
    # All three limits at once -- the idealized-device upper bound.
    "optical_limits_instant_g_perfect_cond": dataclasses.replace(
        _LIMITS, instant_equilibrium=True, condenser_ambient=True,
    ),
}


def scenario_groups(names: list[str] | None = None) -> dict[tuple[bool, bool], list[str]]:
    """Scenario names bucketed by (instant_equilibrium, condenser_ambient).

    One bucket = one compiled code path = one Slurm task's worth of work, so the
    bucket ORDER is load-bearing: sbatch_gpu_sweep_array.sh maps its array index onto
    it. Insertion order of SCENARIOS makes it deterministic -- don't sort here.
    """
    groups: dict[tuple[bool, bool], list[str]] = {}
    for name in names if names is not None else list(SCENARIOS):
        sc = SCENARIOS[name]
        groups.setdefault((sc.instant_equilibrium, sc.condenser_ambient), []).append(name)
    return groups


@dataclass(frozen=True, slots=True)
class GroupRun:
    """How one scenario group is run across Slurm array tasks.

    Per-group rather than global because this was, for a while, badly unequal: with
    instant equilibrium imposed as a stiff g-penalty, those two groups cost ~50x per
    instance-day and had to run a coarser grid in narrow chunks to fit a walltime at all.
    Imposing the limit as a constraint instead removed that gap entirely -- the ideal case
    now costs slightly LESS than finite g, since its absorption half needs no ODE -- so
    every group is back on the 3-degree grid at one width.

    Kept as a per-group structure anyway: it is the natural place for the next scenario
    whose cost is genuinely different, and it is what the array script indexes.
    """

    step_deg: float
    sites_per_chunk: int


# 200 sites/chunk measured ~1.5 h per task for the widest group (600 instances) on the
# serc A100, against a 6 h walltime. Batch width is nearly free below GPU saturation
# (~700 instances), so this sits near the useful ceiling.
_STANDARD_RUN = GroupRun(step_deg=3.0, sites_per_chunk=200)

GROUP_RUNS: dict[tuple[bool, bool], GroupRun] = {
    (False, False): _STANDARD_RUN,
    (True, False): _STANDARD_RUN,
    (False, True): _STANDARD_RUN,
    (True, True): _STANDARD_RUN,
}


def array_tasks(site_count) -> list[tuple[int, float, int, int, list[str]]]:
    """(group index, grid step, site start, site end, scenario names) per Slurm task.

    ``site_count(step_deg)`` returns how many land sites that grid has -- passed in
    rather than imported so this stays testable without the Natural Earth shapefiles.

    Ragged by design: each group is chunked over its own grid at its own width, so index
    *i* of this list is array task *i* and ``len()`` is the --array size covering
    everything. Group order follows ``scenario_groups()``; appending to SCENARIOS is
    safe, reordering it renumbers tasks and invalidates a half-finished sweep.
    """
    tasks: list[tuple[int, float, int, int, list[str]]] = []
    for gid, (key, names) in enumerate(scenario_groups().items()):
        run = GROUP_RUNS[key]
        total = site_count(run.step_deg)
        for start in range(0, total, run.sites_per_chunk):
            tasks.append((gid, run.step_deg, start, min(start + run.sites_per_chunk, total), names))
    return tasks


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
    # Site elevation, from the weather frame (weather.site_elevation_m). Thins the air:
    # more vapor diffusivity and a better-insulated collector, so high sites gain. The
    # 0.0 default keeps every non-site caller (tests, single designs) at sea level.
    site_elevation_m: float = 0.0,
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
        salt_loading=salt_loading,
        hydrogel_thickness_m=combo.hydrogel_thickness_mm * 1e-3,
        vapor_gap_m=combo.vapor_gap_mm * 1e-3,
        insulation_gap_m=insulation_gap_mm * 1e-3,
        fin_area_ratio=combo.fin_area_ratio,
        tilt_deg=tilt_deg,
        thermal=thermal,
        site_elevation_m=site_elevation_m,
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
    # Site elevation drives ambient pressure, which changes yield by ~+2.4%/1000 m, so a
    # row without it is ambiguous. NOTE: adding this broke append-compatibility with
    # pre-elevation CSVs -- those have a different header and must not be appended to.
    "elevation_m",
    "mean_rh_frac",
    "mean_t_amb_c",
    "mean_solar_w_m2",
    "salt",
    # Which entry of SCENARIOS produced this row -- the sweep's only design axis.
    "scenario",
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
    # "ode" (default, Eq. 2) or "ambient" (T_cond == T_amb, infinite-cooling limit).
    "condenser_mode",
    # "finite_g" (default) or "instant" (g -> infinity, equilibrium every instant).
    # See SystemConfig.instant_equilibrium.
    "kinetics",
)


def _existing_scenarios(path: Path) -> set[tuple[float, float, str]]:
    """The (lat, lon, scenario) rows already in *path*, for --resume."""
    if not path.is_file():
        return set()
    import pandas as pd

    df = pd.read_csv(path)
    return {
        (round(float(r["lat"]), 6), round(float(r["lon"]), 6), str(r["scenario"]))
        for _, r in df.iterrows()
    }


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
