"""True physics for the global study: the one design variable, the five daily controls, and
the JAX year runner every physics number in this package comes from.

Design. Hydrogel thickness H0 is the only design variable. Salt loading sits at the top of
its parameters.xlsx sweep range (the previous campaign's optima sat there at a median of
8.0), and the vapor gap is H0 + 7 mm: the cast gel must fit inside the gap, and a fixed
headroom above it keeps every thickness in the 1-10 mm box buildable while the swelling
ceiling (``capped``) still records over-swelling.

Controls, per day. Seal and open offsets (hours from GHI sunrise and sunset, on the
15-minute weather grid) and the tilt in each third of the sealed window. Daily tilt is the
case where the three tilts are equal; three-per-day tilt lets them differ. Tilt reaches the
physics twice per step -- plane-of-array transposition and the Hollands gap convection --
through the desorption phase's per-step tilt channel (weather.PhaseProfile.tilt_deg).
"""

from __future__ import annotations

import numpy as np

from solar_lumped._parameters_xlsx import physics_bounds

THICKNESS_BOUNDS_M: tuple[float, float] = physics_bounds("Hydrogel reference thickness (H0)", mm_to_m=True)
SALT_LOADING: float = physics_bounds("Salt loading (SL)")[1]
GAP_HEADROOM_M = 0.007
OFFSET_BOUNDS_H: tuple[float, float] = physics_bounds("Seal / open offset from sunrise-sunset")
TILT_BOUNDS_DEG: tuple[float, float] = physics_bounds("Tilt angle (theta)")

# Control grids. 0.25 h is the weather time step, so a finer offset is invisible to the
# physics. Daily and seasonal tilt search the 5-degree grid; three-per-day tilt searches the
# 15-degree grid in each segment (5^3 = 125 combinations, against 13^3 = 2197 at 5 degrees).
SEAL_GRID = np.arange(OFFSET_BOUNDS_H[0], OFFSET_BOUNDS_H[1] + 1e-9, 0.25)
OPEN_GRID = np.arange(OFFSET_BOUNDS_H[0], OFFSET_BOUNDS_H[1] + 1e-9, 0.25)
TILT_GRID = np.arange(TILT_BOUNDS_DEG[0], TILT_BOUNDS_DEG[1] + 1e-9, 5.0)
TILT3_GRID = np.arange(TILT_BOUNDS_DEG[0], TILT_BOUNDS_DEG[1] + 1e-9, 15.0)
N_TILT_SEGMENTS = 3
CONTROL_VARS: tuple[str, ...] = ("seal_offset_h", "open_offset_h", "tilt1_deg", "tilt2_deg", "tilt3_deg")
STATE_VARS: tuple[str, ...] = ("c_w_mol_m3", "h_mm")


def simulate_years(frames: list, thickness_m: np.ndarray, controls: np.ndarray, day_index: np.ndarray, *,
                   progress_every: int = 30) -> dict:
    """One chained physics year per instance under per-day controls.

    ``frames[i]`` is instance i's weather frame (instances of one cell share the object),
    ``thickness_m`` (n_inst,), ``controls`` (n_inst, n_days, 5) in CONTROL_VARS order, and
    ``day_index`` the (n_days,) day-of-year indices every frame covers.

    Returns flat per-(instance, day) rows: instance, day, thickness_m, control, state (start
    of day), water (kg/m2), state_end, capped. Instances whose solver hit its step cap on any
    day are dropped whole, since their later states descend from a truncated day. Day 1
    starts from the Aitken cyclic state of day 1 under day 1's controls.

    Profiles are built lazily, one day at a time: a year of them for ~700 instances is tens
    of GB of Python floats. ponytail: that build is serial CPU pandas work interleaved with
    the GPU step; overlap it with a thread pool if it dominates the day time.
    """
    from solar_lumped import jax_daily_cycle as jdc
    from solar_lumped.physics import initial_loading
    from solar_lumped.weather import PHASE_DT_S, _steps_for, profile_from_day_df, site_elevation_m

    n_inst, n_days = len(frames), len(day_index)
    by_frame: dict[int, dict] = {}
    for df in frames:
        if id(df) not in by_frame:
            by_frame[id(df)] = {d.timetuple().tm_yday - 1: g for d, g in df.groupby(df.index.date)}
    groups = [by_frame[id(df)] for df in frames]

    # Either phase is a subset of its day's rows, so the day's own step count bounds both
    # exactly -- and _pad_to truncates silently, so an estimate that ran short would be a
    # wrong answer rather than an error.
    n_max = 0
    for g in by_frame.values():
        for doy in day_index:
            day = g[int(doy)]
            native_dt = float(day.index.to_series().diff().dropna().dt.total_seconds().median())
            n_max = max(n_max, _steps_for(len(day), native_dt))

    configs = [system_config(float(h), elevation_m=site_elevation_m(df)) for df, h in zip(frames, thickness_m)]
    system = jdc.build_system_arrays(configs)
    # SystemConfig's default h_des_mode is "isosteric" on the simple LiCl path, so the JAX
    # step must say so too, or the two backends price desorption differently.
    step = jdc.make_year_step_fn(system, PHASE_DT_S, n_max, n_max, h_des_isosteric=True)

    class _LazyDays:
        """The year's per-day weather tuples, built on access (see docstring)."""

        def __len__(self):
            return n_days

        def __getitem__(self, d):
            if d >= n_days:
                raise IndexError(d)
            profiles = [
                profile_from_day_df(groups[i][int(day_index[d])], seal_offset_h=float(c[0]),
                                    open_offset_h=float(c[1]), poa_tilt_deg=tuple(float(t) for t in c[2:]))
                for i, c in enumerate(controls[:, d])
            ]
            return jdc.build_day_weather(profiles, n_max, n_max, tilt_deg=system["tilt_deg"])

    days = jdc.run_year_batched(
        step, _LazyDays(),
        c_w_initial=np.array([initial_loading(c) for c in configs]),
        h_initial=np.array([c.hydrogel_thickness_m for c in configs]),
        progress_every=progress_every,
    )
    keep = days["ok"].all(axis=0)
    if not keep.all():
        print(f"    dropping {int((~keep).sum())}/{n_inst} instance(s) that hit the step cap", flush=True)
    inst = np.flatnonzero(keep)
    grid_inst, grid_day = np.meshgrid(inst, np.arange(n_days), indexing="ij")
    grid_inst, grid_day = grid_inst.ravel(), grid_day.ravel()

    def per_day(key, scale=1.0):
        return days[key][grid_day, grid_inst] * scale

    return {
        "instance": grid_inst,
        "day": np.asarray(day_index)[grid_day],
        "thickness_m": np.asarray(thickness_m, float)[grid_inst],
        "control": controls[grid_inst, grid_day],
        "state": np.column_stack([per_day("c_w_start"), per_day("h_start", 1000.0)]),
        "water": per_day("water"),
        "state_end": np.column_stack([per_day("c_w_end"), per_day("h_end", 1000.0)]),
        "capped": per_day("capped").astype(bool),
    }


def system_config(thickness_m: float, *, elevation_m: float):
    """The study's SystemConfig for one thickness at one site. Everything not named here is
    SystemConfig's default: Case 2 optics, LiCl, isosteric h_des, 5 mm insulation gap, fin
    area ratio 7.1. ``tilt_deg`` is left at its default because every profile carries its
    own per-step tilt schedule."""
    from solar_lumped.simulation import SystemConfig

    return SystemConfig(hydrogel_thickness_m=thickness_m, vapor_gap_m=thickness_m + GAP_HEADROOM_M,
                        salt_loading=SALT_LOADING, site_elevation_m=elevation_m)


def random_controls(rng: np.random.Generator, n_inst: int, n_days: int) -> np.ndarray:
    """(n_inst, n_days, 5) training controls, drawn independently every day.

    Offsets are uniform on their grids. Tilt is a single 5-degree level for all three
    segments on half the days and three independent levels on the other half: the daily and
    seasonal scenarios only ever query equal tilts, a measure-zero set under fully
    independent draws, so it gets half the data on purpose.
    """
    seal = rng.choice(SEAL_GRID, (n_inst, n_days))
    open_ = rng.choice(OPEN_GRID, (n_inst, n_days))
    tilts = rng.choice(TILT_GRID, (n_inst, n_days, N_TILT_SEGMENTS))
    equal = rng.random((n_inst, n_days)) < 0.5
    tilts[equal] = tilts[equal][:, :1]
    return np.concatenate([seal[..., None], open_[..., None], tilts], axis=-1)


def stratified_thicknesses(rng: np.random.Generator, n_cells: int, per_cell: int) -> np.ndarray:
    """(n_cells, per_cell) thicknesses (m): stratified sampling (Kochenderfer & Wheeler,
    Algorithms for Optimization, 2nd ed., section 16.4) -- one uniform draw inside each of
    ``per_cell`` equal strata of the thickness range, so every cell spans the whole box."""
    lo, hi = THICKNESS_BOUNDS_M
    edges = np.linspace(lo, hi, per_cell + 1)
    return edges[:-1] + rng.random((n_cells, per_cell)) * (edges[1] - edges[0])
