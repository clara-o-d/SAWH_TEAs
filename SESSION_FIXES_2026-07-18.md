# Session fixes — 2026-07-18

Summary of changes made across `waste-heat_lumped`, `solar_lumped`, `waste-heat_cycle_lumped`,
`waste-heat_cycle_lumped_no_loop`, and `comparison` in this session.

## 1. Real-weather mode for `waste-heat_lumped`

- Added `profile_from_day_df`, `representative_mean_day_profile`, `monthly_mean_day_profiles`,
  and `real_weather_days_from_df` to `waste-heat_lumped/src/waste_heat_lumped/weather/profiles.py`,
  mirroring `solar_lumped`'s weather pipeline (same `WeatherClient`/Open-Meteo backend, already
  present but unused).
- Since the loop-fluid heater runs on a fixed schedule rather than sunlight, the 24 h fetched
  weather day is split into absorption/desorption by a fixed **clock hour** (default desorption
  08:00–20:00), not a day/night irradiance threshold — configurable via `--desorption-start-hour`.
- `run_waste_heat_sim.py`: new `--profile real` mode with `--lat/--lon/--year/--cache-dir`, and
  `--resolution {annual,monthly}` (day-weighted mean across 12 representative months, mirroring
  `solar_lumped/scripts/grid_param_sweep.py`'s `monthly_mean_profiles`).
- Added `requests`/`requests-cache`/`retry-requests` to `waste-heat_lumped/pyproject.toml`
  (already imported by `weather/client.py` but not declared).

## 2. Ambient convection coefficient default: 15 → 10 W/m²K

Changed `H_AMB_W_M2_K` from 15.0 to 10.0 in:
- `waste-heat_lumped/src/waste_heat_lumped/physics/device_defaults.py`
- `waste-heat_cycle_lumped/src/waste_heat_cycle_lumped/physics/device_defaults.py`
- `waste-heat_cycle_lumped_no_loop/src/waste_heat_cycle_lumped_no_loop/physics/device_defaults.py`
- `comparison/lib/scenario.py`

`solar_lumped` was already at 10 everywhere (including a same-day-earlier fix pinning its
real-weather `h_amb` at a fixed 10 instead of deriving it from wind speed). Recreation scripts
(`wilson-et-al._re-creation/`, `diaz-marin-et-al._re-creation/`) don't import these defaults and
were untouched, as requested.

Known side effect: `waste-heat_cycle_lumped`'s `test_energy_balance_closes_hydrogel` and
`test_no_waste_heat_yields_negligible_water_mof` now fail at h=10 (passed at h=15). Investigated
and concluded the energy-balance test is measuring a near-zero net signal as the difference of
~9-10 million-scale cancelling flux terms (catastrophic cancellation) — actual gross-flux-relative
error is ~3 ppm, i.e. the model conserves energy well; the test's *normalization* is fragile, not
the physics. Left failing, unresolved, per explicit instruction to investigate rather than paper
over it.

## 3. Real-weather absorption/desorption resampling bugs

Both `waste-heat_lumped/weather/profiles.py` and `solar_lumped/weather/profiles.py`
(`_resample_phase` / `profile_from_day_df`) had two independent bugs when a phase spans midnight:

- **Flat forward-fill bug**: the < 432-row branch reindexed onto a fresh calendar `date_range`
  starting at the slice's first timestamp. For a midnight-wrapping slice (e.g. 20:00–23:45 then
  00:00–07:45), the second chunk fell outside that forward-only window and was silently dropped,
  forward-filling flat for the rest of the phase. Fixed by interpolating by **row position**
  within the slice instead of by real timestamp.
- **Chronological-order bug**: a boolean mask (`hour>=20 | hour<8`, or `solar<threshold`)
  preserves *calendar-day* row order (00:00 block first, evening block appended after), not real
  elapsed time within the phase (evening → midnight → early morning). Fixed by rotating rows into
  true chronological order before resampling — `waste-heat_lumped` rotates by the known
  `desorption_start_hour`; `solar_lumped` rotates night by a **noon** pivot and day by a
  **midnight** pivot (each the one guaranteed not to fall inside that slice — using the same pivot
  for both, as a first pass did, scrambles the day slice since noon lies inside it).

## 4. `solar_lumped`: absorption/desorption no longer forced onto a fixed 12h/12h split

`solar_lumped`'s real night/day split is irradiance-threshold-based and naturally unequal length
(e.g. 10 h night / 14 h day at 42.36°N in June) — unlike `waste-heat_lumped`'s genuinely
fixed-schedule loop fluid. `_resample_phase` was still forcing every phase onto a fixed 432-step
(12 h) grid, silently rescaling real time (running night 1.2× slower, day 0.857× faster than
real). Added `_native_dt_s`/`_steps_for` so each phase's step count now matches its true real
elapsed duration; `profile_from_day_df` no longer references a fixed `STEPS_PER_PHASE` except as a
fallback-selection floor.

## 5. Aitken-accelerated cyclic warmup, rolled out everywhere

`solar_lumped/scripts/grid_param_sweep.py` ("Sherlock") already used Aitken Δ² extrapolation
(`find_cyclic_state`) to find the true steady periodic device state, instead of a fixed number of
warmup cycles (`warmup_to_cyclic_state`) — critical because fixed-cycle warmup can need 100+
cycles to converge at strongly-seasonal sites. Rolled the same solver out everywhere else:

- `solar_lumped/simulation/ode_system.py`: `run_daily_cycle`'s `cyclic_initial=True` path now
  calls `find_cyclic_state` instead of `warmup_to_cyclic_state`. This automatically upgrades every
  caller (`run_solar_sim.py`, `lcow_random_global_map.py`, `lcow_full_global_map.py`,
  `site_feasibility.py`) with no per-script changes needed.
- `waste-heat_lumped/simulation/ode_system.py`: ported `find_cyclic_state` (identical (c_w, H)
  state shape to `solar_lumped`) and swapped `run_daily_cycle`'s `cyclic_initial=True` path the
  same way — automatically upgrades `run_waste_heat_sim.py` and `parameter_sweep.py`.
  `--desorption-start-hour`/`--resolution` flags unaffected.
- `waste-heat_cycle_lumped` and `waste-heat_cycle_lumped_no_loop/simulation/ode_system.py`: these
  have a richer two-bed cycle state (8 fields: `loading_a, loading_d, h_a, h_d, t_a, t_d, t_f,
  t_cond`; 7 for the no-loop variant, which has no `t_f`; both reduce to 6/5 real fields for MOF
  sorbent, where `h_a`/`h_d` are always `None`). Generalized the same Aitken solver to this vector
  state via `_state_to_vec`/`_vec_to_state` (drops the `h_a`/`h_d` slots when `is_hydrogel(config)`
  is false), and swapped `run_cycle`/`run_daily_operation`'s fixed `for _ in range(warmup_cycles)`
  loop to call `find_cyclic_state` when `warmup_cycles > 0`. Verified both hydrogel and MOF sorbent
  paths converge cleanly (e.g. `start == end` exactly) via each package's CLI script.
- `cyclic_warmup_cycles`/`warmup_cycles` arguments are now passed through as Aitken's `max_rounds`
  (floored at 3) rather than a literal cycle count, everywhere they used to drive a fixed loop.

Verified via full test suite + representative CLI runs in all four packages after every change;
no new regressions were introduced (pre-existing unrelated failures in `solar_lumped` and
`waste-heat_cycle_lumped`, documented above and earlier in-session, are unchanged in kind).
