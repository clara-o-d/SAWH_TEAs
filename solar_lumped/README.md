# Solar lumped SAWH

Physics-based forward simulation of the passive solar sorbent atmospheric water harvesting system described by Wilson & Díaz-Marín (*Device*, 2025), with LCOW economics identical to [electrolyte_optimization](https://github.com/clara/electrolyte_optimization). Governing equations: [`docs/governing_eq.tex`](docs/governing_eq.tex).

## Features

- Wilson et al. Eqs. 1–6 (absorber, glass, gel, condenser, mass transfer)
- SciPy `solve_ivp` with **Radau** stiff ODE integration
- Weather modes: **`real`** (year aggregated to mean diurnal Open-Meteo profile), **`baseline`**, **`atacama-replay`**, **`cambridge-replay`**
- LCOW and cost breakdown using the same equation as `lcow_zsr_at_sl`
- Parameter sweeps and tornado plots

## Install

```bash
cd SAWH_TEAs/solar_lumped
pip install -e ".[dev]"
```

## Run

```bash
# Paper baseline (Fig. 2 validation)
python -m solar_lumped.system --weather-mode baseline

# Atacama field test replay (May 8, 2024)
python -m solar_lumped.system --weather-mode atacama-replay

# One real day of real weather (defaults to 15 June of --year)
python -m solar_lumped.system --weather-mode real --lat -23.65 --lon -70.40 --year 2024
python -m solar_lumped.system --weather-mode real --lat -23.65 --lon -70.40 --day 2024-03-07

# Global scenario sweep -- GPU only, all 365 real days x the 8 scenarios in
# site_sweep.SCENARIOS, every one at the parameters.xlsx baseline design
python3 gpu_sweep/run_gpu_sweep.py --lat-lon -23.65 -70.40 --output-csv outputs/gpu_scenario_sweep/site.csv
```

## Architecture

Single day-cycle simulation (`src/solar_lumped/`), driven by CLI entry points, with a
JAX/diffrax fast path (`gpu_sweep/`) for cluster-scale multi-site sweeps and BayesOpt
(`sawh_bayesopt/`, a separate package). Parameter values all come from
`docs/parameters.xlsx` (Physics/Economics sheets) via `_parameters_xlsx.py` — no
hardcoded constants scattered through the physics.

**Call order (a single simulation run):**
`system.py` (CLI) → `weather.py` (build a `DailyWeatherProfile`) → `simulation.py`
(`SystemConfig` + `run_daily_cycle`, which repeatedly evaluates `physics.py`'s coupled
ODE right-hand side via `scipy.integrate.solve_ivp`) → `economics.py` (LCOW from the
resulting daily water yield) → `plotting.py` / CSV writers for output.

**`src/solar_lumped/`**
- `system.py` — CLI entry point (`python -m solar_lumped.system`): parses args, builds
  a `SystemConfig`, runs one daily cycle, writes the LCOW cost breakdown and optional
  detailed/water-inventory CSVs and plots.
- `weather.py` — Open-Meteo client plus `baseline` / `real` / `atacama-replay` /
  `cambridge-replay` weather modes; produces the `DailyWeatherProfile` fed to the
  simulation, and the land-grid sampling used by the GPU sweep.
- `physics.py` — system geometry/material constants, brine/salt thermodynamics,
  heat-transfer correlations, and the coupled thermal + mass-transfer governing
  equations (Wilson et al. Eqs. 1–6). No I/O; pure functions over physical state.
- `complex_model.py` — optional higher-fidelity add-ons (ZSR salt blends, glazing
  stacks, selective absorber coatings, finned/forced condensers, shifted cycle
  schedules), only reached when `SystemConfig.complex` is set; `None` reproduces the
  simple model that `physics.py`/`simulation.py`/`gpu_sweep` use by default. The simple
  model's own default radiative physics is Case 2 (selective surface); pass an explicit
  `thermal=SystemThermalParams(eps_abs_ir=1.0, eps_glass_ir=1.0)` for Case 1's original
  Wilson blackbody/cavity approximation (see `analysis/.../paper_recreation/wilson/`).
- `simulation.py` — `SystemConfig`, the coupled ODE right-hand side, `solve_ivp`
  integration for absorption/desorption phases, cyclic (day-over-day steady-state)
  solving, detailed diagnostics, water-inventory accounting, and annual-yield
  aggregation across sites/months.
- `economics.py` — LCOW/NPV/payback economics identical to `electrolyte_optimization`;
  reads costs from `docs/parameters.xlsx` (no purchased-energy term — this is a passive
  solar system).
- `site_sweep.py` — shared definition of the full-factorial sweep: the combo grid
  (hydrogel thickness × fin area ratio × vapor gap), the per-combo `SystemConfig`, and
  the output CSV schema. Imported by `gpu_sweep/run_gpu_sweep.py`, which is the only
  sweep driver; there is no CPU sweep.
- `plotting.py` — shared matplotlib rcParams matching the paper's MATLAB figure style.
- `_parameters_xlsx.py` — loads `docs/parameters.xlsx` Physics/Economics sheets; the
  single source of truth for every system/economic constant.
- `utils.py` — bracketed root-finding helper shared across `physics.py`/`weather.py`.

**`gpu_sweep/`** — JAX/diffrax port for running the same 125-combo grid across many
sites at once on a GPU (see `GPU_PRIMER.md` / `SHERLOCK_GPU_RUNBOOK.md`).
- `jax_physics.py` — JAX port of the quasi-steady desorption RHS from `physics.py`
  (LiCl-only, fixed-iteration Newton/bisection instead of `scipy.root`/`brentq`).
- `jax_daily_cycle.py` — `diffrax.Tsit5` daily-cycle integrator plus the Aitken
  steady-periodic-state search, the JAX counterpart to `simulation.run_daily_cycle` /
  `find_cyclic_state`.
- `run_gpu_sweep.py` — the sweep driver: reuses `site_sweep.py`'s grid/config/CSV
  schema, vmapped across combos and walked through all 365 real days in lockstep.
- `run_bayesopt_sweep.py` — same site selection as `run_gpu_sweep.py`, but runs
  BayesOpt (via `sawh_bayesopt`) over the sweep's parameter ranges instead of
  brute-forcing the grid.

## Reference

Wilson, C.T., Díaz-Marín, C.D., et al. Solar-driven atmospheric water harvesting in the Atacama Desert through physics-based optimization of a hygroscopic hydrogel device. *Device* (2025). https://doi.org/10.1016/j.device.2025.100798
