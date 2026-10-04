# SAWH_TEAs

Techno-economic analysis of sorbent atmospheric water harvesting (SAWH).

The core is a physics-based simulation of the passive solar PAM-LiCl hydrogel device of
Wilson & Díaz-Marín (*Device*, 2025). Its LCOW economics are identical to
[electrolyte_optimization](https://github.com/clara/electrolyte_optimization). Built on
that are a GPU sweep across the globe, design and operation optimization, and a sibling
waste-heat-driven variant.

## Repository layout

| Path | What it is |
|---|---|
| [`solar_lumped/`](solar_lumped/) | Solar device: physics, weather, LCOW, CLI (`src/`), and the JAX/diffrax GPU fast path (`gpu_sweep/`). Described below. |
| [`sawh_bayesopt/`](sawh_bayesopt/) | Global design + daily-operation optimization on top of `solar_lumped`: the two-stage surrogate pipeline, and per-site true-physics BO. See its [README](sawh_bayesopt/README.md). |
| [`waste-heat/`](waste-heat/) | Waste-heat-driven, two-bed variant (data-center waste heat, vacuum desorption). See its [README](waste-heat/README.md). |
| [`analysis/`](analysis/) | Paper recreations (Wilson, Díaz-Marín), physics checks, sensitivity/tornado sweeps, global maps, BO diagnostics, water-source comparison. |
| [`benchmark/tariff/`](benchmark/tariff/) | Scraped per-country tap-water tariffs and maps, the first comparator for LCOW. |

## Documentation

- [`solar_lumped/docs/governing_eq.tex`](solar_lumped/docs/governing_eq.tex): solar device governing equations.
- [`waste-heat/docs/governing_eq.tex`](waste-heat/docs/governing_eq.tex): waste-heat device governing equations.
- [`sawh_bayesopt/docs/problem_formulation.tex`](sawh_bayesopt/docs/problem_formulation.tex): the optimization problem and its two-stage decomposition.
- [`sawh_bayesopt/docs/design_notes.md`](sawh_bayesopt/docs/design_notes.md): why the optimizers are built the way they are.
- [`solar_lumped/docs/sherlock_param_sweep.tex`](solar_lumped/docs/sherlock_param_sweep.tex): plan for the global LiCl parameter sweep on Sherlock.
- [`solar_lumped/docs/parameters.xlsx`](solar_lumped/docs/parameters.xlsx): the single source of truth for every physics and economics constant, in all packages.

## Install

Each package is installed separately, in editable mode. `sawh_bayesopt` and `waste-heat`
import `solar_lumped`, so install that first.

```bash
pip install -e "solar_lumped[dev]"
pip install -e sawh_bayesopt
pip install -e waste-heat
```

The GPU path (`gpu_sweep/`, `sawh_bayesopt`) also needs `jax`, `diffrax` and `equinox`.
These are already in `solar_lumped/.venv_gpu`. On a GPU node, add `pip install "jax[cuda12]"`.

## Tests

Use the GPU venv for anything that touches `solar_lumped/gpu_sweep/` or `sawh_bayesopt`:

```bash
solar_lumped/.venv_gpu/bin/python -m pytest solar_lumped/tests sawh_bayesopt/tests -q
```

The default `python` doesn't have jax or diffrax. Under it, the CPU/JAX parity tests
(`solar_lumped/tests/test_cpu_jax_parity.py`) and `test_two_stage.py` skip silently, and
the run still reports all-green. Those parity tests are the only guard keeping the two
physics backends in step. [`CLAUDE.md`](CLAUDE.md) has more on this, and on editing
physics in both backends.

```bash
python -m pytest waste-heat/tests -q          # CPU only
SAWH_BAYESOPT_SLOW_TESTS=1 solar_lumped/.venv_gpu/bin/python -m pytest \
  sawh_bayesopt/tests/test_integration_real_model.py -q   # opt-in real-model test
```

---

## Solar lumped SAWH (`solar_lumped/`)

### Features

- Wilson et al. Eqs. 1–6: absorber, glass, gel, condenser, and mass transfer.
- Stiff ODE integration with SciPy `solve_ivp` (**Radau**). The JAX port integrates with
  diffrax `Tsit5`.
- Cyclic steady state found by Aitken Δ² extrapolation, so that a reported day does not
  start from an arbitrary gel loading.
- Weather modes:
  - **`real`**: one real calendar day of Open-Meteo weather, no averaging.
  - **`stanford-measured`**: one day of measured Stanford Met Tower data.
  - **`baseline`**: the synthetic paper profile, with a fixed 12 h/12 h split.
  - **`atacama-replay`**, **`cambridge-replay`**, **`fig-s1-replay`**: validation replays.
- LCOW and cost breakdown, using the same equation as `lcow_zsr_at_sl`.
- Global site × scenario sweeps on GPU, plus parameter sweeps and tornado plots
  (`analysis/sensitivity/`).

### Run

```bash
# Paper baseline (Fig. 2 validation)
python -m solar_lumped.system --weather-mode baseline

# Atacama field test replay (May 8, 2024)
python -m solar_lumped.system --weather-mode atacama-replay

# One real day of real weather (defaults to 15 June of --year)
python -m solar_lumped.system --weather-mode real --lat -23.65 --lon -70.40 --year 2024
python -m solar_lumped.system --weather-mode real --lat -23.65 --lon -70.40 --day 2024-03-07

# One measured Stanford day (defaults to the latest complete day in --year)
python -m solar_lumped.system --weather-mode stanford-measured --year 2025

# Global scenario sweep. GPU only: all 365 real days x the 8 scenarios in
# site_sweep.SCENARIOS, each at the parameters.xlsx baseline design
python3 solar_lumped/gpu_sweep/run_gpu_sweep.py --lat-lon -23.65 -70.40 --output-csv outputs/gpu_scenario_sweep/site.csv
```

### Architecture

The package is a single day-cycle simulation (`src/solar_lumped/`) driven by CLI entry
points. A JAX/diffrax fast path (`gpu_sweep/`) handles cluster-scale multi-site sweeps,
and BayesOpt lives in the separate `sawh_bayesopt/` package. Every parameter value comes
from `docs/parameters.xlsx` (Physics/Economics sheets) via `_parameters_xlsx.py`; there
are no hardcoded constants in the physics.

**Call order of a single simulation run:**
1. `system.py` (CLI) calls `weather.py`, which builds a `DailyWeatherProfile`.
2. `simulation.py` sets up a `SystemConfig` and runs `run_daily_cycle`. This calls
   `scipy.integrate.solve_ivp`, which repeatedly evaluates `physics.py`'s coupled ODE
   right-hand side.
3. `economics.py` computes LCOW from the resulting daily water yield.
4. `plotting.py` and the CSV writers produce the output.

**`src/solar_lumped/`**
- `system.py`: the CLI entry point (`python -m solar_lumped.system`). It parses
  arguments, builds a `SystemConfig` and runs one daily cycle. It writes the LCOW cost
  breakdown, and optionally the detailed and water-inventory CSVs and plots.
- `weather.py`: the Open-Meteo client and the weather modes listed above. It produces the
  `DailyWeatherProfile` fed to the simulation, applies plane-of-array transposition, and
  provides the land-grid sampling used by the GPU sweep.
- `physics.py`: geometry and material constants, brine/salt thermodynamics, heat-transfer
  correlations, and the coupled thermal and mass-transfer equations (Wilson et al.
  Eqs. 1–6). It does no I/O; every function is pure over the physical state.
- `complex_model.py`: optional higher-fidelity add-ons, reached only when
  `SystemConfig.complex` is set. They cover ZSR salt blends, glazing stacks, selective
  absorber coatings, finned and forced condensers, and shifted cycle schedules. `None`
  reproduces the simple model, which `physics.py`, `simulation.py` and `gpu_sweep` use by
  default.
  - The simple model's default radiative physics is Case 2 (selective surface).
  - For Case 1, Wilson's original blackbody/cavity approximation, pass
    `thermal=SystemThermalParams(eps_abs_ir=1.0, eps_glass_ir=1.0)`
    (see `analysis/performance/physics/paper_recreation/wilson/`).
- `simulation.py`: `SystemConfig`, the coupled ODE right-hand side, `solve_ivp`
  integration of the absorption and desorption phases, and the cyclic steady-state
  solver. It also holds the detailed diagnostics, water-inventory accounting, and
  annual-yield aggregation.
- `economics.py`: LCOW, NPV and payback economics, identical to
  `electrolyte_optimization`. Costs are read from `docs/parameters.xlsx`. There is no
  purchased-energy term, since this is a passive solar system.
- `site_sweep.py`: the shared definition of the global sweep. It holds the scenario
  list, the baseline design every scenario runs at, the per-instance `SystemConfig`, and
  the output CSV schema. `gpu_sweep/run_gpu_sweep.py` imports it and is the only sweep
  driver; there is no CPU sweep.
- `plotting.py`: shared matplotlib rcParams matching the paper's MATLAB figure style.
- `_parameters_xlsx.py`: loads the `docs/parameters.xlsx` Physics/Economics/Salts sheets.
- `utils.py`: a bracketed root-finding helper shared by `physics.py` and `weather.py`.

**`gpu_sweep/`**: the JAX/diffrax port, which runs many (site, design, scenario)
instances at once on a GPU. See `GPU_PRIMER.md`, `SHERLOCK_GPU_RUNBOOK.md`, and
`FINDINGS.md` for CPU/JAX parity results.
- `jax_physics.py`: JAX port of the quasi-steady desorption right-hand side from
  `physics.py`. It is LiCl-only, and uses fixed-iteration Newton/bisection in place of
  `scipy.root`/`brentq`.
- `jax_daily_cycle.py`: the `diffrax.Tsit5` daily-cycle integrator, the batched year walk
  and the Aitken steady-periodic-state search. It is the JAX counterpart to
  `simulation.run_daily_cycle` and `find_cyclic_state`.
- `run_gpu_sweep.py`: the global sweep driver. It runs every land site × the
  `site_sweep.SCENARIOS`, vmapped across instances and walked through all 365 real days
  in lockstep.
- `run_bayesopt_sweep.py`: per-site true-physics BayesOpt (via `sawh_bayesopt`) across
  many sites. It is the reference that the two-stage pipeline is validated against.

## Reference

Wilson, C.T., Díaz-Marín, C.D., et al. Solar-driven atmospheric water harvesting in the Atacama Desert through physics-based optimization of a hygroscopic hydrogel device. *Device* (2025). https://doi.org/10.1016/j.device.2025.100798
