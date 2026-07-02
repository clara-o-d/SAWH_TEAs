# Solar lumped SAWH

Physics-based forward simulation of the passive solar sorbent atmospheric water harvesting device described by Wilson & Díaz-Marín (*Device*, 2025), with LCOW economics identical to [electrolyte_optimization](https://github.com/clara/electrolyte_optimization). Governing equations: [`docs/governing_eq.tex`](docs/governing_eq.tex).

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
python scripts/run_solar_sim.py --weather-mode baseline

# Atacama field test replay (May 8, 2024)
python scripts/run_solar_sim.py --weather-mode atacama-replay

# Full year, real weather (mean diurnal profile for the year)
python scripts/run_solar_sim.py --weather-mode real --lat -23.65 --lon -70.40 --year 2024

# Sensitivity
python scripts/parameter_sweep.py --n-points 11
python scripts/tornado_plot.py
```

## Reference

Wilson, C.T., Díaz-Marín, C.D., et al. Solar-driven atmospheric water harvesting in the Atacama Desert through physics-based optimization of a hygroscopic hydrogel device. *Device* (2025). https://doi.org/10.1016/j.device.2025.100798
