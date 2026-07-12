# Fluid-heated daily-cycle SAWH

Physics-based forward simulation of a **single-bed, fluid-heated** atmospheric water harvesting device with a fixed 12 h absorption + 12 h desorption daily cycle. Governing equations: [`docs/governing_eq.tex`](docs/governing_eq.tex).

## Features

- Wilson et al. Eqs. 5–6 mass transport (atmospheric vapor-gap desorption)
- Fixed loop-fluid setpoints \(T_f\), \(\dot{m}_f\) with NTU–ε gel HX during desorption
- Data-center baseline: 58 °C loop fluid, 32 °C / 45 % RH return air
- LCOW economics (zero parasitic electricity)

## Install

```bash
cd waste-heat_lumped
pip install -e ".[dev]"
```

## Run

```bash
python scripts/run_waste_heat_sim.py --profile datacenter-baseline --plot-water-inventory
```

## Related packages

| Package | Cycle | Heat source |
|---------|-------|-------------|
| `solar_lumped` | 12 h + 12 h daily | Solar irradiance |
| **`waste-heat_lumped`** | 12 h + 12 h daily | Fixed HTF → gel HX |
| `waste-heat_cycle_lumped` | Event-driven half-cycles | HTF loop + vacuum desorption |

## Tests

```bash
pytest
```
