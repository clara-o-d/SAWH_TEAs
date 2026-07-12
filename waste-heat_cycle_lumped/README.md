# Waste-heat cycle lumped SAWH

Physics-based forward simulation of a **waste-heat-driven, two-bed** sorbent atmospheric water harvesting device (AirJoule-style latent energy and water harvesting), following [`docs/governing_eq.tex`](docs/governing_eq.tex).

## Features

- Two contactors with alternating adsorption / vacuum desorption half-cycles
- Transient energy balances for adsorbing bed, desorbing bed, HTF loop, and condenser
- **Default sorbent:** Wilson PAM-LiCl hydrogel (Eqs. 5–6 isotherm + swelling)
- **Optional sorbent:** MOF placeholder isotherm (dual-site Langmuir)
- Vacuum desorption mass transfer: `ṁ_des = C_vac(P_sat − P_cond)`
- Variable-speed HTF pump and vacuum pump feedback for cycle matching
- Data-center baseline: 58 °C liquid-cooled waste heat (NTU–ε HX to loop fluid), 32 °C / 45% RH return air

## Sorbent options

| `--sorbent` | Model | Heats |
|---|---|---|
| `hydrogel` (default) | PAM-LiCl brine + DVS isotherm, `c_w` + `H` state | LiCl from `salt_catalog.csv` |
| `mof` | Dual-site Langmuir placeholder | `mof_catalog.csv` |

## Run

```bash
cd waste-heat_cycle_lumped
pip install -e ".[dev]"

# Default LiCl hydrogel
python scripts/run_waste_heat_cycle_sim.py --profile datacenter-baseline

# MOF placeholder
python scripts/run_waste_heat_sim.py --sorbent mof --profile datacenter-baseline

# Full day
python scripts/run_waste_heat_sim.py --daily
```

Outputs: `outputs/water_inventory/water_in_gel_*.png` (hydrogel) or `water_in_mof_*.png` (MOF).

## Tests

```bash
pytest
```
