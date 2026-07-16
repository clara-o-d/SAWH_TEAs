# Waste-heat cycle lumped SAWH (no HTF loop)

Physics-based forward simulation of a **waste-heat-driven, two-bed** sorbent atmospheric water harvesting device (AirJoule-style latent energy and water harvesting), following [`docs/governing_eq.tex`](docs/governing_eq.tex).

This variant removes the pumped heat-transfer-fluid (HTF) loop present in the
sibling `waste-heat_cycle_lumped` package: the desorbing contactor couples
directly to the fixed waste-heat source through a single equivalent UA (the
series combination of the waste-heat-stream HX and the contactor-side UA that
used to sandwich the loop). The two-contactor swap-roles / vacuum-desorption /
RH-switch cycling logic is otherwise unchanged.

## Features

- Two contactors with alternating adsorption / vacuum desorption half-cycles
- Transient energy balances for adsorbing bed, desorbing bed, and condenser (no intermediate HTF loop state)
- **Default sorbent:** Wilson PAM-LiCl hydrogel (Eqs. 5–6 isotherm + swelling)
- **Optional sorbent:** MOF placeholder isotherm (dual-site Langmuir)
- Vacuum desorption mass transfer: `ṁ_des = C_vac(P_sat(T_d) − P_sat(T_cond))`
- Vacuum pump feedback for cycle matching (no HTF pump/loop-flow control)
- Data-center baseline: 58 °C liquid-cooled waste heat (direct NTU–ε HX to the desorbing contactor), 32 °C / 45% RH return air

## Sorbent options

| `--sorbent` | Model | Heats |
|---|---|---|
| `hydrogel` (default) | PAM-LiCl brine + DVS isotherm, `c_w` + `H` state | LiCl from `salt_catalog.csv` |
| `mof` | Dual-site Langmuir placeholder | `mof_catalog.csv` |

## Run

```bash
cd waste-heat_cycle_lumped_no_loop
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
