# Physics checks

Room for conservation checks on the system models:

* **Mass** — water leaving the sorbent equals water reaching the condenser, and the
  integrated flux matches the change in bed inventory.
* **Energy** — the per-component balances sum to the externally-visible terms, with the
  internal exchange terms cancelling.

`waste-heat/tests/test_waste_heat_sim.py` already covers both for the two-bed model
(`test_energy_balance_closes_hydrogel`, `test_mass_balance_half_cycle`); this is where the
standalone, reportable versions go.

## `run_physics_checks.py`

Runs one solar_lumped daily cycle and writes every diagnostic we have for it into
`outputs/` (gitignored):

```
python run_physics_checks.py --weather-mode baseline
python run_physics_checks.py --weather-mode atacama-replay --cycled
```

* `diagnostics_<tag>.csv/.png` — absorber/glass/condenser/gel temperatures, weather, and
  the desorption driving force (c_r, brine a_w, their gap) against the salt's DRH at the
  gel temperature.
* `water_inventory_<tag>.csv/.png` — water in the gel plus cumulative collected water.
* `conservation_<tag>.png` — the mass and enthalpy check: sorbent water and condenser
  enthalpy from LSODA's dense output against what integrating the RHS boundary flows
  (`m_des`, `dT_cond_dt`) predicts, each panel titled with max |drift| as a percentage of
  the day's swing. A flat drift line means state and rates agree; drift that keeps growing
  means they don't, and that day's yield is suspect.

Baseline reference values: mass drift ~1e-5 L/m² (4e-4 % of swing); enthalpy drift ~1.5 kJ/m²
(~3 % of swing), taken almost entirely as one step in the first few minutes where the
condenser's start-up transient is faster than the output spacing the trapezoid rule sees.
