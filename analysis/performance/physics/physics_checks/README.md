# Physics checks

Room for conservation checks on the device models:

* **Mass** — water leaving the sorbent equals water reaching the condenser, and the
  integrated flux matches the change in bed inventory.
* **Energy** — the per-component balances sum to the externally-visible terms, with the
  internal exchange terms cancelling.

`waste-heat/tests/test_waste_heat_sim.py` already covers both for the two-bed model
(`test_energy_balance_closes_hydrogel`, `test_mass_balance_half_cycle`); this is where the
standalone, reportable versions go.

Nothing here yet.
