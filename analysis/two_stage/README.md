# Two-stage SAWH optimization: results so far

This folder summarizes where the two-stage pipeline in `sawh_bayesopt/` stands: what has been
run, what each step found, and what the numbers can and cannot yet be trusted for. Figures are
regenerated from the run directory synced from Sherlock with

```bash
python analysis/two_stage/make_figures.py      # reads sawh_bayesopt/outputs/two_stage/main
```

## The idea in one paragraph

Finding the best device at every place on Earth by running full-year physics optimizations is
far too expensive (~1 GPU-hour per design-year, thousands of locations, dozens of designs each).
Instead the problem is split in two. **Design** (hydrogel thickness, vapor gap, salt loading)
is chosen once per location and must serve the whole year. **Operation** (when to seal and open
the device, and its tilt) is re-chosen every day against that day's weather. A fast neural-network
**surrogate** of the daily physics connects them: given design, today's schedule, today's
weather and the gel's state at dawn, it predicts today's water and tomorrow's starting state. It
is trained once, on a fixed physics campaign over a few hundred representative climates, and
then used to optimize everywhere.

```
weather cache ──► daily climate features ──► representative locations ──► physics campaign
 (14.7k sites)     (PCA, Fig. 1–2)            (stratified, 967)             (15,472 years)
                                                                                 │
     maps ◄── design search per location ◄── daily surrogate (Fig. 3) ◄─────────┘
 (Fig. 4–6)   (GP + EI outer, exhaustive
              daily schedule inner)
```

## Status

| Step | State |
|---|---|
| Featurize every cached 2024 location | done |
| Check how many weather components are needed | done (Fig. 2) → **20** |
| Select representative locations, run physics | done: 15,472 physics-years, 23 GPU chunks |
| Train surrogate, test on held-out locations | done (Fig. 3) |
| Design search on the 822 training ("anchor") locations | done for **daily tilt + hindsight** (Fig. 4–6) |
| Active round (physics re-run at the chosen optima) | not yet |
| Design search on all ~14.7k locations; other control modes | not yet |
| Comparison against per-location true-physics optimization (`validate-bo`) | not yet |

## Fig. 1 — Each day's weather as a few numbers

![Weather PCA](figures/fig1_weather_pca.png)

Each day is its 24-hour temperature, humidity and sunlight cycle (72 numbers), compressed by a
principal-component analysis fitted over days from every cached location. Only T, RH and GHI
are used because they are the only weather the physics responds to (convection is fixed and
there is no sky-temperature term). The first few components carry almost everything: 20 keep
99.7% of the variance. The component shapes are recognisable physics — PC1 is broadly "hot,
dry and sunny vs. cool, humid and dim", PC3 is the size of the solar peak. (PCA signs are
arbitrary.) Seven physical scalars ride alongside the components: RH at the coldest hour,
daily solar total, dew-point depression at peak sun, diurnal temperature swing, elevation,
|latitude| and noon solar zenith.

## Fig. 2 — Does the compression keep what matters for yield?

![Featurization check](figures/fig2_featurization_check.png)

Variance captured is not the same as yield captured, so the physics was run twice on 60
locations (12 per climate type): once on the real year and once on the same year rebuilt from
only k components, at a fixed baseline design. The k = 72 "floor" is the error from resampling
alone. The **median** location is fine at any k, but the **90th-percentile** location is not:
at k = 8–12 a few high-altitude and coastal-humid locations get their annual water wrong by
~45–55%. At **k = 20** high-altitude drops to the floor (~1%) and coastal-humid to 13%, so the
surrogate uses 20. Coastal-humid remains the weakest climate for this featurization.

## Fig. 3 — The surrogate, on locations it never trained on

![Surrogate held-out](figures/fig3_surrogate_holdout.png)

145 locations were held out of training entirely (whole locations, never random rows — random
splits leak each climate across the split and flatter the error). On them:

- **Left:** the surrogate is run through each held-out location's whole year on its own,
  feeding each day's predicted gel state into the next day, under the same schedules the
  physics used. Annual water is within **0.6% median, 3.7% at the 90th percentile** overall,
  and within 1.5% / 5% in every climate type.
- **Right:** what the design search actually relies on: within a location, does the surrogate
  order its 16 simulated designs the same way the physics does? Median rank correlation is
  **0.985**, ≥ 0.97 in every climate, and the two agree on whether thicker gel helps in 98% of
  locations (83% in coastal-humid, where the physics' own thickness effect is weakest).

Also from the held-out data, about the physics itself: thicker gel gives more daily water in
93% of locations (less so in coastal-humid, ~70%), while a longer sealed (desorption) window
*reduces* daily water in 75% of locations — in every hyper-arid and high-altitude location —
because the absorption time it costs outweighs the extra release.

The surrogate is a 5-member ensemble of small networks (35 inputs → 128-wide shared trunk →
separate heads for water, next-day state and swelling-cap probability), each trained on a
bootstrap of locations; the ensemble spread is its uncertainty.

## Fig. 4–6 — Best designs at the anchor locations

Each of the 822 anchor locations ran a Bayesian design search (8 space-filling starting
designs, then 16 chosen by expected improvement). Every candidate design was scored by walking
the whole year on the surrogate, choosing each day the best of ~14,000 combinations of seal
time, open time and tilt.

**These runs use daily tilt and hindsight** — each day's schedule chosen knowing that day's
actual weather. That is an upper bound on what any controller could achieve with this hardware,
not a forecast-driven result.

![LCOW map](figures/fig4_anchor_lcow_map.png)

![Thickness map](figures/fig4b_anchor_thickness_map.png)

The optimal gel is thickest (4–5 mm) across the hot deserts — Sahara, Arabia, the US Southwest
and Mexico, interior Australia, the Kalahari and the Atacama — and thinnest (~1.5–2.5 mm) in the
humid tropics and along coasts (median 2.8 mm overall; hyper-arid 3.8, coastal-humid 2.3). Strong
sun and dry air can fully regenerate a thick gel every day, so its extra capacity pays for its
sorbent cost; where the drying drive is weak, extra gel mostly adds cost. Thickness is the one
design variable whose optimum sits well inside its range everywhere.

![LCOW by climate](figures/fig5_lcow_by_regime.png)

| climate | locations | median LCOW (USD/m³) | median water (kg/m²/day) |
|---|---:|---:|---:|
| monsoonal | 68 | 5.15 | 2.81 |
| other | 528 | 6.31 | 2.34 |
| high-altitude | 73 | 7.85 | 1.78 |
| hyper-arid | 85 | 8.98 | 1.70 |
| coastal-humid | 68 | 12.54 | 1.08 |

The cheapest locations reach about **$3.5/m³ at ~4 kg/m²/day**, all tropical/subtropical.
Cost rises toward high latitudes (short winter days) and across the Sahara–Central Asia dry
belt. Coastal-humid is strikingly bimodal: about a third of its locations are among the
cheapest anywhere, the rest among the most expensive — humidity alone is not enough; the
day–night swing and sunshine decide it. All 822 locations found a design that respects the
swelling-ceiling limit.

![Daily tilt](figures/fig6_daily_tilt.png)

The controller re-tilts by ~16° over a typical year, tracking the sun: steep in winter (up to the
60° limit at 40°N), shallow in summer, mirrored in the southern hemisphere, near flat on the
equator. The fixed- and seasonal-tilt runs will put a number on what that freedom is worth.

## What the numbers can and cannot be trusted for yet

- **They are surrogate predictions.** The surrogate is very accurate on held-out locations, but
  a search that picks the best of many options also picks up the surrogate's optimistic errors.
  The active round re-runs each chosen optimum in true physics and measures that bias
  (`active-check`); `validate-bo` then compares against per-location true-physics optimization.
- **Hindsight + daily tilt is a ceiling.** Persistence (yesterday's weather as the forecast) and
  climatological schedules are the realistic cases.
- **Two design limits bind.** Salt loading sits at its upper bound (8) in 62% of locations, so
  the true optimum lies beyond the explored range; vapor gap sits at 60 mm in 28%, most likely
  because gap height is not priced in the cost model. Neither is the current focus, but both
  shape the absolute LCOW values.
- **One weather year** (2024), from the Open-Meteo cache.

## Next steps

1. Active round on the daily/hindsight optima → `active-check` → re-fit → re-run anchors.
2. Realistic modes: fixed tilt + persistence (headline), then seasonal tilt and climatological.
3. Design search on all ~14.7k locations, with out-of-distribution climates flagged.
4. `validate-bo`: the optimality gap in $/m³ against per-location true-physics BO.

## Files

| file | contents |
|---|---|
| `make_figures.py` | regenerates every figure here from the synced run directory |
| `figures/` | the PNGs above |
| `../../sawh_bayesopt/outputs/two_stage/main/` | the synced run (gitignored): `pca.npz`, `featval*.csv`, `holdout_report.json`, `selection.csv`, `opt/<mode>/anchors.*` |
