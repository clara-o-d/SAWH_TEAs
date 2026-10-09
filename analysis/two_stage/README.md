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
| Active round: true physics at 300 chosen optima | done: surrogate over-predicts water by ≤ 1.2% (median) |
| Design search on all 14,711 locations | done for **daily tilt + hindsight** (Fig. 7–9) |
| Other control modes (fixed / seasonal tilt, persistence, climatological) | not yet |
| Comparison against per-location true-physics optimization (`validate-bo`) | done (Fig. 12): median −1.8%, within 2% at 92% |

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

## Fig. 7–9 — Every mapped location

The same design search, run at all 14,711 cached locations: the 822 anchors with their full
budget, every other location warm-started from the best designs of its three most similar
anchors (by climate) and given 12 evaluations. Locations are gridded onto 0.5° land cells for
the maps. Still **daily tilt + hindsight**, so still the ceiling.

![Global LCOW](figures/fig7_global_lcow.png)

Cost falls almost monotonically toward the equator: median **$4.04/m³ within 15° of the
equator**, 5.23 at 15–30°, 6.39 at 30–45°, 8.72 at 45–60°, and 14.07 above 60°. The other
expensive belts are the Tibet–Central Asia highlands and the Sahara. Ringed points (81) are
climates outside the range the surrogate was trained on — mostly the Tibetan Plateau, the
Andes, Greenland and the East African highlands (median elevation 1.8 km) — where the map is
extrapolation and should be checked with physics before it is quoted.

![Global panels](figures/fig8_global_design_and_operation.png)

Water yield and optimal thickness tell different stories: yield is highest in the wet tropics
(Amazon, Congo, Southeast Asia: 3–3.5 kg/m²/day), while the gel is thickest in the dry subtropics
(Sahara, Arabia, Australia, the Kalahari, the US Southwest), where strong drying can fully
cycle a thick layer. Re-tilting matters least in an equatorial band, where the noon sun barely
moves through the year, and most at mid-to-high latitudes. Surrogate uncertainty is ~1–2%
almost everywhere, rising to ~5% in the Arctic and Greenland.

![LCOW CDF](figures/fig9_lcow_cdf.png)

**26% of mapped locations come in at or below $5/m³ and 73% at or below $10/m³** (median
$7.30/m³, 2.0 kg/m²/day). By climate, the medians are monsoonal 5.34, other 7.04,
high-altitude 9.17, hyper-arid 9.23 and coastal-humid 13.56 USD/m³; coastal-humid has the
longest tail but its cheapest fifth is among the cheapest anywhere. These are shares of the
cached locations, which are not spread evenly over land, so they are not land-area fractions.

![SAWH vs desalination](figures/fig10_sawh_vs_desal.png)

**Against delivered desalination.** Desalination uses the repo's existing model
(`analysis/comparison/desal_vs_sawh_map.py`, after Kocher & Menon 2023): coastal seawater RO
at $1/m³ plus levelized conveyance to the site — $5×10⁻⁴/m³ per m of elevation and
$1.358×10⁻³/m³ per km from the nearest ocean coastline (median 383 km here) — with a multiplier
on both conveyance costs for higher-transport scenarios. At the baseline transport cost
desalination is cheaper **everywhere** (median $1.84 vs $7.30/m³), even against SAWH's
daily-tilt, hindsight ceiling. SAWH starts to win only where moving water inland costs several
times the baseline: at **36% of locations at 5×** and **60% at 10×** (the paper's high-cost
scenario); the median location breaks even at 7.3×. The comparison is location-by-location
(right panel), which the overlaid distributions (left) cannot show on their own. Desalination
here assumes a connected conveyance network and ignores the fixed cost of small, remote
demand, which is where SAWH's case is strongest.

The map's 822 anchor rows come from the anchor search run before the active-round re-fit;
the other ~13,900 used the re-fitted model. The re-fit changed held-out accuracy by under
0.1%, so the mix does not change the picture, but re-running the anchors on the re-fitted
model would make it uniform.

## Fig. 11–12 — How well the two-stage route performs

![Self-scoring](figures/fig11_self_scoring.png)

**Does the surrogate score its own choices correctly?** The 300 optima the active round
re-ran in true physics — each on its exact chosen 366-day schedule — against what the surrogate
predicted for them: median bias +0.3%, 90% within ±3%. A search that picks the best of
thousands of options favours whatever the model overrates, so this is the check that matters
for the optimizer, not ordinary test error. Once `validate_bo.csv` is synced, a second panel
does the same for LCOW at the held-out locations' picks.

The right panel does the same for LCOW at the held-out locations' picks: median bias 0.0%,
90% within ±2.3%.

![vs true-physics BO](figures/fig12_vs_true_physics_bo.png)

**Does it find the best design?** On the held-out locations, per-location Bayesian
optimization run directly on the true physics (50 full-year evaluations per location, over the
same design box plus tilt and one annual schedule) is the reference. The two-stage pick for each
location is replayed in the same true physics under the **same rules** — one tilt, one annual
schedule (left, middle):

- median **−1.8%** LCOW (−$0.11/m³), **within 2% of the BO or better at 92% of locations**;
  by climate the median runs from +0.2% (hyper-arid) to −4.6% (coastal-humid);
- the two-stage route is usually *cheaper* because it splits the problem: operation is solved
  exactly (every one of ~14,000 tilt × seal × open combinations tried for each design), leaving a
  3-D design search, while the BO has to search all 6 dimensions with 50 evaluations — its best
  is good but not fully converged;
- the failures are extreme climates: Tien Shan at 5.7 km (+95%, flagged out-of-distribution),
  the Greenland ice sheet (+17%), the New Guinea highlands (+20%, flagged), one Sahara location
  (+11%). At most of these the surrogate scored its own pick correctly; it misjudged how other
  designs would do, so its search missed the region the BO found.

Six held-out locations, all above 66°N, are excluded: the BO found **no** feasible design there
in 50 evaluations, while the two-stage picks are feasible in true physics at $13.7–22.9/m³.

With daily tilt and hindsight (right), the same locations come out a median **7.3% cheaper than
the BO's fixed operation** (−$0.56/m³; −13% in coastal-humid climates) — the value of the
control freedom, not a test of the surrogate.

## What the numbers can and cannot be trusted for yet

- **They are surrogate predictions, and true physics confirms them.** A search that picks the
  best of many options also picks up the surrogate's optimistic errors, so the active round re-ran
  300 chosen optima, each on its exact 366-day schedule, in the true physics (`active-check`):

  | climate | n | water over-prediction, median | 90th pct (abs) | LCOW surrogate − physics, median |
  |---|---:|---:|---:|---:|
  | other | 192 | 0.1% | 2.7% | −$0.01/m³ |
  | monsoonal | 14 | 0.5% | 2.3% | −$0.02/m³ |
  | high-altitude | 29 | 0.4% | 3.3% | −$0.05/m³ |
  | hyper-arid | 28 | 0.9% | 2.0% | −$0.08/m³ |
  | coastal-humid | 37 | 1.2% | 6.0% | −$0.12/m³ |

  The optimism is real but small — about a percent of water, cents per m³ — and no chosen design
  breaks the swelling cap in physics. `validate-bo` will still compare against per-location
  true-physics optimization, which tests whether the search finds the best design, not only
  whether it scores its choice correctly.
- **Hindsight + daily tilt is a ceiling.** Persistence (yesterday's weather as the forecast) and
  climatological schedules are the realistic cases.
- **Two design limits bind.** Salt loading sits at its upper bound (8) in 62% of locations, so
  the true optimum lies beyond the explored range; vapor gap sits at 60 mm in 28%, most likely
  because gap height is not priced in the cost model. Neither is the current focus, but both
  shape the absolute LCOW values.
- **One weather year** (2024), from the Open-Meteo cache.

## Next steps

1. Re-fit with the active-round rows (one round was enough: bias ≤ 1.2%).
2. Realistic modes: fixed tilt + persistence (headline), then seasonal tilt and climatological.
3. All-locations search for the realistic modes; physics checks at the out-of-distribution locations.
4. `validate-bo`: the optimality gap in $/m³ against per-location true-physics BO.

## Files

| file | contents |
|---|---|
| `make_figures.py` | regenerates every figure here from the synced run directory |
| `figures/` | the PNGs above |
| `../../sawh_bayesopt/outputs/two_stage/main/` | the synced run (gitignored): `pca.npz`, `featval*.csv`, `holdout_report.json`, `selection.csv`, `opt/<mode>/anchors.*` |
