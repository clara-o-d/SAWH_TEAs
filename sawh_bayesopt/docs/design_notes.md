# Design notes

## The two-stage pipeline (`climate.py`, `daily_surrogate.py`, `two_stage.py`)

The per-site BO below costs a full physics campaign per map cell and folds daily
operation into annual-constant design dimensions. The two-stage pipeline splits the two:

- **Design** -- hydrogel thickness, vapor gap, salt loading (and, under fixed tilt, the
  tilt) -- is chosen once per cell and must serve the whole year.
- **Control** -- seal/open offsets on the 15-minute grid, and tilt when it varies -- is
  re-chosen every day against that day's weather.

Between them sits a *daily* surrogate `g(design, control, start-of-day state, day
features) -> (water that day, end-of-day state, P(swelling cap))`, fitted on physics
outputs rather than dollars: LCOW is applied afterwards, so the economics stay auditable
and every map regenerates when cost assumptions change without retraining.

**Why start-of-day state is an input.** The physics chains the gel's loading from one day
to the next. Whether that memory matters depends on whether absorption saturates
overnight, which depends on the climate, so the surrogate is told the state rather than
asked to average over it. Training years are chained under a random control every day
(`daily_surrogate.simulate_years`), so every simulated year yields ~366 training rows that
span the control space, with the state distribution that real operation produces. The
inner loop then walks the year greedily, feeding each day's predicted end state into the
next day.

**Features.** Each day is its hourly [T, RH, GHI] cycle (72 numbers) projected onto a PCA
fitted over every cached day on Earth, plus scalars the decomposition can under-weight:
RH at the diurnal T minimum, daytime GHI integral, dewpoint depression at peak sun, diurnal
T amplitude, elevation, |latitude| and noon zenith. The last two are there because tilt
acts through the sun's path, which the GHI shape alone does not reveal. Only T, RH and GHI
are used because they are the only channels the physics consumes: `h_amb` is fixed at
10 W/m2K and nothing radiates to a sky temperature, so wind or T_sky features would be
variance the surrogate could only overfit. `validate-features` checks the retained PC
count by running the physics on real years and on years rebuilt from the retained
features.

**Sampling.** Realized climates occupy a thin curved manifold, so cells are stratified
over real cells (KMeans in descriptor space, with the member nearest each centroid kept)
rather than over a uniform climate box. Hyper-arid, high-altitude, monsoonal and
coastal-humid cells are rare by area but decisive for the headline result, so each is
topped up to a quota. Each selected cell is crossed with a per-cell scrambled-Sobol set of
designs. The budget is fixed by `select` and is independent of map resolution.

**Hold-out by cell.** Held-out rows are whole cells (split in `select`), never random
rows: a random split leaks each climate across the split and flatters the error by about
an order of magnitude. `holdout_report.json` gives daily error and, more importantly,
annual error with the surrogate chained through the year under the physics' own
controls. Both are broken out by regime.

**Inner loop: enumeration, not gradients.** There are 1089 seal/open schedules per tilt
level (x13 for daily tilt), so a day is one batched forward pass plus an argmax. That gives
the exact discrete optimum with no local-minimum risk, and a discrete schedule is what a
real controller runs. The tilt modes (fixed, seasonal = per quarter, daily) nest, so the
yield gain of each step up is itself a result. Schedule modes are chosen on the decision
features and scored on the real day:
- hindsight: the day's own weather;
- persistence: yesterday's weather, standing in for a day-ahead forecast;
- climatological: the month's mean day;
- constant: one annual schedule, the reference BO's own control space.

**Outer loop.** This is the same GP + constrained EI + Kriging-Believer code the reference
BO uses (`surrogate.py`, `acquisition.py`), over 3 dimensions. Cells run in lockstep so
each round is one vmapped surrogate sweep, and they are warm-started from the optima of
their nearest solved climate neighbours. A design is infeasible if the gel sits on the
swelling ceiling on more than 5% of days
(`two_stage.CAPPED_DAY_FRACTION_MAX`). That is the climate-dependent over-swelling
constraint, driven by high RH at high salt loading. Salt loading enters LCOW through the
yield and through the sorbent cost.

**Active round.** Cells are drawn in proportion to the relative ensemble spread at their
optimum. Physics is then run at and around each optimum under the surrogate's own schedule
with jitter, and the surrogate is refit. Accuracy is needed near the optimum and about its
value, not uniformly.

**Validation.** Cells whose descriptors fall outside the training distribution
(Mahalanobis, 99th percentile of training cells) are flagged in the maps (`ood` column),
not silently extrapolated. `validate-bo` runs the per-site true-physics BO on the
held-out cells and scores the two-stage design and schedule with true physics. It reports
the gap in $/m3: (fixed, constant) is the like-for-like optimality gap, and the daily
modes add the value of daily control.

What follows documents the per-site true-physics BO, which is now the reference the
pipeline is validated against.

## Why Bayesian optimization, not a grid or an evolutionary search

`solar_lumped` has no optimizer -- only brute-force grid/OAT sweeps
(`scripts/parameter_sweep.py`, `scripts/grid_param_sweep.py`). One LCOW
evaluation (all 365 real days, Aitken-converged cyclic state on day 1) costs ~380s on
a laptop (`solar_lumped/docs/sherlock_param_sweep.tex`), which rules out
anything that needs hundreds-to-thousands of evaluations. Bayesian
optimization (EGO: GP surrogate + Expected Improvement) is designed
specifically for this "few, expensive, black-box evaluations" regime, unlike
the local NLP solvers (Ipopt/Bonmin) tried on the earlier, unrelated ZSR
system model, which got stuck in local optima 36-47% worse than what
multistart could find.

## How this actually works, step by step

If the only ML you've seen so far is "fit a model to a fixed dataset, then
evaluate it once," the big mental shift here is: there is no dataset up
front. The "labels" (LCOW numbers) are produced one at a time by running the
real physics simulation, and each one costs minutes. The whole job of this
package is to decide, as cheaply as possible, *which* design is worth
spending the next few minutes of compute on.

There are two models involved, and it's easy to conflate them:

1. **The true model** -- `solar_lumped/gpu_sweep`'s JAX daily-cycle + Aitken
   pipeline (`jax_daily_cycle.py`), wrapped by `evaluator.py`. Feed it 6
   numbers describing a system design (hydrogel thickness, vapor gap, tilt,
   ...) and it simulates a year of operation and returns one number,
   `combined_lcow` (USD/m^3 of water at the one site being optimized).
   This is the function we're minimizing. It agrees with
   `solar_lumped`'s CPU `ode_system.py` to <0.03% (`gpu_sweep/FINDINGS.md`)
   and is ~8x faster even single-threaded on a CPU with no GPU.
2. **The surrogate** -- a Gaussian Process (GP), built in `surrogate.py`
   with scikit-learn's `GaussianProcessRegressor`. This is the "ML model" in
   the familiar sense: fit on whatever (design, LCOW) pairs have been
   measured so far (24 to start, +3 per round), it predicts LCOW for *any*
   design without running the slow simulator. The key property that makes a
   GP specifically useful here (over, say, a small neural net) is that it
   doesn't just output a number -- it outputs a mean guess `mu(x)` *and* a
   standard deviation `sigma(x)`: "here's my best guess, and here's how
   unsure I am." Everything below is built on that uncertainty estimate.

### Why a GP, and why this kernel

- `build_gp()` uses `ConstantKernel * Matern(nu=2.5) + WhiteKernel`. The
  Matern term encodes "designs that are close together in the 6-D space
  should have similar LCOW" -- it's what lets the GP interpolate sensibly
  between the handful of points it's actually seen. `nu=2.5` is a moderate
  smoothness assumption (roughly: twice differentiable), a reasonable
  default for a physically continuous cost surface without assuming it's
  perfectly smooth. The `WhiteKernel` adds a small noise floor so the GP
  doesn't chase numerical jitter in the simulator as if it were signal.
- All 6 design variables get rescaled to a `[0, 1]` unit cube
  (`to_unit_cube`) before fitting, since they live on very different natural
  scales (meters vs. degrees vs. unitless ratios) and a kernel with one
  length-scale per dimension needs comparable ranges to fit well.
  `normalize_y=True` does the same for the LCOW targets.
- `n_restarts_optimizer=10` refits the kernel's hyperparameters (length
  scales, noise level) from 10 random starting points each time the GP is
  fit, because that inner fit is a non-convex optimization and can get stuck.

### Picking where to sample next: Expected Improvement

This is the part a standard ML class doesn't usually cover, since you don't
normally get to choose which point to label next -- here, choosing well is
the entire point, because each label costs ~380s x however many sites.

`acquisition.py::expected_improvement` scores any candidate design `x` as

```
z = (y_best - mu(x) - xi) / sigma(x)
EI(x) = (y_best - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)
```

(`Phi`/`phi` = normal CDF/PDF), which in words is: "how much better than the
best LCOW seen so far (`y_best`) do we expect this point to be, averaged over
everything the GP is still unsure about, floored at zero if it doesn't look
worth trying." A candidate can score well for two different reasons -- its
mean prediction `mu(x)` is good (**exploitation**: "this looks like a good
design"), or its uncertainty `sigma(x)` is large (**exploration**: "no idea
what happens here, and it might be great"). That mean/uncertainty trade-off
*is* Bayesian optimization, and it's why it beats a grid or random search
under a tiny evaluation budget: a grid spends its budget uniformly regardless
of what it's already learned, EI spends it wherever the GP's own model says
looking next is most valuable. `xi=0.01` (`BayesOptConfig.ei_xi`) is a small
margin that nudges EI slightly toward exploitation.

EI is itself a function of `x`, so finding the best next design to try means
*maximizing* EI -- a second, much cheaper optimization problem, solved in
`propose_next` with `scipy.optimize.differential_evolution` (gradient-free,
because the EI surface tends to be flat almost everywhere with a few sharp
spikes, which gradient methods handle poorly -- especially early on when the
GP has seen very few points).

### Getting several candidates at once: Kriging-Believer

`evaluate_requests` stacks every uncached (site, design) instance into
one `jax.vmap`-compiled call, so proposing only one design per round would
waste that batching. `propose_batch` gets `batch_size` diverse candidates via a
trick called Kriging-Believer: propose the best point by EI, *pretend*
("believe") its outcome is exactly the GP's own mean prediction there,
add that fake observation to a scratch copy of the GP, refit, and propose
again. Each later proposal now "sees" the earlier ones as already-explored
(lower uncertainty nearby), so the batch spreads out instead of piling onto
the same peak, without needing a true batch-EI (qEI) implementation.

### The full loop (`bayesopt.py::run_bayesopt`)

1. **Warm start**: draw `n_init=24` designs via Latin-hypercube sampling
   (space-filling -- spreads samples evenly across all 6 dimensions at once,
   unlike uniform random, which tends to clump). No rejection step: the
   worst corner of the box is 6.03 mm of dry gel against the 7 mm minimum
   gap, so no sampled design starts with the gel in the condenser
   (`design_space.latin_hypercube_design`).
2. Evaluate all 24 on the true model (one batched `jax.vmap` call across
   every design, walking all 365 days, cached to disk so a crash doesn't
   lose already-paid-for evaluations).
3. Fit the GP on those 24 (design, LCOW) pairs.
4. Loop: propose a batch of `batch_size=3` next designs by EI, evaluate them
   on the true model, append the results to the GP's training data, refit.
5. Stop when either the evaluation budget (`n_total=50`) runs out, or the
   best LCOW seen hasn't improved by more than `stall_rel_tol=0.5%` for
   `stall_rounds=3` rounds in a row (diminishing returns -- no reason to keep
   spending evaluations once the search has flattened out).

## Why sites run in lockstep (and why batch width is nearly free)

Measured on a `serc` A100, one batched evaluation of the annual objective costs
**60.1 min for 1 design and 68.2 min for 8** — 8× the work for +13% time — and
XLA compilation is only ~20 s of that. The reason is structural: a year is ~366
*sequential* day-steps (`jax_daily_cycle.run_year_batched` walks days in a Python
loop, each warm-starting from the previous day's end state), and the batch axis
is the only parallel one. Each of those steps is a small vmapped solve, so the
GPU is latency-bound, not throughput-bound.

So **cost tracks the number of evaluation calls, not the number of designs.**
Two consequences shape the drivers:

- `bayesopt.run_bayesopt_sites` advances every site's loop in lockstep: each
  round's designs across all sites go into one `evaluator.evaluate_requests`
  call, as do the group's verification and baseline passes. Sites stay fully
  independent optimizations (own GP, own history, own `cache.jsonl`) — only the
  call is shared. A site-at-a-time sweep paid ~12 calls per site; a group of 32
  pays ~12 calls total.
- For a *single-site* run, where width can only come from `batch_size`, raising
  it is the only lever — at the cost of Kriging-Believer's approximation getting
  looser the wider each batch gets. Prefer more sites per group over a wider
  per-site batch when both are available.

Requests are `(site, design)` pairs rather than a cross product, since a design
is scored where it will be built (`evaluator.site_lcow_or_penalty`).

The remaining cost is the sequential day loop itself, and there are two levers.

**`--day-stride N`** (implemented) keeps every Nth calendar day: 366 → 74 days at
stride 5, so ~5× off every evaluation. It is a *different objective*, not a
cheaper estimate of the same one — the mean is over the sampled days, and each
sampled day warm-starts from the previous *sampled* day's end state, so the
sorbent's seasonal history advances in N-day jumps. That is cheap rather than
free because the cyclic state re-equilibrates within about a day. Because it is a
different objective it joins the cache key
(`evaluator.design_vector_hash` → `annual+elev+strideN`) and the `resolution`
column in `summary.csv`; stride 1 keeps the original key, so no existing
`cache.jsonl` is retired. Never merge strided and full-year rows into one
comparison.

**A `lax.scan` over stacked day weather** (untried) would collapse the 366 Python
dispatches — `run_year_batched` currently calls `np.asarray` on every day's
outputs, forcing a host sync per day — into one device-side scan. Unlike the
stride this leaves the objective untouched, so it needs no cache-key change. Its
payoff depends on how much of the ~9.6 s per day-step is dispatch overhead versus
the diffrax solve inside a day, which nobody has measured yet.

## Design-variable provenance

| variable | range used | source |
|---|---|---|
| hydrogel_thickness_m | [0.001, 0.010] | `parameters.xlsx` Physics, `Hydrogel reference thickness (H0)` sweep columns |
| vapor_gap_m | [0.007, 0.060] | `parameters.xlsx` Physics, `Vapor gap (L_g)` sweep columns; the lower bound also matches `Vapor-gap transport floor` |
| insulation_gap_m *(complex only)* | [0.001, 0.020] | `parameters.xlsx` Physics, `Insulation gap (L_ins)` sweep columns |
| fin_area_ratio *(complex only)* | [3.0, 12.0] | `parameters.xlsx` Physics, `Condenser fin area ratio (A_r)` sweep columns |
| tilt_deg | [0.0, 60.0] | `parameters.xlsx` Physics, `Tilt angle (theta)` sweep columns |
| salt_loading | [1.0, 8.0] | `parameters.xlsx` Physics, `Salt loading (SL)` sweep columns |
| eps_abs_ir | [0.05, 0.95] | `parameters.xlsx` Physics, `Absorber IR emissivity (eps_abs_ir)` sweep columns |
| condenser_air_speed_m_s | [0.0, 1.5] | `parameters.xlsx` Physics, `Condenser forced-air speed` sweep columns |
| seal_offset_h / open_offset_h | [-4.0, 4.0] | `parameters.xlsx` Physics, `Seal / open offset from sunrise-sunset` sweep columns |
| glazing_config | [0, 3] | not a workbook row -- the index range of `GLAZING_CONFIGS`, derived from that tuple |
| blend_u / blend_v | [0, 1] | not a workbook row -- the unit square of the stick-breaking map onto the ZSR simplex |

`DesignBounds` reads every one of the workbook rows above directly, so the table
is a description of the sheet, not a second copy of it.

The two rows marked *complex only* are bounds without a dimension in simple
mode: `design_space.SIMPLE_FIXED` pins them at solar_lumped's defaults (5 mm,
A_r = 7.1). The 6-dim simple space is the two-stage pipeline's design block
(thickness, vapor gap, salt loading) plus tilt and the cycle schedule, so the
per-site reference BO and the surrogate pipeline search the same box. `seal_offset_h` / `open_offset_h` are optimized in
both modes; because they move the day/night split, which lives inside the
weather profile, both modes now rebuild per-day profiles per design point rather
than fetching one profile set per site.

`tilt_deg` is in the profile for the same reason: POA transposition is on in both
fidelities, at each design's own tilt (`design_space.to_profile_kwargs`), so one
`tilt_deg` trades solar gain against gap convection instead of only entering the
Hollands `cos(theta)`. Sites also run at their real ambient pressure — elevation
comes off the same weather frame (`evaluator.fetch_site_inputs`), and it sets
every gap air property and the `h_amb` density derate, so the loop, the
verification neighbours and the Wilson baseline all see the same site.

No `condenser_thickness_m` row: it isn't a design variable in this package
(see "Known caveats" below).

## Known caveats

- **No `condenser_thickness_m` dimension.** Two independent reasons: (1)
  `solar_lumped/src/solar_lumped/economics/lcow.py` charges a fixed
  `device_bom_condenser` cost regardless of thickness, so it was a free
  cost-side lever with no downside -- `condenser_thermal_mass_j_m2_k()`
  (physics, not cost) is the only thing that depended on it. (2) The JAX
  `gpu_sweep/` fast path this package now evaluates against (see below)
  hardcodes condenser thermal mass at Table S3's constant
  (`jax_physics.py::CONDENSER_THERMAL_MASS_J_M2_K = RHO_AL * CP_AL * L_C_M`)
  rather than taking it as a per-instance input, so it isn't a real physics
  knob on that path either. `SystemConfig`'s own default for
  `condenser_thickness_m` already matches `table_s3.L_C_M`, i.e. the same
  constant `jax_physics.py` hardcodes -- simply not setting it is correct,
  not an approximation.
- **Evaluates via the JAX `gpu_sweep/` fast path** (`solar_lumped/gpu_sweep/`,
  specifically `jax_daily_cycle.py`), not `solar_lumped`'s CPU `ode_system.py`
  directly. It already matched this package's LiCl+hydrogel+quasi_steady
  scope and is ~8x faster even single-threaded on a CPU with no GPU, agreeing
  with the CPU path to <0.03% (`gpu_sweep/FINDINGS.md` Results 6/7).
  `evaluator.py::evaluate_requests` stacks every (site, design) instance
  in a round -- across every uncached design, not just one -- into one
  `jax.vmap`-compiled call instead of dispatching one CPU process per design.
- **Two independent local checkouts of the SAWH_TEAs GitHub repo exist**
  (`~/github-repos/SAWH_TEAs` and a nested copy inside
  `electrolyte_optimization/`). This package was built against, and its
  `solar-lumped-sawh` dependency resolves to, the **top-level** checkout
  (`~/github-repos/SAWH_TEAs`) -- confirmed byte-identical to the nested copy
  for every file this package actually imports, at the time this was built.
  If that stops being true, `sawh_bayesopt`'s results would silently be
  grounded in different physics than expected -- worth an occasional
  `diff -rq` sanity check between the two trees' `solar_lumped/src/` if both
  keep being edited independently.

## Non-goals for v1

- No `salt_name`/`sorbent` categorical dimensions -- LiCl + hydrogel only.
- No touching `LCOEconomicParams` -- financial parameters are fixed scenario
  inputs, not decision variables.
- No Díaz-Marín Fig. 4 sorption-enthalpy re-extraction -- desorption
  enthalpy stays at Wilson Table S3's `H_DES_J_PER_KG` constant.
