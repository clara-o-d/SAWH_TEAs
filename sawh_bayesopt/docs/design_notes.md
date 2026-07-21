# Design notes

## Why Bayesian optimization, not a grid or an evolutionary search

`solar_lumped` has no optimizer -- only brute-force grid/OAT sweeps
(`scripts/parameter_sweep.py`, `scripts/grid_param_sweep.py`). One LCOW
evaluation (monthly-resolution, Aitken-converged cyclic state) costs ~380s on
a laptop (`solar_lumped/docs/sherlock_param_sweep.tex`), which rules out
anything that needs hundreds-to-thousands of evaluations. Bayesian
optimization (EGO: GP surrogate + Expected Improvement) is designed
specifically for this "few, expensive, black-box evaluations" regime, unlike
the local NLP solvers (Ipopt/Bonmin) tried on the earlier, unrelated ZSR
device model, which got stuck in local optima 36-47% worse than what
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
   numbers describing a device design (hydrogel thickness, vapor gap, tilt,
   ...) and it simulates a year of operation and returns one number,
   `combined_lcow` (USD/m^3 of water, averaged across the two weather
   sites). This is the function we're minimizing. It agrees with
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

`evaluate_batch` stacks every uncached design's (site, month) instances into
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
   unlike uniform random, which tends to clump), with a rejection rule that
   resamples any design whose vapor gap leaves too little clearance over the
   hydrogel thickness (physics-degenerate, not worth an expensive evaluation).
2. Evaluate all 24 on the true model (one batched `jax.vmap` call across
   every design x site x month instance, cached to disk so a crash doesn't
   lose already-paid-for evaluations).
3. Fit the GP on those 24 (design, LCOW) pairs.
4. Loop: propose a batch of `batch_size=3` next designs by EI, evaluate them
   on the true model, append the results to the GP's training data, refit.
5. Stop when either the evaluation budget (`n_total=50`) runs out, or the
   best LCOW seen hasn't improved by more than `stall_rel_tol=0.5%` for
   `stall_rounds=3` rounds in a row (diminishing returns -- no reason to keep
   spending evaluations once the search has flattened out).

## Design-variable provenance

| variable | range used | source |
|---|---|---|
| hydrogel_thickness_m | [0.001, 0.010] | `data/economics/lcow_economic_params.csv`'s `hydrogel_thickness_min/max_m` (explicitly documented as "bound on **optimized** hydrogel thickness"), matches `params.py`'s `HYDROGEL_THICKNESS_MIN/MAX_M` module constants and `parameter_sweep.py`'s sweep range |
| vapor_gap_m | [0.007, 0.060] | `parameter_sweep.py::make_sweep_params` (`vapor_gap_mm`, 7-60mm); lower bound also matches `table_s3.VAPOR_GAP_TRANSPORT_MIN_M` |
| insulation_gap_m | [0.001, 0.020] | `parameter_sweep.py` (`insulation_gap_mm`, 1-20mm) |
| fin_area_ratio | [3.0, 12.0] | `parameter_sweep.py` and `grid_param_sweep.py`'s swept range |
| tilt_deg | [0.0, 60.0] | `parameter_sweep.py`'s swept range |
| salt_to_polymer_ratio | [1.0, 8.0] | `parameter_sweep.py` (`salt_to_polymer_ratio`) |

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
  knob on that path either. `DeviceConfig`'s own default for
  `condenser_thickness_m` already matches `table_s3.L_C_M`, i.e. the same
  constant `jax_physics.py` hardcodes -- simply not setting it is correct,
  not an approximation.
- **Evaluates via the JAX `gpu_sweep/` fast path** (`solar_lumped/gpu_sweep/`,
  specifically `jax_daily_cycle.py`), not `solar_lumped`'s CPU `ode_system.py`
  directly. It already matched this package's LiCl+hydrogel+quasi_steady
  scope and is ~8x faster even single-threaded on a CPU with no GPU, agreeing
  with the CPU path to <0.03% (`gpu_sweep/FINDINGS.md` Results 6/7).
  `evaluator.py::evaluate_batch` stacks every (design, site, month) instance
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
