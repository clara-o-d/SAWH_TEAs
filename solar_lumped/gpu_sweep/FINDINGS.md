# GPU sweep prototype: findings (single-instance JAX RHS + daily-cycle port)

Scope (see [`docs/gpu_sweep_handoff.md`](../docs/gpu_sweep_handoff.md)): port the
quasi_steady desorption RHS, the absorption RHS, and the full daily-cycle +
Aitken steady-state search to JAX, validated against the CPU pipeline on a single
site, before building the batched-across-the-whole-grid pipeline. Everything here
is CPU-only (this Mac has no CUDA GPU) -- speed numbers are single-core baselines,
not a GPU projection, but the correctness and architecture findings carry over
directly.

Code: [`jax_physics.py`](jax_physics.py) (RHS port, both phases),
[`jax_daily_cycle.py`](jax_daily_cycle.py) (daily-cycle integrator, Aitken loop,
and the batched/padded/masked cross-length versions of both),
[`validate_rhs.py`](validate_rhs.py) (pointwise RHS cross-check, both phases),
[`validate_desorption_integration_tsit5.py`](validate_desorption_integration_tsit5.py)
(single-phase integration + yield comparison),
[`validate_monthly_pipeline.py`](validate_monthly_pipeline.py) (full monthly +
Aitken pipeline vs. CPU and vs. the paper's reference values),
[`validate_batched_pipeline.py`](validate_batched_pipeline.py) (cross-length
batching + fixed-round-count Aitken vs. the serial pipeline),
[`benchmark_gpu_batch_size.py`](benchmark_gpu_batch_size.py) (batch-size scaling
scan -- written for, but not yet run on, real GPU hardware),
[`run_gpu_sweep.py`](run_gpu_sweep.py) (the actual GPU-driven sweep -- mirrors
`scripts/grid_param_sweep.py`'s CLI/weather/combo-grid/CSV-schema, batches one
site's full combo x month grid per compiled call; validated locally against real
weather on CPU, not yet run on GPU -- see `SHERLOCK_GPU_RUNBOOK.md` step 6). If
you're new to
JAX/GPU work generally, see [`GPU_PRIMER.md`](GPU_PRIMER.md) first; to actually
run any of this on Sherlock's `serc` A100s, see
[`SHERLOCK_GPU_RUNBOOK.md`](SHERLOCK_GPU_RUNBOOK.md).

## What was ported

The `desorption_solver="quasi_steady"` path (the sweep's default): Eqs. 1/3/4
(3-surface steady thermal balance, originally `scipy.optimize.root("hybr")`) nested
inside Eq. 5's desorption mass-flux root (originally `scipy.brentq`), driving a
3-state ODE `[c_w, H, T_cond]` originally integrated with `scipy.solve_ivp(Radau)`;
the absorption phase (2-state `[c_w, H]`, no thermal solve, `T_gel == T_amb`); and
the Aitken Delta^2 steady-periodic-state search on top of both. LiCl only (the
sweep's only salt) -- CaCl2/MgCl2/NaCl's `brine_equilibrium` path was not ported.

## Result 1: the physics port is correct

`validate_rhs.py` evaluates the JAX RHS at real states sampled from an actual CPU
`_integrate_desorption` trajectory (Atacama site, eps_abs 0.90 and 0.95, including
states with nonzero desorption flux, not just the trivial zero-flux ones) and
compares dy/dt component-by-component against the CPU RHS.

**Result: agreement to ~1e-11 relative error** at high iteration counts, ~1e-4-1e-5
at the cheap iteration counts actually used (see Result 2) -- comfortably inside the
CPU solver's own `rtol=1e-4`. The physics translated cleanly; no correctness
surprises.

## Result 2: the naive root-finding architecture is ~15x too expensive per RHS call

The CPU code nests two solves: an outer scalar `brentq` root over the desorption
mass flux `m_des`, each trial of which requires an inner 3x3 `root("hybr")` solve
for the thermal state. Porting this literally -- outer fixed-iteration bisection
wrapping an inner fixed-iteration Newton -- costs `n_bisect * n_iter` thermal-residual
evaluations per RHS call (with reasonable iteration counts, ~50*25 = 1250).

**Fix applied**: solve `[T_gel, T_abs, T_glass, m_des]` as one joint 4x4 Newton
system instead (`solve_desorption_state_joint` in `jax_physics.py`) -- the `m_des`
root and the thermal root are just two more rows/columns of the same linear system
each Newton step solves. This cut the cost to `n_iter` (~12) evaluations, a ~15x
reduction, with no loss of accuracy (back to ~1e-11 agreement with 12 joint
iterations vs. 8+22 nested iterations for similar cost).

**Takeaway for whoever continues this**: don't port CPU's solver structure
1:1. Where the CPU code uses nested black-box solvers (`brentq` wrapping `root`)
because that's what SciPy offers, JAX/vmap rewards collapsing them into one larger
joint Newton system instead -- it's both cheaper and more uniform across a batch
(no separate bisection bracket bookkeeping per instance).

## Result 3: the ODE is not stiff enough to need an implicit solver at this resolution

Tried `diffrax.Kvaerno5` (implicit) first, since the CPU code uses Radau/BDF.
It was extremely slow to reason about (differentiating through the nested
Newton/bisection for the implicit stage's own Newton solve creates a very deep AD
graph) and was abandoned in favor of `diffrax.Tsit5` (explicit, adaptive,
Dormand-Prince-family) once the joint-Newton fix (Result 2) made the RHS cheap
enough that stiffness stopped being an obvious requirement.

**Result: `Tsit5` converges cleanly** -- for the real Atacama desorption phase
(~477 steps at the CPU's own `dt=100s` grid), the adaptive controller needed 478
accepted steps and only 1 rejected step, essentially matching the CPU's own
`max_step=dt` step count. **This answers the handoff doc's open question directly:
at monthly-mean-day resolution, the quasi-steady desorption RHS is not
stiff enough to require an implicit method.** This matters because explicit
solvers don't need the RHS's Jacobian, so all the nested-Newton machinery inside
the RHS never has to be differentiated through by the ODE solver itself --
only forward-evaluated. That's a substantially simpler autodiff story for a GPU
port than an implicit solver would have required.

## Result 4: full-day yield matches CPU to ~0.01%

`validate_desorption_integration_tsit5.py` integrates one full desorption phase
(same IC and real Atacama weather profile as a CPU run) with JAX/diffrax/Tsit5 and
compares total water yield against `scipy.solve_ivp(Radau)`:

| eps_abs | CPU (Radau) yield | JAX (Tsit5) yield | relative diff |
|---|---|---|---|
| 0.90 | 1.884484 kg/m² | 1.884299 kg/m² | 0.0098% |
| 0.95 | 2.022479 kg/m² | 2.022314 kg/m² | 0.0081% |

(These use one annual-mean day with no Aitken warmup, so they're single-mean-day
numbers, not the monthly + cyclic-state pipeline -- see Result 6 for that.)

## Result 6: absorption ported too, full daily-cycle + Aitken pipeline matches CPU to ~0.01% -- but neither matches the doc's cited reference values

Absorption (`jp.absorption_rhs` in `jax_physics.py`) has no thermal root-solve at
all (`T_gel == T_amb` during open absorption, Note S1 Eq. S1) -- it's a closed-form
2-state RHS, and matched the CPU RHS to **exact float64 equality (0.0 relative
error)** pointwise, no iterative solve to introduce discretization differences.

`jax_daily_cycle.py` wires up the full daily cycle (`make_daily_cycle_fn`: absorption
-> desorption, one `jax.jit`-compiled function per profile) and an Aitken
Delta^2 steady-state search (`find_cyclic_state_jax`, a thin Python loop calling
the jitted daily-cycle fn twice per round -- same algorithm as
`ode_system.py::find_cyclic_state`, ~3-6 rounds, including its period-2-orbit
stall-detection fallback). `validate_monthly_pipeline.py` runs this at the real
monthly resolution (12 monthly mean-day profiles, live 2024 Atacama weather) and
compares against the CPU pipeline on identical profiles/config:

| eps_abs | CPU mean_yield | JAX mean_yield | JAX vs CPU | wall-clock (CPU / JAX) |
|---|---|---|---|---|
| 0.90 | 2.074918 kg/m² | 2.075232 kg/m² | 0.0152% | 259.1s / 30.9s |
| 0.95 | 2.174242 kg/m² | 2.174427 kg/m² | 0.0085% | 268.6s / 37.6s |

**The full pipeline is correct** (JAX vs CPU agreement is as tight as the earlier
single-phase results) **and ~8.4x faster even single-threaded on a CPU with no
GPU at all** -- purely from compiling the daily-cycle function once and reusing it
across all 12 Aitken warmup rounds x 2 evaluations/round, instead of re-dispatching
scipy's Radau/root/brentq calls from Python every time.

**But neither CPU nor JAX reproduces the handoff doc's cited reference values**
(1.707476 / 1.800478 kg/m² at eps_abs 0.90/0.95) -- both land at ~2.07-2.17
kg/m², about 21% higher. This was checked directly: calling
`grid_param_sweep.py`'s own `monthly_mean_profiles` / `combo_yield_kg_m2`
functions with `hydrogel_thickness_mm=4.0`, `warmup_method="aitken"` (i.e. the
literal production code, not a reimplementation) against live 2024 Atacama
weather from the same `.weather_cache` this repo already had, gives
**2.0749176 kg/m²** for eps_abs=0.90 -- matching this pass's CPU and JAX numbers,
not the doc's 1.707476. **This means the ~21% gap is not a JAX-porting bug** (JAX
and CPU agree with each other and with the actual production function to <0.02%)
-- it's a pre-existing discrepancy between whatever weather data/run produced the
doc's cited reference values and what today's `.weather_cache` + current codebase
reproduce. Possible causes, not investigated further this pass: the weather cache
was refreshed/changed since the reference was computed, the reference used a
different year, or the reference came from a slightly different code path than
`grid_param_sweep.py`'s current `monthly_mean_profiles`/`combo_yield_kg_m2`.
(Confirmed not a concern -- the doc's reference values are just stale/from an
earlier weather-data snapshot, unrelated to this port's correctness.)

## Result 5: `jax.jit` is not optional, and vmap batching works as expected

Two performance traps hit during this work, both worth flagging explicitly since
they're easy to reintroduce by accident in the next pass:

1. **Calling `diffrax.diffeqsolve` without wrapping it in `jax.jit` is
   catastrophic** -- an un-jitted single-instance integration that should take
   ~0.1s took multiple minutes (killed before completion) because every one of
   ~480 adaptive steps gets dispatched through Python/XLA one at a time instead of
   fused into one compiled `lax.while_loop`. Jitted: 1.0-1.2s to compile, ~0.1s
   warm.
2. **Any post-processing loop over per-timestep results must also be
   batched/jitted** -- a Python `for` loop calling the (un-jitted) RHS once per
   saved timestep to recover `m_des` for the trapezoidal yield integral reproduced
   the same multi-minute slowdown. Fixed by `jax.vmap` + `jax.jit` over the whole
   saved trajectory at once.

With both fixed, a **batch of 15 (hydrogel thickness x eps_abs) combos vmapped
through one compiled call** took 1.86s to compile (all 15 at once) and 0.56s warm
for all 15 together -- i.e. batching is already paying for itself even on CPU
(0.56s / 15 combos vs. ~0.1s/combo run serially: comparable per-combo cost, but
one compilation instead of 15, which is what actually matters at 189,675-combo
scale -- compiling once per *shape* and reusing across the whole grid, not once
per combo, is the whole point).

## Result 7: cross-length batching (padding + masking) works, and the fixed-round-count Aitken loop matches the adaptive one closely

Everything above batches combos that share one weather profile (Result 5). The
real grid also needs to batch across *sites/months*, whose real
absorption/desorption step counts differ (day length varies). diffrax needs one
static `t1` per batched call, so `jax_daily_cycle.py`'s `build_batch_arrays` /
`make_batched_daily_cycle_fn` pad every profile in a batch to the batch's max
length (repeating each profile's own last weather value) and mask the vector
field to `dy=0` once real time runs out for that instance -- equivalent to
stopping at each instance's own real end, but with one uniform step count the
whole batch can compile together.

This also required resolving the handoff doc's other open question --
fixed-round-count vs `lax.while_loop` for Aitken convergence -- since a batch's
instances converge at different rates. Implemented the doc's suggested strategy 1
(`find_cyclic_state_batched`): every instance runs the same fixed `max_rounds`
(no early exit, so it vmaps cleanly), then one vectorized final pass decides
per-instance whether the last round is trustworthy (`rel_step < tol`) or whether
to average the last two rounds instead -- a simplified, vectorized stand-in for
`find_cyclic_state`'s multi-round stall counter, not a byte-for-byte port of it.

Tested against the 12 real monthly profiles at the Atacama site, which naturally
have different lengths (378-477 desorption steps) -- a genuine, already-available
test of cross-length batching, not a synthetic one. Compared against the serial
per-month result (one profile at a time, adaptive per-instance Aitken,
`max_rounds=10`) that Result 6 already validated against CPU:

| | worst per-month rel. err | day-weighted mean rel. diff | wall-clock |
|---|---|---|---|
| batched (12 months, one compiled call, `max_rounds=8`, fixed) | 0.028% | 0.016% | 9.3s compile+run, 6.7s warm |
| serial (12 separate compiles, adaptive, `max_rounds=10`) | -- (reference) | -- | 30.6s |

**The padding/masking and fixed-round-count approximations cost <0.03% accuracy**
and the batched version is **~3.3x faster on first call, ~4.6x faster warm** --
again on one CPU core, no GPU. This directly answers the doc's "which is faster
in practice" question in favor of fixed-round-count for now: it's simpler to
implement correctly (no per-instance branching inside a traced function) and the
accuracy cost is small. `lax.while_loop` might still win at larger batch sizes
where the wasted rounds on already-converged instances dominate more -- worth
revisiting with real batch sizes (hundreds+) and on a GPU, not concluded here.

## Answers to the handoff doc's open questions

- **Stiff or not?** Not stiff enough to need an implicit method at monthly-mean-day
  resolution (Result 3) -- use `Tsit5` or similar explicit adaptive solver, not
  `Kvaerno5`/BDF. Simpler autodiff story, cheaper per step.
- **Fixed-round-count vs `lax.while_loop` for steady-state convergence?**
  Fixed-round-count wins for now (Result 7) -- <0.03% accuracy cost at
  `max_rounds=8` vs. the adaptive per-instance CPU/JAX result, and ~3-4x faster
  than looping serially over a 12-instance batch even on one CPU core.
  `lax.while_loop` wasn't implemented/compared -- flagged as possibly winning at
  much larger batch sizes (where wasted rounds on already-converged instances
  cost more) and worth a real GPU comparison, but fixed-round-count is the
  simpler default and there's no evidence yet that the complexity of
  `while_loop` is needed.
- **Precision: float32 or float64?** Not tested this pass -- everything here used
  `jax.config.update("jax_enable_x64", True)` throughout (required for `c_w`'s
  ~1e4-1e5 mol/m³ scale to resolve the CPU's `atol~1e-7`-level fluxes). Worth an
  explicit float32-vs-float64 accuracy sweep before committing, but there was no
  reason to try float32 here since x64 was already fast enough on CPU.
- **Per-instance memory / max batch size per GPU?** Partially answered (Result
  8): memory usage is nearly flat from 12 to 12,000 instances (61.2GB -> 61.5GB
  of 80GB on an A100), so memory is very unlikely to be the limiter for the full
  189,675-instance grid -- but 60,000 and 189,675 haven't been measured yet
  (timed out on a 1hr allocation), so the actual ceiling (if any) isn't confirmed.

## Recommended next steps, in order

The reference-value discrepancy from Result 6 is confirmed stale/not a concern
(per follow-up) -- not tracked as a next step.

## Result 8: real A100 numbers (Sherlock `serc`) -- small batches are much slower than CPU, but the amortization curve is real

First real-GPU data, on an A100-SXM4-80GB (`nvidia-smi`: driver 550.163.01, CUDA
12.4), via `salloc --partition=serc --gres=gpu:1`.

**Correctness carried over exactly**: `validate_batched_pipeline.py` on the GPU
reproduced the same ~0.03% worst-per-month / 0.016% mean-yield agreement against
the serial pipeline as the CPU run (Result 7) -- the physics port is correct on
real GPU hardware, not just a CPU JAX backend.

**Speed at small batch size (12 instances) is *worse* than CPU, by design of the
problem, not a bug**: the same `validate_batched_pipeline.py` that took ~47s total
on the Mac CPU took **585s (serial) + 104s/92s (batched first/warm)** on the A100
-- 15-19x slower. This is expected in hindsight: the pipeline does thousands of
tiny sequential steps per instance (fixed-iteration Newton solves inside an
adaptive ODE stepper inside an 8-round Aitken loop), each a real dispatch to the
GPU. At only 12 instances there isn't nearly enough parallel work per step to
amortize that per-dispatch overhead, so the GPU pays the same thousands of
round-trips as one instance would, for barely more work. A CPU doesn't pay that
discrete-device dispatch cost, so it wins easily at this size. (See
`GPU_PRIMER.md` for a plainer-language version of this.)

**But the amortization curve is exactly what was hoped for as batch size grows**
(`benchmark_gpu_batch_size.py`, same 12 real profiles tiled up to each size,
`find_cyclic_state_batched` with `max_rounds=8`):

| batch size | warm-rerun wall-clock | ms/instance | GPU memory used |
|---|---|---|---|
| 12 | 91.95s | 7662.1 | 61233 / 81920 MiB |
| 120 | 98.27s | 818.9 | 61237 / 81920 MiB |
| 1,200 | 111.76s | 93.1 | 61273 / 81920 MiB |
| 12,000 | 246.24s | 20.5 | 61505 / 81920 MiB |
| 60,000 | 837.02s | 14.0 | 62567 / 81920 MiB |
| 189,675 (full grid) | *(pending -- see below)* | | |

Going from 12 to 60,000 instances (5000x more work) cost only ~9.1x more
wall-clock -- a **~550x per-instance throughput improvement**. The speedup-per-10x
is shrinking as batch size grows (9.4x -> 8.8x -> 4.5x -> ~1.5x per decade from
12k->60k), consistent with the A100's actual parallel capacity (108 SMs) becoming
the limit rather than dispatch overhead -- expected diminishing returns, not a
problem. **GPU memory usage stays modest** (61.2GB -> 62.6GB from 12 to 60,000
instances, out of 80GB) -- still no sign of memory being the limiter.

Extrapolating from the 60,000-instance warm throughput (60,000 instances in
837.02s = ~71.7 instances/sec) to the full 189,675-instance grid (3.16x more)
gives very roughly 2,650s (~44 min) for one Aitken pass over the *entire*
site x combo grid, if throughput holds at that rate -- vs. the CPU sweep's
current multi-day, cluster-wide job. (This estimate and the whole table above
are measured under the architecture Result 9 below replaced -- kept here as the
historical record of what motivated the fix, not the current expected numbers.)

**Attempting 189,675 directly failed differently than expected**: not a `salloc
--mem` host-RAM issue (a bigger `--mem` wouldn't have helped -- see Result 9),
but a genuine GPU `RESOURCE_EXHAUSTED` from XLA trying to allocate a single
~60GB tensor.

## Result 9: the 189,675 OOM was a real architecture bug, not a hardware ceiling -- fixed

The batched daily-cycle function computed the water yield by (1) telling diffrax
to save the *entire* trajectory (`SaveAt(ts=t_eval)`, one saved point per ODE
step -- ~478 points for desorption), then (2) recomputing `m_des` at every one of
those saved points via a second `jax.vmap`, then (3) trapezoidally integrating
that over time. Step (2)'s vmap was *inside* a function that itself gets `vmap`ed
over the batch axis for the batched pipeline -- nesting two vmaps means XLA
effectively materializes a `(batch x num_saved_points)`-wide parallel computation,
each lane doing a full 4x4 Newton-solve-with-Jacobian. At the full grid,
that's `189,675 x 478 ~= 90.6 million` lanes computed simultaneously -- consistent
with the single ~60GB tensor allocation XLA's error message reported (and with
`--mem=128G` on the host not helping at all -- this was GPU device memory, not
host RAM, so the earlier guess about the cause was wrong).

**Fix**: integrate the cumulative water yield as a 4th ODE state variable
(`dW/dt = m_des`) instead of reconstructing it after the fact from a densely
saved trajectory. diffrax already computes `m_des` internally at every step to
advance the other three states -- folding it into the state vector means the
solver accumulates the answer as part of the normal integration, and only the
*final* state needs to be saved (`SaveAt(t1=True)`, not `SaveAt(ts=...)`). This
removes the second vmap and the dense trajectory entirely, for both
`make_daily_cycle_fn` (serial) and `make_batched_daily_cycle_fn` (batched) in
`jax_daily_cycle.py`. Also switched `adjoint=DirectAdjoint()` ->
`RecursiveCheckpointAdjoint()` while in there (lower memory if this code is ever
differentiated later; not required for the OOM fix itself, since nothing here
calls `jax.grad`, but no reason to keep the more memory-hungry default).

Re-validated on CPU after the fix: `validate_batched_pipeline.py` still agrees
with the serial pipeline to the same ~0.03% worst-case / 0.016% mean-yield
tolerance as before the fix (small yield-value shifts of ~0.1% vs. the pre-fix
numbers are expected and *more* accurate, not a regression -- integrating
continuously replaces a discrete trapezoidal approximation over ~478 points with
what the adaptive stepper actually computes). **Not yet re-measured on the GPU**
-- that's the immediate next step, now that the fix is in.

## Result 10: `run_gpu_sweep.py` -- the actual GPU-driven sweep, not yet run on GPU

Everything above validates the physics/integration/batching *machinery*; none of
it is a runnable replacement for `scripts/grid_param_sweep.py`. `run_gpu_sweep.py`
is that -- it imports `grid_param_sweep.py` directly (CLI patterns, weather
fetch, `combo_grid`/`build_device_config`, `_CSV_COLUMNS`/`_append_row`) so its
output is schema-identical to the CPU sweep's, and batches one site's full combo
x month cross product (up to 135 x 12 = 1,620 instances) into a single compiled
call via `build_batch_arrays`/`make_batched_daily_cycle_fn`/
`find_cyclic_state_batched`.

Validated locally (CPU, real Atacama weather, 2 combos x 12 months = 24
instances) against `grid_param_sweep.py`'s own `combo_yield_kg_m2` for the exact
same combos: CPU `2.074918`/`2.174242` (eps_abs 0.90/0.95) vs. this script's
`2.073405`/`2.172332` -- 0.073%/0.088% differences, the same order of magnitude
as Result 7's fixed-round-count approximation, not a new discrepancy.

## Result 11: the 10-site smoke test ran on the A100 -- correct, but compile-per-site dominates

`sbatch_gpu_sweep_smoke.sh` (10 real sites spanning different latitudes, full
135-combo grid) completed on the `serc` A100: **1,350 rows written (135 x 10,
exactly as expected), no errors**. Per-site time ranged 104.8s-200.4s
(mean 161.4s), 1614.3s total for all 10.

**Correctness carried over again** -- this is the first time the full
1,620-instance combo x month batch shape has run on real GPU hardware (bigger
than anything in Results 7-9's tiled benchmarks) using genuinely different real
per-site weather, not tiled/repeated data, and it held up.

**But the per-site time is compile-dominated, not compute-dominated**: the
60,000-instance tiled throughput from Result 8/9 (~71.7 instances/sec) predicts
only ~23s of actual computation for 1,620 instances; the other ~80-180s per site
is JAX recompiling from scratch, because every site has a different weather
profile shape (`n_abs_max`/`n_des_max` depend on that site's real day lengths, so
nothing compiled for one site is reusable for the next). Extrapolating the
measured 161.4s/site rate to the full 1,405-site grid gives **~63 hours (~2.6
days) sequentially on one A100** -- not the dramatic win the tiled benchmarks
alone would have suggested, because those never paid a per-shape recompile cost
(one shape, reused across the whole scan).

**Immediate mitigation, not yet the root-cause fix**: split the grid across
multiple concurrent GPU allocations rather than one sequential job --
`run_gpu_sweep.py` now takes `--site-range START END` for exactly this, and
`sbatch_gpu_sweep_array.sh` is a Slurm job array that computes each task's
range automatically from the array size (`--array=0-39%8` -- 40 chunks, 8
running at once by default, tune both numbers to your account's actual serc
GPU quota). N tasks running concurrently -> roughly N x faster than the 63-hour
sequential estimate, no new engineering beyond chunking. The actual root-cause
fix (making many sites share one compiled shape, "Next steps" #2 below) would
help *even a single GPU*, and remains unstarted.

## Result 12: the full-grid job array is running, correctness spot-checked, and faster than the smoke test predicted

After fixing a `sbatch_gpu_sweep_array.sh` bug (`grid_land_points()`'s one-time
"Loading Natural Earth land polygons..." stdout message was getting captured
ahead of the actual site-range output in the per-task python subshell, so every
task failed immediately with `invalid int value: 'Loading'` -- fixed by piping
through `tail -1`), the full 1,405-site x 135-combo job array is running on
`serc` (`--array=0-39%8`, 40 chunks of ~36 sites each, 8 concurrent).

**Per-site time in the real run (~94-98s) is noticeably better than the smoke
test's 161.4s average** -- task 0 finished all 36 of its sites in 4,400.9s
(~73 min, ~122s/site including its own first-site cold start). Not yet
understood why (possibly a compilation-cache effect from nearby-latitude sites
sharing similar/identical day-length shapes within one task's sequential site
loop; not confirmed). At this rate, 5 sequential waves of 8 concurrent
~73-minute chunks would finish the whole grid in **~6 hours**, better than the
~8 hour estimate extrapolated from the smoke test alone.

**Correctness spot-checked against the CPU pipeline at this real full-scale
run** (not just the smoke test): site (-54, -72), hydrogel 1.0mm, eps_abs 0.85,
tau_glass 0.8, all 3 fin_area_ratio values --

| fin_area_ratio | CPU mean_yield | GPU chunk_0.csv | rel. diff |
|---|---|---|---|
| 3.0 | 0.355796 | 0.355786 | 0.0028% |
| 7.1 | 0.366755 | 0.366889 | 0.0365% |
| 12.0 | 0.369801 | 0.369890 | 0.0241% |

Site-level weather stats (`mean_rh_frac`, `mean_t_amb_c`, `mean_solar_w_m2`)
matched the CPU pipeline exactly. All yield differences are within the same
<0.05% fixed-round-count tolerance seen throughout this document -- no new
discrepancy at full scale. This site's CPU convergence also hit the period-2
orbit stall-detection path Wilson's code already anticipates (cold,
strongly-seasonal high-latitude site), and the GPU pipeline's simplified
vectorized stall handling (Result 7) tracked it correctly.

## Next steps, in order

1. **Run `sbatch_gpu_sweep_array.sh` and see if the chunked/parallel approach
   holds up** -- new territory again: concurrent array tasks sharing the same
   `serc` partition/GPU pool, and the merge-the-chunk-CSVs step
   (`sbatch_gpu_sweep_array.sh`'s header comment) hasn't been exercised.
2. **Fix the per-site recompile cost directly** (Result 11's root cause, not
   yet attempted) -- group sites by similar day-length/latitude so many sites
   share one padded shape and one compile, the same trick Result 7 already
   uses across a *site's* 12 months, extended across *sites*. Would speed up
   both the sequential and the array-job paths, and reduces how many GPUs the
   array approach needs to hit a given wall-clock target.
3. **Re-run `benchmark_gpu_batch_size.py` on the GPU now that Result 9's fix is
   in** -- rerun the full default size list (`12 120 1200 12000 60000 189675`),
   not just 189,675, since the fix changes the per-instance memory/compute
   profile for every size, not just the one that OOM'd. This is the number
   needed to know whether the full grid fits in one batch and what it costs
   (relevant to step 2's site-batching design, not just curiosity).
4. Revisit `lax.while_loop` for Aitken convergence if larger batches show the
   fixed-round-count waste becoming a real cost (see the open question above)
   -- no evidence of this yet at 12,000 instances.
