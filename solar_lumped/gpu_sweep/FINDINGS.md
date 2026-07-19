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
scan -- written for, but not yet run on, real GPU hardware). If you're new to
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
- **Per-instance memory / max batch size per GPU?** Not tested -- no GPU available
  in this environment. The 15-combo vmap above is a correctness/architecture
  smoke test, not a sizing test.

## Recommended next steps, in order

The reference-value discrepancy from Result 6 is confirmed stale/not a concern
(per follow-up) -- not tracked as a next step.

1. **Scale the batch up from 12 instances toward the real grid size** (189,675
   site x combo pairs) and check whether compile time, memory, and the
   fixed-round-count accuracy cost (Result 7) hold up at hundreds or thousands of
   instances per batch, not just 12 -- everything validated so far is at small
   scale, on CPU.
2. **Batch across sites, not just months at one site** -- Result 7 batched 12
   months at the *same* site (same device config, different weather only). The
   real grid varies device config too (135 combos x however many sites fit in one
   batch); `build_batch_arrays`/`make_batched_daily_cycle_fn` already accept
   per-instance config arrays, so this should compose directly, but hasn't been
   tried.
3. Decide how to split the 1405 sites x 135 combos = 189,675 total instances into
   batches (one shape per batch after padding to that batch's max weather-profile
   length -- sites vary a lot in day length depending on latitude/season, so
   padding *all* 189,675 into one shape would waste a lot of compute on short-day
   sites; grouping by latitude band before padding would help).
4. Get access to an actual GPU (`serc` A100 partition, see
   [`SHERLOCK_GPU_RUNBOOK.md`](SHERLOCK_GPU_RUNBOOK.md) -- in progress) and
   re-measure everything above -- every number in this document is a single CPU
   core, no GPU at all. The ~3-8x speedups already seen from compile-once
   batching are a lower bound; the real win is expected once thousands of
   instances batch into one compiled
   call on a GPU instead of 12.
5. Revisit `lax.while_loop` for Aitken convergence if step 1's larger batches show
   the fixed-round-count waste becoming a real cost (see the open question above).
