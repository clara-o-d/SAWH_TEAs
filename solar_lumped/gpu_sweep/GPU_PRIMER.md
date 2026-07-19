# GPU computing, for someone who's never touched one: what this folder is doing and why

You don't usually work with GPUs, so this explains the concepts behind
`gpu_sweep/` from scratch, tied to what was actually built and measured (see
[`FINDINGS.md`](FINDINGS.md) for the results themselves). No prior GPU/JAX
knowledge assumed.

## The problem, in one sentence

The device-parameter sweep (`solar_lumped/scripts/grid_param_sweep.py`) needs to
run the same small physics simulation ~2.27 million times (1405 sites x 135
device-parameter combos x 12 months x ~2 simulated days per steady-state search),
and doing that on a regular computer (a CPU) is slow enough that it's currently
running as a multi-day job spread across a shared university cluster.

## What's actually different about a GPU

A CPU (the chip in a normal laptop) is built to do a handful of different things
one after another, very fast and flexibly -- it's good at "do this, then depending
on the result do that, then this other thing." A GPU (graphics card chip) is built
to do the *same* simple arithmetic operation on *thousands of numbers at once* --
originally because that's what rendering a screen full of pixels requires (each
pixel needs the same lighting-and-color math). It's much less flexible than a CPU
per-operation, but if your problem really is "do the same thing to a huge pile of
independent numbers," a GPU can be 10-100x faster than a CPU at it.

The sweep's ~2.27 million simulations are independent of each other (site A's
result doesn't depend on site B's) and are all *identically shaped* (same
equations, same number of variables, just different input numbers) -- that's
exactly the shape of problem a GPU is good at. The CPU version runs them one at a
time (or a few dozen at a time across cluster nodes); a GPU can run thousands
simultaneously as one "batched" operation.

## Why you can't just point the existing code at a GPU

The existing simulation code uses SciPy (`scipy.integrate.solve_ivp`,
`scipy.optimize.root`, `scipy.optimize.brentq`) -- these are CPU-only, one-at-a-time
solvers. There's no flag that makes them run on a GPU; the *algorithm* has to be
rewritten in a language a GPU can execute. That's what this folder does, using a
library called **JAX**.

## The tools: JAX, `vmap`, and "compiling"

**JAX** is a Python library (from Google) for writing numerical code that can run
on a CPU or a GPU with the same source code, and that can be *compiled* --
converted once into a fast low-level program instead of being re-interpreted by
Python every time it runs. Two JAX features matter most here:

- **`jax.jit`** ("just-in-time compile"): the first time you call a JAX function,
  it traces through your Python code once and produces an optimized, compiled
  version. Every call after that reuses the compiled version instead of
  re-running your Python line by line. This is the difference between "interpret
  this recipe from scratch every time" and "write the recipe down as an optimized
  assembly line once, then just feed ingredients through it."
- **`jax.vmap`** ("vectorizing map"): takes a function written for *one* input and
  automatically turns it into a function that runs on a *whole batch* of inputs at
  once, with no per-item Python loop. This is the actual mechanism for "run the
  same simulation on thousands of sites simultaneously" -- you write the physics
  once for one site, and `vmap` handles turning that into the batched, GPU-shaped
  version.

For the ODE integration itself (advancing the simulation forward in time step by
step), this uses a companion library called **diffrax**, which provides
GPU-compatible versions of the numerical integrators SciPy would otherwise
provide.

## What this specific prototype did

1. **Rewrote the physics equations in JAX** (`jax_physics.py`). The original code
   solves several nonlinear equations at every simulated time step (e.g. "what
   temperature makes this energy balance equation true?") using SciPy's
   general-purpose root-finders. JAX doesn't have a drop-in replacement for those,
   so this reimplements the same math using **Newton's method** run for a *fixed*
   number of iterations (as opposed to SciPy's "keep iterating until it's precise
   enough, however many steps that takes") -- fixed iteration counts are what let
   `vmap` batch thousands of these solves together, since every one takes exactly
   the same number of steps.
2. **Checked the rewrite against the original** (`validate_rhs.py`) by plugging in
   the exact same numbers to both and comparing outputs. They agreed to about 11
   decimal places, which says the physics translated correctly.
3. **Wired the rewritten physics into a full-day simulation** (`jax_daily_cycle.py`)
   using diffrax, and checked the whole day's water-yield output against the
   original code -- agreement to ~0.01%.
4. **Actually ran things in a batch** -- proved that 12+ different simulations
   (different months, different device settings) can be combined into *one*
   compiled program and run together, rather than one at a time. This is the part
   that would give the real speedup on a GPU.
5. **All of this ran on a Mac CPU**, since there's no GPU in this environment.
   Every speed number in `FINDINGS.md` is "how much faster is a smarter CPU
   approach," not "how much faster is a GPU" -- those are different questions.
   The reason CPU speedups already showed up is #1 above (`jax.jit`): compiling a
   simulation once and reusing it for thousands of calls is faster than
   re-interpreting SciPy calls from Python every time, even with no GPU involved.
   A real GPU test still needs to happen (see FINDINGS.md's next steps) to know
   the actual GPU speedup, but the building blocks are now proven correct.

## Two mistakes worth knowing about (both fixed, but instructive)

- **Forgetting to compile.** Calling the JAX/diffrax simulation function without
  wrapping it in `jax.jit` made it *slower* than the original SciPy code -- because
  every one of ~480 tiny simulation steps was being re-interpreted by Python
  individually instead of running as one compiled program. This is a common
  beginner trap: JAX code that isn't compiled doesn't get JAX's speed benefit at
  all, and can be worse than not using JAX.
- **Batching only some of the work.** After compiling the main simulation, a
  separate small step (recomputing a byproduct value at each saved time point)
  was still running one point at a time in a Python loop, uncompiled -- and that
  alone caused the same multi-minute slowdown. The fix was to batch (`vmap`) that
  step too. The lesson: *every* piece of the pipeline has to be compiled/batched,
  not just the obviously expensive part -- one un-batched loop anywhere in the
  chain can dominate the total time.

## What "batching across different-length simulations" means, and why it was tricky

Different sites/months have different day lengths (more or less daylight), so
each simulation naturally runs for a different number of time steps. But GPU
batching wants every simulation in a batch to be the *same shape* (same number of
steps), so they can be laid out as one uniform block of numbers.

The fix (also standard in other GPU/ML work, e.g. batching sentences of different
lengths for a language model): **pad** every simulation to the length of the
longest one in the batch (by repeating its last real weather value), and add a
switch that **freezes** each simulation's state once it passes its own real
day-length, so the padding doesn't actually change anything -- it's just
bookkeeping to make the shapes match. This was checked against running each
simulation separately (unpadded) and matched to within 0.03%.

## Where this leaves things

Everything in this folder is a **correctness and architecture prototype**: proof
that the physics can be rewritten for GPU-style batched execution and gives the
same answers as the existing SciPy code, plus early evidence (from CPU-only
compiling/batching) that the approach should pay off. It is not yet: running on an
actual GPU, sized to the real 189,675-combo grid, or optimized for real GPU memory
limits. `FINDINGS.md`'s "Recommended next steps" section is the order to tackle
those in.
