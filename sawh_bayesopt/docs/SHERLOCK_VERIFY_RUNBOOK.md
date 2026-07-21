# Verifying sawh_bayesopt on Sherlock's `serc` A100s

This package now evaluates through `solar_lumped/gpu_sweep`'s JAX/diffrax fast
path (see `evaluator.py`, `docs/design_notes.md`) instead of `solar_lumped`'s
CPU `ode_system.py` directly. Everything so far (tests, a tiny local run) has
been checked on a Mac CPU with no GPU. This runbook is what to run on Sherlock
to (a) confirm it actually works on a real GPU, and (b) run the regression /
optimization-loop / baseline diagnostics that need a real, full-sized
(`n_init=24, n_total=50`) run to be meaningful -- a 6-point local smoke test is
too small for k-fold CV or a fair BayesOpt-vs-random comparison.

Mirrors `solar_lumped/gpu_sweep/SHERLOCK_GPU_RUNBOOK.md`'s structure, since
it's the exact same JAX/diffrax dependency this package now shares.

## 1. Environment setup (on a login node)

```bash
cd /home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt   # sibling of solar_lumped in the same checkout
ml python/3.12.1 uv
uv venv .venv_gpu && source .venv_gpu/bin/activate
uv pip install -e ../solar_lumped   # sawh_bayesopt depends on solar-lumped-sawh, installed editable
uv pip install -e .                 # pulls scikit-learn, scipy, pandas, matplotlib, plain jax/diffrax
uv pip install "jax[cuda12]"        # replace the CPU jax wheel with the CUDA build
```

`jax[cuda12]` pulls a self-contained CUDA/cuDNN runtime via pip -- it doesn't
need a matching `ml load cuda/...` system module, only an NVIDIA driver new
enough for CUDA 12 (A100 nodes should already satisfy this).

**Weather cache**: same as the gpu_sweep runbook -- if `solar_lumped`'s
`.weather_cache/` already has Cambridge and Atacama fetched (it should, from
prior CPU-sweep or gpu_sweep work), `sawh_bayesopt`'s
`--weather-cache-dir` can point straight at it (`../solar_lumped/.weather_cache`)
to skip re-fetching:

```bash
python3 scripts/run_bayesopt.py --weather-cache-dir ../solar_lumped/.weather_cache ...
```

## 2. Get an interactive GPU allocation on `serc`

```bash
salloc --partition=serc --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00
cd /home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt
source .venv_gpu/bin/activate
```

## 3. Sanity check: does JAX actually see the GPU?

```bash
python3 -c "import jax; print(jax.devices())"
nvidia-smi
```

**This is the single most important line in this whole runbook.** If
`jax.devices()` prints something with `Cpu` in it instead of `Gpu`/`cuda`,
nothing below is actually testing the GPU.

## 4. Confirm correctness carries over (should already be true)

```bash
SAWH_BAYESOPT_SLOW_TESTS=1 python3 -m pytest tests/test_integration_real_model.py -v
```

This is the same tiny end-to-end wiring test that already passed on CPU
(config -> `jax_daily_cycle`'s batched daily-cycle/Aitken pipeline ->
`lcow_from_daily_yield` -> cache -> GP fit); running it here just confirms
nothing about the GPU backend changes the result.

## 5. Run a real, full-sized optimization

This is the run the diagnostics below actually need -- a 6-point local smoke
test doesn't have enough data for meaningful cross-validation or a fair
BayesOpt-vs-random comparison.

```bash
python3 scripts/run_bayesopt.py \
    --n-init 24 --n-total 50 --batch-size 3 \
    --weather-cache-dir ../solar_lumped/.weather_cache \
    --run-id sherlock_gpu_run_1
```

Watch the per-round timing that gets printed -- on the GPU this should be
dramatically faster than the CPU-path's old ~380s/site/design estimate (see
`docs/design_notes.md` and `solar_lumped/gpu_sweep/FINDINGS.md`'s ~8x
CPU-vs-CPU number; the GPU number is the new information this run provides).
Outputs land in `outputs/runs/sherlock_gpu_run_1/`, including `config.json`
and `gp_state.joblib` (both needed by the diagnostics scripts below).

## 6. GP surrogate regression diagnostics

```bash
python3 scripts/diagnostics/gp_diagnostics.py --run-dir outputs/runs/sherlock_gpu_run_1
```

Writes `outputs/runs/sherlock_gpu_run_1/diagnostics/gp_regression_report.json`
and `gp_slices.png`. What "good" looks like:

- `standardized_residual_mean` near 0, `standardized_residual_std` near 1
  (see the report's own `interpretation` field for what over/under-confident
  looks like).
- `msll_gp_minus_trivial` clearly negative -- the GP should explain held-out
  points better than a constant mean/std baseline.
- Check `n_penalized_points` in the report before trusting the above: a run
  with many infeasible (`combined_lcow` == the 1e4 USD/m^3 penalty) designs
  will have those outliers dominate `cv_mse`/residuals in a small sample.
- `gp_slices.png`: the 95% CI band should visibly pinch down near the
  scattered black dots (actually-evaluated points) and widen away from them.
  If it's uniformly wide everywhere, the kernel's length scales are probably
  stuck at a bound (check `final_fit_hyperparameters` in the report).

## 7. Optimization-loop diagnostics

```bash
python3 scripts/diagnostics/loop_diagnostics.py --run-dir outputs/runs/sherlock_gpu_run_1
```

Writes `loop_best_so_far.png`, `loop_hyperparameters.png`,
`loop_acquisition_and_exploration.png`, and `loop_diagnostics_report.json`.
What "good" looks like:

- `loop_best_so_far.png`: steep early drop, then a flat tail. Still steeply
  dropping at evaluation 50 means the budget was too small.
- `loop_hyperparameters.png`: length scales / signal variance / noise settle
  into a band in the later rounds rather than continuing to swing between
  the kernel's bounds (`1e-2`/`1e2` for length scale, `1e-3`/`1e3` for signal
  variance -- see `surrogate.py::build_gp`).
- `loop_acquisition_and_exploration.png`: achieved EI of each round's
  proposed batch trending down (log scale) toward 0, and within-batch
  spread / distance-to-incumbent both shrinking over rounds (early rounds
  spread out and far from the incumbent = exploration; late rounds tight and
  close = exploitation).

## 8. Baseline: does it actually beat random search?

```bash
python3 scripts/diagnostics/baseline_random_search.py \
    --bayesopt-run-dir outputs/runs/sherlock_gpu_run_1 \
    --run-id sherlock_gpu_run_1_random --eval-batch-size 6
```

Runs the same total number of evaluations (`n_total` from
`sherlock_gpu_run_1/config.json`) as pure IID-uniform random search against
the same true model and sites, then writes
`outputs/runs/sherlock_gpu_run_1_random/random_search_report.json` and
`.../diagnostics/bayesopt_vs_random.png` comparing both best-so-far curves.
Prints a one-line verdict (`BayesOpt BEATS/DOES NOT BEAT random search`).
This costs a second `n_total`-evaluation budget, so it's the priciest of the
three diagnostics -- run it last, and only once steps 6-7 already look
reasonable, since a failing BayesOpt run failing this comparison too doesn't
tell you anything new that step 6/7 wouldn't have already flagged.

## What to send back

- `jax.devices()` and `nvidia-smi` output (confirms it actually ran on the
  A100).
- Wall-clock for the `n_init=24, n_total=50` run in step 5, and how that
  compares to the CPU number.
- The three diagnostics reports (`gp_regression_report.json`,
  `loop_diagnostics_report.json`, `random_search_report.json`) and their
  plots.
- Anything that looks wrong against the "what good looks like" bullets above
  -- especially a `bayesopt_beats_random: false` verdict, which would mean
  the surrogate/acquisition loop isn't adding value over blind sampling and
  is worth debugging before trusting any `report.json` recommendation from
  this package.
