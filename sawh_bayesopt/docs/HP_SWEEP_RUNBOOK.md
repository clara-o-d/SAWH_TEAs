# Running the ei_xi / stall_rel_tol / n_init hyperparameter sweep on Sherlock

Context: `outputs/runs/sherlock_gpu_run_1` stalled after 33-39 evaluations
with EI-proposed points clustering tightly around the incumbent best design
(see `outputs/runs/sherlock_gpu_run_1/diagnostics/gp_slices_near_xbest.png`).
That's consistent with a working exploitation phase converging quickly under
`ei_xi=0.01` and a stall rule (`stall_rel_tol=0.005`, `stall_rounds=3`) that
can stop the search early -- but it's also consistent with those settings
just being too exploitation-heavy for this problem. `scripts/hp_sweep.py`
runs a 3x3x3 grid over `ei_xi`, `stall_rel_tol`, and `n_init` (see its module
docstring for the exact values and why `n_total` scales with `n_init` rather
than staying fixed) to tell those two explanations apart.

Before this sweep can run at all, two things needed fixing/adding, already
done on this branch:
- `scripts/diagnostics/gp_diagnostics.py` now also reports, per run, whether
  `acquisition.py`'s inner `differential_evolution` EI-maximization calls
  actually converged (`result.success`) or silently exhausted `maxiter`
  (`result.nit >= maxiter`) -- see its new `summarize_de_diagnostics()` and
  `de_diagnostics_summary` report field. A high hit-maxiter rate would mean
  apparent under-exploration might be an artifact of an under-budgeted inner
  optimizer, not a real property of the acquisition function -- worth ruling
  out before trusting any of this sweep's results.
- It already computed standardized residuals and MSLL vs. a trivial
  mean/std baseline (`cross_validate()`'s `standardized_residual_std` /
  `msll_gp_minus_trivial`) -- unchanged, just confirming these feed the
  sweep's aggregated table too (`scripts/hp_sweep.py` pulls both, plus the
  new DE summary, out of `gp_regression_report.json` for every combination).

## 1. Environment setup

Same as `docs/SHERLOCK_VERIFY_RUNBOOK.md` section 1 -- if you already have
`.venv_gpu` set up from that runbook, skip this.

```bash
cd /home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt
ml python/3.12.1 uv
uv venv .venv_gpu && source .venv_gpu/bin/activate
uv pip install --only-binary :all: -e ../solar_lumped
uv pip install --only-binary :all: -e .
uv pip install "jax[cuda12]"
```

## 2. Pull this branch

```bash
git pull
```

`scripts/hp_sweep.py`, `scripts/plot_hp_sweep.py`,
`scripts/sbatch_hp_sweep_smoke.sh`, `scripts/sbatch_hp_sweep_full.sh`, the
`gp_diagnostics.py` DE-diagnostics update, and the new `--case` flag on
`scripts/run_bayesopt.py` all need to be present -- if `python3
scripts/hp_sweep.py --help` doesn't exist yet, the pull didn't pick them up.

## 3. Smoke test first

```bash
sbatch scripts/sbatch_hp_sweep_smoke.sh
```

2 tiny combinations (single site, single-day resolution, ~12 evaluations
each), 2 workers sharing 1 GPU -- this validates the whole
sweep-plus-diagnostics-plus-plotting pipeline (including the GPU-memory-
sharing setup described in `hp_sweep.py`'s module docstring) in well under
the 30-minute time limit, before committing GPU-hours to the real grid.

Check `logs/hp_sweep_smoke_<jobid>.out` for:
- `jax.devices()` printing `Gpu`/`cuda`, not `Cpu`.
- Both combinations finishing with a `best=... stopped=... n_eval=...` line,
  not an `ERROR: ...` line.
- `outputs/runs/hp_sweep_smoke/sweep_results.csv` has 2 rows, neither with a
  non-empty `error` column.
- `outputs/runs/hp_sweep_smoke/hp_sweep_marginals.png` and
  `hp_sweep_heatmap_*.png` exist and aren't blank/all-NaN (with only 2
  combinations most subplots will be sparse -- that's expected, this is
  checking the plotting code runs end-to-end, not that the smoke grid is
  scientifically meaningful).

## 4. The full sweep

Only after step 3 looks clean:

```bash
sbatch scripts/sbatch_hp_sweep_full.sh
```

27 combinations (3 ei_xi x 3 stall_rel_tol x 3 n_init values, full two-site
monthly evaluations each), 8 workers split across 2 GPUs, 12-hour time
limit -- adjust `--gres=gpu:2`/`--cpus-per-task`/`--time` in the sbatch
script if your allocation differs. Each combination writes its own
`outputs/runs/hp_sweep_1/<combo-tag>/` (same layout as any single
`run_bayesopt.py` run -- `config.json`, `cache.jsonl`, `history.csv`,
`report.json`, `diagnostics/gp_regression_report.json`,
`diagnostics/de_diagnostics.json`, `diagnostics/gp_slices.png`).

## 5. What to look at

```bash
column -s, -t outputs/runs/hp_sweep_1/sweep_results.csv | less -S
```

- Any row with a non-empty `error` (or `verify_error`/`diagnostics_error`)
  column -- a failed combination, worth reading its own `run_dir` before
  trusting the rest.
- `hp_sweep_marginals.png`: does `best_combined_lcow_usd_per_m3` actually
  trend down (better) as `ei_xi` increases, or is the baseline `ei_xi=0.01`
  already fine and this is flat/noisy? Does raising `stall_rel_tol` (stops
  earlier) visibly hurt `best_combined_lcow_usd_per_m3` vs. lowering it
  (runs longer)? Does higher `n_init` change `standardized_residual_std`
  (more initial coverage -> better-calibrated GP) even though every
  combination gets the same post-init EI budget?
- `frac_de_hit_maxiter` row: if this is elevated (>10-20%) for the
  higher-`ei_xi` combinations specifically, that's a confound -- a larger
  `ei_xi` can flatten/broaden the EI landscape DE has to search, so a fixed
  `maxiter`/`popsize` that was fine at `ei_xi=0.01` might not be at
  `ei_xi=0.1`. If you see this, rerun the affected combinations with
  `acquisition.propose_next`'s `maxiter`/`popsize` raised before trusting
  their `best_combined_lcow_usd_per_m3`.
- `hp_sweep_heatmap_best_combined_lcow_usd_per_m3.png`: the full 3-way
  interaction -- e.g. does a high `ei_xi` only help when `stall_rel_tol` is
  also loose enough to let the extra exploration actually run?

## Send back

- `jax.devices()`/`nvidia-smi` from the smoke test log.
- `outputs/runs/hp_sweep_1/sweep_results.csv`, `hp_sweep_marginals.png`, and
  both `hp_sweep_heatmap_*.png` files.
- Total wall-clock for the full sweep job (`sacct -j <jobid>
  --format=Elapsed`), so the smoke test's per-combination timing can be
  sanity-checked against it.
