# Running the GPU sweep prototype on Sherlock's `serc` A100s

Everything in `FINDINGS.md` was measured on a Mac CPU (no GPU available). This is
the single biggest gap in that document -- this runbook is what to run on the
real `serc` A100 partition to fill it in. Nothing here needs new code beyond what
already exists in this folder; JAX picks CPU vs. GPU automatically based on what's
installed and visible, so the same scripts run as-is.

## 1. Environment setup (on a login node)

Reuses the exact recipe that already worked for this repo's CPU sweep
(`docs/sherlock_param_sweep.tex`, "Sherlock-side smoke test"), plus the two new
GPU packages:

```bash
cd /home/groups/cdiazm/SAWH_TEAs/solar_lumped   # same repo path the CPU sweep uses
ml python/3.12.1 uv
uv venv .venv_gpu && source .venv_gpu/bin/activate
uv pip install --only-binary :all: numpy scipy pandas requests-cache retry-requests shapely cartopy
uv pip install -e .
uv pip install "jax[cuda12]" diffrax
```

`jax[cuda12]` pulls a self-contained CUDA/cuDNN runtime via pip -- it does **not**
need a matching `ml load cuda/...` system module, only an NVIDIA driver new enough
for CUDA 12 (A100 nodes should already satisfy this). If `pip install "jax[cuda12]"`
fails to find a wheel, check `python3 --version` is still 3.12.x from the `ml`
load above, and check Sherlock's docs for whichever `jax[cuda12X]` suffix matches
whatever driver version `nvidia-smi` reports on a GPU node.

**Weather cache**: don't re-fetch anything. The CPU sweep already produced
`.weather_cache/openmeteo_cache.sqlite` in this same repo path (visible in `git
status` as untracked, ~21GB) from fetching every site in the real grid, including
Atacama (-23.6, -70.4) -- since Sherlock's home/group storage is shared between
login and compute nodes, the GPU job will see this cache automatically with no
transfer step, as long as you run from this same repo checkout.

## 2. Get an interactive GPU allocation on `serc`

```bash
salloc --partition=serc --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00
```

(Group `sh_o-serc` -- confirmed accessible per `docs/gpu_sweep_handoff.md` -- so no
extra `--account=` flag should be needed, matching the CPU smoke test's sbatch
header, which also didn't set one.) Once the allocation lands you on a GPU node:

```bash
cd /home/groups/cdiazm/SAWH_TEAs/solar_lumped
source .venv_gpu/bin/activate
```

## 3. Sanity check: does JAX actually see the GPU?

```bash
python3 -c "import jax; print(jax.devices())"
nvidia-smi
```

**This is the most important single line in this whole runbook.** If
`jax.devices()` prints something with `Cpu` in it instead of `Gpu`/`cuda`, nothing
below will actually be testing the GPU -- stop and fix the JAX install before
running anything else (send me exactly what it prints if this happens).

## 4. Re-run the existing validation scripts

These should behave identically to the CPU results in `FINDINGS.md` (same
correctness), but the timing numbers are the new, real information:

```bash
python3 gpu_sweep/validate_rhs.py
python3 gpu_sweep/validate_desorption_integration_tsit5.py
python3 gpu_sweep/validate_monthly_pipeline.py
python3 gpu_sweep/validate_batched_pipeline.py
```

`validate_monthly_pipeline.py` and `validate_batched_pipeline.py` are the slowest
(each does a real CPU run for comparison too, ~5 minutes each on the CPU side per
`FINDINGS.md`) -- if you're short on allocation time, `validate_batched_pipeline.py`
alone is the most informative one (it's the cross-length batching + fixed-round
Aitken test, i.e. the actual architecture the full sweep would use).

## 5. The real question: how big a batch fits, and how fast?

This is new -- `benchmark_gpu_batch_size.py` doesn't exist in the CPU findings
because there was no GPU to run it on. It tiles the same 12 real Atacama monthly
profiles + a few device configs up to increasingly large batch sizes and reports
compile time, per-instance throughput, and GPU memory at each size, stopping at
whatever size first fails (out-of-memory or otherwise):

```bash
python3 gpu_sweep/benchmark_gpu_batch_size.py
```

Default sizes are `12 120 1200 12000 60000 189675` (the last one is the *actual*
full grid size). If it OOMs partway through, that's useful information, not a
failure -- it tells us the real per-A100 ceiling. You can also pass custom sizes,
e.g. to binary-search around wherever it starts struggling:

```bash
python3 gpu_sweep/benchmark_gpu_batch_size.py --sizes 20000 30000 40000
```

## What to send back (steps 1-5)

For each script: the full printed output (accuracy/`rel_err` numbers should match
the CPU findings; the timing numbers are the new data). Specifically useful:

- `jax.devices()` and `nvidia-smi` output (confirms it actually ran on the A100).
- `benchmark_gpu_batch_size.py`'s full table -- this directly answers the open
  "max batch size per GPU" and "real GPU speedup" questions in `FINDINGS.md`.
- Whatever batch size (if any) it fails at, and the error message.

I'll fold whatever comes back into `FINDINGS.md` as a real GPU data point.

## 6. Once steps 1-5 check out: the actual sweep, on a small subset of real sites

`run_gpu_sweep.py` is the GPU counterpart to `scripts/grid_param_sweep.py` (see
its module docstring) -- it reuses that script's CLI, weather fetch, combo grid,
and CSV schema directly, but batches one site's full 135-combo x 12-month grid
(up to 1,620 instances) into a single compiled call instead of looping calls to
SciPy. This has only been validated on real weather data locally on a Mac CPU so
far (matches `grid_param_sweep.py`'s own `combo_yield_kg_m2` to ~0.02-0.07%,
consistent with Result 7's fixed-round-count tolerance) -- **not yet run on a
real GPU with the actual 1,620-instance-per-site batch size**, which is
meaningfully bigger than anything validated in steps 1-5 for a *single* site (all
the batch-size scaling data so far *tiled* one small profile set; this uses 12
genuinely different real months x up to 135 real combos at once).

Submit the smoke test (10 real sites, not tiled data) as a batch job rather than
running it interactively -- more realistic for something that might take a while,
and matches how the CPU sweep itself runs:

```bash
sbatch gpu_sweep/sbatch_gpu_sweep_smoke.sh
squeue --me                              # watch it queue/run
tail -f gpu_sweep/logs/smoke_<jobid>.out  # jobid from squeue or the sbatch output
```

Output lands in `outputs/gpu_grid_sweep/smoke_10sites.csv` -- a **separate**
directory from the live CPU sweep's `outputs/grid_sweep/`, so there's no risk of
the two runs touching the same files. When it finishes, send back:

- The full log output (per-site timing, any errors/OOMs -- 1,620 instances/site
  is 27x bigger than the 60,000-instance *tiled* test from step 5, but it's also
  a much smaller total batch than 189,675, so this is genuinely new territory,
  not a strict subset of what's already been measured).
- A few rows of `smoke_10sites.csv` so I can spot-check them against
  `grid_param_sweep.py`'s own CPU function for the same sites (I'll need the
  specific lat/lon values it picked, which the log prints).

If this holds up, next is scaling `--num-sites`/`--site-indices` up toward the
full 1,405-site grid -- not attempted yet.
