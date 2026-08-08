#!/bin/bash
# Full-grid COMPLEX-fidelity BayesOpt sweep, split across parallel Slurm array
# tasks (each its own GPU allocation). Same site split as
# sbatch_bayesopt_sweep_array.sh -- this is that script with --complex and a
# budget sized for 13 design dims instead of 6.
#
# Parallelism is across sites, not within one optimization: each site's BayesOpt
# loop is sequential (round N's EI proposal needs round N-1's fit), but sites are
# fully independent, so the grid splits cleanly into contiguous [start, end)
# ranges -- one GPU per array task, no communication.
#
# --time=10:00:00 rather than the simple sweep's 4h, for two compounding reasons:
# complex mode rebuilds each site's daily profiles per design point (A1's
# seal/open offsets, B4's condenser air and POA tilt live *inside* the profile --
# evaluator.py::_profiles_for_design), and the budget below is 130 evaluations
# per site rather than 50 because a 13-dim anisotropic Matern needs more points
# than 6 dims did. Both multiply the per-site cost.
#
# Each task writes its own output-dir/summary.csv (no concurrent-write
# contention) -- merge afterward:
#   (head -1 outputs/gpu_bayesopt_sweep_complex/task_0/summary.csv; \
#    tail -n +2 -q outputs/gpu_bayesopt_sweep_complex/task_*/summary.csv) \
#     > outputs/gpu_bayesopt_sweep_complex/full_sweep_summary.csv
#
# Passes --resume, so if a task hits the 10h wall just resubmit this exact
# script: sites already in that task's summary.csv are skipped, and within the
# site that was mid-flight, cache.jsonl replays every completed design for free.
#
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_bayesopt_sweep_complex_array.sh
#
# Sizing, measured: a 3-design complex run at one site cost 159s of loop time on
# a laptop CPU, i.e. ~53s/design, so a 130-design site is ~1.9h there and
# plausibly ~40min on an A100 (the profile rebuild is numpy, and doesn't
# accelerate). 140 array tasks over step-3.0's 1405 land sites is ~10 sites/task,
# ~7h -- inside the 10h wall with margin. The simple sweep's 40 tasks would put
# ~35 sites in a task and blow through it.
#
# %8 throttles to 8 concurrent tasks, so raising the task count shortens each
# task but does NOT shorten total wall clock -- that is bound by concurrent GPU
# allocations (1405 sites / 8 at a time). Expect this to span several
# resubmissions; --resume is what makes that cheap. Raise %8 if your account can
# actually hold more serc allocations, or drop to --step 5.0 (513 sites) for a
# coarser map -- see sbatch_gpu_sweep_array.sh's %K comment.
#SBATCH --job-name=sawh-bayesopt-sweep-complex
#SBATCH --time=10:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-139%8
#SBATCH --output=gpu_sweep/logs/bayesopt_complex_%A_%a.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_bayesopt_sweep_complex

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

STEP=3.0
TASK_ID="${SLURM_ARRAY_TASK_ID}"
NUM_TASKS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

# grid_land_points() prints a one-time diagnostic ("Loading Natural Earth land
# polygons...") to stdout on first call -- `tail -1` keeps only the final
# "start end" line so that diagnostic can't get parsed as the range.
RANGE=$(python3 -c "
import sys
sys.path.insert(0, 'src')
from solar_lumped.weather import grid_land_points
total = len(grid_land_points(${STEP}))
num_tasks = ${NUM_TASKS}
chunk = -(-total // num_tasks)  # ceil division
start = ${TASK_ID} * chunk
end = min(start + chunk, total)
print(start, end)
" | tail -1)
read -r START END <<< "${RANGE}"
echo "Task ${TASK_ID}/${NUM_TASKS}: sites [${START}, ${END})"

if [ "${START}" -ge "${END}" ]; then
  echo "Empty range for this task (more array tasks than sites) -- nothing to do."
  exit 0
fi

python3 -c "import jax; print('jax.devices():', jax.devices())"

python3 gpu_sweep/run_bayesopt_sweep.py \
  --complex \
  --site-range "${START}" "${END}" --step "${STEP}" \
  --n-init 60 --n-total 130 --batch-size 6 \
  --output-dir "outputs/gpu_bayesopt_sweep_complex/task_${TASK_ID}" \
  --resume
