#!/bin/bash
# Full-grid BayesOpt sweep, split across parallel Slurm array tasks (each its
# own GPU allocation) -- same split as sbatch_gpu_sweep_array.sh, just handing
# each task's site range to run_bayesopt_sweep.py instead of the brute-force
# run_gpu_sweep.py.
#
# Each task computes its own contiguous [start, end) site range (from its
# array index and the array's total size -- edit only --array below, nothing
# else needs to stay in sync) and writes to its own output-dir/summary
# (avoids concurrent-write contention between tasks) -- merge afterward:
#   (head -1 outputs/gpu_bayesopt_sweep/task_0/summary.csv; \
#    tail -n +2 -q outputs/gpu_bayesopt_sweep/task_*/summary.csv) \
#     > outputs/gpu_bayesopt_sweep/full_sweep_summary.csv
#
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_bayesopt_sweep_array.sh
#
# Tune the --array range/throttle below to how many concurrent serc GPU
# allocations your account can realistically get -- see
# sbatch_gpu_sweep_array.sh's comment on the %K suffix.
#SBATCH --job-name=sawh-bayesopt-sweep-full
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-39%8
#SBATCH --output=gpu_sweep/logs/bayesopt_%A_%a.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_bayesopt_sweep

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
  --site-range "${START}" "${END}" --step "${STEP}" \
  --output-dir "outputs/gpu_bayesopt_sweep/task_${TASK_ID}" \
  --resume
