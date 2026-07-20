#!/bin/bash
# Full 1,405-site GPU sweep, split across parallel Slurm array tasks (each its
# own GPU allocation) instead of one ~2.6-day sequential job -- see
# FINDINGS.md Result 11 for why the per-site recompile cost makes splitting
# across GPUs the right first move before optimizing the recompile itself.
#
# Each task computes its own contiguous [start, end) site range (from its
# array index and the array's total size -- edit only --array below, nothing
# else needs to stay in sync) and writes to its own chunk_<task_id>.csv
# (avoids concurrent-write contention between tasks) -- merge them afterward:
#   (head -1 outputs/gpu_grid_sweep/chunk_0.csv; tail -n +2 -q outputs/gpu_grid_sweep/chunk_*.csv) \
#     > outputs/gpu_grid_sweep/full_sweep.csv
#
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_array.sh
#
# Tune the --array range/throttle below to how many concurrent serc GPU
# allocations your account can realistically get. The %K suffix caps how many
# array tasks run *simultaneously* -- e.g. --array=0-39%8 submits 40 chunks
# (~35 sites each) but only runs 8 at once, letting Slurm queue and run the
# rest automatically as slots free up, without you needing to know an exact
# quota up front. More tasks = smaller, more independently resumable chunks.
#SBATCH --job-name=sawh-gpu-sweep-full
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-39%8
#SBATCH --output=gpu_sweep/logs/full_%A_%a.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_grid_sweep

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

STEP=3.0
TASK_ID="${SLURM_ARRAY_TASK_ID}"
NUM_TASKS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

RANGE=$(python3 -c "
import sys
sys.path.insert(0, 'src')
from solar_lumped.weather.land_grid import grid_land_points
total = len(grid_land_points(${STEP}))
num_tasks = ${NUM_TASKS}
chunk = -(-total // num_tasks)  # ceil division
start = ${TASK_ID} * chunk
end = min(start + chunk, total)
print(start, end)
")
read -r START END <<< "${RANGE}"
echo "Task ${TASK_ID}/${NUM_TASKS}: sites [${START}, ${END})"

if [ "${START}" -ge "${END}" ]; then
  echo "Empty range for this task (more array tasks than sites) -- nothing to do."
  exit 0
fi

python3 -c "import jax; print('jax.devices():', jax.devices())"

python3 gpu_sweep/run_gpu_sweep.py \
  --site-range "${START}" "${END}" --step "${STEP}" \
  --output-csv "outputs/gpu_grid_sweep/chunk_${TASK_ID}.csv" \
  --resume
