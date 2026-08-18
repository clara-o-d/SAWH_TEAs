#!/bin/bash
# Full 1,405-site x 8-scenario GPU sweep, split across parallel Slurm array tasks
# (each its own GPU allocation). One scenario sweep replaces the old per-case
# scripts (Case 1/2/3, the ambient-condenser and DRH variants): the scenario list
# lives in site_sweep.SCENARIOS and every scenario runs the parameters.xlsx
# baseline design, so there are no sweep flags to keep in sync here.
#
# Each task computes its own contiguous [start, end) site range (from its array
# index and the array's total size -- edit only --array below) and writes to its
# own chunk_<task_id>.csv, avoiding concurrent-write contention. Merge afterward:
#   (head -1 outputs/gpu_scenario_sweep/chunk_0.csv; tail -n +2 -q outputs/gpu_scenario_sweep/chunk_*.csv) \
#     > outputs/gpu_scenario_sweep/full_sweep.csv
#
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_array.sh
# Smoke-test first with sbatch_gpu_sweep_smoke.sh.
#
# The %K suffix caps how many array tasks run *simultaneously*. Sites within one
# task now share a single compilation (they are the batch axis, see
# run_gpu_sweep.py), so FEWER, BIGGER tasks are cheaper here than they were for
# the old per-site combo sweep -- the trade is GPU memory, which grows with
# sites-per-task x scenarios-in-group x days. 20 tasks is ~70 sites each.
#SBATCH --job-name=sawh-gpu-scenarios
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-19%8
#SBATCH --output=gpu_sweep/logs/scenarios_%A_%a.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_scenario_sweep

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

python3 gpu_sweep/run_gpu_sweep.py \
  --site-range "${START}" "${END}" --step "${STEP}" \
  --output-csv "outputs/gpu_scenario_sweep/chunk_${TASK_ID}.csv" \
  --resume
