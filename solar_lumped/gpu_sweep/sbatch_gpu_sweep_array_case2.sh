#!/bin/bash
# Case 2 (selective-surface radiative physics: eps_abs_ir=0.05, eps_glass_ir=0.95)
# full 1,405-site GPU sweep -- identical structure to sbatch_gpu_sweep_array.sh
# (Case 1), just with the two new flags and a separate output directory (the
# CSV schema gained 2 columns for Case 2/3, so don't mix outputs with Case 1's
# files -- see docs/gpu_sweep_handoff.md).
#
# Each task computes its own contiguous [start, end) site range and writes to
# its own chunk_<task_id>.csv -- merge them afterward:
#   (head -1 outputs/gpu_grid_sweep_case2/chunk_0.csv; tail -n +2 -q outputs/gpu_grid_sweep_case2/chunk_*.csv) \
#     > outputs/gpu_grid_sweep_case2/full_sweep_case2.csv
#
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_array_case2.sh
#
# Tune the --array range/throttle to how many concurrent serc GPU allocations
# your account can realistically get -- see sbatch_gpu_sweep_array.sh's header
# comment for how the %K throttle works. Run the Case 2 smoke test
# (--num-sites 10, same eps-abs-ir/eps-glass-ir flags) before this and confirm
# it looks right first.
#SBATCH --job-name=sawh-gpu-sweep-case2
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-39%8
#SBATCH --output=gpu_sweep/logs/case2_%A_%a.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_grid_sweep_case2

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
  --eps-abs-ir 0.05 --eps-glass-ir 0.95 \
  --output-csv "outputs/gpu_grid_sweep_case2/chunk_${TASK_ID}.csv" \
  --resume
