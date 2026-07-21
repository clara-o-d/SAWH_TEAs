#!/bin/bash
# Case 3 ("optical material limits": eps_abs_ir=0, eps_abs=1, eps_glass_ir=0,
# tau_glass=1 -- the idealized-device upper bound) full 1,405-site GPU sweep.
# Identical structure to sbatch_gpu_sweep_array.sh (Case 1) / _case2.sh, with
# Case 3's flags and its own output directory (don't mix with Case 1/2 output
# -- see docs/gpu_sweep_handoff.md).
#
# Note: eps_abs=1 and tau_glass=1 are fixed idealized values, not the usual
# swept 0.85-0.95/0.80-0.90 ranges -- pass --eps-abs 1.0 --tau-glass 1.0
# explicitly (single values, not lists) so the combo grid doesn't sweep them.
# hydrogel-thickness-mm and fin-area-ratio still sweep normally.
#
# Merge afterward:
#   (head -1 outputs/gpu_grid_sweep_case3/chunk_0.csv; tail -n +2 -q outputs/gpu_grid_sweep_case3/chunk_*.csv) \
#     > outputs/gpu_grid_sweep_case3/full_sweep_case3.csv
#
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_array_case3.sh
#
# Tune the --array range/throttle to your account's real serc GPU quota (see
# sbatch_gpu_sweep_array.sh's header comment for how %K works). Smoke-test
# Case 3 first (--num-sites 10, same flags) -- eps=0 is a genuinely different
# numerical regime than anything Case 1/2 exercised (see
# docs/gpu_sweep_handoff.md's note on this), worth confirming separately.
#SBATCH --job-name=sawh-gpu-sweep-case3
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-39%8
#SBATCH --output=gpu_sweep/logs/case3_%A_%a.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_grid_sweep_case3

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
  --eps-abs 1.0 --tau-glass 1.0 --eps-abs-ir 0.0 --eps-glass-ir 0.0 \
  --output-csv "outputs/gpu_grid_sweep_case3/chunk_${TASK_ID}.csv" \
  --resume
