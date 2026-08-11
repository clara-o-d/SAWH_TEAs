#!/bin/bash
# Shared body for the three Case 2 global-sweep variants (see the sbatch_gpu_sweep_case2_*
# scripts that source this). Everything except the #SBATCH header and the two physics
# flags is identical between them, so it lives here once -- three copies of the chunking
# math is three chances for the variants to stop being comparable.
#
# The sourcing script must set, before sourcing:
#   OUT_DIR    output directory for this variant's chunk_<task>.csv files
#   RUN_FLAGS  array of extra run_gpu_sweep.py flags that define the variant
#
# Merge one variant's chunks afterward:
#   (head -1 "$OUT_DIR"/chunk_0.csv; tail -n +2 -q "$OUT_DIR"/chunk_*.csv) > "$OUT_DIR"/full_sweep.csv

set -euo pipefail
mkdir -p gpu_sweep/logs "${OUT_DIR}"

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

STEP=3.0
TASK_ID="${SLURM_ARRAY_TASK_ID}"
NUM_TASKS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

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
echo "Task ${TASK_ID}/${NUM_TASKS}: sites [${START}, ${END}) -> ${OUT_DIR}"
echo "Variant flags: ${RUN_FLAGS[*]:-<none>}"

if [ "${START}" -ge "${END}" ]; then
  echo "Empty range for this task (more array tasks than sites) -- nothing to do."
  exit 0
fi

python3 -c "import jax; print('jax.devices():', jax.devices())"

python3 gpu_sweep/run_gpu_sweep.py \
  --site-range "${START}" "${END}" --step "${STEP}" \
  --eps-abs-ir 0.05 --eps-glass-ir 0.95 \
  "${RUN_FLAGS[@]}" \
  --output-csv "${OUT_DIR}/chunk_${TASK_ID}.csv" \
  --resume
