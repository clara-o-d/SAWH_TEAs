#!/bin/bash
# Full 1,405-site x 8-scenario GPU sweep. One array task = one site chunk x one
# scenario group, because a task's cost is ~366 sequential day-steps PER GROUP and
# four groups in one task overruns any sane walltime.
#
# Measured on the serc A100: ~9.5 s per day-step at 30 instances, i.e. ~320 ms per
# instance-day against the ~14 ms Result 8/9 measured at 60,000 instances. The GPU is
# nowhere near saturated at these widths, so per-step time is set by one instance's
# serial integration and **extra batch width is nearly free** until the two costs
# cross over, around ~700 instances. Chunking aims at that crossover: 7 site chunks
# (~200 sites) x 3 scenarios = ~600 instances in the widest group. Wider chunks stop
# being free; narrower ones pay the ~9.5 s/day floor more times than necessary.
#
# ONE KNOB: --array below. Tasks must be a multiple of the scenario-group count (4),
# and chunks = tasks / groups -- the script fails loudly rather than silently
# mis-chunking, so only multiples of 4 are valid array sizes. 0-27 = 7 chunks x 4
# groups. 0-15 = 4 chunks (~350 sites, ~1,050-wide groups -- past the crossover, so
# more GPU-hours but fewer waves); 0-55 = 14 smaller, cheaper-to-lose chunks.
#
# Each task writes its own chunk_<chunk>_group_<group>.csv (no write contention).
# Merge afterward:
#   (head -1 outputs/gpu_scenario_sweep/chunk_0_group_0.csv; \
#    tail -n +2 -q outputs/gpu_scenario_sweep/chunk_*_group_*.csv) \
#     > outputs/gpu_scenario_sweep/full_sweep.csv
#
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_array.sh
# Smoke-test first with sbatch_gpu_sweep_smoke.sh.
#SBATCH --job-name=sawh-gpu-scenarios
# 6 h against a ~1-2 h expected task: a task that dies writes NOTHING (rows are only
# appended once a group's whole year finishes), so headroom is cheaper than a retry.
#SBATCH --time=06:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-27%8
#SBATCH --output=gpu_sweep/logs/scenarios_%A_%a.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_scenario_sweep

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

STEP=3.0
TASK_ID="${SLURM_ARRAY_TASK_ID}"
NUM_TASKS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

# Group count, total land sites, and this task's scenario names, from the one place
# they are defined (site_sweep.scenario_groups). grid_land_points() prints a one-time
# diagnostic on first call, so `tail -1` keeps only the line we asked for.
INFO=$(python3 -c "
import sys
sys.path.insert(0, 'src')
from solar_lumped.site_sweep import scenario_groups
from solar_lumped.weather import grid_land_points
groups = list(scenario_groups().values())
print(len(groups), len(grid_land_points(${STEP})), ' '.join(groups[${TASK_ID} % len(groups)]))
" | tail -1)
read -r NUM_GROUPS TOTAL_SITES SCENARIOS <<< "${INFO}"

if (( NUM_TASKS % NUM_GROUPS != 0 )); then
  echo "ERROR: --array size ${NUM_TASKS} is not a multiple of the ${NUM_GROUPS} scenario groups."
  echo "Set --array to a multiple of ${NUM_GROUPS} (e.g. 0-$(( NUM_GROUPS * 7 - 1 ))%8)."
  exit 1
fi

NUM_CHUNKS=$(( NUM_TASKS / NUM_GROUPS ))
GROUP_ID=$(( TASK_ID % NUM_GROUPS ))
CHUNK_ID=$(( TASK_ID / NUM_GROUPS ))
CHUNK_SIZE=$(( (TOTAL_SITES + NUM_CHUNKS - 1) / NUM_CHUNKS ))   # ceil division
START=$(( CHUNK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))
if (( END > TOTAL_SITES )); then END=${TOTAL_SITES}; fi

echo "Task ${TASK_ID}: chunk ${CHUNK_ID}/${NUM_CHUNKS} sites [${START}, ${END}) x group ${GROUP_ID}/${NUM_GROUPS}: ${SCENARIOS}"

if (( START >= END )); then
  echo "Empty site range for this chunk (more chunks than sites) -- nothing to do."
  exit 0
fi

python3 -c "import jax; print('jax.devices():', jax.devices())"

python3 gpu_sweep/run_gpu_sweep.py \
  --site-range "${START}" "${END}" --step "${STEP}" \
  --scenarios ${SCENARIOS} \
  --output-csv "outputs/gpu_scenario_sweep/chunk_${CHUNK_ID}_group_${GROUP_ID}.csv" \
  --resume
