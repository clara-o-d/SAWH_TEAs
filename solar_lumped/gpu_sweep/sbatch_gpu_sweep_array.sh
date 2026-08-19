#!/bin/bash
# 8-scenario GPU sweep. One array task = one site chunk x one scenario group, from
# site_sweep.array_tasks() -- 61 tasks, RAGGED in both grid and width because the groups
# do not cost remotely the same (see site_sweep.GroupRun):
#
#   tasks  0-7   group 0  finite_g/ode      3 deg, 1405 sites, 200/chunk (600 inst) ~1.5 h
#   tasks  8-37  group 1  instant/ode       6 deg,  360 sites,  12/chunk  (24 inst)
#   tasks 38-45  group 2  finite_g/ambient  3 deg, 1405 sites, 200/chunk (400 inst) ~1.3 h
#   tasks 46-60  group 3  instant/ambient   6 deg,  360 sites,  24/chunk  (24 inst)
#
# The instant groups run the COARSER 6-degree grid: even with the step-cap fix they cost
# ~50x per instance-day (stiff c_w relaxation -> stability-limited explicit steps) and,
# unlike finite-g, do not amortize with batch width, so full 3-degree coverage would be
# ~1,000 GPU-hours. 6 deg is a strict subset of 3 deg, so every instant row still pairs
# with finite-g rows at the same site -- tests/test_sweep_task_table.py pins that.
#
# Submit everything:
#   sbatch gpu_sweep/sbatch_gpu_sweep_array.sh
# Or one group at a time (--array on the command line overrides the header) -- e.g. just
# the instant groups, if the finite-g rows are already in hand:
#   sbatch --array=8-37,46-60%8 gpu_sweep/sbatch_gpu_sweep_array.sh
#
# Each task writes outputs/gpu_scenario_sweep/step<deg>_sites<start>-<end>_group<gid>.csv.
# Merge, de-duplicating in case runs with different chunk boundaries were mixed:
#   python3 -c "
#   import glob, pandas as pd
#   f = glob.glob('outputs/gpu_scenario_sweep/*_group_*.csv')
#   d = pd.concat(map(pd.read_csv, f)).drop_duplicates(['lat','lon','scenario'])
#   d.to_csv('outputs/gpu_scenario_sweep/full_sweep.csv', index=False)
#   print(len(d), 'rows from', len(f), 'files')"
#SBATCH --job-name=sawh-gpu-scenarios
# 12 h serves the whole array: the finite-g tasks need ~1.5 h, but the instant tasks'
# per-instance-day cost is extrapolated, not measured post-fix, and a task that dies
# writes NOTHING (rows land only when a group's whole year completes). If serc refuses
# 12 h, lower it and shrink GROUP_RUNS[...].sites_per_chunk in the same proportion.
#SBATCH --time=12:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-60%8
#SBATCH --output=gpu_sweep/logs/scenarios_%A_%a.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_scenario_sweep

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

TASK_ID="${SLURM_ARRAY_TASK_ID}"

# This task's group and site range, from the one place the table is defined.
# grid_land_points() prints a one-time diagnostic, so `tail -1` keeps our line only.
INFO=$(python3 -c "
import sys
from functools import lru_cache
sys.path.insert(0, 'src')
from solar_lumped.site_sweep import array_tasks
from solar_lumped.weather import grid_land_points
count = lru_cache(None)(lambda step: len(grid_land_points(step)))
tasks = array_tasks(count)
i = ${TASK_ID}
if i < len(tasks):
    gid, step, start, end, names = tasks[i]
    print(len(tasks), gid, step, start, end, ' '.join(names))
else:
    print(len(tasks), 'OUT_OF_RANGE')
" | tail -1)
read -r TOTAL_TASKS GROUP_ID STEP START END SCENARIOS <<< "${INFO}"

if [ "${GROUP_ID}" = "OUT_OF_RANGE" ]; then
  echo "ERROR: task ${TASK_ID} is past the task table (${TOTAL_TASKS} tasks: 0-$(( TOTAL_TASKS - 1 )))."
  echo "Fix --array, or the grid would be silently under-covered."
  exit 1
fi

echo "Task ${TASK_ID}/${TOTAL_TASKS}: group ${GROUP_ID}, ${STEP} deg grid, sites [${START}, ${END}): ${SCENARIOS}"

python3 -c "import jax; print('jax.devices():', jax.devices())"

python3 gpu_sweep/run_gpu_sweep.py \
  --site-range "${START}" "${END}" --step "${STEP}" \
  --scenarios ${SCENARIOS} \
  --output-csv "outputs/gpu_scenario_sweep/step${STEP}_sites${START}-${END}_group${GROUP_ID}.csv" \
  --resume
