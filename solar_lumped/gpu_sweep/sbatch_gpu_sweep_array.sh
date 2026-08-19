#!/bin/bash
# Full 1,405-site x 8-scenario GPU sweep. One array task = one site chunk x one scenario
# group, from site_sweep.array_tasks() -- 32 tasks, all four groups on the 3-degree grid:
#
#   tasks  0-7   group 0  finite_g/ode      200 sites x 3 scenarios = 600 instances  ~1.5 h
#   tasks  8-15  group 1  instant/ode       200 sites x 2 scenarios = 400 instances
#   tasks 16-23  group 2  finite_g/ambient  200 sites x 2 scenarios = 400 instances  ~1.3 h
#   tasks 24-31  group 3  instant/ambient   200 sites x 1 scenario  = 200 instances
#
# The instant groups used to need a coarser grid and 4x narrower chunks, when g -> infinity
# was a stiff penalty costing ~50x per instance-day. They are now imposed as the
# equilibrium constraint and cost slightly LESS than finite g (their absorption half needs
# no ODE at all), so all eight scenarios are back at full 3-degree resolution and pair
# site-by-site.
#
# Submit everything:
#   sbatch gpu_sweep/sbatch_gpu_sweep_array.sh
# Or one group (--array on the command line overrides the header) -- e.g. just the instant
# groups, if the finite-g rows are already in hand:
#   sbatch --array=8-15,24-31%8 gpu_sweep/sbatch_gpu_sweep_array.sh
#
# Each task writes outputs/gpu_scenario_sweep/step<deg>_sites<start>-<end>_group<gid>.csv.
# Merge, de-duplicating in case runs with different chunk boundaries were mixed:
#   python3 -c "
#   import glob, pandas as pd
#   f = glob.glob('outputs/gpu_scenario_sweep/*_group*.csv')
#   d = pd.concat(map(pd.read_csv, f)).drop_duplicates(['lat','lon','scenario'])
#   d.to_csv('outputs/gpu_scenario_sweep/full_sweep.csv', index=False)
#   print(len(d), 'rows from', len(f), 'files')"
#SBATCH --job-name=sawh-gpu-scenarios
# 6 h against ~1.5 h expected: a task that dies writes NOTHING (rows land only when a
# group's whole year completes), so headroom is cheaper than a retry.
#SBATCH --time=06:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-31%8
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
