#!/bin/bash
# Full 1,405-site x 8-scenario GPU sweep. One array task = one site chunk x one
# scenario group. The task table is site_sweep.array_tasks() -- 74 tasks, RAGGED:
# each group is chunked at its own width (GROUP_SITES_PER_CHUNK) because the groups
# do not cost the same per instance-day.
#
#   tasks  0-7   group 0  finite_g/ode      200 sites (600 instances)  ~1.5 h
#   tasks  8-36  group 1  instant/ode        50 sites (100 instances)
#   tasks 37-44  group 2  finite_g/ambient  200 sites (400 instances)  ~1.3 h
#   tasks 45-73  group 3  instant/ambient    50 sites  (50 instances)
#
# Why the instant groups get 4x narrower chunks: under vmap, diffrax steps the whole
# batch until EVERY instance has finished its day, so a group's cost tracks its worst
# site's adaptive step count. Finite-g is uniform enough that width is nearly free
# (measured: 20x the width for 1.5x the time). Instant equilibrium's 1e6-scaled g has
# a heavy tail -- a 402-instance batch measured >90 s/day, ~7x finite-g, and would
# overrun any sane walltime. Confirm the narrow-chunk cost with
# sbatch_gpu_sweep_probe.sh before submitting the instant ranges at full scale.
#
# Submit all 74:
#   sbatch gpu_sweep/sbatch_gpu_sweep_array.sh
# Or just some groups (--array on the command line overrides the header):
#   sbatch --array=8-36,45-73%8 gpu_sweep/sbatch_gpu_sweep_array.sh
#
# Each task writes outputs/gpu_scenario_sweep/chunk_<start>-<end>_group_<gid>.csv.
# Merge, de-duplicating in case runs with different chunk boundaries were mixed:
#   python3 -c "
#   import glob, pandas as pd
#   f = glob.glob('outputs/gpu_scenario_sweep/*_group_*.csv')
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
#SBATCH --array=0-73%8
#SBATCH --output=gpu_sweep/logs/scenarios_%A_%a.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_scenario_sweep

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

STEP=3.0
TASK_ID="${SLURM_ARRAY_TASK_ID}"

# This task's group and site range, from the one place the table is defined.
# grid_land_points() prints a one-time diagnostic, so `tail -1` keeps our line only.
INFO=$(python3 -c "
import sys
sys.path.insert(0, 'src')
from solar_lumped.site_sweep import array_tasks
from solar_lumped.weather import grid_land_points
tasks = array_tasks(len(grid_land_points(${STEP})))
i = ${TASK_ID}
if i < len(tasks):
    gid, start, end, names = tasks[i]
    print(len(tasks), gid, start, end, ' '.join(names))
else:
    print(len(tasks), 'OUT_OF_RANGE')
" | tail -1)
read -r TOTAL_TASKS GROUP_ID START END SCENARIOS <<< "${INFO}"

if [ "${GROUP_ID}" = "OUT_OF_RANGE" ]; then
  echo "ERROR: task ${TASK_ID} is past the task table (${TOTAL_TASKS} tasks: 0-$(( TOTAL_TASKS - 1 )))."
  echo "Fix --array, or the grid would be silently under-covered."
  exit 1
fi

echo "Task ${TASK_ID}/${TOTAL_TASKS}: group ${GROUP_ID}, sites [${START}, ${END}): ${SCENARIOS}"

python3 -c "import jax; print('jax.devices():', jax.devices())"

python3 gpu_sweep/run_gpu_sweep.py \
  --site-range "${START}" "${END}" --step "${STEP}" \
  --scenarios ${SCENARIOS} \
  --output-csv "outputs/gpu_scenario_sweep/chunk_${START}-${END}_group_${GROUP_ID}.csv" \
  --resume
