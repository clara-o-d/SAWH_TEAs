#!/bin/bash
# Optimality gap on the held-out cells, in three steps (submit from the package root,
# /home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt):
#
# 0. LOGIN NODE (compute nodes have no outbound network): list the held-out cells and
#    fetch their weather into the cache. MISSING must read 0 before going on.
#      python3 scripts/run_two_stage.py --run-dir outputs/two_stage/main validate-bo --emit-sites
#      python3 ../solar_lumped/gpu_sweep/warm_weather_cache.py \
#        --sites-file outputs/two_stage/main/bo_sites.txt
#
# 1. Per-site true-physics BO over the same design box (run_bayesopt_sweep.py), split
#    across array tasks. A task's sites share one lockstep group, so a task costs ~12
#    full-year calls (~20 h) whatever its width; more tasks only cut wall time. Each site
#    keeps a cache.jsonl, so a timed-out task resumes when resubmitted -- alone, with the
#    original task count: NUM_TASKS=5 sbatch --array=1 scripts/sbatch_two_stage_validate_bo.sh bo
#      sbatch --array=0-4 scripts/sbatch_two_stage_validate_bo.sh bo
#
# 2. Merge the tasks' summaries and score the two-stage designs + schedules in true
#    physics against them, one physics chunk (~2.5 h) per mode. Extra arguments go to
#    validate-bo, e.g. --modes; fixed:constant is the like-for-like gap.
#      sbatch --time=08:00:00 scripts/sbatch_two_stage_validate_bo.sh score --modes fixed:constant,daily:hindsight
#
# Results: ${RUN_DIR}/validate_bo.csv and validate_bo_summary.csv.
#SBATCH --job-name=sawh-two-stage-validate-bo
#SBATCH --time=48:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/two_stage_validate_bo_%A_%a.out

set -euo pipefail
mkdir -p logs
ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True
RUN_DIR="${RUN_DIR:-outputs/two_stage/main}"
SITES="${RUN_DIR}/bo_sites.txt"

case "${1:-}" in
  bo)
    [ -f "${SITES}" ] || { echo "no ${SITES}: run step 0 on a login node first"; exit 1; }
    # NUM_TASKS must be the size of the ORIGINAL split, also when resubmitting one task
    # (NUM_TASKS=5 sbatch --array=1 ...): from a one-task array it would read 1, and the
    # split below would hand task 1 no sites at all.
    NUM_TASKS="${NUM_TASKS:-$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))}"
    # Task i takes every NUM_TASKS-th site, so tasks get a similar spread of climates.
    LAT_LON_ARGS=$(awk -v n="${NUM_TASKS}" -v i="${SLURM_ARRAY_TASK_ID}" \
      '(NR - 1) % n == i {printf "--lat-lon %s %s ", $1, $2}' "${SITES}")
    echo "task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS}: $(echo "${LAT_LON_ARGS}" | grep -o -- --lat-lon | wc -l) site(s)"
    # shellcheck disable=SC2086
    python3 ../solar_lumped/gpu_sweep/run_bayesopt_sweep.py ${LAT_LON_ARGS} \
      --sites-per-group 200 --output-dir "${RUN_DIR}/bo_reference/task_${SLURM_ARRAY_TASK_ID}" --resume
    ;;
  score)
    python3 -c "
import glob, pandas as pd
parts = sorted(glob.glob('${RUN_DIR}/bo_reference/task_*/summary.csv'))
df = pd.concat([pd.read_csv(p) for p in parts])
df.to_csv('${RUN_DIR}/bo_reference/summary.csv', index=False)
print(f'merged {len(parts)} task summaries: {len(df)} sites')"
    shift
    python3 scripts/run_two_stage.py --run-dir "${RUN_DIR}" validate-bo \
      --bo-summary "${RUN_DIR}/bo_reference/summary.csv" "$@"
    ;;
  *)
    echo "usage: sbatch [--array=0-4] $0 bo|score"; exit 1 ;;
esac
