#!/bin/bash
# sawh_bayesopt.run_global for all 12 scenarios after training (bo_config.json must exist).
# Submit from the package root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt):
#
#   bash scripts/submit_global.sh                # all scenarios, then merge and figures
#   N_CHUNKS=60 bash scripts/submit_global.sh    # smaller cell blocks per task
#
# Task k runs scenario k // N_CHUNKS (optimize.SCENARIOS order: three_per_day, daily,
# seasonal, each under hindsight, persistence, monthly, yearly) on cell block k % N_CHUNKS.
# Three-per-day tasks score ~10x the candidates of the others; parts that exist are skipped,
# so resubmitting the same array finishes whatever timed out.
set -euo pipefail
export RUN_DIR="${RUN_DIR:-outputs/main}"
N_CHUNKS="${N_CHUNKS:-30}"
mkdir -p logs
RUN=$(sbatch --parsable --job-name=global --time=12:00:00 --cpus-per-task=16 \
  --array=0-$(( 12 * N_CHUNKS - 1 ))%16 \
  scripts/sbatch_gpu.sh sawh_bayesopt.run_global run --task-index TASK --n-chunks "$N_CHUNKS")
MERGE=$(sbatch --parsable --dependency=afterok:"$RUN" --job-name=merge --time=01:00:00 \
  scripts/sbatch_cpu.sh sawh_bayesopt.run_global merge)
sbatch --dependency=afterok:"$MERGE" --job-name=plots --time=01:00:00 scripts/sbatch_cpu.sh sawh_bayesopt.plots
echo "submitted: run $RUN, merge $MERGE"
