#!/bin/bash
# sawh_bayesopt.validate: BO with and without the physics surrogate at held-out test cells.
# Needs training done (model/, bo_config.json); the score stage also wants the merged global
# maps (submit_global.sh), and skips scenarios not merged yet. Submit from the package root:
#
#   bash scripts/submit_validate.sh
#
# physics-bo runs 40 evaluations per cell in batches of 8 -> 5 rounds of one physics year.
# Measured on the training campaign: ~40 s per day-step, so ~4-4.5 h per year with the
# Aitken warm-up, ~22 h for the search. It caches every evaluation, so if it times out,
# resubmit just that stage:
#   sbatch --time=48:00:00 scripts/sbatch_gpu.sh sawh_bayesopt.validate physics-bo
set -euo pipefail
export RUN_DIR="${RUN_DIR:-outputs/main}"
mkdir -p logs
V=sawh_bayesopt.validate
CELLS=$(sbatch --parsable --job-name=v-cells --time=00:20:00 --cpus-per-task=2 --mem=16G scripts/sbatch_cpu.sh $V cells)
SBO=$(sbatch --parsable --dependency=afterok:"$CELLS" --job-name=v-surrogate-bo --time=04:00:00 scripts/sbatch_gpu.sh $V surrogate-bo)
PBO=$(sbatch --parsable --dependency=afterok:"$CELLS" --job-name=v-physics-bo --time=48:00:00 scripts/sbatch_gpu.sh $V physics-bo)
SCORE=$(sbatch --parsable --dependency=afterok:"$SBO" --job-name=v-score --time=10:00:00 scripts/sbatch_gpu.sh $V score)
REP=$(sbatch --parsable --dependency=afterok:"$PBO":"$SCORE" --job-name=v-report --time=00:30:00 scripts/sbatch_cpu.sh $V report)
sbatch --dependency=afterok:"$REP" --job-name=plots --time=01:00:00 scripts/sbatch_cpu.sh sawh_bayesopt.plots --only validation
echo "submitted: cells $CELLS, surrogate-bo $SBO, physics-bo $PBO, score $SCORE, report $REP"
