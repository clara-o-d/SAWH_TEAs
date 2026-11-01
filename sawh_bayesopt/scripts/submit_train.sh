#!/bin/bash
# sawh_bayesopt.train as one dependency chain. Submit from the package root
# (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt):
#
#   bash scripts/submit_train.sh                   # features -> select -> ... -> tune-bo
#   SKIP_FEATURES=1 bash scripts/submit_train.sh   # features.npz already in $RUN_DIR
#   SKIP_SELECT=1 bash scripts/submit_train.sh     # selection done too: start at simulate
#
# Wall time is dominated by the physics campaign: 5,000 years in 8 tasks, all at once. A
# full 700-year task measured 3.0-4+ h on the first run (2.2 h for the 100-year last task):
# the per-day weather profiles are built serially on the CPU, so time grows with width.
# A task killed mid-year loses the whole year. Every stage skips work whose output exists,
# so resubmitting resumes -- finished physics chunks exit at once.
set -euo pipefail
export RUN_DIR="${RUN_DIR:-outputs/main}"
mkdir -p logs "${RUN_DIR}"
after() { echo "--dependency=afterok:$(IFS=:; echo "$*")"; }
GPU=scripts/sbatch_gpu.sh
CPU=scripts/sbatch_cpu.sh
T=sawh_bayesopt.train

SIM_DEP=()
if [ -z "${SKIP_SELECT:-}" ]; then
  if [ -z "${SKIP_FEATURES:-}" ]; then
    FEAT=$(sbatch --parsable --job-name=features --time=06:00:00 $CPU $T features)
    SEL=$(sbatch --parsable $(after "$FEAT") --job-name=select --time=02:00:00 $CPU $T select)
  else
    SEL=$(sbatch --parsable --job-name=select --time=02:00:00 $CPU $T select)
  fi
  SIM_DEP=($(after "$SEL"))
fi
# 1,000 cells x 5 thicknesses = 5,000 years at 700 per task -> tasks 0-7 (select prints it).
SIM=$(sbatch --parsable ${SIM_DEP[@]+"${SIM_DEP[@]}"} --job-name=simulate --time=08:00:00 --array=0-7%8 \
  $GPU $T simulate --chunk-index TASK)
# One task per hyperparameter config (surrogates.GRIDS): 9 ridge alphas run in one job.
RIDGE=$(sbatch --parsable $(after "$SIM") --job-name=fit-ridge --time=02:00:00 $CPU $T fit --family ridge)
RF=$(sbatch --parsable $(after "$SIM") --job-name=fit-rf --time=12:00:00 --array=0-8 \
  $CPU $T fit --family random_forest --config-index TASK)
GB=$(sbatch --parsable $(after "$SIM") --job-name=fit-gb --time=12:00:00 --array=0-8 \
  $CPU $T fit --family gradient_boosting --config-index TASK)
MLP=$(sbatch --parsable $(after "$SIM") --job-name=fit-mlp --time=02:00:00 --array=0-23%8 \
  $GPU $T fit --family mlp --config-index TASK)
CMP=$(sbatch --parsable $(after "$RIDGE" "$RF" "$GB" "$MLP") --job-name=compare --time=01:00:00 $CPU $T compare)
GRID=$(sbatch --parsable $(after "$CMP") --job-name=tune-grid --time=08:00:00 $GPU $T tune-grid)
TUNE=$(sbatch --parsable $(after "$GRID") --job-name=tune-bo --time=12:00:00 $CPU $T tune-bo)
echo "submitted: select ${SEL:-skipped}, simulate $SIM, fits $RIDGE $RF $GB $MLP, compare $CMP, tune $GRID $TUNE"
