#!/bin/bash
# Per-cell design BO on the surrogate for one tilt x schedule mode, split into contiguous
# cell ranges across array tasks. Run the anchors (training cells) first, as a plain job --
# every other cell warm-starts from its climate neighbours among them:
#
#   sbatch --array=0 scripts/sbatch_two_stage_optimize.sh anchors fixed persistence
#   sbatch --array=0-29%8 scripts/sbatch_two_stage_optimize.sh all fixed persistence
#   python3 scripts/run_two_stage.py --run-dir outputs/two_stage/main maps \
#          --tilt-mode fixed --schedule-mode persistence
#
# The proposals are CPU work (a GP + small DE per cell per round, joblib over cores), the
# inner enumeration GPU work, hence the extra cores.
#SBATCH --job-name=sawh-two-stage-optimize
#SBATCH --time=12:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/two_stage_optimize_%A_%a.out

set -euo pipefail
mkdir -p logs
ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True
RUN_DIR="${RUN_DIR:-outputs/two_stage/main}"
CELLS="$1"; TILT="$2"; SCHEDULE="$3"

if [ "${CELLS}" = "anchors" ]; then
  python3 scripts/run_two_stage.py --run-dir "${RUN_DIR}" optimize --cells anchors \
    --tilt-mode "${TILT}" --schedule-mode "${SCHEDULE}"
  exit 0
fi

N_CELLS=$(python3 -c "import numpy as np; print(len(np.load('${RUN_DIR}/features.npz')['lat']))")
NUM_TASKS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
CHUNK=$(( (N_CELLS + NUM_TASKS - 1) / NUM_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
echo "task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS}: cells [${START}, ${END}) of ${N_CELLS}"
python3 scripts/run_two_stage.py --run-dir "${RUN_DIR}" optimize --cells all \
  --cell-range "${START}" "${END}" --tilt-mode "${TILT}" --schedule-mode "${SCHEDULE}"
