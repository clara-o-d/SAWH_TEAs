#!/bin/bash
# Baseline design and operation at every cell, for the gain-over-baseline map (Fig. 14).
# One schedule search per cell at the baseline design (vs the optimize stage's 12-24), so
# roughly a twentieth of an optimize run per mode. Split into contiguous cell ranges:
#
#   sbatch --array=0-7 scripts/sbatch_two_stage_baseline.sh fixed persistence
#   sbatch --array=0-7 scripts/sbatch_two_stage_baseline.sh daily hindsight
#   sbatch --array=0-7 scripts/sbatch_two_stage_baseline.sh fixed persistence --base-only
#   python3 scripts/run_two_stage.py --run-dir outputs/two_stage/main baseline --merge \
#          --tilt-mode fixed --schedule-mode persistence
#
# True-physics check of the surrogate's baseline at 50 held-out cells (one chunk of
# physics years, ~2 h on an A100; independent of the mode):
#
#   sbatch --array=0 --time=04:00:00 scripts/sbatch_two_stage_baseline.sh check 50
#
# Parts whose output exists are skipped, so resubmitting the same array resumes it.
# Submit from the package root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt).
#SBATCH --job-name=sawh-two-stage-baseline
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/two_stage_baseline_%A_%a.out

set -euo pipefail
mkdir -p logs
ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True
RUN_DIR="${RUN_DIR:-outputs/two_stage/main}"

if [ "$1" = "check" ]; then
  python3 scripts/run_two_stage.py --run-dir "${RUN_DIR}" baseline --physics-check "$2"
  exit 0
fi
TILT="$1"; SCHEDULE="$2"; shift 2

N_CELLS=$(python3 -c "import numpy as np; print(len(np.load('${RUN_DIR}/features.npz')['lat']))")
NUM_TASKS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
CHUNK=$(( (N_CELLS + NUM_TASKS - 1) / NUM_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
echo "task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS}: cells [${START}, ${END}) of ${N_CELLS}"
python3 scripts/run_two_stage.py --run-dir "${RUN_DIR}" baseline \
  --cell-range "${START}" "${END}" --tilt-mode "${TILT}" --schedule-mode "${SCHEDULE}" "$@"
