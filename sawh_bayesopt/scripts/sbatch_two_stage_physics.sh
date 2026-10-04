#!/bin/bash
# Physics campaign for the two-stage pipeline, one chunk of ~700 instance-years per array
# task (the width at which an A100 saturates). A year is ~366 sequential day-steps at
# ~20 s each on an A100-40GB (smoke run 46425285): twice the fixed-schedule sweeps'
# ~10 s, because random +-4 h seal/open shifts pad both phases to nearly a full day.
# So a task is ~2.1 h whatever the width; a chunk killed mid-year loses the whole year.
#
#   sbatch --array=0-22%8 scripts/sbatch_two_stage_physics.sh simulate
#   sbatch --array=0-5%6  scripts/sbatch_two_stage_physics.sh active --round 0 \
#          --tilt-mode fixed --schedule-mode persistence
#
# `select` prints how many simulate chunks the campaign has -- size --array to it. Chunks
# whose output exists are skipped, so resubmitting the same array resumes it.
#
# Submit from the package root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt).
#SBATCH --job-name=sawh-two-stage-physics
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/two_stage_physics_%A_%a.out

set -euo pipefail
mkdir -p logs
ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True
RUN_DIR="${RUN_DIR:-outputs/two_stage/main}"

STAGE="$1"; shift
python3 -c "import jax; print('jax.devices():', jax.devices())"
python3 scripts/run_two_stage.py --run-dir "${RUN_DIR}" \
  "${STAGE}" --chunk-index "${SLURM_ARRAY_TASK_ID}" "$@"
