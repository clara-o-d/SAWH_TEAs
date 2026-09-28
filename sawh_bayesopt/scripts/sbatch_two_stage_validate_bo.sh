#!/bin/bash
# Optimality gap on held-out cells: per-site true-physics BO over the same design box
# (run_bayesopt_sweep.py, lockstep groups), then the two-stage designs + schedules scored
# by true physics against it. Needs `fit` done; anchors for each VALIDATE_MODES mode are
# used as warm starts when present.
#
#   sbatch scripts/sbatch_two_stage_validate_bo.sh
#
# The BO half is the expensive one (~1 h per round of calls, ~10 rounds); it writes
# ${RUN_DIR}/bo_reference/summary.csv and resumes with --resume if resubmitted.
#SBATCH --job-name=sawh-two-stage-validate-bo
#SBATCH --time=24:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/two_stage_validate_bo_%j.out

set -euo pipefail
mkdir -p logs
ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True
RUN_DIR="${RUN_DIR:-outputs/two_stage/main}"

python3 scripts/run_two_stage.py --run-dir "${RUN_DIR}" validate-bo --emit-sites
LAT_LON_ARGS=$(awk '{printf "--lat-lon %s %s ", $1, $2}' "${RUN_DIR}/bo_sites.txt")
# shellcheck disable=SC2086
python3 ../solar_lumped/gpu_sweep/run_bayesopt_sweep.py ${LAT_LON_ARGS} \
  --sites-per-group 64 --output-dir "${RUN_DIR}/bo_reference" --resume
python3 scripts/run_two_stage.py --run-dir "${RUN_DIR}" validate-bo \
  --bo-summary "${RUN_DIR}/bo_reference/summary.csv"
