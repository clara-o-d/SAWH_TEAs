#!/bin/bash
# Run any one non-chunked stage of the two-stage pipeline as a GPU batch job -- the
# stages too long for sh_dev's 2 h cap (validate-features, fit) or that want a GPU.
# Arguments after the script name go straight to run_two_stage.py after --run-dir:
#
#   sbatch scripts/sbatch_two_stage_run.sh validate-features --n-cells 300
#   sbatch scripts/sbatch_two_stage_run.sh fit
#   sbatch --time=00:30:00 scripts/sbatch_two_stage_run.sh select --n-components 12
#
# Chunked stages have their own wrappers (sbatch_two_stage_physics.sh,
# sbatch_two_stage_optimize.sh). RUN_DIR defaults to outputs/two_stage/main.
# Submit from the package root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt).
#SBATCH --job-name=sawh-two-stage
#SBATCH --time=08:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/two_stage_%x_%j.out

set -euo pipefail
mkdir -p logs
ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True
RUN_DIR="${RUN_DIR:-outputs/two_stage/main}"

echo "stage: $*  run dir: ${RUN_DIR}"
python3 -c "import jax; print('jax.devices():', jax.devices())"
python3 scripts/run_two_stage.py --run-dir "${RUN_DIR}" "$@"
