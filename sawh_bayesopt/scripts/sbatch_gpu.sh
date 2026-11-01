#!/bin/bash
# Any sawh_bayesopt module stage on one GPU. Arguments go straight to `python -m`:
#
#   sbatch scripts/sbatch_gpu.sh sawh_bayesopt.train fit --family mlp --config-index 3
#
# A literal TASK argument becomes the array index, so the same wrapper serves arrays:
#
#   sbatch --array=0-10 scripts/sbatch_gpu.sh sawh_bayesopt.train simulate --chunk-index TASK
#
# RUN_DIR defaults to outputs/main.
# Submit from the package root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt).
#SBATCH --job-name=sawh-gpu
#SBATCH --time=08:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
mkdir -p logs
ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True
python3 -c "import jax; print('jax.devices():', jax.devices())"
MODULE="$1"; shift
python3 -m "${MODULE}" --run-dir "${RUN_DIR:-outputs/main}" "${@//TASK/${SLURM_ARRAY_TASK_ID:-TASK}}"
