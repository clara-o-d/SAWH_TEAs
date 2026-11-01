#!/bin/bash
# Any sawh_bayesopt module stage on CPU cores only (selection, tree surrogates, BO tuning,
# merges, figures). Arguments go straight to `python -m`:
#
#   sbatch --array=0-8 scripts/sbatch_cpu.sh sawh_bayesopt.train fit --family random_forest --config-index TASK
#
# A literal TASK argument becomes the array index. RUN_DIR defaults to outputs/main.
# Submit from the package root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt).
#SBATCH --job-name=sawh-cpu
#SBATCH --time=12:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
mkdir -p logs
ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True
MODULE="$1"; shift
python3 -m "${MODULE}" --run-dir "${RUN_DIR:-outputs/main}" "${@//TASK/${SLURM_ARRAY_TASK_ID:-TASK}}"
