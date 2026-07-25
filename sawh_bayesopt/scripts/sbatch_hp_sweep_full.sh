#!/bin/bash
# hp_sweep.py full grid: ei_xi x stall_rel_tol x n_init, 3 values each (27
# combinations), real two-site/monthly evaluations, 8 workers split across 2
# GPUs (4 combinations sharing each GPU's memory at a time -- see
# hp_sweep.py's module docstring for why XLA_PYTHON_CLIENT_MEM_FRACTION and
# max_tasks_per_child=1 make that safe). Run scripts/sbatch_hp_sweep_smoke.sh
# first and confirm it produces sane output before submitting this.
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt):
#   sbatch scripts/sbatch_hp_sweep_full.sh
#SBATCH --job-name=sawh-hp-sweep-full
#SBATCH --time=12:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/hp_sweep_full_%j.out

set -euo pipefail
mkdir -p logs

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

python3 -c "import jax; print('jax.devices():', jax.devices())"
nvidia-smi

python3 scripts/hp_sweep.py \
  --sweep-id hp_sweep_1 \
  --n-workers 8 --gpu-ids 0,1 \
  --weather-cache-dir ../solar_lumped/.weather_cache

python3 scripts/plot_hp_sweep.py --sweep-dir outputs/runs/hp_sweep_1
