#!/bin/bash
# Case 2 smoke test -- 10 real sites, full combo grid, eps_abs_ir=0.05/
# eps_glass_ir=0.95. Run this and sanity-check the output before submitting
# sbatch_gpu_sweep_array_case2.sh for the full 1,405-site grid.
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_smoke_case2.sh
#SBATCH --job-name=sawh-gpu-sweep-smoke-case2
#SBATCH --time=02:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=gpu_sweep/logs/smoke_case2_%j.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_grid_sweep_case2

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

python3 -c "import jax; print('jax.devices():', jax.devices())"

python3 gpu_sweep/run_gpu_sweep.py \
  --num-sites 10 \
  --eps-abs-ir 0.05 --eps-glass-ir 0.95 \
  --output-csv outputs/gpu_grid_sweep_case2/smoke_10sites.csv \
  --resume
