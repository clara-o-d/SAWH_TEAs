#!/bin/bash
# GPU sweep smoke test -- a handful of real sites, full combo grid, to validate
# run_gpu_sweep.py on actual hardware before scaling to the full 1,405-site grid.
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_smoke.sh
#
# Mirrors docs/sherlock_param_sweep.tex's CPU smoke test structure, plus --gres=gpu:1.
#SBATCH --job-name=sawh-gpu-sweep-smoke
#SBATCH --time=02:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=gpu_sweep/logs/smoke_%j.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_grid_sweep

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

python3 -c "import jax; print('jax.devices():', jax.devices())"

# 10 real sites spanning different latitudes/day-lengths (not tiled/synthetic
# data -- this is the point of the smoke test: does the per-site combo x month
# batching architecture hold up on real, varied inputs on the actual GPU).
python3 gpu_sweep/run_gpu_sweep.py \
  --num-sites 10 \
  --output-csv outputs/gpu_grid_sweep/smoke_10sites.csv \
  --resume
