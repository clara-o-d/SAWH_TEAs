#!/bin/bash
# Case 3 ("optical material limits") smoke test -- 10 real sites, eps_abs=1.0/
# tau_glass=1.0 fixed (not swept lists), eps_abs_ir=0.0/eps_glass_ir=0.0. Run
# this and sanity-check the output before submitting
# sbatch_gpu_sweep_array_case3.sh for the full 1,405-site grid -- eps=0 is a
# genuinely different numerical regime than Case 1/2, worth confirming here
# first (see docs/gpu_sweep_handoff.md).
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_smoke_case3.sh
#SBATCH --job-name=sawh-gpu-sweep-smoke-case3
#SBATCH --time=02:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=gpu_sweep/logs/smoke_case3_%j.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_grid_sweep_case3

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

python3 -c "import jax; print('jax.devices():', jax.devices())"

python3 gpu_sweep/run_gpu_sweep.py \
  --num-sites 10 \
  --eps-abs 1.0 --tau-glass 1.0 --eps-abs-ir 0.0 --eps-glass-ir 0.0 \
  --output-csv outputs/gpu_grid_sweep_case3/smoke_10sites.csv \
  --resume
