#!/bin/bash
# hp_sweep.py smoke test -- 2 tiny combinations (single site, single-day
# resolution, ~12 evaluations each) run across 2 workers sharing 1 GPU, to
# validate the whole sweep+diagnostics+plotting pipeline on real Sherlock
# hardware before committing to the full 27-combination sweep.
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt):
#   sbatch scripts/sbatch_hp_sweep_smoke.sh
#
# Mirrors solar_lumped/gpu_sweep/sbatch_gpu_sweep_smoke.sh's structure and
# docs/SHERLOCK_VERIFY_RUNBOOK.md's venv/module conventions.
#SBATCH --job-name=sawh-hp-sweep-smoke
#SBATCH --time=00:30:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/hp_sweep_smoke_%j.out

set -euo pipefail
mkdir -p logs

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

python3 -c "import jax; print('jax.devices():', jax.devices())"
nvidia-smi

python3 scripts/hp_sweep.py \
  --sweep-id hp_sweep_smoke --resume \
  --ei-xi-values 0.02,0.1 --stall-rel-tol-values 0.005 --n-init-values 8 \
  --bo-budget 4 --batch-size 2 --sites cambridge --resolution single \
  --n-workers 2 --gpu-ids 0 \
  --weather-cache-dir ../solar_lumped/.weather_cache

python3 scripts/plot_hp_sweep.py --sweep-dir outputs/runs/hp_sweep_smoke
