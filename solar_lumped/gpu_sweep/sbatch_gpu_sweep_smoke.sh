#!/bin/bash
# Scenario-sweep smoke test -- 10 real sites x all 8 scenarios, to validate
# run_gpu_sweep.py on actual hardware before scaling to the full 1,405-site grid.
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_smoke.sh
#
# Worth re-running whenever the scenario list changes: the optical-limits
# scenarios put eps=0 through the radiative solve, a genuinely different
# numerical regime from the Wilson baseline (see docs/gpu_sweep_handoff.md).
#SBATCH --job-name=sawh-gpu-scenarios-smoke
#SBATCH --time=02:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=gpu_sweep/logs/smoke_%j.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_scenario_sweep

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

python3 -c "import jax; print('jax.devices():', jax.devices())"

# 10 real sites spanning different latitudes/day-lengths (not tiled/synthetic data --
# the point is whether the site x scenario batching holds up on real, varied inputs).
python3 gpu_sweep/run_gpu_sweep.py \
  --num-sites 10 \
  --output-csv outputs/gpu_scenario_sweep/smoke_10sites.csv \
  --resume
