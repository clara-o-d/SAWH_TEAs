#!/bin/bash
# Full complex-fidelity BayesOpt run: 13 design dims (VAR_ORDER + A1/B1/B2/B3/B4/B8,
# see design_space.py::COMPLEX_VAR_ORDER) on the JAX fast path, both validated field
# sites, all 365 real days.
#
# One GPU, not two: this is a single sequential BayesOpt loop -- each round's EI
# proposal depends on the previous round's fit, so there is nothing to split across
# devices the way hp_sweep.py's independent combinations are.
#
# --time=10:00:00 rather than the hp_sweep scripts' 4h. Complex mode rebuilds each
# site's daily profiles per design point (A1's seal/open offsets, B4's condenser air,
# and POA tilt all live inside the profile -- evaluator.py::_profiles_for_design), so
# per-round cost is well above simple mode's ~160s/call, and the budget here is
# 130 evaluations rather than 50 because a 13-dim anisotropic Matern needs more
# points than 6 dims did.
#
# If it hits the wall anyway, resubmit this exact script: cache.jsonl is fsync'd
# after every completed design and every loop seed derives from cfg.seed +
# len(history), so the rerun replays the identical design sequence and everything
# already evaluated is a cache hit. Only the batch that was mid-flight is re-paid.
#
# Submit from the package root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt):
#   sbatch scripts/sbatch_complex_run.sh
#SBATCH --job-name=sawh-complex-run
#SBATCH --time=10:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/complex_run_%j.out

set -euo pipefail
mkdir -p logs

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

python3 -c "import jax; print('jax.devices():', jax.devices())"
nvidia-smi

python3 scripts/run_bayesopt.py \
  --complex --backend jax \
  --n-init 60 --n-total 130 --batch-size 6 \
  --weather-cache-dir ../solar_lumped/.weather_cache \
  --run-id complex_gpu_run_1
