#!/bin/bash
# Full complex-fidelity BayesOpt run: 13 design dims (BASE_VAR_ORDER + A1/B1/B2/B3/B4/B8,
# see design_space.py::COMPLEX_VAR_ORDER), both validated field sites, all 365 real days.
#
# One GPU, not two: this is a single sequential BayesOpt loop -- each round's EI
# proposal depends on the previous round's fit, so there is nothing to split across
# devices the way hp_sweep.py's independent combinations are.
#
# --time=10:00:00 rather than the hp_sweep scripts' 4h. Both fidelities rebuild each
# site's daily profiles per design point (the seal/open offsets and POA tilt live inside
# the profile, plus B4's condenser air here -- evaluator.py::_profiles_for_design), and
# complex mode also runs solar_lumped's CPU ODE path, so per-round cost is well above
# simple mode's. The budget here is 130 evaluations rather than 50 because a 13-dim
# anisotropic Matern needs more points than 5 dims does.
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
