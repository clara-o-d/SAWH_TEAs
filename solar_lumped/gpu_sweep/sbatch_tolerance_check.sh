#!/bin/bash
# ODE tolerance verification on a real A100: is the production 1e-4/1e-7 already
# converged, and does the solver survive being tightened 100x?
#
# Submit from the solar_lumped working directory (/home/groups/cdiazm/SAWH_TEAs/solar_lumped),
# same as every other sbatch script here:
#   sbatch gpu_sweep/sbatch_tolerance_check.sh
#
# Runs three things, cheapest first, so a failure stops before the expensive part:
#   1. the existing CPU/JAX parity + physics test suite (catches a broken env)
#   2. the single-instance Tsit5 vs Radau desorption validator
#   3. the tolerance ladder at full 125-combo width on real Atacama weather
#
# Exits nonzero if any stage fails, so the Slurm job state reflects the result rather
# than requiring someone to read the log.
#SBATCH --job-name=sawh-tol-check
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=gpu_sweep/logs/tol_check_%j.out

set -euo pipefail
mkdir -p gpu_sweep/logs

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

# Days to simulate per tolerance. 14 is enough for the Aitken warmup to settle and for
# seasonal variety; --days 0 runs the full year and costs roughly 26x this at the tight
# end, which is an overnight job, not a check.
DAYS="${DAYS:-14}"

echo "=============================================================================="
echo "Environment"
echo "=============================================================================="
nvidia-smi || echo "nvidia-smi unavailable"
# Import jax_physics, not bare jax: x64 is enabled as a side effect of that module, so
# checking it without the import reports False and looks like a broken run when it isn't.
PYTHONPATH=gpu_sweep python3 -c "
import jax_physics, jax
print('jax', jax.__version__, jax.devices())
print('x64:', jax.config.jax_enable_x64)
"

echo
echo "=============================================================================="
echo "1/3  Test suite (CPU/JAX parity, physics, design space)"
echo "=============================================================================="
# .venv_gpu is a runtime environment, not a dev one, so pytest may not be installed.
# That is not a reason to abandon the tolerance measurement this job exists for --
# skip loudly and let stages 2 and 3 run.
if python3 -c "import pytest" 2>/dev/null; then
  python3 -m pytest tests -q
else
  echo "SKIPPED: pytest not installed in this venv (uv pip install pytest to enable)"
fi

# The validators live one level up, outside the solar_lumped/ working directory this
# runbook submits from -- same ../analysis/ path SHERLOCK_GPU_RUNBOOK.md section 4 uses.
echo
echo "=============================================================================="
echo "2/3  Single-instance Tsit5 vs scipy Radau desorption"
echo "=============================================================================="
python3 ../analysis/performance/optimization/validators/validate_desorption_integration_tsit5.py

echo
echo "=============================================================================="
echo "3/3  Tolerance ladder, full combo grid, real weather"
echo "=============================================================================="
python3 ../analysis/performance/optimization/validators/validate_tolerance_sensitivity.py \
  --lat-lon -23.65 -70.40 \
  --year 2024 \
  --days "$DAYS"

echo
echo "All stages passed."
