#!/bin/bash
# Can the instant-equilibrium groups be made affordable? Measured on the serc A100 they
# cost ~1.0 s per instance-day and stay there as the batch widens (20 instances:
# 23.6 s/day; 100: ~106 s/day), against ~0.023 s/instance-day for the finite-g groups at
# 603 wide. That puts the two instant groups at ~430 GPU-hours for the full grid, and it
# is not a chunking problem: under vmap diffrax steps the whole batch until every
# instance is done, so the cost is the batch's WORST adaptive step count, and by ~20
# instances that worst case is already in every batch. Narrower chunks split the bill
# without shrinking it.
#
# What is left is the stiffness itself: instant equilibrium multiplies g by
# _INSTANT_EQUILIBRIUM_G_SCALE = 1e6 (parameters.xlsx calls it "numerical, not
# physical"), which is what forces the tiny explicit steps. probe_instant_g_scale.py
# runs the same 20 sites x 20 days at 1e4/1e5/1e6 and reports yield drift against cost,
# so the factor is chosen by measurement.
#
#   sbatch gpu_sweep/sbatch_gpu_sweep_probe.sh
#   grep -E "s/day|drift" gpu_sweep/logs/probe_<jobid>.out
#
# If a smaller factor holds the yield to well under the 1e-4 ODE tolerance while costing
# much less, change the workbook value (both backends read that one cell) and re-run
# tests/test_cpu_jax_parity.py before sweeping. If nothing below 1e6 is safe, the instant
# groups need a stiff solver (diffrax Kvaerno) or an algebraic isotherm constraint
# instead of a 1e6 penalty -- a real physics change, not a knob.
#SBATCH --job-name=sawh-g-scale-probe
#SBATCH --time=02:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=gpu_sweep/logs/probe_%j.out

set -euo pipefail
mkdir -p gpu_sweep/logs outputs/gpu_scenario_sweep/probe

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

python3 -c "import jax; print('jax.devices():', jax.devices())"
python3 gpu_sweep/probe_instant_g_scale.py
