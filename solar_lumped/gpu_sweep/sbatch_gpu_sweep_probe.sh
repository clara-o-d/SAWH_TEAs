#!/bin/bash
# Timing probe: how does the instant-sorption group's per-day cost scale with batch
# width? That one curve sets GROUP_SITES_PER_CHUNK in site_sweep.py, and getting it
# wrong costs either walltime-killed tasks (too wide) or wasted GPU-hours (too narrow).
#
# Runs the instant/ode group at three widths, 20 days each with a 1-round warm-up --
# enough for a stable s/day, ~1 h total instead of the ~10 h a real chunk would take.
# The rows it writes are 20-day means, NOT annual: they go to a probe/ directory and
# n_periods records 20. Do not merge them into a real sweep.
#
#   sbatch gpu_sweep/sbatch_gpu_sweep_probe.sh
#   grep -E "instance|s/day" gpu_sweep/logs/probe_<jobid>.out
#
# Read the result as: s/day per instance. If 100 instances costs ~1/4 of what 400 did
# (measured >90 s/day at 402), the tail is width-driven and narrow chunks pay off. If
# it barely drops, the cost is per-instance and the instant groups are simply ~100
# GPU-hours -- at which point tightening the solver for that path is the better lever
# than re-chunking.
#SBATCH --job-name=sawh-probe
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

for SITES in 10 50 200; do
  echo "=== instant/ode, ${SITES} site(s) x 2 scenarios ==="
  python3 gpu_sweep/run_gpu_sweep.py \
    --site-range 0 "${SITES}" --step 3.0 \
    --scenarios improved_instant_g optical_limits_instant_g \
    --max-days 20 --max-rounds 1 --progress-every 5 \
    --output-csv "outputs/gpu_scenario_sweep/probe/instant_${SITES}sites.csv"
done

echo "=== finite_g/ode at 200 sites, same settings, as the baseline to divide by ==="
python3 gpu_sweep/run_gpu_sweep.py \
  --site-range 0 200 --step 3.0 \
  --scenarios wilson improved optical_limits \
  --max-days 20 --max-rounds 1 --progress-every 5 \
  --output-csv "outputs/gpu_scenario_sweep/probe/finite_200sites.csv"
