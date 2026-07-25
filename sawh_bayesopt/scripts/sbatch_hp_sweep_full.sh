#!/bin/bash
# hp_sweep.py full grid: ei_xi x stall_rel_tol x n_init, 3 values each (27
# combinations), real two-site/monthly evaluations, 4 workers split across 2
# GPUs (2 combinations sharing each GPU's memory at a time -- see
# hp_sweep.py's module docstring for why XLA_PYTHON_CLIENT_MEM_FRACTION and
# max_tasks_per_child=1 make that safe). Deliberately only 2-way, not 4-way,
# sharing per GPU: each evaluate_batch() call costs ~160s almost regardless
# of batch size (JAX/XLA per-call dispatch overhead dominates, not compute --
# see sherlock_gpu_run_1/history.csv), and Sherlock's serc GPUs likely don't
# have NVIDIA MPS enabled, so concurrent CUDA contexts mostly time-slice
# rather than truly parallelize -- heavier oversubscription risks contention
# actively hurting throughput rather than just not helping it.
#
# Per-combo cost if it runs its full budget without stalling (~12
# evaluate_batch calls: 1 init + ceil(bo_budget/batch_size) BO rounds + 1
# baseline + 1 verify) is ~30 min; 27 combos / 2 real GPUs is ~13-14
# combos/GPU even with zero benefit from the 2-way sharing above, i.e.
# worst-case ~7-8 hours -- under this job's --time, but not by a wide
# margin, especially since testing *higher* ei_xi values specifically means
# testing settings likely to stall less than sherlock_gpu_run_1 did (i.e.
# closer to the full-budget cost, not the early-stall cost that run hit).
#
# Because of that margin, this job passes --resume: if it does hit the time
# limit, just resubmit this exact script with the same --sweep-id
# (hp_sweep_1) -- combinations whose run_dir already has a complete
# report.json + gp_regression_report.json are skipped instead of re-run, so
# a resubmission picks up where the last one left off rather than starting
# over. Check your partition's actual max time (`sinfo -p serc -o "%l"` or
# `scontrol show partition serc`) and raise --time here if it allows more.
#
# Run scripts/sbatch_hp_sweep_smoke.sh first and confirm it produces sane
# output before submitting this.
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt):
#   sbatch scripts/sbatch_hp_sweep_full.sh
#SBATCH --job-name=sawh-hp-sweep-full
#SBATCH --time=12:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/hp_sweep_full_%j.out

set -euo pipefail
mkdir -p logs

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

python3 -c "import jax; print('jax.devices():', jax.devices())"
nvidia-smi

python3 scripts/hp_sweep.py \
  --sweep-id hp_sweep_1 --resume \
  --n-workers 4 --gpu-ids 0,1 \
  --weather-cache-dir ../solar_lumped/.weather_cache

python3 scripts/plot_hp_sweep.py --sweep-dir outputs/runs/hp_sweep_1
