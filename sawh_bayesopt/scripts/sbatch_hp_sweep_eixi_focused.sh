#!/bin/bash
# Targeted follow-up to hp_sweep_1 (the full 27-combo ei_xi x stall_rel_tol x
# n_init sweep): hp_sweep_1 found (a) stall_rel_tol and ei_xi had
# essentially no effect within their tested ranges, while (b) 33-67% of
# every combo's differential_evolution EI-proposal calls hit maxiter without
# declaring success -- meaning that near-invariance was likely an artifact
# of an under-converged inner optimizer, not a real property of the
# acquisition function. acquisition.py's propose_next/propose_batch have
# since had maxiter/popsize raised and tol loosened (see that file's
# docstring) -- validated locally to drop frac_de_hit_maxiter to 0 on a tiny
# smoke case. This reruns *just* the ei_xi axis, fixed at n_init=42 (the best
# performer in hp_sweep_1) and stall_rel_tol=0.005 (the original default,
# shown not to matter), and adds ei_xi=0.01 -- the *original* baseline
# (never actually in hp_sweep_1's grid, but the run every one of hp_sweep_1's
# 27 combos underperformed, per outputs/runs/case2_local_run) -- so this
# table gives a real, apples-to-apples answer: with a properly-converging
# DE, does raising ei_xi actually help, hurt, or do nothing?
#
# 4 combinations, cheap enough for --n-workers 2 across 2 GPUs (1 combo per
# GPU at a time, no oversubscription needed) well within the time limit below.
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt):
#   sbatch scripts/sbatch_hp_sweep_eixi_focused.sh
#SBATCH --job-name=sawh-hp-sweep-eixi-focused
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/hp_sweep_eixi_focused_%j.out

set -euo pipefail
mkdir -p logs

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True

python3 -c "import jax; print('jax.devices():', jax.devices())"
nvidia-smi

python3 scripts/hp_sweep.py \
  --sweep-id hp_sweep_eixi_focused --resume \
  --ei-xi-values 0.01,0.02,0.05,0.1 --stall-rel-tol-values 0.005 --n-init-values 42 \
  --n-workers 2 --gpu-ids 0,1 \
  --weather-cache-dir ../solar_lumped/.weather_cache

python3 scripts/plot_hp_sweep.py --sweep-dir outputs/runs/hp_sweep_eixi_focused
