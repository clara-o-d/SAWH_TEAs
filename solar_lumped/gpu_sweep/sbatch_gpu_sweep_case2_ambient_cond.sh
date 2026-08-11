#!/bin/bash
# Case 2 global sweep, variant 3 of 3: AMBIENT-PINNED condenser.
#   condenser: T_cond == T_amb    c_w floor: hydrate (n * c_s)
# Differs from _base.sh in exactly one thing: Eq. 2's condenser ODE is dropped and the
# condenser is held at ambient -- the infinite-cooling-capacity limit, not a buildable
# design. The difference against _base.sh is the yield penalty imposed by the real
# condenser warming up under its own latent load.
#
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_case2_ambient_cond.sh
#SBATCH --job-name=sawh-c2-ambcond
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-39%8
#SBATCH --output=gpu_sweep/logs/case2_ambcond_%A_%a.out

OUT_DIR=outputs/gpu_grid_sweep_case2_ambient_cond
RUN_FLAGS=(--cw-floor hydrate --condenser-ambient)

source gpu_sweep/_case2_sweep_body.sh
