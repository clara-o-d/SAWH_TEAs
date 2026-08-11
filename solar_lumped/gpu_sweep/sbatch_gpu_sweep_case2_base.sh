#!/bin/bash
# Case 2 global sweep, variant 1 of 3: BASE / default physics.
#   condenser: full Eq. 2 ODE      c_w floor: hydrate (n * c_s)
# This is the configuration every other run in the repo has been using; it is the
# reference the other two variants are differenced against. All other parameters at
# their defaults (1,405-site 3-deg land grid, full thickness/fin/gap combo grid).
#
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_case2_base.sh
# Smoke-test first with sbatch_gpu_sweep_smoke_case2.sh.
#SBATCH --job-name=sawh-c2-base
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-39%8
#SBATCH --output=gpu_sweep/logs/case2_base_%A_%a.out

# Written out even though they are the defaults: the CSV records what it was told, and
# a variant sweep whose distinguishing flags are implicit is one rename from ambiguous.
OUT_DIR=outputs/gpu_grid_sweep_case2_base
RUN_FLAGS=(--cw-floor hydrate)

source gpu_sweep/_case2_sweep_body.sh
