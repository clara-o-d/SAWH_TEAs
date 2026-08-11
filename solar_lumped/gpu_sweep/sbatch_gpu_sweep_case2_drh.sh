#!/bin/bash
# Case 2 global sweep, variant 2 of 3: DELIQUESCENCE-RH gel-water floor.
#   condenser: full Eq. 2 ODE      c_w floor: drh (equilibrium c_w at the DRH)
# Differs from _base.sh in exactly one thing: desorption stops where the LiCl brine
# saturates (~2.78 H2O per LiCl) instead of at the monohydrate (1.0 H2O per LiCl), so
# the gel is held ~2.8x wetter at the dry end. The conservative bound -- it can only
# lower yield -- and the difference against _base.sh is how much of the yield depends
# on drying past brine saturation, where the Conde/ZSR activity correlations are
# describing a slurry rather than a solution.
#
# Submit from the repo root (/home/groups/cdiazm/SAWH_TEAs/solar_lumped):
#   sbatch gpu_sweep/sbatch_gpu_sweep_case2_drh.sh
#SBATCH --job-name=sawh-c2-drh
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-39%8
#SBATCH --output=gpu_sweep/logs/case2_drh_%A_%a.out

OUT_DIR=outputs/gpu_grid_sweep_case2_drh
RUN_FLAGS=(--cw-floor drh)

source gpu_sweep/_case2_sweep_body.sh
