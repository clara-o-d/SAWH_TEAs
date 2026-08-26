#!/bin/bash
# Global BayesOpt campaign: every land point on a 12-degree grid (87 sites), all 8
# site_sweep.SCENARIOS, day_stride 10. One array task per scenario.
#
# Submit from /home/groups/cdiazm/SAWH_TEAs/solar_lumped:
#   sbatch gpu_sweep/sbatch_bayesopt_global_12deg.sh
#
# WHY 8 TASKS AND NOT MORE. Sites inside a task run in ONE lockstep group: every site
# keeps its own GP/history/cache, but each round's designs across the whole group go into
# a single batched evaluation. A call costs ~the same at any width (a year is ~366
# *sequential* day-steps; measured 60.1 min for 1 design vs 68.2 min for 8 on an A100), so
# a task's cost is set by how many CALLS it makes, not how many sites it holds. Splitting
# the 87 sites across more tasks would multiply total GPU-hours while barely improving wall
# clock. The scenario axis is the only one worth parallelizing: instant_equilibrium and
# condenser_ambient each select a *code path* in the JAX step, so they cannot share a
# compiled batch anyway (see site_sweep.scenario_groups). 8 scenarios -> 8 tasks is the
# maximum useful parallelism here, and it is also the cheapest total.
#
# SIZING. At stride 10 a year walk is 37 days, not 366:
#   1 init + 5 infill (batch 6) + 1 verify + 1 baseline = ~8 calls x ~7 min = ~1 h GPU
# But see --cpus-per-task below: at stride 10 the physics stops dominating.
#
# STALE CACHE WARNING. case3 changed meaning when CASE_SOLAR_OPTICS landed: it now sets
# eps_abs=1.0/tau_glass=1.0 as site_sweep._LIMITS always did, where before it kept a real
# absorber behind real glass. design_vector_hash keys on the case NAME, so any pre-existing
# case3 cache.jsonl under --output-dir is stale-but-key-identical and --resume would mix two
# physics. Wipe ${OUT_ROOT} before the first submission of this campaign.
#
# Merge the per-scenario summaries once the whole array is done:
#   (head -1 outputs/bayesopt_global_12deg/wilson/summary.csv; \
#    tail -n +2 -q outputs/bayesopt_global_12deg/*/summary.csv) \
#     > outputs/bayesopt_global_12deg/global_summary.csv
#
#SBATCH --job-name=sawh-bo-global-12deg
#SBATCH --time=12:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
# 16 not 4: with stride 10 the GPU year walk drops to ~7 min/call while the CPU-side
# acquisition does not shrink at all -- 87 sites x ~6 rounds of differential_evolution at
# maxiter=1000/popsize=40, plus 87 GP fits per round. That is now comparable to the
# physics, so the old --cpus-per-task=4 (sized when physics was ~100% of cost) would leave
# the GPU idle waiting on scipy. Drop back to 4 only if the queue wait becomes the problem.
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --output=gpu_sweep/logs/bo_global_12deg_%A_%a.out

set -euo pipefail

STEP=12.0
DAY_STRIDE=10
SITES_PER_GROUP=96   # >= the 87 land points, so every scenario is a single lockstep group
OUT_ROOT="outputs/bayesopt_global_12deg"

mkdir -p gpu_sweep/logs "${OUT_ROOT}"

ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True
# scipy's differential_evolution is single-threaded per call but BLAS-parallel inside the
# GP predicts; leave it to the 16 cores rather than letting each of many threads oversubscribe.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# --- Resolve this task's scenario --------------------------------------------------------
# The scenario table is site_sweep.SCENARIOS, not a copy in bash: array index -> name -> the
# driver flags that reproduce it. The case must match all FOUR optics numbers (both IR and
# both solar-side); a scenario with no exact match aborts this task rather than running the
# wrong physics under the right name. test_every_scenario_is_reachable_through_the_case_flag
# is the same check at test time.
SPEC=$(python3 -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '../sawh_bayesopt/src')
from solar_lumped.site_sweep import SCENARIOS
from sawh_bayesopt.design_space import CASE_EPS_IR, CASE_SOLAR_OPTICS

name = list(SCENARIOS)[${SLURM_ARRAY_TASK_ID}]
sc = SCENARIOS[name]
case = next(
    (c for c in CASE_EPS_IR
     if CASE_EPS_IR[c] == (sc.eps_abs_ir, sc.eps_glass_ir)
     and CASE_SOLAR_OPTICS[c] == (sc.eps_abs, sc.tau_glass)),
    None,
)
if case is None:
    sys.exit(f'No case reproduces {name} optics '
             f'(eps_abs={sc.eps_abs}, tau_glass={sc.tau_glass}, '
             f'eps_abs_ir={sc.eps_abs_ir}, eps_glass_ir={sc.eps_glass_ir})')
flags = []
if sc.instant_equilibrium:
    flags.append('--instant-equilibrium')
if sc.condenser_ambient:
    flags.append('--condenser-ambient')
print(name, case, ' '.join(flags))
" | tail -1)
read -r SCENARIO CASE EXTRA_FLAGS <<< "${SPEC}"

echo "Task ${SLURM_ARRAY_TASK_ID}: scenario=${SCENARIO} case=${CASE} flags='${EXTRA_FLAGS:-none}'"

# --- Pre-flight ---------------------------------------------------------------------------
# The GPU check is the single most important line here: JAX silently falls back to CPU, which
# turns a 1 h task into a walltime kill.
python3 -c "
import jax, sys
d = jax.devices()
print('jax.devices():', d)
if d[0].platform != 'gpu':
    sys.exit('JAX sees no GPU -- refusing to run the campaign on CPU.')
"

# Compute nodes have no outbound network. Every site's weather must already be in the
# shared .weather_cache from a login-node warm (see the runbook step below), or the run
# fails site by site with an opaque fetch error hours in.
python3 -c "
import sys
sys.path.insert(0, 'src')
from solar_lumped.weather import grid_land_points
print('land points at ${STEP} deg:', len(grid_land_points(${STEP})))
" | tail -1

# --- Run ----------------------------------------------------------------------------------
# Diagnostics are per-site and on by default in the driver: history.csv, convergence.png,
# gp_state.joblib, diagnostics/de_diagnostics.json, diagnostics/gp_regression_report.json,
# gp_slices.png, report.json, plus cv_rmse / standardized residuals / msll /
# n_hyperparameter_warnings / DE convergence fractions folded into summary.csv. Verification
# re-evaluates each optimum and 5 perturbed neighbors against the true model, so GP
# artifacts show up as a verified row rather than a surrogate guess.
python3 gpu_sweep/run_bayesopt_sweep.py \
  --step "${STEP}" --site-range 0 10000 \
  --day-stride "${DAY_STRIDE}" \
  --case "${CASE}" ${EXTRA_FLAGS} \
  --sites-per-group "${SITES_PER_GROUP}" \
  --n-init 24 --n-total 50 --batch-size 6 --seed 0 \
  --n-verify-neighbors 5 \
  --output-dir "${OUT_ROOT}/${SCENARIO}" \
  --resume

echo "Task ${SLURM_ARRAY_TASK_ID} (${SCENARIO}) done."
