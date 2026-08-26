#!/bin/bash
# Global BayesOpt campaign: every land point on a 12-degree grid (87 sites), all 8
# site_sweep.SCENARIOS, day_stride 10. One array task per scenario.
#
# Submit from /home/groups/cdiazm/SAWH_TEAs/solar_lumped:
#   sbatch gpu_sweep/sbatch_bayesopt_global_12deg.sh
#
# WHY 8 TASKS AND NOT MORE. Sites inside a task run in lockstep groups: every site keeps
# its own GP/history/cache, but each round's designs across a whole group go into a single
# batched evaluation. A call costs ~the same at any width up to GPU saturation (a year is
# ~366 *sequential* day-steps; measured 60.1 min for 1 design vs 68.2 min for 8 on an
# A100), so a task's cost is set by how many CALLS it makes, not how many sites it holds --
# which is why the site axis is the wrong one to parallelize across tasks: splitting it
# would multiply total GPU-hours while barely improving wall clock. It is capped only by
# batch width, not by task count (see SITES_PER_GROUP).
#
# The scenario axis is the one worth parallelizing: instant_equilibrium and
# condenser_ambient each select a *code path* in the JAX step, so they cannot share a
# compiled batch anyway (see site_sweep.scenario_groups). 8 scenarios -> 8 tasks.
#
# SIZING -- MEASURED, job 40830060 on serc (8 tasks, 87 sites, stride 10, 3 groups of 29):
#   completed tasks ran 7:42, 7:58, 8:39, 8:32; four others passed 8:57 still running
#   MaxRSS 10.9-12.4 GB against --mem=64G, so width 696 was never memory-bound
# Walltime is 20 h on that evidence, not the 12 h this was first submitted with -- four
# tasks came within ~3 h of truncation.
#
# That job predates SITES_PER_GROUP=29: its progress lines read n_target 4350 = 50 x 87, so
# all 87 sites ran as ONE group -- a 2088-instance init call, 3x past the ~700 saturation
# figure, which completed fine at 11-12 GB. Width was not the problem, and the reason to
# group is checkpointing, not memory.
#
# Where the 8.5 h actually went: ~7 calls, so ~73 min/call against the ~7 min a 37-day walk
# at ~11 s/day predicts. Physics cannot account for that. The sequential per-site EI
# proposals can -- batch_size 6 means Kriging-Believer runs 6 SEQUENTIAL differential_
# evolution optimizations per site per round at maxiter=1000/popsize=40, and 87 sites x ~4
# rounds x ~60 s is ~5.8 h on its own. See the ponytail note at bayesopt.py's propose_batch
# call, which predicted exactly this ("parallelize over sites with joblib if a task ever
# carries hundreds"). A bigger GPU allocation would not have helped; joblib over sites, or
# a lower --de-maxiter, is where the hours are.
#
# Corollary: grouping is ~cost-neutral, not 3x. Acquisition cost is per-site and invariant
# to how sites are grouped, and 3 groups triples the call count while cutting each call's
# width by 3x. So 29 buys its checkpointing for free.
#
# The GPU is not the whole story: see --cpus-per-task below.
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
#SBATCH --time=20:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
# 16 not 4: with stride 10 the GPU year walk drops to ~7 min/call while the CPU-side
# acquisition does not shrink at all -- across a task that is still 87 sites x ~6 rounds of
# differential_evolution at maxiter=1000/popsize=40 plus a GP refit each, and bayesopt.py's
# own note flags those proposals as sequential across sites. That is now comparable to the
# physics, so the old --cpus-per-task=4 (sized when physics was ~100% of cost) would leave
# the GPU idle waiting on scipy. Drop back to 4 only if the queue wait becomes the problem.
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --output=gpu_sweep/logs/bo_global_12deg_%A_%a.out

set -euo pipefail

STEP=12.0
DAY_STRIDE=10
# 29, not 96. bayesopt._evaluate flattens every (site, design) pair into ONE batched call,
# so the widest call is the LHS init at sites_per_group x n_init -- 96 x 24 would be a
# 2088-instance vmap against the ~700-instance saturation measured in site_sweep.py, which
# is what --sites-per-group exists to lower on OOM. 29 x 24 = 696 fits just under it, and
# the infill rounds are a comfortable 29 x 6 = 174.
#
# Splitting 87 sites into 3 groups costs 3 sets of rounds instead of 1 (~24 calls, not 8),
# which is the price of staying inside a width that has actually been measured. It buys
# back two things: summary.csv is written after each group rather than only at the very end,
# and --resume can pick up at a group boundary instead of restarting the scenario.
SITES_PER_GROUP=29
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
# LOOP PARAMS, all four changed on evidence from job 40830060's 435 finished sites:
#
#   --n-total 100 (was 50)      Most sites never converged: stopped_reason was "budget" for
#                               63/87 sites in `improved` and 58/87 in `wilson`, so 50 was
#                               binding rather than sufficient. Median improvement over
#                               baseline was only 11.8%.
#   --stall-rounds 5            10 sites finished WORSE than baseline, every one of them
#   --stall-rel-tol 0.002       "stalled" at exactly n_evals=42. With the GP overconfident
#     (was 3 / 0.005)           (standardized_residual_std 1.57-1.76 where 1.0 is
#                               calibrated), EI collapses early and the old 3-round/0.5%
#                               trigger quit before beating the starting design.
#   --de-maxiter 300 (was 1000) frac_de_hit_maxiter was 0.000 at every one of 435 sites, and
#                               a smoke run converged at nit/maxiter=0.14 (~140 iters). 1000
#                               was never reached, so this is free -- and it is the lever
#                               that matters, since ~70% of wall clock was sequential
#                               per-site DE, not physics.
#
# Net cost: ~13 infill rounds instead of ~4 (more physics calls), against ~3x cheaper
# acquisition per round. Expect ~10-12 h/task rather than 8.5 -- still inside the 20 h
# walltime, but check task 0's first rounds before assuming the whole array fits.
python3 gpu_sweep/run_bayesopt_sweep.py \
  --step "${STEP}" --site-range 0 10000 \
  --day-stride "${DAY_STRIDE}" \
  --case "${CASE}" ${EXTRA_FLAGS} \
  --sites-per-group "${SITES_PER_GROUP}" \
  --n-init 24 --n-total 100 --batch-size 6 --seed 0 \
  --stall-rounds 5 --stall-rel-tol 0.002 \
  --de-maxiter 300 \
  --n-verify-neighbors 5 \
  --output-dir "${OUT_ROOT}/${SCENARIO}" \
  --resume

echo "Task ${SLURM_ARRAY_TASK_ID} (${SCENARIO}) done."
