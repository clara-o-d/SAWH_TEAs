#!/bin/bash
# Two-stage pipeline smoke test: every stage end to end at toy scale (~16 cells x 4
# designs = ~64 physics years), to check the CUDA setup, the per-day GPU timing and the
# stage plumbing before committing ~23 GPU-hours to the full campaign.
#
#   sbatch scripts/sbatch_two_stage_smoke.sh
#
# Stages whose output already exists are skipped, so resubmitting after a timeout picks
# up where it stopped. Delete outputs/two_stage/smoke to start over. What to look at:
#   logs/two_stage_smoke_<jobid>.out         the "s/day" lines from simulate
#   outputs/two_stage/smoke/holdout_report.json
#   outputs/two_stage/smoke/opt/fixed_persistence/anchors.csv
#
# Submit from the package root (/home/groups/cdiazm/SAWH_TEAs/sawh_bayesopt).
#SBATCH --job-name=sawh-two-stage-smoke
#SBATCH --time=06:00:00
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/two_stage_smoke_%j.out

set -euo pipefail
mkdir -p logs
ml python/3.12.1 uv
source .venv_gpu/bin/activate
export PYTHONUNBUFFERED=True
RUN_DIR="${RUN_DIR:-outputs/two_stage/smoke}"
run() { python3 scripts/run_two_stage.py --run-dir "${RUN_DIR}" "$@"; }

python3 -c "import jax, equinox; print('jax.devices():', jax.devices(), 'equinox', equinox.__version__)"
nvidia-smi

# features is CPU work (~25 min for the full cache) and the slowest step here; the
# result is reusable by the main run (copy features.npz + pca.npz across).
[ -f "${RUN_DIR}/features.npz" ] || { echo "== features"; run features; }

[ -f "${RUN_DIR}/selection.csv" ] || { echo "== select"; run select --n-components 12 --n-clusters 20 \
  --quota high_altitude=2,hyper_arid=2,monsoonal=2,coastal_humid=2 --designs-per-cell 4; }

echo "== simulate"  # skips itself if the chunk exists
run simulate --chunk-index 0

[ -f "${RUN_DIR}/model/hyper.json" ] || { echo "== fit"; run fit --steps 4000; }

[ -f "${RUN_DIR}/opt/fixed_persistence/anchors.csv" ] || { echo "== optimize anchors"; \
  run optimize --cells anchors --tilt-mode fixed --schedule-mode persistence --n-total 8; }

echo "== done"
python3 -c "
import json; r = json.load(open('${RUN_DIR}/holdout_report.json'))
print('annual held-out error by regime:', {k: round(v['median_rel_err'], 3) for k, v in r['holdout']['annual'].items() if v['median_rel_err'] is not None})
print('thickness sign agreement:', r['monotonicity_heldout']['hydrogel_thickness']['all'])"
