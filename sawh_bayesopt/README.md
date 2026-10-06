# sawh-bayesopt

Global design and operation optimization of the solar-driven PAM-LiCl hydrogel
SAWH system (Wilson et al. 2025 / Díaz-Marín et al. 2024), built on
`solar_lumped`'s physics and its JAX fast path (`solar_lumped/gpu_sweep`).

There are two optimizers:

1. **Two-stage surrogate pipeline** (`climate.py`, `daily_surrogate.py`,
   `two_stage.py`, driven by `scripts/run_two_stage.py`). This is the main
   pipeline and produces the maps.
   - *Design* (hydrogel thickness, vapor gap, salt loading) is chosen once per
     cell.
   - *Control* (seal/open times on a 15-minute grid, plus tilt when it varies)
     is re-chosen every day by exact enumeration.
   - A daily surrogate `g(design, control, state, day features) -> water, end
     state, P(swelling cap)` connects the two. It is trained on a fixed physics
     campaign (~5k–50k instance-years) over a few hundred to ~2,000
     representative climate cells.
   - Tilt is handled at three levels (fixed, seasonal, daily), and schedules
     under four information regimes (hindsight, persistence/day-ahead,
     climatological, constant).
2. **Per-site true-physics BO** (`evaluator.py`, `bayesopt.py`, and
   `solar_lumped/gpu_sweep/run_bayesopt_sweep.py` across many sites). This is
   the validation reference.
   - GP + constrained EI directly on full-year JAX physics LCOW, one site at a
     time.
   - It covers the same design box plus tilt and one annual schedule, so its
     optimum is the like-for-like benchmark for the pipeline's
     (fixed tilt, constant schedule) mode.

`docs/design_notes.md` explains both. It covers why the pipeline is split in
two, why held-out validation is by whole cell, the enumeration/BO split, and
what each validation measures.

## Scope

- LiCl + hydrogel only. The simple JAX path hardcodes LiCl's isotherm.
- Physics sees T, RH and GHI only: `h_amb` is fixed, and there is no
  sky-temperature radiation. Features follow the physics.
- Financial parameters (`LCOEconomicParams`) are fixed scenario inputs, applied
  *after* the surrogate, so maps regenerate when they change.
- `insulation_gap_m` and `fin_area_ratio` are pinned at solar_lumped's
  defaults (`design_space.SIMPLE_FIXED`). The reference BO's `--complex` mode
  frees them along with 7 further dimensions.
- One weather year (2024) from the Open-Meteo requests-cache, about 14.7k
  sites.

## Install

`solar_lumped` must already be installed (editable, from this same
`SAWH_TEAs` checkout). Then:

```bash
pip install -e .
pip install "jax[cuda12]"   # on a GPU node (e.g. Sherlock)
```

The two-stage pipeline also needs `equinox`, which is already in
`solar_lumped/.venv_gpu`.

## Run the two-stage pipeline

Every stage writes under `--run-dir`. Chunked stages are resumable, and the
`scripts/sbatch_two_stage_*.sh` wrappers split them into Slurm arrays.

```bash
R="--run-dir outputs/two_stage/main"
python scripts/run_two_stage.py $R features                        # scan the cache (CPU, once)
python scripts/run_two_stage.py $R validate-features               # pick the retained PC count
python scripts/run_two_stage.py $R select --n-components 12        # cells, split, campaign
sbatch --array=0-N%8 scripts/sbatch_two_stage_physics.sh simulate  # physics campaign
python scripts/run_two_stage.py $R fit                             # surrogate + holdout_report.json
sbatch --array=0 scripts/sbatch_two_stage_optimize.sh anchors fixed persistence
sbatch --array=0-5 scripts/sbatch_two_stage_physics.sh active --round 0 \
    --tilt-mode fixed --schedule-mode persistence                  # then re-fit, 1-2 rounds
sbatch --array=0-29%8 scripts/sbatch_two_stage_optimize.sh all fixed persistence
python scripts/run_two_stage.py $R maps --tilt-mode fixed --schedule-mode persistence
python scripts/run_two_stage.py $R validate-bo --emit-sites      # login node: held-out sites
python ../solar_lumped/gpu_sweep/warm_weather_cache.py --sites-file outputs/two_stage/main/bo_sites.txt
sbatch --array=0-4 scripts/sbatch_two_stage_validate_bo.sh bo     # per-site true-physics BO
sbatch --time=16:00:00 scripts/sbatch_two_stage_validate_bo.sh score   # $/m3 gap vs per-site BO
```

Outputs, all in the run directory:
- `holdout_report.json`: error on never-seen climates, broken out by regime,
  plus whether the surrogate ranks each cell's designs, and responds to thickness
  and the sealed window, the way the physics does.
- `featval_summary.csv`: real vs reconstructed-year yields.
- `maps_<tilt>_<schedule>.csv`: one row per cell with the design, yield,
  LCOW, ensemble spread and an `ood` flag.
- `validate_bo_summary.csv`: the optimality gap in $/m3, by mode and regime.

## Run the per-site reference BO

```bash
python scripts/run_bayesopt.py --n-init 24 --n-total 50            # one site
python ../solar_lumped/gpu_sweep/run_bayesopt_sweep.py --lat-lon -23.65 -70.40 --output-dir out/
```

Outputs are written to `outputs/runs/<run_id>/`:
- `cache.jsonl`: every evaluated design; the run resumes from it.
- `gp_state.joblib`
- `convergence.png`
- `report.json`

## Tests

```bash
solar_lumped/.venv_gpu/bin/python -m pytest solar_lumped/tests sawh_bayesopt/tests -q   # from SAWH_TEAs/
```

Run them with the GPU venv. Under the default interpreter, the JAX parity tests
and `test_two_stage.py` skip, and the run still reports green.
