# sawh-bayesopt

Surrogate-assisted (Gaussian process + Expected Improvement) global optimization
of solar-driven SAWH device design, built on top of `solar_lumped`'s forward
ODE simulation of the Wilson et al. 2025 / Díaz-Marín et al. 2024 PAM-LiCl
hydrogel device.

`solar_lumped` itself has no optimizer — only brute-force grid/OAT sweeps, and
its CPU ODE solver is too slow (~380s/site per evaluation) for a dense grid or
evolutionary search to find a global optimum. This package instead runs
Bayesian optimization (EGO: a small Latin-hypercube initial design, then
adaptive batched Expected-Improvement infill), evaluated against
`solar_lumped/gpu_sweep`'s JAX/diffrax fast path (see `evaluator.py`) rather
than the CPU path directly — `gpu_sweep/FINDINGS.md` shows the two agree to
<0.03% and the JAX path is ~8x faster even single-threaded on a CPU with no
GPU, and considerably more once run on an actual GPU. Every round batches all
of that round's (design, site, month) instances into one `jax.vmap`-compiled
call. Designs are evaluated at both of the paper's experimentally
field-validated sites (Cambridge, MA and the Atacama Desert, Chile) and
combined via the mean of the two sites' LCOW.

## Scope (v1)

- Design variables: `hydrogel_thickness_m`, `vapor_gap_m`, `insulation_gap_m`,
  `fin_area_ratio`, `tilt_deg`, `salt_to_polymer_ratio`. No
  `condenser_thickness_m`: `economics/lcow.py` charges a flat condenser BOM
  cost regardless of thickness (a free cost-side lever with no downside), and
  the JAX fast path hardcodes condenser thermal mass at Table S3's constant
  rather than taking it as a per-instance input — it isn't a real physics
  knob on the path this package now evaluates against.
- LiCl + hydrogel sorbent only (no salt-choice or MOF categorical dimensions).
- Financial parameters (`LCOEconomicParams`) are fixed scenario inputs, not
  decision variables.
- Desorption enthalpy fixed at Wilson Table S3's constant (no Díaz-Marín
  Fig. 4 composition-dependent enthalpy model — that's future work).

## Install

`solar_lumped` must already be installed (editable, from this same `SAWH_TEAs`
checkout — verify with `python -c "import solar_lumped; print(solar_lumped.__file__)"`
before installing this package, since a stale/different `solar_lumped` install
would silently ground every optimization result in the wrong physics):

```bash
pip install -e .
```

`jax` and `diffrax` are regular dependencies (see `pyproject.toml`) and install
along with everything else. That gets you a working CPU backend. For a real
GPU node (e.g. Sherlock), reinstall JAX with its CUDA extra afterward:

```bash
pip install "jax[cuda12]"
```

## Run

```bash
python scripts/run_bayesopt.py --n-init 24 --n-total 50
```

Outputs land in `outputs/runs/<run_id>/`: `cache.jsonl` (every evaluated design,
resumable), `gp_state.joblib`, `convergence.png`, `report.json`.

## Known caveats (see `docs/design_notes.md`)

- `combined_lcow` comes from `solar_lumped/gpu_sweep`'s JAX fast path, not
  `solar_lumped`'s CPU `ode_system.py` directly. `gpu_sweep/FINDINGS.md`
  documents <0.03% worst-case disagreement between the two paths, so this
  isn't expected to be a meaningfully different physics model, but it's a
  different code path than the one `solar_lumped`'s own sweep scripts use.
