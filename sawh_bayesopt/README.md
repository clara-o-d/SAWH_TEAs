# sawh-bayesopt

Global design and operation of the solar PAM–LiCl hydrogel SAWH panel, on `solar_lumped`'s
physics (its JAX path, `solar_lumped.jax_daily_cycle`).

- **Design**, once per location: the hydrogel thickness H₀ (1–10 mm). The salt loading is
  held at the top of its range. The vapor gap is H₀ + 7 mm, so the cast gel always fits.
- **Operation**, every day: the seal and open offsets (15-minute grid), plus the tilt in
  each third of the sealed window.
- A **surrogate** of the daily physics map, (H₀, control, gel state, day weather) →
  (water, next state, P(swelling cap)), connects the two. The outer loop is a constrained
  Bayesian optimization over H₀. The inner loop enumerates every daily control on the
  surrogate.

## Modules (`python -m sawh_bayesopt.<module> <stage>`)

| Module | What it does |
|---|---|
| `train` | Featurize climates. Pick 750 space-filling locations (Kochenderfer & Wheeler alg. 16.8) plus 250 extra Köppen BWh/BWk desert cells, with a 70/15/15 split by cell. Run 5 thicknesses × 1 physics year each under random daily controls. Fit ridge, random forest, gradient boosting and MLP grids, then compare them on the validation cells, score them on the test cells and deploy one. Tune the BO kernel, acquisition and GP settings against grid-search truth. |
| `run_global` | The tuned BO at every cell, for 12 scenarios (tilt 3×/day, daily or seasonal × hindsight, persistence, monthly or yearly-mean weather). Saves LCOW, yield, thickness, tilt and the full daily schedule. |
| `validate` | BO with the surrogate against BO on true physics, on the same annual-schedule problem at held-out test cells. Also physics-scores every scenario's optimum. |
| `plots` | Every methodology and result figure, from whatever is on disk. |

Library modules: `climate` (features, sampling, Köppen, OOD), `simulate` (physics years),
`surrogates` (the four model families), and `optimize` (inner enumeration, constrained BO).

## Running on Sherlock

Submit from this directory, in order. Each script chains its stages with dependencies, and
every stage skips work whose output already exists.

```bash
bash scripts/submit_train.sh      # ~1 day; the physics campaign is 8 parallel ~2.1 h tasks
bash scripts/submit_global.sh     # 12 scenarios x 30 cell blocks
bash scripts/submit_validate.sh   # physics BO: 5 rounds of ~2.1 h
```

## Tests

```bash
solar_lumped/.venv_gpu/bin/python -m pytest sawh_bayesopt/tests -q   # from the repo root
```
