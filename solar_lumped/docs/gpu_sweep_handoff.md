# GPU-parallelized device-parameter sweep: handoff + status

This started as a scoping doc for building a GPU-accelerated alternative to the
CPU sweep (`sherlock_param_sweep.tex`/`session_summary.md`). **That build is now
done, validated on real A100 hardware, and has completed a full 1,405-site x
135-combo run.** This doc now records what was built and learned (Case 1), plus
exact instructions for the next phase (Case 2/3, a modified radiative-heat-transfer
model). Full step-by-step results, numbers, and the reasoning behind every
architecture decision live in [`../gpu_sweep/FINDINGS.md`](../gpu_sweep/FINDINGS.md)
(12 dated "Results") — this doc is the summary; that one is the detailed log.

## What the sweep computes

`solar_lumped` is a lumped-parameter simulation of a solar-driven atmospheric
water harvesting (SAWH) device — the Wilson model, absorption/desorption
half-cycles, a hydrogel bed loaded with a hygroscopic salt (LiCl only). The
physics right-hand-side is a small system of ODEs integrated over one simulated
day, repeated until the day-to-day state converges to a steady periodic cycle.

The sweep: **for every land grid point on Earth, evaluate every combination of 4
device design parameters, and report water yield + thermal efficiency.**

- **Grid**: 1,405 land sites (3° lat/lon spacing, Natural Earth land polygons).
  `solar_lumped/src/solar_lumped/weather/land_grid.py`'s `grid_land_points()`.
- **Swept parameters**, 135 combinations per site ($5\times3\times3\times3$):
  hydrogel thickness (1.0/3.25/5.5/7.75/10.0 mm), absorber solar absorptivity
  `eps_abs` (0.85/0.90/0.95), glass transmittance `tau_glass` (0.80/0.85/0.90),
  condenser fin area ratio (3.0/7.1/12.0).
- **Fixed**: salt = LiCl, salt:polymer ratio = 4.0, insulation gap = 5mm, vapor
  gap = 40mm, tilt = 35°, `h_amb` = 10 W/m²K (flat constant).
- **Time resolution**: 12 monthly representative mean days per site
  (day-weighted), each solved to its steady periodic state via Aitken Δ²
  acceleration (`ode_system.py`'s `find_cyclic_state`).
- **Total problem size**: $1405 \times 135 = 189{,}675$ (site, combo) evaluations
  x 12 months = ~2.27M leaf "converge this ODE to a steady cycle" problems.
- **Output schema**: `lat, lon, mean_rh_frac, mean_t_amb_c, mean_solar_w_m2,
  salt, hydrogel_thickness_mm, eps_abs, tau_glass, eps_abs_ir, eps_glass_ir,
  fin_area_ratio, vapor_gap_mm, warmup_method, resolution, mean_yield_kg_m2,
  mean_eta_thermal, n_periods` (`grid_param_sweep.py`'s `_CSV_COLUMNS` — the last
  two columns, `eps_abs_ir`/`eps_glass_ir`, were added for Case 2/3 below; Case 1
  output has them blank).

## What was built and learned (Case 1 — original blackbody radiative physics)

Everything lives in `solar_lumped/gpu_sweep/`. Built with JAX + `diffrax`, as
originally proposed, but the two biggest lessons weren't about the choice of
library — they were about how the CPU physics translates to batched execution:

1. **Don't port nested black-box solvers 1:1.** The CPU code nests a scalar
   `brentq` root (desorption mass flux) around a 3x3 `root("hybr")` solve
   (thermal state) because that's what SciPy offers. Porting that structure
   literally cost ~15x more per-step work than necessary. Collapsing both into
   one joint 4x4 Newton system (`jax_physics.py`'s `solve_desorption_state_joint`)
   was both cheaper and more uniform across a batch. *(FINDINGS.md Result 2.)*
2. **The RHS turned out not to be stiff** at monthly-mean-day resolution — an
   explicit adaptive solver (`diffrax.Tsit5`) converges cleanly, no implicit
   method needed. This matters because explicit solvers never need the RHS's
   Jacobian, so the nested-Newton machinery inside the RHS only has to be
   forward-evaluated, not differentiated through. *(Result 3.)*
3. **`jax.jit` is not optional, and forgetting it is catastrophic** — an
   un-jitted integration that should take ~0.1s can take multiple minutes,
   because every adaptive step gets dispatched through Python one at a time
   instead of fused into one compiled program. This bit twice: once for the main
   integration, once for a leftover unbatched post-processing loop. *(Result 5.)*
4. **Cross-length batching** (different sites/months have different real day
   lengths) works by padding every profile in a batch to the batch's max length
   and masking the vector field to freeze state (`dy=0`) past each instance's
   real end. Validated to <0.03% accuracy against the unpadded/serial result.
   *(Result 7.)*
5. **Fixed-round-count beats `lax.while_loop`** for the Aitken steady-state
   search, at least at the batch sizes tested (up to 12,000): run every instance
   the same number of rounds (no early exit, so it vmaps cleanly), then one
   vectorized final pass decides per-instance whether to trust the last round or
   average the last two (covers the CPU code's period-2-orbit fallback too).
   *(Result 7.)*
6. **A real memory bug, not a hardware ceiling**: computing the water-yield
   integral by densely saving every ODE step and recomputing the mass flux via a
   second `vmap` — nested inside the outer batch `vmap` — quietly materialized a
   `(batch x num_saved_steps)`-wide parallel computation, ~90 million-wide at the
   full grid size, and exhausted GPU memory. Fixed by integrating the yield as a
   4th ODE state (`dW/dt = m_des`) instead of reconstructing it after the fact.
   *(Result 9 — worth reading if you extend this to any new derived-quantity
   integral; the same trap is easy to reintroduce.)*
7. **First real A100 numbers** (`serc` partition): small batches (12 instances)
   are 15-19x *slower* than CPU (per-dispatch overhead dominates with nothing to
   amortize it over); by 60,000 instances, per-instance cost dropped ~550x
   relative to the 12-instance case, with GPU memory still nearly flat. *(Result 8.)*
8. **Compile-per-site, not compute, dominated the real full-grid run.** Every
   site has a different weather-profile shape, so nothing compiled for one site
   is reusable for the next — measured ~95-161s/site (varied by run), overwhelmingly
   compile time (actual compute alone predicts ~20-25s). This is *why* the full
   1,405-site sweep was split across a Slurm job array (`sbatch_gpu_sweep_array.sh`,
   `run_gpu_sweep.py --site-range`) rather than run as one sequential job — an
   ~8x mitigation via parallelism, not a fix of the root cause (sites sharing one
   compiled shape, still unfixed). *(Results 10-12.)*
9. **The full grid ran successfully** and was spot-checked against the CPU
   pipeline at multiple sites/combos (including a cold, strongly-seasonal
   high-latitude site that triggers the period-2-orbit fallback) — all
   differences <0.05%, consistent with the fixed-round-count approximation, no
   new discrepancy at full scale. *(Result 12.)* Merged output:
   `outputs/gpu_grid_sweep/full_sweep.csv`.

Code map: `jax_physics.py` (RHS port), `jax_daily_cycle.py` (integrator + Aitken
loop, serial and batched), `run_gpu_sweep.py` (the actual sweep, CLI-compatible
with `grid_param_sweep.py`), `sbatch_gpu_sweep_array.sh` (job array),
`validate_*.py` (the checks that produced every claim above), `GPU_PRIMER.md`
(GPU/JAX concepts from scratch), `SHERLOCK_GPU_RUNBOOK.md` (exact commands).

## Next: Case 2 and Case 3 — modified absorber/glass radiative exchange

Case 1 used the original Wilson Eqs. 3/4, which treat absorber→glass and
glass→ambient radiative exchange as blackbody (`eps=1`, a cavity
approximation). Cases 2 and 3 replace that with real surface IR emissivities:

**Modified glass energy balance:**
$$0 = \frac{k_{air}}{L_c}(T_{abs}-T_{glass}) + \varepsilon_{a\text{-}g}\sigma(T_{abs}^4-T_{glass}^4) - h_{amb}(T_{glass}-T_{amb}) - \varepsilon_{glass,IR}\sigma(T_{glass}^4-T_{amb}^4)$$

**Modified absorber energy balance:**
$$0 = \varepsilon_{abs}\tau_{glass}Q_{solar} - \frac{k_{air}}{L_c}(T_{abs}-T_{glass}) - \varepsilon_{a\text{-}g}\sigma(T_{abs}^4-T_{glass}^4) - U_{gel}(T_{abs}-T_{gel,s})$$

**With:** $\varepsilon_{a\text{-}g} = \left(\dfrac{1}{\varepsilon_{abs,IR}}+\dfrac{1}{\varepsilon_{glass,IR}}-1\right)^{-1}$ (the same parallel-plate formula already used elsewhere for gel↔condenser exchange)

| | $\varepsilon_{abs,IR}$ | $\varepsilon_{abs}$ | $\varepsilon_{glass,IR}$ | $\tau_{glass}$ |
|---|---|---|---|---|
| Case 2 (selective surface) | 0.05 | 0.95 | 0.95 | 0.9 |
| Case 3 ("optical material limits") | 0 | 1 | 0 | 1 |

`eps_abs`/`tau_glass` keep sweeping the existing 135-combo grid for Case 2 (0.05
and 0.95 in the table above are just the existing baseline defaults, shown for
context, not new fixed values); `eps_abs_ir`/`eps_glass_ir` are **new fixed
constants**, not swept. Case 3 fixes all four as the theoretical ideal-device
limit. This is a strict generalization of Case 1, not a breaking change:
$\varepsilon_{abs,IR}=\varepsilon_{glass,IR}=1$ reduces the formula to Case 1's
exact blackbody behavior ($\varepsilon_{a\text{-}g}=1$, glass emits at 1).

### What was changed (already done, validated, committed)

- `src/solar_lumped/physics/device_balances.py`: `DeviceThermalParams` gained
  `eps_abs_ir`/`eps_glass_ir` (default `None` each — both `None` reproduces Case
  1 exactly). `_residuals` branches on whether both are set.
- `gpu_sweep/jax_physics.py`: `ThermalParams` gained the same two fields
  (default `1.0` each — the JAX-side equivalent of Case-1-reproducing). Also
  fixed a real bug found while testing Case 3's `eps=0` values: dividing by a
  plain Python `0.0` (as opposed to a JAX array `0.0`) raises `ZeroDivisionError`
  eagerly, before `jnp.where` gets a chance to mask the invalid branch —
  `parallel_plate_emissivity` now coerces inputs to JAX arrays first.
- `jax_daily_cycle.py`, `scripts/grid_param_sweep.py`, `gpu_sweep/run_gpu_sweep.py`:
  plumbed through as `--eps-abs-ir`/`--eps-glass-ir` CLI flags and two new CSV
  columns (`eps_abs_ir`, `eps_glass_ir`).
- **Validated**: pointwise CPU-vs-JAX thermal residuals match exactly (0.0
  relative error) at both Case 2 and Case 3 parameter values across several
  representative states, and end-to-end `run_gpu_sweep.py` output at a real site
  matches `grid_param_sweep.py`'s own CPU function for the same combos (checked
  locally; see conversation/commit history for exact numbers).

### How to run Case 2 (and later Case 3)

**Important**: the CSV schema changed (2 new columns) — use a **new**
`--output-csv`/output directory for Case 2/3, don't append to Case 1's files.

Same GPU approach as Case 1, just with the two new flags:

```bash
# Smoke-test a few real sites first (mirrors sbatch_gpu_sweep_smoke.sh):
python3 gpu_sweep/run_gpu_sweep.py --num-sites 10 \
  --eps-abs-ir 0.05 --eps-glass-ir 0.95 \
  --output-csv outputs/gpu_grid_sweep_case2/smoke_10sites.csv --resume
```

Then the full grid via a copy of `sbatch_gpu_sweep_array.sh` with
`--eps-abs-ir 0.05 --eps-glass-ir 0.95` added to its `run_gpu_sweep.py` call and
its `--output-csv`/chunk paths pointed at a Case-2-specific output directory
(e.g. `outputs/gpu_grid_sweep_case2/`). For Case 3, same again with
`--eps-abs-ir 0 --eps-glass-ir 0` and its own output directory.

Before trusting a full Case 2/3 run: re-run the smoke test and spot-check a few
rows against the CPU `grid_param_sweep.py`/`build_device_config(...,
eps_abs_ir=..., eps_glass_ir=...)` path for the same sites/combos, the same way
Case 1's full run was checked (`FINDINGS.md` Result 12) — the physics-level
validation above confirms the *equations* are right, not that the full
monthly+Aitken+batching pipeline behaves as expected at scale for this new
parameter regime (e.g. Case 3's `eps=0` extremes are a genuinely different
numerical regime than anything Case 1 exercised — worth confirming the Aitken
convergence and fixed-round-count tolerance still hold up there specifically).
