# SAWH_TEAs

## Running tests: use the GPU venv for anything touching `gpu_sweep/`

There are two interpreters, and the default one silently under-tests the repo.

```bash
# CPU-only. Fine for solar_lumped/src, sawh_bayesopt, waste-heat.
python -m pytest solar_lumped/tests sawh_bayesopt/tests -q

# Has jax + diffrax. REQUIRED for anything touching solar_lumped/gpu_sweep/.
solar_lumped/.venv_gpu/bin/python -m pytest solar_lumped/tests sawh_bayesopt/tests -q
```

`tests/test_cpu_jax_parity.py` opens with `pytest.importorskip("diffrax")` /
`importorskip("jax")`. The default `python` on PATH (miniforge 3.12) has neither, so
those tests **skip silently** and the run still reports all-green. That guard is the
only thing keeping the CPU and JAX physics in lockstep — `SIMPLE_PARITY_TOL` is 5e-4,
tight enough to catch sub-0.1% divergence — so a green CPU-only run means nothing about
the GPU path.

Two extra checks worth running before trusting a sweep:

```bash
# Opt-in real-model integration test (skipped by default even in the GPU venv)
SAWH_BAYESOPT_SLOW_TESTS=1 solar_lumped/.venv_gpu/bin/python -m pytest \
  sawh_bayesopt/tests/test_integration_real_model.py -q
```

### Editing physics in both backends

`physics.py` and `gpu_sweep/jax_physics.py` implement the same equations twice, so any
constitutive change has to land in both. Grep is not sufficient to find every site: a
clamp can be re-applied *downstream* of the function that computes it under a different
variable name — the gel thickness floor is applied in three separate places in
`jax_physics.py` alone, the last as `dh_masked` inside `solve_desorption_state_joint`,
which greps for `dh` will not surface. Static checks — `py_compile`, AST argument-order
verification — will not catch a missed one. Only the parity tests will.

`jax_daily_cycle.build_system_arrays` returns a dict whose **insertion order must match
the positional signature** of `_make_single`'s inner `single()`, since
`make_year_step_fn` does `tuple(system.values())` into a `vmap`. Adding a per-instance
parameter means editing both, in the same position.
