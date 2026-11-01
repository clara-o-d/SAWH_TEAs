# SAWH_TEAs

## Running tests: use `.venv_gpu` for anything touching the JAX physics or `sawh_bayesopt`

There are two interpreters, and the default one silently under-tests the repo.

```bash
# CPU-only. Fine for solar_lumped/src and the per-site BO's cpu backend.
python -m pytest solar_lumped/tests sawh_bayesopt/tests -q

# Has jax + diffrax + equinox. REQUIRED for the JAX physics and all of sawh_bayesopt.
solar_lumped/.venv_gpu/bin/python -m pytest solar_lumped/tests sawh_bayesopt/tests -q
```

`tests/test_cpu_jax_parity.py` opens with `pytest.importorskip("diffrax")` /
`importorskip("jax")`. The default `python` on PATH (miniforge 3.12) has neither, so
those tests **skip silently** and the run still reports all-green. That guard is the
only thing keeping the CPU and JAX physics in lockstep — `SIMPLE_PARITY_TOL` is 5e-4,
tight enough to catch sub-0.1% divergence — so a green CPU-only run means nothing about
the JAX path (which every sawh_bayesopt physics year runs on, CPU or GPU).

### Editing physics in both backends

`physics.py` and `jax_physics.py` (both in `solar_lumped/src/solar_lumped/`) implement the same equations twice, so any
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

### Intra-day tilt

Tilt can change within the sealed (desorption) phase: `weather.PhaseProfile.tilt_deg` is a
per-step tilt, set by `profile_from_day_df(poa_tilt_deg=(t1, t2, t3))`. Both backends read it
where they read `SystemConfig.tilt_deg` -- CPU per step in `simulation._integrate_desorption`,
JAX through the `tilt_des` weather channel (`jax_daily_cycle.WEATHER_KEYS`) -- and
`build_day_weather` now takes the system tilt (`tilt_deg=system["tilt_deg"]`) to fill the
channel for profiles without a schedule. `test_intraday_tilt_schedule_backends_agree` is the
parity guard.
