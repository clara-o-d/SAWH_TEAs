"""Full daily-cycle JAX integrator plus the Aitken steady-periodic-state search, matching
run_daily_cycle / find_cyclic_state for the quasi_steady solver.

Both phases use diffrax.Tsit5 (non-stiff at daily-cycle resolution, see FINDINGS.md).
The Aitken loop stays a thin Python loop -- only ~3-6 rounds, not worth fusing into JIT."""

from __future__ import annotations

import time

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

import jax_physics as jp

_T0 = time.perf_counter()


# --- Batched cross-length daily cycle ---
# diffrax needs one static t1 per batch, so profiles are padded to the batch max and the
# vector field freezes state (dy=0) past each instance's real end. See FINDINGS.md 7.


def _pad_to(arr, n_max):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n >= n_max:
        return arr[:n_max]
    return np.concatenate([arr, np.full(n_max - n, arr[-1])])


# The first 8 per-instance arrays are weather (they change day to day); the rest describe
# the system and are constant across a year.
WEATHER_KEYS: tuple[str, ...] = (
    "t_amb_abs", "rh_abs", "h_amb_abs", "n_abs_real",
    "t_amb_des", "solar_des", "h_amb_des", "n_des_real",
)


# Both fidelities use the same tolerance. Complex mode was briefly run at 1e-7/1e-9
# on the theory that its hotter operating points needed it; measured against the CPU
# path that changed the answer by <1e-6 and cost ~36x the wall clock on real weather
# (the constant-profile synthetic day hides this -- its ODE is trivially smooth).
# Parity is carried by matching physics, not by oversolving.
_RTOL, _ATOL = 1e-4, 1e-7

# Per-solve step ceiling. Both solves run throw=False, so exceeding it is SILENT: the
# instance returns a truncated day that reads like a result. That is not hypothetical --
# the instant-equilibrium path used to blow this cap and swing yields by ~60%, which is
# why single() now returns whether both solves reached t1 and run_year_batched NaNs any
# instance that ever did not. The cause was the g-penalty formulation's stiffness; with
# the limit imposed as a constraint instead (physics.equilibrium_t_gel_desorption_c,
# jax_physics.desorption_driving_force) the ideal case costs LESS than finite g and this
# ceiling is ample for both. Kvaerno3 was also tried and is ~5x slower: the vector field
# carries an inner Newton thermal solve, so implicit stages cost more than they save.
_MAX_STEPS = 16384


def _make_single(dt, n_abs_max, n_des_max, *, complex_mode=False, condenser_tracks_ambient=False,
                 h_des_isosteric=False, instant_equilibrium=False):
    """The per-instance daily cycle, taking weather and system parameters as arguments.

    ``complex_mode`` is static (it fixes how many unknowns the thermal Newton solve
    carries), so simple mode keeps its exact 3-unknown cost and code path. The
    per-instance complex parameters arrive as trailing positional arguments, in the
    order ``build_system_arrays`` appends them. ``condenser_tracks_ambient`` is also
    static and uniform across the whole batch -- see jax_physics.desorption_rhs."""

    def single(
        c_w_initial, h_initial,
        t_amb_abs, rh_abs, h_amb_abs, n_abs_real,
        t_amb_des, solar_des, h_amb_des, n_des_real,
        c_s_mol_m3, formula_weight_g_mol, c_w_min_mol_m3, c_w_max_mol_m3, g_conv_m_s, eps_abs, tau_glass,
        eps_abs_ir, eps_glass_ir,
        h0_ref_m, h_floor_m, vapor_gap_m, insulation_gap_m, tilt_deg, fin_area_ratio,
        salt_loading, h_fg_j_per_kg, p_atm_pa,
        *complex_extras,
    ):
        if complex_mode:
            (
                n_glazing_panes, evacuated_gap, fin_thickness_m, fin_height_m,
                h_amb_cond, aw_t_grid, aw_fb_grid, aw_values,
            ) = complex_extras
            aw_table = (aw_t_grid, aw_fb_grid, aw_values)
            fin_geom = dict(fin_thickness_m=fin_thickness_m, fin_height_m=fin_height_m)
        else:
            aw_table, h_amb_cond, fin_geom = None, None, {}
            n_glazing_panes, evacuated_gap = 1.0, 0.0

        mass = jp.MassParams(
            h0_ref_m=h0_ref_m, vapor_gap_m=vapor_gap_m, tilt_deg=tilt_deg,
            c_s_mol_m3=c_s_mol_m3, formula_weight_g_mol=formula_weight_g_mol,
            c_w_min_mol_m3=c_w_min_mol_m3, c_w_max_mol_m3=c_w_max_mol_m3,
            g_conv_m_s=g_conv_m_s,
            aw_table=aw_table,
            instant_equilibrium=instant_equilibrium,
            p_atm_pa=p_atm_pa,
        )
        thermal = jp.ThermalParams(
            insulation_gap_m=insulation_gap_m, vapor_gap_m=vapor_gap_m,
            eps_abs=eps_abs, tau_glass=tau_glass, tilt_deg=tilt_deg,
            eps_abs_ir=eps_abs_ir, eps_glass_ir=eps_glass_ir,
            n_glazing_panes=n_glazing_panes, evacuated_gap=evacuated_gap,
            complex_mode=complex_mode,
            h_des_isosteric=h_des_isosteric,
            p_atm_pa=p_atm_pa,
        )
        h_max_m = jnp.maximum(vapor_gap_m - jp.VAPOR_GAP_TRANSPORT_MIN_M, h0_ref_m + 1e-6)

        def idx_abs(t):
            return jnp.clip((t / dt).astype(jnp.int32), 0, n_abs_max - 1)

        def idx_des(t):
            return jnp.clip((t / dt).astype(jnp.int32), 0, n_des_max - 1)

        def abs_vf(t, y, args):
            i = idx_abs(t)
            dy = jp.absorption_rhs(
                y, t_amb_c=t_amb_abs[i], rh=rh_abs[i], h0_ref_m=h0_ref_m,
                h_floor_m=h_floor_m, h_max_m=h_max_m,
                mass=mass, salt_loading=salt_loading,
            )
            return jnp.where(i < n_abs_real, dy, 0.0)

        def des_vf(t, y, args):
            """y = [c_w, H, T_cond, W], with W integrated directly (dW/dt = m_des). The
            dense-trajectory alternative needs a second vmap that, nested inside this
            batch vmap, exhausted GPU memory at ~90M-wide (FINDINGS.md 9)."""
            i = idx_des(t)
            t_amb_c = t_amb_des[i]
            q_solar = solar_des[i]
            h_amb = h_amb_des[i]
            t_cond_c = t_amb_c if condenser_tracks_ambient else jp.clamp_temperature_c(y[2])
            t_gel0 = jnp.maximum(t_amb_c + 5.0, t_cond_c + 5.0)
            guess = [t_gel0, t_gel0 + jnp.clip(q_solar / 40.0, 5.0, 30.0), t_amb_c + 2.0]
            if complex_mode:
                # Outer pane starts between the inner pane and ambient, which is the
                # physically correct ordering for a 2-pane stack.
                guess.append(t_amb_c + 1.0)
            x0_guess = jnp.array(guess)
            dy, aux = jp.desorption_rhs(
                y[:3], t_amb_c=t_amb_c, q_solar_w_m2=q_solar,
                h_amb=h_amb, thermal=thermal, mass=mass,
                h0_ref_m=h0_ref_m, h_floor_m=h_floor_m, h_fg_j_per_kg=h_fg_j_per_kg,
                fin_area_ratio=fin_area_ratio,
                x0_guess=x0_guess,
                # B4: forced condenser air is a separate channel from the absorber's
                # h_amb, so a fan-cooled condenser is not tied to ambient wind.
                h_amb_cond=h_amb_cond,
                condenser_tracks_ambient=condenser_tracks_ambient,
                **fin_geom,
            )
            m_des = aux[3]
            dy4 = jnp.concatenate([dy, jnp.array([m_des])])
            return jnp.where(i < n_des_real, dy4, 0.0)

        if instant_equilibrium:
            # No ODE on this half-cycle. T_gel is T_amb (nothing heat-limited) and the
            # weather is piecewise-constant on dt, so instant kinetics means the gel sits
            # at the current interval's equilibrium loading: the exact trajectory is a
            # staircase, one isotherm inversion per interval. This is what removes the
            # last stiff term -- the penalty route integrated a rate-~g relaxation toward
            # this same staircase and needed >16,384 steps a day to do it.
            ratio = jp.WATER_MOLAR_MASS_KG_MOL * h0_ref_m / jp.RHO_GEL_KG_M3

            def absorb_interval(carry, i):
                c_w_k, h_k = carry
                c_eq = jp.equilibrium_c_w_absorption(
                    rh=rh_abs[i], t_gel_c=t_amb_abs[i], mass=mass, salt_loading=salt_loading,
                )
                c_eq = jnp.clip(c_eq, c_w_min_mol_m3, c_w_max_mol_m3)
                h_next = jnp.clip(h_k + (c_eq - c_w_k) * ratio, h_floor_m, h_max_m)
                # Past the instance's real day length the state freezes, exactly as the
                # padded vector fields do (dy = 0 there).
                real = i < n_abs_real
                return (jnp.where(real, c_eq, c_w_k), jnp.where(real, h_next, h_k)), None

            (c_w_mid, h_mid), _ = jax.lax.scan(
                absorb_interval,
                (c_w_initial, jnp.maximum(h_initial, h_floor_m)),
                jnp.arange(n_abs_max),
            )
            abs_ok = jnp.array(True)
        else:
            y0_abs = jnp.array([c_w_initial, jnp.maximum(h_initial, h_floor_m)])
            sol_abs = diffrax.diffeqsolve(
                diffrax.ODETerm(abs_vf), diffrax.Tsit5(), t0=0.0, t1=dt * n_abs_max, dt0=dt, y0=y0_abs, args=None,
                saveat=diffrax.SaveAt(t1=True),
                stepsize_controller=diffrax.PIDController(rtol=_RTOL, atol=_ATOL, dtmax=dt),
                max_steps=_MAX_STEPS, adjoint=diffrax.DirectAdjoint(), throw=False,
            )
            c_w_mid = jnp.clip(sol_abs.ys[0, 0], c_w_min_mol_m3, c_w_max_mol_m3)
            h_mid = jnp.clip(sol_abs.ys[0, 1], h_floor_m, h_max_m)
            abs_ok = sol_abs.result == diffrax.RESULTS.successful

        t_cond0 = jp.clamp_temperature_c(t_amb_des[0])
        y0_des = jnp.array([c_w_mid, h_mid, t_cond0, 0.0])
        sol_des = diffrax.diffeqsolve(
            diffrax.ODETerm(des_vf), diffrax.Tsit5(), t0=0.0, t1=dt * n_des_max, dt0=dt, y0=y0_des, args=None,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(rtol=_RTOL, atol=_ATOL, dtmax=dt),
            max_steps=_MAX_STEPS, adjoint=diffrax.DirectAdjoint(), throw=False,
        )
        water = jnp.maximum(0.0, sol_des.ys[0, 3])

        solar_sum = jnp.sum(jnp.where(jnp.arange(n_des_max) < n_des_real, solar_des, 0.0)) * dt
        eta = jnp.where(solar_sum > 0.0, water * h_fg_j_per_kg / solar_sum, 0.0)

        c_w_end = jnp.clip(sol_des.ys[0, 0], c_w_min_mol_m3, c_w_max_mol_m3)
        h_end = jnp.maximum(sol_des.ys[0, 1], h_floor_m)
        # Both solves run throw=False so one bad instance cannot abort a whole batch --
        # but that means an instance which exhausts max_steps returns a TRUNCATED day and
        # no error, i.e. a plausible-looking wrong yield. Report it instead: this flag is
        # what run_year_batched turns into NaN rather than a silently short year.
        ok = abs_ok & (sol_des.result == diffrax.RESULTS.successful)
        return water, eta, c_w_end, h_end, ok

    return single


def year_padding(profiles_by_instance):
    """Shared (dt, n_abs_max, n_des_max) across every day of every instance, so each day's
    weather pads to one shape and the compiled step is reused all year."""
    flat = [p for profiles in profiles_by_instance for p in profiles]
    return (
        flat[0].absorption.dt_s,
        max(len(p.absorption.temperature_c) for p in flat),
        max(len(p.desorption.temperature_c) for p in flat),
    )


def _h_amb_cond_for(cx) -> float:
    """B4 condenser-side convection coefficient, from ComplexOptions itself.

    Deliberately not a second implementation: both backends call the same method,
    so forced cooling cannot drift between them. ``None`` (passive) becomes the
    baseline h_amb here, which is what sharing the absorber's coefficient means.
    """
    from solar_lumped.physics import H_AMB_W_M2_K

    h = cx.condenser_h_amb_w_m2_k()
    return float(H_AMB_W_M2_K) if h is None else float(h)


def build_system_arrays(configs, *, complex_mode=False, instant_equilibrium=False):
    """Per-instance system parameters -- constant across the year.

    In complex mode the dict gains B2/B3/B8's per-instance parameters, appended in
    the order ``_make_single`` unpacks them from ``*complex_extras``. The ZSR
    inverse table is built on the *host* (blend weights are constant per instance),
    which is what lets B8 run inside a jitted right-hand side at all -- the CPU
    path root-solves the same inversion, from the same builder.
    """
    mass_ps = [c.mass_params() for c in configs]
    thermal_ps = [c.thermal_params() for c in configs]
    # instant_equilibrium selects a code path (like complex_mode), so it is static: it
    # reaches the RHS through make_year_step_fn, not through these arrays. Checking it
    # here is the only place that catches a caller who set it on the configs and then
    # ran a whole sweep at finite g without noticing.
    if {c.instant_equilibrium for c in configs} != {instant_equilibrium}:
        raise ValueError(
            "instant_equilibrium must be uniform across the batch and match the flag "
            "passed to build_system_arrays / make_year_step_fn"
        )
    arrays = dict(
        c_s_mol_m3=jnp.array([m.c_s_mol_m3 for m in mass_ps]),
        formula_weight_g_mol=jnp.array([m.formula_weight_g_mol for m in mass_ps]),
        # Gel-water bounds, per instance -- salt- and blend-dependent, so neither can be
        # the module constant the guards used to read. Both come straight off
        # MassTransferParams so the two backends bound at the same numbers by construction.
        c_w_min_mol_m3=jnp.array([m.c_w_min_mol_m3 for m in mass_ps]),
        c_w_max_mol_m3=jnp.array([m.c_w_max_mol_m3 for m in mass_ps]),
        g_conv_m_s=jnp.array([m.g_conv_m_s for m in mass_ps]),
        eps_abs=jnp.array([t.eps_abs for t in thermal_ps]),
        tau_glass=jnp.array([t.tau_glass for t in thermal_ps]),
        eps_abs_ir=jnp.array([t.eps_abs_ir if t.eps_abs_ir is not None else 1.0 for t in thermal_ps]),
        eps_glass_ir=jnp.array([t.eps_glass_ir if t.eps_glass_ir is not None else 1.0 for t in thermal_ps]),
        h0_ref_m=jnp.array([c.hydrogel_thickness_m for c in configs]),
        # Hydrate-floor thickness, not H0: mirrors SystemConfig.hydrogel_floor_thickness_m
        # so the CPU and JAX backends bottom the gel out at the same place.
        h_floor_m=jnp.array([c.hydrogel_floor_thickness_m() for c in configs]),
        vapor_gap_m=jnp.array([c.vapor_gap_m for c in configs]),
        insulation_gap_m=jnp.array([t.insulation_gap_m for t in thermal_ps]),
        tilt_deg=jnp.array([c.tilt_deg for c in configs]),
        fin_area_ratio=jnp.array([c.fin_area_ratio for c in configs]),
        salt_loading=jnp.array([c.salt_loading for c in configs]),
        h_fg_j_per_kg=jnp.array([c.h_fg_j_per_kg for c in configs]),
        # Site ambient pressure. Must stay LAST in this dict: single() unpacks these
        # positionally, so insertion order here is the signature there.
        p_atm_pa=jnp.array([t.p_atm_pa for t in thermal_ps]),
    )
    if not complex_mode:
        return arrays

    from solar_lumped.complex_model import zsr_inverse_table

    cxs = [c.complex for c in configs]
    if any(cx is None for cx in cxs):
        raise ValueError("complex_mode=True requires every SystemConfig to carry .complex")
    tables = [zsr_inverse_table(tuple(cx.blend_weights)) for cx in cxs]
    arrays.update(
        n_glazing_panes=jnp.array([float(cx.n_glazing_panes) for cx in cxs]),
        evacuated_gap=jnp.array([1.0 if cx.evacuated_gap else 0.0 for cx in cxs]),
        fin_thickness_m=jnp.array([cx.fin_thickness_m for cx in cxs]),
        fin_height_m=jnp.array([cx.fin_height_m for cx in cxs]),
        # Forced condenser air is design-constant, so it is a scalar per instance
        # rather than a weather channel: the fans hold their speed regardless of day.
        h_amb_cond=jnp.array([_h_amb_cond_for(cx) for cx in cxs]),
        aw_t_grid=jnp.array(np.stack([t for t, _f, _v in tables])),
        aw_fb_grid=jnp.array(np.stack([f for _t, f, _v in tables])),
        aw_values=jnp.array(np.stack([v for _t, _f, v in tables])),
    )
    return arrays


def build_day_weather(profiles, n_abs_max, n_des_max):
    """One day's weather for every instance, padded to the year-wide shape."""
    return (
        jnp.array(np.stack([_pad_to(p.absorption.temperature_c, n_abs_max) for p in profiles])),
        jnp.array(np.stack([_pad_to(p.absorption.relative_humidity, n_abs_max) for p in profiles])),
        jnp.array(np.stack([_pad_to(p.absorption.h_amb_w_m2_k, n_abs_max) for p in profiles])),
        jnp.array(np.array([len(p.absorption.temperature_c) for p in profiles], dtype=np.int32)),
        jnp.array(np.stack([_pad_to(p.desorption.temperature_c, n_des_max) for p in profiles])),
        jnp.array(np.stack([_pad_to(p.desorption.solar_w_m2, n_des_max) for p in profiles])),
        jnp.array(np.stack([_pad_to(p.desorption.h_amb_w_m2_k, n_des_max) for p in profiles])),
        jnp.array(np.array([len(p.desorption.temperature_c) for p in profiles], dtype=np.int32)),
    )


def make_year_step_fn(system, dt, n_abs_max, n_des_max, *, complex_mode=False, condenser_tracks_ambient=False,
                      h_des_isosteric=False, instant_equilibrium=False):
    """One compiled step reused for every day of the year: (c_w, h, weather) -> (water,
    eta, c_w_end, h_end). Weather is an argument, not a closure constant, so all 365 days
    share a single compilation as long as they are padded to the same shape."""
    single = _make_single(
        dt, n_abs_max, n_des_max, complex_mode=complex_mode,
        condenser_tracks_ambient=condenser_tracks_ambient,
        h_des_isosteric=h_des_isosteric,
        instant_equilibrium=instant_equilibrium,
    )
    n_weather = len(WEATHER_KEYS)
    batched = jax.vmap(single, in_axes=(0, 0) + (0,) * (n_weather + len(system)))
    system_vals = tuple(system.values())

    @jax.jit
    def step(c_w, h, weather):
        return batched(c_w, h, *weather, *system_vals)

    return step


def run_year_batched(step_fn, day_weathers, *, c_w_initial, h_initial, aitken_max_rounds=8,
                     progress_every=0):
    """Simulate a full year per instance and return (mean daily yield, mean eta).

    Day 1 is Aitken-extrapolated to its steady periodic state so the year does not start
    from an arbitrary loading; every later day warm-starts from the previous day's end
    state, so the sorbent carries real seasonal history and no mean-day approximation is
    made. Days are inherently sequential -- the batch axis (design x site) is where the
    parallelism lives.

    ``progress_every`` > 0 prints a day counter and ETA that often. The year is one
    uninterruptible block from the caller's side -- nothing reaches disk until it
    returns -- so without this a multi-hour run is indistinguishable from a hung one."""
    c_w = np.asarray(c_w_initial, dtype=float)
    h = np.asarray(h_initial, dtype=float)
    c_w, h = find_cyclic_state_batched(
        lambda cw, hh: step_fn(cw, hh, day_weathers[0]),
        c_w_initial=c_w, h_initial=h, max_rounds=aitken_max_rounds,
    )
    if progress_every:
        print(f"    warm-up done ({time.perf_counter() - _T0:.0f}s in), "
              f"{len(day_weathers)} day(s) to walk", flush=True)

    water_sum = np.zeros_like(c_w)
    eta_sum = np.zeros_like(c_w)
    failed_days = np.zeros_like(c_w, dtype=int)
    t_days = time.perf_counter()
    for day, weather in enumerate(day_weathers, start=1):
        water, eta, c_w, h, ok = step_fn(jnp.asarray(c_w), jnp.asarray(h), weather)
        failed_days += ~np.asarray(ok, dtype=bool)
        if progress_every and (day % progress_every == 0 or day == len(day_weathers)):
            per_day = (time.perf_counter() - t_days) / day
            print(f"    day {day}/{len(day_weathers)}  {per_day:.2f}s/day  "
                  f"~{per_day * (len(day_weathers) - day) / 60:.0f} min left", flush=True)
        water_sum += np.asarray(water, dtype=float)
        eta_sum += np.asarray(eta, dtype=float)
        c_w = np.asarray(c_w, dtype=float)
        h = np.asarray(h, dtype=float)
    n_days = len(day_weathers)
    mean_water, mean_eta = water_sum / n_days, eta_sum / n_days
    if failed_days.any():
        bad = failed_days > 0
        # NaN, not a raise: the instances that did converge are still good data, and a
        # raise here would throw away a whole chunk over one pathological site. NaN
        # reaches the CSV as an empty/NaN yield, which cannot be mistaken for a result.
        mean_water[bad] = np.nan
        mean_eta[bad] = np.nan
        print(f"    WARNING: {int(bad.sum())}/{len(bad)} instance(s) hit the ODE step cap "
              f"on at least one day (worst: {int(failed_days.max())}/{n_days} days). "
              f"Their yields are NaN, not truncated years.", flush=True)
    return mean_water, mean_eta


def find_cyclic_state_batched(
    daily_cycle_fn, *, c_w_initial, h_initial, tol=1e-6, max_rounds=8,
):
    """Batched Aitken Delta^2 search: every instance runs the same max_rounds (no early
    exit, which is what makes it vmap-friendly), then one vectorized pass either trusts the
    extrapolated state (rel_step < tol) or averages the last two rounds. A simplified
    stand-in for find_cyclic_state's stall counter, not a byte-for-byte port."""
    c_w_initial = np.asarray(c_w_initial, dtype=float)
    h_initial = np.asarray(h_initial, dtype=float)
    x = np.stack([c_w_initial, h_initial], axis=1)

    def step(state):
        out = daily_cycle_fn(jnp.asarray(state[:, 0]), jnp.asarray(state[:, 1]))
        return np.stack([np.asarray(out[2]), np.asarray(out[3])], axis=1)

    x_prev = x
    x_star = x
    for _ in range(max(1, max_rounds)):
        x1 = step(x)
        x2 = step(x1)
        d0 = x1 - x
        d1 = x2 - x1
        dd = d1 - d0
        denom = np.sum(dd * dd, axis=1)
        safe = denom > 1e-30
        x_star_new = np.where(
            safe[:, None],
            x - d0 * (np.sum(d0 * dd, axis=1) / np.where(safe, denom, 1.0))[:, None],
            x2,
        )
        x_prev, x_star, x = x, x_star_new, x_star_new

    rel_step = np.linalg.norm(x_star - x_prev, axis=1) / np.maximum(np.linalg.norm(x_star, axis=1), 1e-12)
    converged = rel_step < tol
    x_final = np.where(converged[:, None], x_star, 0.5 * (x_star + x_prev))
    return x_final[:, 0], x_final[:, 1]
