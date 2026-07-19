"""Full daily-cycle JAX integrator (absorption -> desorption) and the Aitken
steady-periodic-state search on top of it, matching ode_system.py's
run_daily_cycle / find_cyclic_state for the quasi_steady desorption solver.

Absorption and desorption are each integrated with diffrax.Tsit5 (both proved
non-stiff at monthly-mean-day resolution -- see FINDINGS.md). The Aitken Delta^2
loop itself stays a thin Python loop calling one jitted daily-cycle function
twice per round, exactly mirroring ode_system.py::find_cyclic_state's structure
-- it's only ~3-6 rounds, not worth fusing into the JIT boundary.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

import jax_physics as jp


def _phase_arrays(phase_profile):
    return (
        jnp.array(phase_profile.temperature_c),
        jnp.array(phase_profile.relative_humidity),
        jnp.array(phase_profile.solar_w_m2),
        jnp.array(phase_profile.h_amb_w_m2_k),
        phase_profile.dt_s,
        len(phase_profile.temperature_c),
    )


def make_daily_cycle_fn(profile, config):
    """Build a single jax.jit-able function y0 -> (yield_kg, eta, c_w_end, h_end)
    for one (weather profile, device config) pair. Compile once, reuse across
    Aitken rounds and (eventually) vmap across a batch of configs sharing the
    same profile shape.
    """
    t_amb_abs, rh_abs, _solar_abs, h_amb_abs, dt_abs, n_abs = _phase_arrays(profile.absorption)
    t_amb_des, _rh_des, solar_des, h_amb_des, dt_des, n_des = _phase_arrays(profile.desorption)

    mass_p = config.mass_params()
    thermal_p = config.thermal_params()
    h0_ref_m = config.hydrogel_thickness_m
    vapor_gap_m = config.vapor_gap_m
    h_max_m = max(vapor_gap_m - jp.VAPOR_GAP_TRANSPORT_MIN_M, h0_ref_m + 1e-6)
    fin_area_ratio = config.fin_area_ratio
    h_fg = config.h_fg_j_per_kg
    salt_to_polymer_ratio = config.salt_to_polymer_ratio

    mass = jp.MassParams(
        h0_ref_m=h0_ref_m, vapor_gap_m=vapor_gap_m, tilt_deg=config.tilt_deg,
        c_s_mol_m3=mass_p.c_s_mol_m3, formula_weight_g_mol=mass_p.formula_weight_g_mol,
        g_conv_m_s=mass_p.g_conv_m_s,
    )
    thermal = jp.ThermalParams(
        insulation_gap_m=thermal_p.insulation_gap_m, vapor_gap_m=vapor_gap_m,
        eps_abs=thermal_p.eps_abs, tau_glass=thermal_p.tau_glass, tilt_deg=config.tilt_deg,
    )

    def idx_abs(t):
        return jnp.clip((t / dt_abs).astype(jnp.int32), 0, n_abs - 1)

    def idx_des(t):
        return jnp.clip((t / dt_des).astype(jnp.int32), 0, n_des - 1)

    def abs_vector_field(t, y, args):
        i = idx_abs(t)
        dy = jp.absorption_rhs(
            y, t_amb_c=t_amb_abs[i], rh=rh_abs[i], h0_ref_m=h0_ref_m, h_max_m=h_max_m,
            mass=mass, salt_to_polymer_ratio=salt_to_polymer_ratio,
        )
        return dy

    def des_vector_field(t, y, args):
        i = idx_des(t)
        t_amb_c = t_amb_des[i]
        q_solar = solar_des[i]
        h_amb = h_amb_des[i]
        t_cond_c = jnp.clip(y[2], -40.0, 120.0)
        x0_guess = jnp.array(
            [
                jnp.maximum(t_amb_c + 5.0, t_cond_c + 5.0),
                jnp.maximum(t_amb_c + 5.0, t_cond_c + 5.0) + jnp.clip(q_solar / 40.0, 5.0, 30.0),
                t_amb_c + 2.0,
            ]
        )
        dy, _aux = jp.desorption_rhs(
            y, t_amb_c=t_amb_c, q_solar_w_m2=q_solar, h_amb=h_amb, thermal=thermal, mass=mass,
            h0_ref_m=h0_ref_m, h_fg_j_per_kg=h_fg, fin_area_ratio=fin_area_ratio, x0_guess=x0_guess,
        )
        return dy

    def aux_at_des(y_k, i):
        t_amb_c = t_amb_des[i]
        q_solar = solar_des[i]
        h_amb = h_amb_des[i]
        t_cond_c = jnp.clip(y_k[2], -40.0, 120.0)
        x0_guess = jnp.array(
            [
                jnp.maximum(t_amb_c + 5.0, t_cond_c + 5.0),
                jnp.maximum(t_amb_c + 5.0, t_cond_c + 5.0) + jnp.clip(q_solar / 40.0, 5.0, 30.0),
                t_amb_c + 2.0,
            ]
        )
        _, aux = jp.desorption_rhs(
            y_k, t_amb_c=t_amb_c, q_solar_w_m2=q_solar, h_amb=h_amb, thermal=thermal, mass=mass,
            h0_ref_m=h0_ref_m, h_fg_j_per_kg=h_fg, fin_area_ratio=fin_area_ratio, x0_guess=x0_guess,
        )
        return aux[3]

    t_eval_des = jnp.linspace(0.0, dt_des * n_des, n_des + 1)
    idx_arr_des = jnp.clip((t_eval_des / dt_des).astype(jnp.int32), 0, n_des - 1)
    solar_sum_des = jnp.sum(solar_des) * dt_des

    abs_term = diffrax.ODETerm(abs_vector_field)
    des_term = diffrax.ODETerm(des_vector_field)
    abs_controller = diffrax.PIDController(rtol=1e-4, atol=1e-7, dtmax=dt_abs)
    des_controller = diffrax.PIDController(rtol=1e-4, atol=1e-7, dtmax=dt_des)

    def daily_cycle(c_w_initial, h_initial):
        y0_abs = jnp.array([c_w_initial, jnp.maximum(h_initial, h0_ref_m)])
        sol_abs = diffrax.diffeqsolve(
            abs_term, diffrax.Tsit5(), t0=0.0, t1=dt_abs * n_abs, dt0=dt_abs, y0=y0_abs, args=None,
            saveat=diffrax.SaveAt(t1=True), stepsize_controller=abs_controller,
            max_steps=16384, adjoint=diffrax.DirectAdjoint(), throw=False,
        )
        c_w_mid, h_mid = sol_abs.ys[0, 0], sol_abs.ys[0, 1]
        c_w_mid = jnp.clip(c_w_mid, jp.C_W_MIN_MOL_M3, jp.C_W_MAX_MOL_M3)
        h_mid = jnp.clip(h_mid, h0_ref_m, h_max_m)

        t_cond0 = jnp.clip(t_amb_des[0], -40.0, 120.0)
        y0_des = jnp.array([c_w_mid, h_mid, t_cond0])
        sol_des = diffrax.diffeqsolve(
            des_term, diffrax.Tsit5(), t0=0.0, t1=dt_des * n_des, dt0=dt_des, y0=y0_des, args=None,
            saveat=diffrax.SaveAt(ts=t_eval_des), stepsize_controller=des_controller,
            max_steps=16384, adjoint=diffrax.DirectAdjoint(), throw=False,
        )
        ys_des = sol_des.ys
        m_des_hist = jax.vmap(aux_at_des)(ys_des, idx_arr_des)
        water = jnp.trapezoid(m_des_hist, dx=dt_des)
        water = jnp.maximum(0.0, water)
        eta = jnp.where(solar_sum_des > 0.0, water * h_fg / solar_sum_des, 0.0)

        c_w_end = jnp.clip(ys_des[-1, 0], jp.C_W_MIN_MOL_M3, jp.C_W_MAX_MOL_M3)
        h_end = jnp.maximum(ys_des[-1, 1], h0_ref_m)
        return water, eta, c_w_end, h_end

    return jax.jit(daily_cycle)


def find_cyclic_state_jax(
    daily_cycle_fn, *, c_w_initial, h_initial, tol=1e-6, max_rounds=10, stall_ratio=0.5, stall_rounds=2,
):
    """Aitken Delta^2 steady-periodic-state search -- same algorithm and
    stall/period-2 fallback as ode_system.py::find_cyclic_state, calling the
    jitted JAX daily-cycle function instead of the CPU one.
    """
    x = np.array([c_w_initial, h_initial], dtype=float)

    def step(state):
        _, _, cw_end, h_end = daily_cycle_fn(float(state[0]), float(state[1]))
        return np.array([float(cw_end), float(h_end)])

    prev_rel_step = None
    prev_x_star = None
    stall_count = 0
    for _round_idx in range(1, max(1, max_rounds) + 1):
        x1 = step(x)
        x2 = step(x1)
        d0 = x1 - x
        d1 = x2 - x1
        dd = d1 - d0
        denom = float(np.dot(dd, dd))
        x_star = x2 if denom < 1e-30 else x - d0 * (np.dot(d0, dd) / denom)
        rel_step = float(np.linalg.norm(x_star - x2) / max(float(np.linalg.norm(x2)), 1e-12))
        if rel_step < tol:
            x = x_star
            break
        if prev_rel_step is not None and rel_step > stall_ratio * prev_rel_step:
            stall_count += 1
            if stall_count >= stall_rounds:
                x = 0.5 * (x_star + x)
                break
        else:
            stall_count = 0
        prev_rel_step = rel_step
        prev_x_star = x
        x = x_star
    else:
        if prev_x_star is not None:
            x = 0.5 * (x + prev_x_star)
    return float(x[0]), float(x[1])


# ---------------------------------------------------------------------------
# Batched, cross-length daily cycle -- vmaps the daily cycle across a batch of
# (weather profile, device config) pairs whose real absorption/desorption
# lengths differ (e.g. different months at one site, or different sites).
# diffrax needs one static t1 shared by the whole batch, so every profile is
# padded to the batch's max length with its own last value repeated, and the
# vector field is masked to freeze the state (dy=0) once real time runs out --
# equivalent to just stopping at each instance's own real end, but with a
# uniform loop trip count that vmap/XLA can compile once. See FINDINGS.md
# "Result 7" for the accuracy check on this masking approach.
# ---------------------------------------------------------------------------


def _pad_to(arr, n_max):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n >= n_max:
        return arr[:n_max]
    return np.concatenate([arr, np.full(n_max - n, arr[-1])])


def build_batch_arrays(profiles, configs):
    """Stack a list of (DailyWeatherProfile, DeviceConfig) pairs into padded
    JAX arrays for make_batched_daily_cycle_fn. Every profile's own PHASE_DT_S
    is assumed equal (always 100s in this codebase -- see weather/profiles.py);
    only the step *count* varies by real day length.
    """
    n_abs_max = max(len(p.absorption.temperature_c) for p in profiles)
    n_des_max = max(len(p.desorption.temperature_c) for p in profiles)
    dt = profiles[0].absorption.dt_s

    t_amb_abs = np.stack([_pad_to(p.absorption.temperature_c, n_abs_max) for p in profiles])
    rh_abs = np.stack([_pad_to(p.absorption.relative_humidity, n_abs_max) for p in profiles])
    h_amb_abs = np.stack([_pad_to(p.absorption.h_amb_w_m2_k, n_abs_max) for p in profiles])
    n_abs_real = np.array([len(p.absorption.temperature_c) for p in profiles], dtype=np.int32)

    t_amb_des = np.stack([_pad_to(p.desorption.temperature_c, n_des_max) for p in profiles])
    solar_des = np.stack([_pad_to(p.desorption.solar_w_m2, n_des_max) for p in profiles])
    h_amb_des = np.stack([_pad_to(p.desorption.h_amb_w_m2_k, n_des_max) for p in profiles])
    n_des_real = np.array([len(p.desorption.temperature_c) for p in profiles], dtype=np.int32)

    mass_ps = [c.mass_params() for c in configs]
    thermal_ps = [c.thermal_params() for c in configs]

    cfg = dict(
        t_amb_abs=jnp.array(t_amb_abs), rh_abs=jnp.array(rh_abs), h_amb_abs=jnp.array(h_amb_abs),
        n_abs_real=jnp.array(n_abs_real),
        t_amb_des=jnp.array(t_amb_des), solar_des=jnp.array(solar_des), h_amb_des=jnp.array(h_amb_des),
        n_des_real=jnp.array(n_des_real),
        c_s_mol_m3=jnp.array([m.c_s_mol_m3 for m in mass_ps]),
        formula_weight_g_mol=jnp.array([m.formula_weight_g_mol for m in mass_ps]),
        g_conv_m_s=jnp.array([m.g_conv_m_s for m in mass_ps]),
        eps_abs=jnp.array([t.eps_abs for t in thermal_ps]),
        tau_glass=jnp.array([t.tau_glass for t in thermal_ps]),
        h0_ref_m=jnp.array([c.hydrogel_thickness_m for c in configs]),
        vapor_gap_m=jnp.array([c.vapor_gap_m for c in configs]),
        insulation_gap_m=jnp.array([t.insulation_gap_m for t in thermal_ps]),
        tilt_deg=jnp.array([c.tilt_deg for c in configs]),
        fin_area_ratio=jnp.array([c.fin_area_ratio for c in configs]),
        salt_to_polymer_ratio=jnp.array([c.salt_to_polymer_ratio for c in configs]),
        h_fg_j_per_kg=jnp.array([c.h_fg_j_per_kg for c in configs]),
    )
    return cfg, dt, n_abs_max, n_des_max


def make_batched_daily_cycle_fn(batch, dt, n_abs_max, n_des_max):
    """Build a jax.jit(jax.vmap(...))-compiled function
    (c_w_initial, h_initial) [each shape (batch,)] -> (yield, eta, c_w_end, h_end)
    [each shape (batch,)], compiled once for the whole batch regardless of size.
    """
    t_eval_des = jnp.linspace(0.0, dt * n_des_max, n_des_max + 1)

    def single(
        c_w_initial, h_initial,
        t_amb_abs, rh_abs, h_amb_abs, n_abs_real,
        t_amb_des, solar_des, h_amb_des, n_des_real,
        c_s_mol_m3, formula_weight_g_mol, g_conv_m_s, eps_abs, tau_glass,
        h0_ref_m, vapor_gap_m, insulation_gap_m, tilt_deg, fin_area_ratio,
        salt_to_polymer_ratio, h_fg_j_per_kg,
    ):
        mass = jp.MassParams(
            h0_ref_m=h0_ref_m, vapor_gap_m=vapor_gap_m, tilt_deg=tilt_deg,
            c_s_mol_m3=c_s_mol_m3, formula_weight_g_mol=formula_weight_g_mol, g_conv_m_s=g_conv_m_s,
        )
        thermal = jp.ThermalParams(
            insulation_gap_m=insulation_gap_m, vapor_gap_m=vapor_gap_m,
            eps_abs=eps_abs, tau_glass=tau_glass, tilt_deg=tilt_deg,
        )
        h_max_m = jnp.maximum(vapor_gap_m - jp.VAPOR_GAP_TRANSPORT_MIN_M, h0_ref_m + 1e-6)

        def idx_abs(t):
            return jnp.clip((t / dt).astype(jnp.int32), 0, n_abs_max - 1)

        def idx_des(t):
            return jnp.clip((t / dt).astype(jnp.int32), 0, n_des_max - 1)

        def abs_vf(t, y, args):
            i = idx_abs(t)
            dy = jp.absorption_rhs(
                y, t_amb_c=t_amb_abs[i], rh=rh_abs[i], h0_ref_m=h0_ref_m, h_max_m=h_max_m,
                mass=mass, salt_to_polymer_ratio=salt_to_polymer_ratio,
            )
            return jnp.where(i < n_abs_real, dy, 0.0)

        def des_vf(t, y, args):
            i = idx_des(t)
            t_amb_c = t_amb_des[i]
            q_solar = solar_des[i]
            h_amb = h_amb_des[i]
            t_cond_c = jnp.clip(y[2], -40.0, 120.0)
            x0_guess = jnp.array(
                [
                    jnp.maximum(t_amb_c + 5.0, t_cond_c + 5.0),
                    jnp.maximum(t_amb_c + 5.0, t_cond_c + 5.0) + jnp.clip(q_solar / 40.0, 5.0, 30.0),
                    t_amb_c + 2.0,
                ]
            )
            dy, _aux = jp.desorption_rhs(
                y, t_amb_c=t_amb_c, q_solar_w_m2=q_solar, h_amb=h_amb, thermal=thermal, mass=mass,
                h0_ref_m=h0_ref_m, h_fg_j_per_kg=h_fg_j_per_kg, fin_area_ratio=fin_area_ratio, x0_guess=x0_guess,
            )
            return jnp.where(i < n_des_real, dy, 0.0)

        def aux_at(y_k, i):
            t_amb_c = t_amb_des[i]
            q_solar = solar_des[i]
            h_amb = h_amb_des[i]
            t_cond_c = jnp.clip(y_k[2], -40.0, 120.0)
            x0_guess = jnp.array(
                [
                    jnp.maximum(t_amb_c + 5.0, t_cond_c + 5.0),
                    jnp.maximum(t_amb_c + 5.0, t_cond_c + 5.0) + jnp.clip(q_solar / 40.0, 5.0, 30.0),
                    t_amb_c + 2.0,
                ]
            )
            _, aux = jp.desorption_rhs(
                y_k, t_amb_c=t_amb_c, q_solar_w_m2=q_solar, h_amb=h_amb, thermal=thermal, mass=mass,
                h0_ref_m=h0_ref_m, h_fg_j_per_kg=h_fg_j_per_kg, fin_area_ratio=fin_area_ratio, x0_guess=x0_guess,
            )
            return aux[3]

        y0_abs = jnp.array([c_w_initial, jnp.maximum(h_initial, h0_ref_m)])
        sol_abs = diffrax.diffeqsolve(
            diffrax.ODETerm(abs_vf), diffrax.Tsit5(), t0=0.0, t1=dt * n_abs_max, dt0=dt, y0=y0_abs, args=None,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-7, dtmax=dt),
            max_steps=16384, adjoint=diffrax.DirectAdjoint(), throw=False,
        )
        c_w_mid = jnp.clip(sol_abs.ys[0, 0], jp.C_W_MIN_MOL_M3, jp.C_W_MAX_MOL_M3)
        h_mid = jnp.clip(sol_abs.ys[0, 1], h0_ref_m, h_max_m)

        t_cond0 = jnp.clip(t_amb_des[0], -40.0, 120.0)
        y0_des = jnp.array([c_w_mid, h_mid, t_cond0])
        sol_des = diffrax.diffeqsolve(
            diffrax.ODETerm(des_vf), diffrax.Tsit5(), t0=0.0, t1=dt * n_des_max, dt0=dt, y0=y0_des, args=None,
            saveat=diffrax.SaveAt(ts=t_eval_des),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-7, dtmax=dt),
            max_steps=16384, adjoint=diffrax.DirectAdjoint(), throw=False,
        )
        ys_des = sol_des.ys
        idx_arr = jnp.clip((t_eval_des / dt).astype(jnp.int32), 0, n_des_max - 1)
        m_des_hist = jax.vmap(aux_at)(ys_des, idx_arr)
        m_des_hist = jnp.where(idx_arr < n_des_real, m_des_hist, 0.0)
        water = jnp.maximum(0.0, jnp.trapezoid(m_des_hist, dx=dt))

        solar_sum = jnp.sum(jnp.where(jnp.arange(n_des_max) < n_des_real, solar_des, 0.0)) * dt
        eta = jnp.where(solar_sum > 0.0, water * h_fg_j_per_kg / solar_sum, 0.0)

        c_w_end = jnp.clip(ys_des[-1, 0], jp.C_W_MIN_MOL_M3, jp.C_W_MAX_MOL_M3)
        h_end = jnp.maximum(ys_des[-1, 1], h0_ref_m)
        return water, eta, c_w_end, h_end

    in_axes = (0, 0) + (0,) * len(batch)  # c_w_initial, h_initial, then every batch dict value
    batched = jax.vmap(single, in_axes=in_axes)

    def fn(c_w_initial, h_initial):
        return batched(c_w_initial, h_initial, *batch.values())

    return jax.jit(fn)


def find_cyclic_state_batched(
    daily_cycle_fn, *, c_w_initial, h_initial, tol=1e-6, max_rounds=8,
):
    """Batched Aitken Delta^2 search using the handoff doc's "fixed number of
    rounds for every instance" strategy: every instance in the batch runs the
    same max_rounds (no early exit -- that's what makes this vmap-friendly),
    then a single vectorized final pass decides, per instance, whether the
    last round's extrapolated state is trustworthy (rel_step < tol) or whether
    to fall back to averaging the last two rounds (covers both slow-but-real
    convergence and period-2 stalls, per ode_system.py::find_cyclic_state's own
    fallback) -- a simplified, vectorized stand-in for that function's
    multi-round stall counter, not a byte-for-byte port of it.
    """
    c_w_initial = np.asarray(c_w_initial, dtype=float)
    h_initial = np.asarray(h_initial, dtype=float)
    x = np.stack([c_w_initial, h_initial], axis=1)

    def step(state):
        _, _, cw_end, h_end = daily_cycle_fn(jnp.asarray(state[:, 0]), jnp.asarray(state[:, 1]))
        return np.stack([np.asarray(cw_end), np.asarray(h_end)], axis=1)

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
