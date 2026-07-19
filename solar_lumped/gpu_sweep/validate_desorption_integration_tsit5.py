#!/usr/bin/env python3
"""Step 2: integrate the JAX desorption RHS with diffrax over a full phase and
compare the resulting water yield against the CPU scipy.solve_ivp(Radau) result,
for the same (IC, weather profile) -- isolating the ODE-integration layer from
the Aitken steady-state search (still done on CPU here).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import diffrax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solar_lumped.physics.device_balances import DeviceThermalParams  # noqa: E402
from solar_lumped.physics.sorbent import initial_loading  # noqa: E402
from solar_lumped.simulation.device_config import DeviceConfig  # noqa: E402
from solar_lumped.simulation.ode_system import _integrate_absorption, _integrate_desorption  # noqa: E402
from solar_lumped.weather.client import WeatherClient  # noqa: E402
from solar_lumped.weather.climate import representative_mean_day_df  # noqa: E402
from solar_lumped.weather.profiles import profile_from_day_df  # noqa: E402

import jax_physics as jp  # noqa: E402


def _atacama_annual_mean_profile():
    client = WeatherClient(cache_dir=str(_REPO / ".weather_cache"))
    lat, lon = -23.6, -70.4
    start, end = "2024-01-01", "2024-12-31"
    try:
        _, df = client.get_historical_forecast_site_weather(lat, lon, start, end)
    except Exception:
        df = client.get_historical(lat, lon, start, end)
    ref_day = df.index[len(df) // 2].date()
    mean_day_df = representative_mean_day_df(df, reference_day=ref_day)
    return profile_from_day_df(mean_day_df)


def integrate_desorption_jax(c_w0, h0, profile, config: DeviceConfig):
    mass_p = config.mass_params()
    thermal_p = config.thermal_params()
    mass = jp.MassParams(
        h0_ref_m=config.hydrogel_thickness_m,
        vapor_gap_m=config.vapor_gap_m,
        tilt_deg=config.tilt_deg,
        c_s_mol_m3=mass_p.c_s_mol_m3,
        formula_weight_g_mol=mass_p.formula_weight_g_mol,
    )
    thermal = jp.ThermalParams(
        insulation_gap_m=thermal_p.insulation_gap_m,
        vapor_gap_m=config.vapor_gap_m,
        eps_abs=thermal_p.eps_abs,
        tau_glass=thermal_p.tau_glass,
        tilt_deg=config.tilt_deg,
    )
    h0_ref_m = config.hydrogel_thickness_m
    h_fg = config.h_fg_j_per_kg
    fin_area_ratio = config.fin_area_ratio
    dt = profile.dt_s
    n = len(profile.temperature_c)
    t_amb_arr = jnp.array(profile.temperature_c)
    solar_arr = jnp.array(profile.solar_w_m2)
    h_amb_arr = jnp.array(profile.h_amb_w_m2_k)
    t_cond0 = float(np.clip(profile.temperature_c[0], -40.0, 120.0))

    def idx(t):
        return jnp.clip((t / dt).astype(jnp.int32), 0, n - 1)

    def vector_field(t, y, args):
        i = idx(t)
        t_amb_c = t_amb_arr[i]
        q_solar = solar_arr[i]
        h_amb = h_amb_arr[i]
        t_cond_c = jnp.clip(y[2], -40.0, 120.0)
        x0_guess = jnp.array(
            [
                jnp.maximum(t_amb_c + 5.0, t_cond_c + 5.0),
                jnp.maximum(t_amb_c + 5.0, t_cond_c + 5.0) + jnp.clip(q_solar / 40.0, 5.0, 30.0),
                t_amb_c + 2.0,
            ]
        )
        dy, _aux = jp.desorption_rhs(
            y,
            t_amb_c=t_amb_c,
            q_solar_w_m2=q_solar,
            h_amb=h_amb,
            thermal=thermal,
            mass=mass,
            h0_ref_m=h0_ref_m,
            h_fg_j_per_kg=h_fg,
            fin_area_ratio=fin_area_ratio,
            x0_guess=x0_guess,
        )
        return dy

    y0 = jnp.array([c_w0, max(h0, h0_ref_m), t_cond0])
    t_span = (0.0, dt * n)
    t_eval = jnp.linspace(0.0, t_span[1], n + 1)

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-7, dtmax=dt)
    saveat = diffrax.SaveAt(ts=t_eval)

    def _solve(y0):
        return diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_span[1],
            dt0=dt,
            y0=y0,
            args=None,
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=200_000,
            adjoint=diffrax.DirectAdjoint(),
        )

    jitted_solve = jax.jit(_solve)
    t0 = time.perf_counter()
    sol = jitted_solve(y0)
    jax.block_until_ready(sol.ys)
    compile_and_first_run = time.perf_counter() - t0

    t0 = time.perf_counter()
    sol = jitted_solve(y0)
    jax.block_until_ready(sol.ys)
    warm_run = time.perf_counter() - t0
    elapsed = f"compile+run={compile_and_first_run:.2f}s warm={warm_run:.3f}s"

    ys = sol.ys  # (n+1, 3)
    idx_arr = jnp.clip((t_eval / dt).astype(jnp.int32), 0, n - 1)

    def _aux_at(y_k, i):
        t_amb_c = t_amb_arr[i]
        q_solar = solar_arr[i]
        h_amb = h_amb_arr[i]
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

    m_des_hist = jax.jit(jax.vmap(_aux_at))(ys, idx_arr)
    m_des_hist = np.asarray(m_des_hist)

    water = 0.0
    for k in range(n):
        water += 0.5 * (m_des_hist[k] + m_des_hist[k + 1]) * dt
    water = max(0.0, water)
    return water, elapsed, sol.stats


def main() -> int:
    print("Fetching Atacama annual-mean weather profile (cached)...", flush=True)
    profile = _atacama_annual_mean_profile()

    for eps_abs, ref_yield in ((0.90, 1.707476), (0.95, 1.800478)):
        config = DeviceConfig(
            tilt_deg=35.0,
            fin_area_ratio=7.1,
            thermal=DeviceThermalParams(
                insulation_gap_m=0.005,
                vapor_gap_m=0.04,
                eps_abs=eps_abs,
                tau_glass=0.85,
                tilt_deg=35.0,
            ),
        )
        cw0 = initial_loading(config)
        abs_res = _integrate_absorption(cw0, config.hydrogel_thickness_m, profile.absorption, config)
        c_w_start = float(abs_res.c_w[-1])
        h_start = float(abs_res.H[-1])

        des_res_cpu = _integrate_desorption(c_w_start, h_start, profile.desorption, config)
        cpu_yield = des_res_cpu.water_collected_kg_m2

        jax_yield, elapsed, stats = integrate_desorption_jax(
            c_w_start, h_start, profile.desorption, config
        )

        print(f"\n=== eps_abs={eps_abs} (single mean-day, not the 12-month reference) ===")
        print(f"  CPU (Radau)   yield: {cpu_yield:.6f} kg/m^2")
        print(f"  JAX (Tsit5) yield: {jax_yield:.6f} kg/m^2   ({elapsed}, stats={stats})")
        print(f"  Wilson 12-month reference (not directly comparable, see note): {ref_yield:.6f} kg/m^2")
        rel_diff = abs(jax_yield - cpu_yield) / max(abs(cpu_yield), 1e-9)
        print(f"  JAX vs CPU relative difference: {rel_diff:.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
