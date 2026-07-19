#!/usr/bin/env python3
"""Cross-check the JAX desorption RHS against the CPU (scipy) implementation.

Step 1 of validating the GPU-sweep prototype (see docs/gpu_sweep_handoff.md):
before trusting any integrated trajectory, confirm the pointwise RHS agrees at
a handful of representative (c_w, H, T_cond, T_amb, Q_solar) states spanning
the physically relevant range for the Atacama baseline device.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from solar_lumped.physics.device_balances import DeviceThermalParams  # noqa: E402
from solar_lumped.physics.sorbent import initial_loading  # noqa: E402
from solar_lumped.simulation.coupled_dynamics import evaluate_coupled_rates  # noqa: E402
from solar_lumped.simulation.device_config import DeviceConfig  # noqa: E402
from solar_lumped.simulation.ode_system import _integrate_absorption, _integrate_desorption  # noqa: E402
from solar_lumped.weather.client import WeatherClient  # noqa: E402
from solar_lumped.weather.climate import representative_mean_day_df  # noqa: E402
from solar_lumped.weather.profiles import profile_from_day_df  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax_physics as jp  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def cpu_rhs(config: DeviceConfig, *, c_w, h_m, t_cond_c, t_amb_c, q_solar_w_m2, h_amb):
    mass = config.mass_params()
    thermal = config.thermal_params()
    rates = evaluate_coupled_rates(
        c_w=c_w,
        h_m=h_m,
        t_cond_c=t_cond_c,
        t_amb_c=t_amb_c,
        rh=0.0,
        q_solar_w_m2=q_solar_w_m2,
        h_amb=h_amb,
        phase="desorption",
        mass=mass,
        thermal=thermal,
        vapor_gap_m=config.vapor_gap_m,
        condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
        fin_area_ratio=config.fin_area_ratio,
        h_fg_j_per_kg=config.h_fg_j_per_kg,
        config=config,
        t_guess=None,
    )
    dc = min(0.0, rates.dc_w_dt)
    dh = rates.dH_dt if h_m > config.hydrogel_thickness_m + 1e-12 else 0.0
    dh = min(0.0, dh)
    return (
        np.array([dc, dh, rates.dT_cond_dt]),
        (rates.t_gel_c, rates.thermal.t_abs_c, rates.thermal.t_glass_c, rates.m_des_kg_s_m2),
    )


def jax_rhs(config: DeviceConfig, *, c_w, h_m, t_cond_c, t_amb_c, q_solar_w_m2, h_amb, x0_guess):
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
    y = jnp.array([c_w, h_m, t_cond_c])
    dy, aux = jp.desorption_rhs(
        y,
        t_amb_c=t_amb_c,
        q_solar_w_m2=q_solar_w_m2,
        h_amb=h_amb,
        thermal=thermal,
        mass=mass,
        h0_ref_m=config.hydrogel_thickness_m,
        h_fg_j_per_kg=config.h_fg_j_per_kg,
        fin_area_ratio=config.fin_area_ratio,
        x0_guess=jnp.array(x0_guess),
    )
    return np.array(dy), tuple(float(v) for v in aux)


def cpu_absorption_rhs(config: DeviceConfig, *, c_w, h_m, t_amb_c, rh):
    mass = config.mass_params()
    rates = evaluate_coupled_rates(
        c_w=c_w, h_m=h_m, t_cond_c=0.0, t_amb_c=t_amb_c, rh=rh, q_solar_w_m2=0.0,
        h_amb=10.0, phase="absorption", mass=mass, thermal=config.thermal_params(),
        vapor_gap_m=config.vapor_gap_m, condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
        fin_area_ratio=config.fin_area_ratio, h_fg_j_per_kg=config.h_fg_j_per_kg, config=config, t_guess=None,
    )
    h_min = config.hydrogel_thickness_m
    dh = rates.dH_dt if h_m > h_min + 1e-12 else max(0.0, rates.dH_dt)
    return np.array([rates.dc_w_dt, dh])


def jax_absorption_rhs(config: DeviceConfig, *, c_w, h_m, t_amb_c, rh):
    mass_p = config.mass_params()
    mass = jp.MassParams(
        h0_ref_m=config.hydrogel_thickness_m, vapor_gap_m=config.vapor_gap_m, tilt_deg=config.tilt_deg,
        c_s_mol_m3=mass_p.c_s_mol_m3, formula_weight_g_mol=mass_p.formula_weight_g_mol,
        g_conv_m_s=mass_p.g_conv_m_s,
    )
    h_max_m = max(config.vapor_gap_m - jp.VAPOR_GAP_TRANSPORT_MIN_M, config.hydrogel_thickness_m + 1e-6)
    y = jnp.array([c_w, h_m])
    dy = jp.absorption_rhs(
        y, t_amb_c=t_amb_c, rh=rh, h0_ref_m=config.hydrogel_thickness_m, h_max_m=h_max_m,
        mass=mass, salt_to_polymer_ratio=config.salt_to_polymer_ratio,
    )
    return np.array(dy)


def _validate_absorption(config: DeviceConfig, profile) -> float:
    cw0 = initial_loading(config)
    abs_res = _integrate_absorption(cw0, config.hydrogel_thickness_m, profile.absorption, config)
    n = len(abs_res.time_s)
    sample_idx = sorted(set(np.linspace(1, n - 2, 6).astype(int)))
    dt_abs = profile.absorption.dt_s
    n_abs = len(profile.absorption.temperature_c)
    worst = 0.0
    for k in sample_idx:
        t = float(abs_res.time_s[k])
        i = min(int(t / dt_abs), n_abs - 1)
        state = dict(
            c_w=float(abs_res.c_w[k]),
            h_m=float(abs_res.H[k]),
            t_amb_c=profile.absorption.temperature_c[i],
            rh=profile.absorption.relative_humidity[i],
        )
        dy_cpu = cpu_absorption_rhs(config, **state)
        dy_jax = jax_absorption_rhs(config, **state)
        rel_err = np.abs(dy_jax - dy_cpu) / np.maximum(np.abs(dy_cpu), 1e-12)
        worst = max(worst, float(np.max(rel_err)))
        print(f"  [absorption] k={k:3d} t={t:6.0f}s dy_cpu={dy_cpu} dy_jax={dy_jax} rel_err={rel_err}")
    return worst


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


def main() -> int:
    print("Fetching Atacama annual-mean weather profile (cached)...", flush=True)
    profile = _atacama_annual_mean_profile()

    for eps_abs in (0.90, 0.95):
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
        print(f"\n=== eps_abs={eps_abs} ===")

        worst_abs = _validate_absorption(config, profile)
        print(f"  worst rel_err (absorption): {worst_abs:.3e}")

        cw0 = initial_loading(config)
        abs_res = _integrate_absorption(cw0, config.hydrogel_thickness_m, profile.absorption, config)
        des_res = _integrate_desorption(
            float(abs_res.c_w[-1]), float(abs_res.H[-1]), profile.desorption, config
        )
        n = len(des_res.time_s)
        sample_idx = sorted(set(np.linspace(1, n - 2, 6).astype(int)))

        n_des = len(profile.desorption.temperature_c)
        dt_des = profile.desorption.dt_s
        worst = 0.0
        for k in sample_idx:
            t = float(des_res.time_s[k])
            i = min(int(t / dt_des), n_des - 1)
            state = dict(
                c_w=float(des_res.c_w[k]),
                h_m=float(des_res.H[k]),
                t_cond_c=float(des_res.t_cond_c[k]),
                t_amb_c=profile.desorption.temperature_c[i],
                q_solar_w_m2=profile.desorption.solar_w_m2[i],
                h_amb=profile.desorption.h_amb_w_m2_k[i],
            )
            dy_cpu, aux_cpu = cpu_rhs(config, **state)
            x0_guess = (aux_cpu[0], aux_cpu[1], aux_cpu[2])
            dy_jax, aux_jax = jax_rhs(config, **state, x0_guess=x0_guess)

            rel_err = np.abs(dy_jax - dy_cpu) / np.maximum(np.abs(dy_cpu), 1e-12)
            worst = max(worst, float(np.max(rel_err)))
            print(
                f"  k={k:3d} t={t:6.0f}s  m_des cpu={aux_cpu[3]:.6e} jax={aux_jax[3]:.6e}\n"
                f"           dy_cpu={dy_cpu}\n"
                f"           dy_jax={dy_jax}\n"
                f"           rel_err={rel_err}"
            )
        print(f"  worst rel_err this config: {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
