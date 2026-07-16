"""SciPy Radau integration for two-bed half-cycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from scipy.integrate import solve_ivp

from waste_heat_cycle_lumped_no_loop.physics import device_defaults as dd
from waste_heat_cycle_lumped_no_loop.physics.contactor_balances import ThermalEnvironment
from waste_heat_cycle_lumped_no_loop.physics.mass_transfer import rh_outside_desorber
from waste_heat_cycle_lumped_no_loop.physics.sorbent import (
    initial_bed_states,
    is_hydrogel,
    mass_state_size,
    water_kg_m2_bed,
)
from waste_heat_cycle_lumped_no_loop.simulation.control import (
    ControllerState,
    advance_controller_integrals,
    reset_controller_state,
)
from waste_heat_cycle_lumped_no_loop.simulation.coupled_dynamics import (
    controls_for_state,
    evaluate_coupled_rates,
)
from waste_heat_cycle_lumped_no_loop.simulation.device_config import DeviceConfig
from waste_heat_cycle_lumped_no_loop.weather.profiles import HalfCycleProfile

_ODE_RTOL = 1e-4
_ODE_ATOL = 1e-7
_INVENTORY_SAMPLE_DT_S = 6.0

# loading_a, loading_d, h_a, h_d, t_a, t_d, t_cond
CycleState: TypeAlias = tuple[float, float, float | None, float | None, float, float, float]


@dataclass
class HalfCycleResult:
    time_s: np.ndarray
    q_a: np.ndarray
    q_d: np.ndarray
    t_a_c: np.ndarray
    t_d_c: np.ndarray
    t_cond_c: np.ndarray
    m_ads_kg_s_m2: np.ndarray
    m_des_kg_s_m2: np.ndarray
    water_collected_kg_m2: float
    integral_ads_kg_m2: float
    integral_des_kg_m2: float
    h_a: np.ndarray | None = None
    h_d: np.ndarray | None = None


@dataclass
class CycleResult:
    half_a: HalfCycleResult
    half_b: HalfCycleResult
    water_collected_kg_m2: float


def _env_at(profile: HalfCycleProfile, i: int) -> ThermalEnvironment:
    return ThermalEnvironment(
        t_amb_c=profile.temperature_c[i],
        rh_amb=profile.relative_humidity[i],
        h_amb_w_m2_k=profile.h_amb_w_m2_k[i],
        t_wh_in_c=profile.t_wh_in_c[i],
        m_dot_wh_kg_s_m2=profile.m_dot_wh_kg_s_m2[i],
    )


def _pack_y0(
    config: DeviceConfig,
    *,
    loading_a: float,
    loading_d: float,
    h_a: float,
    h_d: float,
    t_a: float,
    t_d: float,
    t_cond: float,
) -> np.ndarray:
    if is_hydrogel(config):
        return np.array([loading_a, h_a, loading_d, h_d, t_a, t_d, t_cond], dtype=float)
    return np.array([loading_a, loading_d, t_a, t_d, t_cond], dtype=float)


def _initial_state(
    config: DeviceConfig,
    *,
    loading_a0: float | None,
    loading_d0: float | None,
    h_a0: float | None,
    h_d0: float | None,
    t_a0: float | None,
    t_d0: float | None,
) -> tuple[float, float, float, float, float, float, float]:
    bed_a, bed_d = initial_bed_states(config)
    loading_a = loading_a0 if loading_a0 is not None else bed_a.loading
    loading_d = loading_d0 if loading_d0 is not None else bed_d.loading
    h_a = h_a0 if h_a0 is not None else (bed_a.h_m or config.hydrogel_thickness_m)
    h_d = h_d0 if h_d0 is not None else (bed_d.h_m or config.hydrogel_thickness_m)
    t_amb = dd.T_AMB_C
    t_a = t_a0 if t_a0 is not None else t_amb
    t_d = t_d0 if t_d0 is not None else t_amb + 5.0
    t_cond = t_amb
    return loading_a, loading_d, h_a, h_d, t_a, t_d, t_cond


def _run_one_cycle(
    profile: HalfCycleProfile,
    config: DeviceConfig,
    state: CycleState,
) -> tuple[CycleResult, CycleState]:
    la, ld, ha, hd, ta, td, tc = state
    half_a = run_half_cycle(
        profile,
        config,
        loading_a0=la,
        loading_d0=ld,
        h_a0=ha,
        h_d0=hd,
        t_a0=ta,
        t_d0=td,
        t_cond0=tc,
    )
    la, ld, ha, hd, ta, td, tc = swap_roles(half_a, config)
    half_b = run_half_cycle(
        profile,
        config,
        loading_a0=la,
        loading_d0=ld,
        h_a0=ha,
        h_d0=hd,
        t_a0=ta,
        t_d0=td,
        t_cond0=tc,
    )
    water = half_a.water_collected_kg_m2 + half_b.water_collected_kg_m2
    cyc = CycleResult(half_a=half_a, half_b=half_b, water_collected_kg_m2=water)
    la, ld, ha, hd, ta, td, tc = swap_roles(half_b, config)
    return cyc, (la, ld, ha, hd, ta, td, tc)


def _clip_mass_state(y: np.ndarray, config: DeviceConfig) -> np.ndarray:
    out = y.copy()
    if is_hydrogel(config):
        h_min = config.hydrogel_thickness_m
        out[1] = max(float(out[1]), h_min)
        out[3] = max(float(out[3]), h_min)
        if out[1] > h_min + 1e-12 and float(out[0]) >= dd.C_W_MIN_HYDROGEL:
            pass
        elif float(out[1]) <= h_min + 1e-12:
            out[1] = h_min
    return out


def _unpack_half_result(y_stack: np.ndarray, config: DeviceConfig) -> dict:
    if is_hydrogel(config):
        return {
            "q_a": y_stack[:, 0],
            "h_a": y_stack[:, 1],
            "q_d": y_stack[:, 2],
            "h_d": y_stack[:, 3],
            "t_a_c": y_stack[:, 4],
            "t_d_c": y_stack[:, 5],
            "t_cond_c": y_stack[:, 6],
        }
    return {
        "q_a": y_stack[:, 0],
        "q_d": y_stack[:, 1],
        "h_a": None,
        "h_d": None,
        "t_a_c": y_stack[:, 2],
        "t_d_c": y_stack[:, 3],
        "t_cond_c": y_stack[:, 4],
    }


def _mass_offset(config: DeviceConfig) -> int:
    return mass_state_size(config)


def _desorber_rh_from_state(y: np.ndarray, config: DeviceConfig) -> float:
    n_mass = _mass_offset(config)
    t_d_c = float(y[n_mass + 1])
    t_cond_c = float(y[n_mass + 2])
    return rh_outside_desorber(t_d_c, t_cond_c)


def _half_cycle_complete(y: np.ndarray, config: DeviceConfig) -> bool:
    return _desorber_rh_from_state(y, config) <= config.rh_desorber_switch


def _step_t_eval(t0: float, t1: float) -> np.ndarray:
    n_pts = max(2, int(round((t1 - t0) / _INVENTORY_SAMPLE_DT_S)) + 1)
    return np.linspace(t0, t1, n_pts)


def _record_half_cycle_state(
    *,
    t_s: float,
    y: np.ndarray,
    env: ThermalEnvironment,
    config: DeviceConfig,
    ctrl: ControllerState,
    times: list[float],
    ys: list[np.ndarray],
    m_ads_series: list[float],
    m_des_series: list[float],
    n_mass: int,
) -> None:
    if times and abs(t_s - times[-1]) < 1e-9:
        return
    y = _clip_mass_state(y, config)
    controls = controls_for_state(
        mass_state=y[:n_mass],
        t_a_c=float(y[n_mass]),
        t_d_c=float(y[n_mass + 1]),
        t_cond_c=float(y[n_mass + 2]),
        env=env,
        config=config,
        integral_ads_kg_m2=ctrl.integral_ads_kg_m2,
        integral_des_kg_m2=ctrl.integral_des_kg_m2,
    )
    rates = evaluate_coupled_rates(
        mass_state=y[:n_mass],
        t_a_c=float(y[n_mass]),
        t_d_c=float(y[n_mass + 1]),
        t_cond_c=float(y[n_mass + 2]),
        env=env,
        config=config,
        controls=controls,
    )
    times.append(float(t_s))
    ys.append(y.copy())
    m_ads_series.append(rates.m_ads_kg_s_m2)
    m_des_series.append(rates.m_des_kg_s_m2)


def _integrate_desorption_kg_m2(times: list[float], m_des_series: list[float]) -> float:
    water = 0.0
    for k in range(len(times) - 1):
        dt_k = times[k + 1] - times[k]
        water += 0.5 * (m_des_series[k] + m_des_series[k + 1]) * dt_k
    return max(0.0, water)


def run_half_cycle(
    profile: HalfCycleProfile,
    config: DeviceConfig,
    *,
    loading_a0: float,
    loading_d0: float,
    h_a0: float | None = None,
    h_d0: float | None = None,
    t_a0: float,
    t_d0: float,
    t_cond0: float,
    controller_state: ControllerState | None = None,
) -> HalfCycleResult:
    """Integrate one half-cycle: A adsorbs, B desorbs."""
    n = len(profile.temperature_c)
    dt = profile.dt_s
    ctrl = controller_state if controller_state is not None else ControllerState()
    reset_controller_state(ctrl)
    h_a = h_a0 if h_a0 is not None else config.hydrogel_thickness_m
    h_d = h_d0 if h_d0 is not None else config.hydrogel_thickness_m

    y = _pack_y0(
        config,
        loading_a=loading_a0,
        loading_d=loading_d0,
        h_a=h_a,
        h_d=h_d,
        t_a=t_a0,
        t_d=t_d0,
        t_cond=t_cond0,
    )
    times: list[float] = []
    ys: list[np.ndarray] = []
    m_ads_series: list[float] = []
    m_des_series: list[float] = []
    n_mass = _mass_offset(config)
    env0 = _env_at(profile, 0)
    _record_half_cycle_state(
        t_s=0.0,
        y=y,
        env=env0,
        config=config,
        ctrl=ctrl,
        times=times,
        ys=ys,
        m_ads_series=m_ads_series,
        m_des_series=m_des_series,
        n_mass=n_mass,
    )

    for i in range(n):
        env = _env_at(profile, i)
        t0 = i * dt
        t1 = (i + 1) * dt

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            state = _clip_mass_state(state, config)
            controls = controls_for_state(
                mass_state=state[:n_mass],
                t_a_c=float(state[n_mass]),
                t_d_c=float(state[n_mass + 1]),
                t_cond_c=float(state[n_mass + 2]),
                env=env,
                config=config,
                integral_ads_kg_m2=ctrl.integral_ads_kg_m2,
                integral_des_kg_m2=ctrl.integral_des_kg_m2,
            )
            rates = evaluate_coupled_rates(
                mass_state=state[:n_mass],
                t_a_c=float(state[n_mass]),
                t_d_c=float(state[n_mass + 1]),
                t_cond_c=float(state[n_mass + 2]),
                env=env,
                config=config,
                controls=controls,
            )
            dy = np.concatenate([rates.dy_mass, np.array(
                [rates.dT_a_dt, rates.dT_d_dt, rates.dT_cond_dt]
            )])
            if is_hydrogel(config):
                h_min = config.hydrogel_thickness_m
                if float(state[1]) <= h_min + 1e-12:
                    dy[1] = max(0.0, dy[1])
            return dy

        y = _clip_mass_state(y, config)
        controls0 = controls_for_state(
            mass_state=y[:n_mass],
            t_a_c=float(y[n_mass]),
            t_d_c=float(y[n_mass + 1]),
            t_cond_c=float(y[n_mass + 2]),
            env=env,
            config=config,
            integral_ads_kg_m2=ctrl.integral_ads_kg_m2,
            integral_des_kg_m2=ctrl.integral_des_kg_m2,
        )
        rates0 = evaluate_coupled_rates(
            mass_state=y[:n_mass],
            t_a_c=float(y[n_mass]),
            t_d_c=float(y[n_mass + 1]),
            t_cond_c=float(y[n_mass + 2]),
            env=env,
            config=config,
            controls=controls0,
        )

        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0=y,
            method="Radau",
            t_eval=_step_t_eval(t0, t1),
            max_step=dt,
            rtol=_ODE_RTOL,
            atol=_ODE_ATOL,
        )
        if not sol.success:
            raise RuntimeError(f"Half-cycle step {i} failed: {sol.message}")

        for k in range(len(sol.t)):
            _record_half_cycle_state(
                t_s=float(sol.t[k]),
                y=sol.y[:, k],
                env=env,
                config=config,
                ctrl=ctrl,
                times=times,
                ys=ys,
                m_ads_series=m_ads_series,
                m_des_series=m_des_series,
                n_mass=n_mass,
            )

        y = _clip_mass_state(sol.y[:, -1], config)
        controls1 = controls_for_state(
            mass_state=y[:n_mass],
            t_a_c=float(y[n_mass]),
            t_d_c=float(y[n_mass + 1]),
            t_cond_c=float(y[n_mass + 2]),
            env=env,
            config=config,
            integral_ads_kg_m2=ctrl.integral_ads_kg_m2,
            integral_des_kg_m2=ctrl.integral_des_kg_m2,
        )
        rates1 = evaluate_coupled_rates(
            mass_state=y[:n_mass],
            t_a_c=float(y[n_mass]),
            t_d_c=float(y[n_mass + 1]),
            t_cond_c=float(y[n_mass + 2]),
            env=env,
            config=config,
            controls=controls1,
        )
        advance_controller_integrals(
            ctrl,
            m_ads_kg_s_m2=0.5 * (rates0.m_ads_kg_s_m2 + rates1.m_ads_kg_s_m2),
            m_des_kg_s_m2=0.5 * (rates0.m_des_kg_s_m2 + rates1.m_des_kg_s_m2),
            dt_s=dt,
        )
        if _half_cycle_complete(y, config):
            break

    y_stack = np.array(ys)
    t_arr = np.array(times)
    water = _integrate_desorption_kg_m2(times, m_des_series)
    unpacked = _unpack_half_result(y_stack, config)

    return HalfCycleResult(
        time_s=t_arr,
        m_ads_kg_s_m2=np.array(m_ads_series),
        m_des_kg_s_m2=np.array(m_des_series),
        water_collected_kg_m2=water,
        integral_ads_kg_m2=ctrl.integral_ads_kg_m2,
        integral_des_kg_m2=ctrl.integral_des_kg_m2,
        **unpacked,
    )


def swap_roles(
    res: HalfCycleResult,
    config: DeviceConfig,
) -> tuple[float, float, float | None, float | None, float, float, float]:
    """After half-cycle: bed that adsorbed now desorbs (swap loading, H, and T)."""
    loading_a = float(res.q_d[-1])
    loading_d = float(res.q_a[-1])
    h_a = float(res.h_d[-1]) if res.h_d is not None else None
    h_d = float(res.h_a[-1]) if res.h_a is not None else None
    t_a = float(res.t_d_c[-1])
    t_d = float(res.t_a_c[-1])
    t_cond = float(res.t_cond_c[-1])
    return loading_a, loading_d, h_a, h_d, t_a, t_d, t_cond


def run_cycle(
    profile: HalfCycleProfile,
    config: DeviceConfig,
    *,
    loading_a0: float | None = None,
    loading_d0: float | None = None,
    h_a0: float | None = None,
    h_d0: float | None = None,
    t_a0: float | None = None,
    t_d0: float | None = None,
    warmup_cycles: int = 0,
) -> CycleResult:
    state = _initial_state(
        config,
        loading_a0=loading_a0,
        loading_d0=loading_d0,
        h_a0=h_a0,
        h_d0=h_d0,
        t_a0=t_a0,
        t_d0=t_d0,
    )
    for _ in range(warmup_cycles):
        _, state = _run_one_cycle(profile, config, state)
    cyc, _ = _run_one_cycle(profile, config, state)
    return cyc


def run_daily_operation(
    profile: HalfCycleProfile,
    config: DeviceConfig,
    *,
    n_cycles: int | None = None,
    loading_a0: float | None = None,
    loading_d0: float | None = None,
    warmup_cycles: int = 0,
) -> tuple[float, float, list[CycleResult]]:
    state = _initial_state(
        config,
        loading_a0=loading_a0,
        loading_d0=loading_d0,
        h_a0=None,
        h_d0=None,
        t_a0=None,
        t_d0=None,
    )
    for _ in range(warmup_cycles):
        _, state = _run_one_cycle(profile, config, state)

    results: list[CycleResult] = []
    total_water = 0.0
    q_wh_total = 0.0
    elapsed_s = 0.0
    day_s = 86400.0

    def _integrate_wh_energy(half: HalfCycleResult) -> None:
        nonlocal q_wh_total
        n_steps = max(0, len(half.time_s) - 1)
        dt = profile.dt_s
        q_wh_step = (
            profile.m_dot_wh_kg_s_m2[0]
            * dd.CP_WH_J_KG_K
            * max(0.0, profile.t_wh_in_c[0] - dd.T_AMB_C)
            * dt
            * 2.0
        )
        q_wh_total += q_wh_step * n_steps

    cycle_count = 0
    while True:
        if n_cycles is not None and cycle_count >= n_cycles:
            break
        if n_cycles is None and elapsed_s >= day_s - 1e-9:
            break

        cyc, state = _run_one_cycle(profile, config, state)
        _integrate_wh_energy(cyc.half_a)
        _integrate_wh_energy(cyc.half_b)
        results.append(cyc)
        total_water += cyc.water_collected_kg_m2
        elapsed_s += float(cyc.half_a.time_s[-1]) + float(cyc.half_b.time_s[-1])
        cycle_count += 1

    eta = (total_water * config.thermal_params().h_fg_j_per_kg / q_wh_total) if q_wh_total > 0 else 0.0
    return total_water, eta, results


def loading_kg_m2(loading: float, config: DeviceConfig, *, h_m: float | None = None) -> float:
    return water_kg_m2_bed(loading, config=config, h_m=h_m)
