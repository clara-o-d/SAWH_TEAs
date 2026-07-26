"""Device configuration, feedback control, coupled two-bed dynamics, SciPy ODE
integration, operating-hours accounting, water-inventory and detailed-plot time
series, and annual-yield aggregation for the two-bed waste-heat SAWH.

Consolidated from the former simulation/{device_config, control, coupled_dynamics,
ode_system, operation_hours, water_inventory, detailed_plots, annual_yield}.py.
Section headers below mark each former module's boundary for traceability.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from waste_heat_cycle_lumped.economics import (
    ParasiticLoadOptions,
    SpecificEnergyBreakdown,
    specific_energy_breakdown_from_daily_operation,
)
from waste_heat_cycle_lumped.physics import (
    CP_WH_J_KG_K,
    C_VAC_BASE_KG_S_PA_M2,
    C_VAC_MAX_KG_S_PA_M2,
    C_VAC_MIN_KG_S_PA_M2,
    C_W_MIN_HYDROGEL,
    DEFAULT_MOF_NAME,
    DEFAULT_SALT_NAME,
    DEFAULT_SORBENT,
    G_CHAMBER_M_S,
    H0_M,
    H_AMB_W_M2_K,
    K_M_PER_KG_M2,
    K_P_PER_KG_S_M2,
    K_T_PER_K,
    M_F_BASE_KG_S_M2,
    M_F_MAX_KG_S_M2,
    M_F_MIN_KG_S_M2,
    M_WH_KG_S_M2,
    P_COND_PA,
    RHO_COMPOSITE_KG_M3,
    RH_AMB,
    RH_DESORBER_SWITCH,
    SALT_TO_POLYMER_RATIO,
    TAU_HALF_S,
    TILT_DEG,
    T_AMB_C,
    T_WH_IN_C,
    VAPOR_GAP_M,
    ContactorThermalParams,
    MofProperties,
    ThermalEnvironment,
    clamp_temperature_c,
    dT_a_dt,
    dT_cond_dt,
    dT_d_dt,
    dT_f_dt,
    fluxes_for_control,
    get_mof,
    h_ads_j_per_kg,
    h_des_j_per_kg,
    initial_bed_states,
    inventory_column,
    inventory_ylabel,
    is_hydrogel,
    loop_heat_fluxes,
    mass_rates,
    mass_state_size,
    rh_outside_desorber,
    water_in_gel_l_m2,
    water_kg_m2_bed,
)
from waste_heat_cycle_lumped.weather import HalfCycleProfile

SorbentKind = Literal["hydrogel", "mof"]


@dataclass(frozen=True, slots=True)
class ControllerParams:
    m_f_base_kg_s_m2: float = M_F_BASE_KG_S_M2
    m_f_min_kg_s_m2: float = M_F_MIN_KG_S_M2
    m_f_max_kg_s_m2: float = M_F_MAX_KG_S_M2
    c_vac_base_kg_s_pa_m2: float = C_VAC_BASE_KG_S_PA_M2
    c_vac_min_kg_s_pa_m2: float = C_VAC_MIN_KG_S_PA_M2
    c_vac_max_kg_s_pa_m2: float = C_VAC_MAX_KG_S_PA_M2
    k_t_per_k: float = K_T_PER_K
    k_m_per_kg_m2: float = K_M_PER_KG_M2
    k_p_per_kg_s_m2: float = K_P_PER_KG_S_M2


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    sorbent: SorbentKind = DEFAULT_SORBENT  # type: ignore[assignment]
    salt_name: str = DEFAULT_SALT_NAME
    salt_to_polymer_ratio: float = SALT_TO_POLYMER_RATIO
    hydrogel_thickness_m: float = H0_M
    hydrogel_density_kg_m3: float = RHO_COMPOSITE_KG_M3
    g_conv_m_s: float = G_CHAMBER_M_S
    vapor_gap_m: float = VAPOR_GAP_M
    tilt_deg: float = TILT_DEG
    mof_name: str = DEFAULT_MOF_NAME
    tau_half_s: float = TAU_HALF_S
    rh_desorber_switch: float = RH_DESORBER_SWITCH
    p_cond_pa: float = P_COND_PA
    controller: ControllerParams | None = None
    thermal: ContactorThermalParams | None = None

    def mof(self) -> MofProperties:
        return get_mof(self.mof_name)

    def thermal_params(self) -> ContactorThermalParams:
        if self.thermal is not None:
            return self.thermal
        return ContactorThermalParams(p_vacuum_pa=self.p_cond_pa)

    def controller_params(self) -> ControllerParams:
        return self.controller if self.controller is not None else ControllerParams()

    def condenser_thermal_mass_j_m2_k(self) -> float:
        return self.thermal_params().condenser_thermal_mass_j_m2_k

    @classmethod
    def datacenter_baseline(cls, **overrides: object) -> DeviceConfig:
        return cls(**overrides)  # type: ignore[arg-type]

    @classmethod
    def mof_baseline(cls, **overrides: object) -> DeviceConfig:
        base = {"sorbent": "mof"}
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]
@dataclass
class ControllerState:
    integral_ads_kg_m2: float = 0.0
    integral_des_kg_m2: float = 0.0


@dataclass(frozen=True, slots=True)
class ControlOutputs:
    m_dot_f_kg_s_m2: float
    c_vac_kg_s_pa_m2: float


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_controls(
    *,
    t_a_c: float,
    t_d_c: float,
    m_ads_kg_s_m2: float,
    m_des_kg_s_m2: float,
    params: ControllerParams,
    integral_ads_kg_m2: float,
    integral_des_kg_m2: float,
) -> ControlOutputs:
    """Temperature matching → HTF flow; vacuum tracks adsorption-limited desorption.

    Half-cycles end when vapor-gap RH outside the desorber falls below the switch
    threshold. Instantaneous ṁ_ads = ṁ_des is enforced in ``sorbent.mass_rates``;
    vacuum conductance is scaled so the natural desorption capacity follows uptake.
    """
    del integral_ads_kg_m2, integral_des_kg_m2
    delta_t = t_a_c - t_d_c
    m_f = clip(
        params.m_f_base_kg_s_m2 * (1.0 + params.k_t_per_k * delta_t),
        params.m_f_min_kg_s_m2,
        params.m_f_max_kg_s_m2,
    )
    c_vac = clip(
        params.c_vac_base_kg_s_pa_m2
        * max(0.1, m_ads_kg_s_m2 / max(m_des_kg_s_m2, 1e-12)),
        params.c_vac_min_kg_s_pa_m2,
        params.c_vac_max_kg_s_pa_m2,
    )
    return ControlOutputs(m_dot_f_kg_s_m2=m_f, c_vac_kg_s_pa_m2=c_vac)


@dataclass(frozen=True, slots=True)
class CoupledRates:
    dy_mass: np.ndarray
    dT_a_dt: float
    dT_d_dt: float
    dT_f_dt: float
    dT_cond_dt: float
    m_ads_kg_s_m2: float
    m_des_kg_s_m2: float
    controls: ControlOutputs
    fluxes: object


def _parse_mass_state(
    state: np.ndarray,
    *,
    config: DeviceConfig,
) -> tuple[float, float, float, float]:
    if is_hydrogel(config):
        return float(state[0]), float(state[1]), float(state[2]), float(state[3])
    return float(state[0]), float(state[1]), config.hydrogel_thickness_m, config.hydrogel_thickness_m


def evaluate_coupled_rates(
    *,
    mass_state: np.ndarray,
    t_a_c: float,
    t_d_c: float,
    t_f_c: float,
    t_cond_c: float,
    env: ThermalEnvironment,
    config: DeviceConfig,
    controls: ControlOutputs,
) -> CoupledRates:
    """Rates for half-cycle: contactor A adsorbs, contactor B desorbs."""
    thermal = config.thermal_params()
    loading_a, h_a, loading_d, h_d = _parse_mass_state(mass_state, config=config)

    t_a = clamp_temperature_c(t_a_c)
    t_d = clamp_temperature_c(t_d_c)
    t_f = clamp_temperature_c(t_f_c)
    t_cond = clamp_temperature_c(t_cond_c)

    sorbent = mass_rates(
        loading_a=loading_a,
        loading_d=loading_d,
        h_a=h_a,
        h_d=h_d,
        t_a_c=t_a,
        t_d_c=t_d,
        t_cond_c=t_cond,
        rh_amb=env.rh_amb,
        c_vac_kg_s_pa_m2=controls.c_vac_kg_s_pa_m2,
        config=config,
    )

    if is_hydrogel(config):
        dy_mass = np.array(
            [sorbent.d_loading_a, sorbent.d_h_a, sorbent.d_loading_d, sorbent.d_h_d],
            dtype=float,
        )
    else:
        dy_mass = np.array([sorbent.d_loading_a, sorbent.d_loading_d], dtype=float)

    dta = dT_a_dt(
        t_a_c=t_a,
        t_f_c=t_f,
        m_ads_kg_s_m2=sorbent.m_ads_kg_s_m2,
        h_ads_j_per_kg=h_ads_j_per_kg(config),
        m_dot_f_kg_s_m2=controls.m_dot_f_kg_s_m2,
        params=thermal,
        env=env,
    )
    dtd = dT_d_dt(
        t_d_c=t_d,
        t_f_c=t_f,
        t_cond_c=t_cond,
        m_des_kg_s_m2=sorbent.m_des_kg_s_m2,
        h_des_j_per_kg=h_des_j_per_kg(config),
        m_dot_f_kg_s_m2=controls.m_dot_f_kg_s_m2,
        params=thermal,
        env=env,
    )
    dtf = dT_f_dt(
        t_a_c=t_a,
        t_d_c=t_d,
        t_f_c=t_f,
        m_dot_f_kg_s_m2=controls.m_dot_f_kg_s_m2,
        params=thermal,
        env=env,
    )
    dtcond = dT_cond_dt(
        t_d_c=t_d,
        t_cond_c=t_cond,
        t_amb_c=env.t_amb_c,
        m_des_kg_s_m2=sorbent.m_des_kg_s_m2,
        h_amb_w_m2_k=env.h_amb_w_m2_k,
        params=thermal,
    )
    fluxes = loop_heat_fluxes(
        t_a_c=t_a,
        t_d_c=t_d,
        t_f_c=t_f,
        m_dot_f_kg_s_m2=controls.m_dot_f_kg_s_m2,
        params=thermal,
        env=env,
    )

    return CoupledRates(
        dy_mass=dy_mass,
        dT_a_dt=dta,
        dT_d_dt=dtd,
        dT_f_dt=dtf,
        dT_cond_dt=dtcond,
        m_ads_kg_s_m2=sorbent.m_ads_kg_s_m2,
        m_des_kg_s_m2=sorbent.m_des_kg_s_m2,
        controls=controls,
        fluxes=fluxes,
    )


def controls_for_state(
    *,
    mass_state: np.ndarray,
    t_a_c: float,
    t_d_c: float,
    t_cond_c: float,
    env: ThermalEnvironment,
    config: DeviceConfig,
    integral_ads_kg_m2: float,
    integral_des_kg_m2: float,
) -> ControlOutputs:
    loading_a, h_a, loading_d, h_d = _parse_mass_state(mass_state, config=config)
    ctrl_p = config.controller_params()
    t_a = clamp_temperature_c(t_a_c)
    t_d = clamp_temperature_c(t_d_c)
    m_ads, m_des = fluxes_for_control(
        loading_a=loading_a,
        loading_d=loading_d,
        h_a=h_a,
        h_d=h_d,
        t_a_c=t_a,
        t_d_c=t_d,
        t_cond_c=t_cond_c,
        rh_amb=env.rh_amb,
        c_vac_kg_s_pa_m2=ctrl_p.c_vac_base_kg_s_pa_m2,
        config=config,
    )
    return compute_controls(
        t_a_c=t_a,
        t_d_c=t_d,
        m_ads_kg_s_m2=m_ads,
        m_des_kg_s_m2=m_des,
        params=ctrl_p,
        integral_ads_kg_m2=integral_ads_kg_m2,
        integral_des_kg_m2=integral_des_kg_m2,
    )
_ODE_RTOL = 1e-4
_ODE_ATOL = 1e-7
_INVENTORY_SAMPLE_DT_S = 6.0

# loading_a, loading_d, h_a, h_d, t_a, t_d, t_f, t_cond
CycleState: TypeAlias = tuple[float, float, float | None, float | None, float, float, float, float]


@dataclass
class HalfCycleResult:
    time_s: np.ndarray
    q_a: np.ndarray
    q_d: np.ndarray
    t_a_c: np.ndarray
    t_d_c: np.ndarray
    t_f_c: np.ndarray
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
    t_f: float,
    t_cond: float,
) -> np.ndarray:
    if is_hydrogel(config):
        return np.array([loading_a, h_a, loading_d, h_d, t_a, t_d, t_f, t_cond], dtype=float)
    return np.array([loading_a, loading_d, t_a, t_d, t_f, t_cond], dtype=float)


def _clip_mass_state(y: np.ndarray, config: DeviceConfig) -> np.ndarray:
    out = y.copy()
    if is_hydrogel(config):
        h_min = config.hydrogel_thickness_m
        out[1] = max(float(out[1]), h_min)
        out[3] = max(float(out[3]), h_min)
        if out[1] > h_min + 1e-12 and float(out[0]) >= C_W_MIN_HYDROGEL:
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
            "t_f_c": y_stack[:, 6],
            "t_cond_c": y_stack[:, 7],
        }
    return {
        "q_a": y_stack[:, 0],
        "q_d": y_stack[:, 1],
        "h_a": None,
        "h_d": None,
        "t_a_c": y_stack[:, 2],
        "t_d_c": y_stack[:, 3],
        "t_f_c": y_stack[:, 4],
        "t_cond_c": y_stack[:, 5],
    }


def _half_cycle_complete(y: np.ndarray, config: DeviceConfig) -> bool:
    n_mass = mass_state_size(config)
    rh = rh_outside_desorber(float(y[n_mass + 1]), float(y[n_mass + 3]))
    return rh <= config.rh_desorber_switch


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
        t_cond_c=float(y[n_mass + 3]),
        env=env,
        config=config,
        integral_ads_kg_m2=ctrl.integral_ads_kg_m2,
        integral_des_kg_m2=ctrl.integral_des_kg_m2,
    )
    rates = evaluate_coupled_rates(
        mass_state=y[:n_mass],
        t_a_c=float(y[n_mass]),
        t_d_c=float(y[n_mass + 1]),
        t_f_c=float(y[n_mass + 2]),
        t_cond_c=float(y[n_mass + 3]),
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
    t_f0: float,
    t_cond0: float,
    controller_state: ControllerState | None = None,
) -> HalfCycleResult:
    """Integrate one half-cycle: A adsorbs, B desorbs."""
    n = len(profile.temperature_c)
    dt = profile.dt_s
    ctrl = controller_state if controller_state is not None else ControllerState()
    ctrl.integral_ads_kg_m2 = 0.0
    ctrl.integral_des_kg_m2 = 0.0
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
        t_f=t_f0,
        t_cond=t_cond0,
    )
    times: list[float] = []
    ys: list[np.ndarray] = []
    m_ads_series: list[float] = []
    m_des_series: list[float] = []
    n_mass = mass_state_size(config)
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
                t_cond_c=float(state[n_mass + 3]),
                env=env,
                config=config,
                integral_ads_kg_m2=ctrl.integral_ads_kg_m2,
                integral_des_kg_m2=ctrl.integral_des_kg_m2,
            )
            rates = evaluate_coupled_rates(
                mass_state=state[:n_mass],
                t_a_c=float(state[n_mass]),
                t_d_c=float(state[n_mass + 1]),
                t_f_c=float(state[n_mass + 2]),
                t_cond_c=float(state[n_mass + 3]),
                env=env,
                config=config,
                controls=controls,
            )
            dy = np.concatenate([rates.dy_mass, np.array(
                [rates.dT_a_dt, rates.dT_d_dt, rates.dT_f_dt, rates.dT_cond_dt]
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
            t_cond_c=float(y[n_mass + 3]),
            env=env,
            config=config,
            integral_ads_kg_m2=ctrl.integral_ads_kg_m2,
            integral_des_kg_m2=ctrl.integral_des_kg_m2,
        )
        rates0 = evaluate_coupled_rates(
            mass_state=y[:n_mass],
            t_a_c=float(y[n_mass]),
            t_d_c=float(y[n_mass + 1]),
            t_f_c=float(y[n_mass + 2]),
            t_cond_c=float(y[n_mass + 3]),
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
            t_cond_c=float(y[n_mass + 3]),
            env=env,
            config=config,
            integral_ads_kg_m2=ctrl.integral_ads_kg_m2,
            integral_des_kg_m2=ctrl.integral_des_kg_m2,
        )
        rates1 = evaluate_coupled_rates(
            mass_state=y[:n_mass],
            t_a_c=float(y[n_mass]),
            t_d_c=float(y[n_mass + 1]),
            t_f_c=float(y[n_mass + 2]),
            t_cond_c=float(y[n_mass + 3]),
            env=env,
            config=config,
            controls=controls1,
        )
        ctrl.integral_ads_kg_m2 += max(0.0, 0.5 * (rates0.m_ads_kg_s_m2 + rates1.m_ads_kg_s_m2)) * dt
        ctrl.integral_des_kg_m2 += max(0.0, 0.5 * (rates0.m_des_kg_s_m2 + rates1.m_des_kg_s_m2)) * dt
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
) -> tuple[float, float, float | None, float | None, float, float, float, float]:
    """After half-cycle: bed that adsorbed now desorbs (swap loading, H, and T)."""
    loading_a = float(res.q_d[-1])
    loading_d = float(res.q_a[-1])
    h_a = float(res.h_d[-1]) if res.h_d is not None else None
    h_d = float(res.h_a[-1]) if res.h_a is not None else None
    t_a = float(res.t_d_c[-1])
    t_d = float(res.t_a_c[-1])
    t_f = float(res.t_f_c[-1])
    t_cond = float(res.t_cond_c[-1])
    return loading_a, loading_d, h_a, h_d, t_a, t_d, t_f, t_cond


def _state_to_vec(state: CycleState, config: DeviceConfig) -> np.ndarray:
    """Drop the h_a/h_d slots for MOF configs, where they're always None."""
    la, ld, ha, hd, ta, td, tf, tc = state
    if is_hydrogel(config):
        return np.array([la, ld, ha, hd, ta, td, tf, tc], dtype=float)
    return np.array([la, ld, ta, td, tf, tc], dtype=float)


def _vec_to_state(vec: np.ndarray, config: DeviceConfig) -> CycleState:
    if is_hydrogel(config):
        la, ld, ha, hd, ta, td, tf, tc = (float(v) for v in vec)
        return la, ld, ha, hd, ta, td, tf, tc
    la, ld, ta, td, tf, tc = (float(v) for v in vec)
    return la, ld, None, None, ta, td, tf, tc


def _initial_state(
    config: DeviceConfig,
    *,
    loading_a0: float | None,
    loading_d0: float | None,
    h_a0: float | None,
    h_d0: float | None,
    t_a0: float | None,
    t_d0: float | None,
) -> tuple[float, float, float, float, float, float, float, float]:
    bed_a, bed_d = initial_bed_states(config)
    loading_a = loading_a0 if loading_a0 is not None else bed_a.loading
    loading_d = loading_d0 if loading_d0 is not None else bed_d.loading
    h_a = h_a0 if h_a0 is not None else (bed_a.h_m or config.hydrogel_thickness_m)
    h_d = h_d0 if h_d0 is not None else (bed_d.h_m or config.hydrogel_thickness_m)
    t_amb = T_AMB_C
    t_a = t_a0 if t_a0 is not None else t_amb
    t_d = t_d0 if t_d0 is not None else t_amb + 5.0
    t_f = 0.5 * (t_a + t_d)
    t_cond = t_amb
    return loading_a, loading_d, h_a, h_d, t_a, t_d, t_f, t_cond


def _run_one_cycle(
    profile: HalfCycleProfile,
    config: DeviceConfig,
    state: CycleState,
) -> tuple[CycleResult, CycleState]:
    la, ld, ha, hd, ta, td, tf, tc = state
    half_a = run_half_cycle(
        profile,
        config,
        loading_a0=la,
        loading_d0=ld,
        h_a0=ha,
        h_d0=hd,
        t_a0=ta,
        t_d0=td,
        t_f0=tf,
        t_cond0=tc,
    )
    la, ld, ha, hd, ta, td, tf, tc = swap_roles(half_a, config)
    half_b = run_half_cycle(
        profile,
        config,
        loading_a0=la,
        loading_d0=ld,
        h_a0=ha,
        h_d0=hd,
        t_a0=ta,
        t_d0=td,
        t_f0=tf,
        t_cond0=tc,
    )
    water = half_a.water_collected_kg_m2 + half_b.water_collected_kg_m2
    cyc = CycleResult(half_a=half_a, half_b=half_b, water_collected_kg_m2=water)
    la, ld, ha, hd, ta, td, tf, tc = swap_roles(half_b, config)
    return cyc, (la, ld, ha, hd, ta, td, tf, tc)


def find_cyclic_state(
    profile: HalfCycleProfile,
    config: DeviceConfig,
    *,
    initial_state: CycleState | None = None,
    tol: float = 1e-6,
    max_rounds: int = 10,
    stall_ratio: float = 0.5,
    stall_rounds: int = 2,
    verbose: bool = True,
) -> CycleState:
    """Find the steady periodic post-cycle state for a profile repeated indefinitely,
    without brute-force warmup cycling.

    Plain fixed-point iteration (looping ``_run_one_cycle``) can need 100+ cycles
    to converge at sites where the one-cycle map's slowest eigenvalue is close to
    1. This instead accelerates convergence with restarted vector Aitken Δ²
    extrapolation: each round applies the real map twice, then extrapolates the
    fixed point from those two real evaluations, typically converging in ~3-6
    rounds. Same algorithm as solar_lumped/waste_heat_lumped's find_cyclic_state,
    generalized from the (c_w, H) pair to this device's 8-field cycle state
    (6 fields for MOF, which has no hydrogel thickness).

    Some (profile, config) pairs have no single fixed point: the one-cycle map
    bifurcates into a stable period-2 orbit, so ``rel_step`` plateaus instead of
    shrinking toward ``tol``. Detected as ``stall_rounds`` consecutive rounds
    where ``rel_step`` fails to shrink by at least ``stall_ratio`` relative to
    the previous round, handled by returning the average of the two most recent
    extrapolated states (also the fallback if ``max_rounds`` is exhausted).
    """
    if initial_state is None:
        initial_state = _initial_state(
            config,
            loading_a0=None,
            loading_d0=None,
            h_a0=None,
            h_d0=None,
            t_a0=None,
            t_d0=None,
        )
    x = _state_to_vec(initial_state, config)

    def step(vec: np.ndarray) -> np.ndarray:
        _, state = _run_one_cycle(profile, config, _vec_to_state(vec, config))
        return _state_to_vec(state, config)

    prev_rel_step: float | None = None
    prev_x_star: np.ndarray | None = None
    stall_count = 0
    for round_idx in range(1, max(1, max_rounds) + 1):
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
                if verbose:
                    print(
                        f"    find_cyclic_state: rel_step stalled at {rel_step:.2e} "
                        f"(round {round_idx}) -- not a single fixed point (likely a "
                        "period-2 orbit); returning the average of the two "
                        "alternating states instead.",
                        flush=True,
                    )
                x = 0.5 * (x_star + x)
                break
        else:
            stall_count = 0
        prev_rel_step = rel_step
        prev_x_star = x
        x = x_star
    else:
        if prev_x_star is not None:
            if verbose:
                print(
                    f"    find_cyclic_state: did not converge within {max_rounds} rounds "
                    "(no stall detected either -- non-periodic drift); returning the "
                    "average of the last two states.",
                    flush=True,
                )
            x = 0.5 * (x + prev_x_star)
    return _vec_to_state(x, config)


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
    warmup_cycles: int = 2,
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
    if warmup_cycles > 0:
        state = find_cyclic_state(
            profile, config, initial_state=state, max_rounds=max(3, warmup_cycles), verbose=False
        )
    cyc, _ = _run_one_cycle(profile, config, state)
    return cyc


def run_daily_operation(
    profile: HalfCycleProfile,
    config: DeviceConfig,
    *,
    n_cycles: int | None = None,
    loading_a0: float | None = None,
    loading_d0: float | None = None,
    warmup_cycles: int = 2,
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
    if warmup_cycles > 0:
        state = find_cyclic_state(
            profile, config, initial_state=state, max_rounds=max(3, warmup_cycles), verbose=False
        )

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
            * CP_WH_J_KG_K
            * max(0.0, profile.t_wh_in_c[0] - T_AMB_C)
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
_DAY_HOURS = 24.0


@dataclass(frozen=True, slots=True)
class DailyOperationHours:
    """Per-day operating hours per m² footprint."""

    n_cycles: int
    desorption_hours_per_day: float
    absorption_hours_per_day: float
    operating_hours_per_day: float


def daily_operating_hours_from_results(results: list[CycleResult]) -> DailyOperationHours:
    """Hours when beds are actively cycling (one adsorbing, one desorbing at all times).

    Desorption and absorption each span the full cycle duration because the two
    beds alternate roles every half-cycle.
    """
    operating_s = 0.0
    for cyc in results:
        operating_s += float(cyc.half_a.time_s[-1]) + float(cyc.half_b.time_s[-1])
    operating_h = min(operating_s / 3600.0, _DAY_HOURS)
    return DailyOperationHours(
        n_cycles=len(results),
        desorption_hours_per_day=operating_h,
        absorption_hours_per_day=operating_h,
        operating_hours_per_day=operating_h,
    )
MassTransferLimit = Literal["absorption", "desorption", "balanced"]
TrackedPhase = Literal["absorption", "desorption"]


@dataclass(frozen=True, slots=True)
class WaterInventorySeries:
    time_s: np.ndarray
    water_l_m2: np.ndarray
    phase: np.ndarray
    half_cycle_end_s: float
    cycle_index: np.ndarray
    half_cycle: np.ndarray
    m_ads_kg_s_m2: np.ndarray
    m_des_kg_s_m2: np.ndarray
    m_eq_kg_s_m2: np.ndarray
    m_ads_natural_kg_s_m2: np.ndarray
    m_des_natural_kg_s_m2: np.ndarray
    mass_transfer_limit: np.ndarray
    operating_flux_role: np.ndarray
    c_w_mol_m3: np.ndarray
    h_m: np.ndarray
    t_tracked_c: np.ndarray
    t_partner_c: np.ndarray
    t_f_c: np.ndarray
    t_cond_c: np.ndarray
    rh_vapor_gap: np.ndarray
    c_vac_kg_s_pa_m2: np.ndarray
    m_dot_f_kg_s_m2: np.ndarray
    collected_water_l_m2: np.ndarray


def cumulative_desorption_yield_l_m2(
    time_s: np.ndarray,
    m_des_kg_s_m2: np.ndarray,
) -> np.ndarray:
    """Trapezoidal cumulative integral of desorption flux (kg/m² ≈ L/m²)."""
    n = len(time_s)
    out = np.zeros(n, dtype=float)
    for k in range(n - 1):
        dt = float(time_s[k + 1] - time_s[k])
        out[k + 1] = out[k] + 0.5 * (m_des_kg_s_m2[k] + m_des_kg_s_m2[k + 1]) * dt
    return out


def _env_at_time(t_s: float, profile: HalfCycleProfile | None) -> ThermalEnvironment:
    if profile is None:
        return ThermalEnvironment(
            t_amb_c=T_AMB_C,
            rh_amb=RH_AMB,
            h_amb_w_m2_k=H_AMB_W_M2_K,
            t_wh_in_c=T_WH_IN_C,
            m_dot_wh_kg_s_m2=M_WH_KG_S_M2,
        )
    dt = profile.dt_s
    i = min(max(int(t_s / dt), 0), len(profile.temperature_c) - 1)
    return ThermalEnvironment(
        t_amb_c=profile.temperature_c[i],
        rh_amb=profile.relative_humidity[i],
        h_amb_w_m2_k=profile.h_amb_w_m2_k[i],
        t_wh_in_c=profile.t_wh_in_c[i],
        m_dot_wh_kg_s_m2=profile.m_dot_wh_kg_s_m2[i],
    )


def _tracked_half_series(
    half: HalfCycleResult,
    *,
    config: DeviceConfig,
    profile: HalfCycleProfile | None,
    tracked_phase: TrackedPhase,
    half_label: Literal["A", "B"],
    cycle_index: int,
    t_offset_s: float,
) -> WaterInventorySeries:
    """Build detailed inventory for one physical bed through one half-cycle."""
    n = len(half.time_s)
    if tracked_phase == "absorption":
        q = half.q_a
        h = half.h_a
        t_tracked = half.t_a_c
        t_partner = half.t_d_c
    else:
        q = half.q_d
        h = half.h_d
        t_tracked = half.t_d_c
        t_partner = half.t_a_c

    ctrl_p = config.controller_params()
    if is_hydrogel(config):
        assert h is not None
        water = np.array(
            [water_in_gel_l_m2(float(q_k), float(h_k), config=config) for q_k, h_k in zip(q, h)]
        )
    else:
        water = np.array([water_kg_m2_bed(float(q_k), config=config) for q_k in q])

    m_ads = np.asarray(half.m_ads_kg_s_m2, dtype=float)
    m_des = np.asarray(half.m_des_kg_s_m2, dtype=float)
    m_eq = np.minimum(m_ads, m_des)

    m_ads_nat = np.zeros(n)
    m_des_nat = np.zeros(n)
    limits: list[MassTransferLimit] = []
    c_vac = np.zeros(n)
    m_dot_f = np.zeros(n)
    rh_gap = np.zeros(n)

    for k in range(n):
        env = _env_at_time(float(half.time_s[k]), profile)
        h_a = float(half.h_a[k]) if half.h_a is not None else config.hydrogel_thickness_m
        h_d = float(half.h_d[k]) if half.h_d is not None else config.hydrogel_thickness_m
        m_ads_ctrl, m_des_ctrl = fluxes_for_control(
            loading_a=float(half.q_a[k]),
            loading_d=float(half.q_d[k]),
            h_a=h_a,
            h_d=h_d,
            t_a_c=float(half.t_a_c[k]),
            t_d_c=float(half.t_d_c[k]),
            t_cond_c=float(half.t_cond_c[k]),
            rh_amb=env.rh_amb,
            c_vac_kg_s_pa_m2=ctrl_p.c_vac_base_kg_s_pa_m2,
            config=config,
        )
        controls = compute_controls(
            t_a_c=float(half.t_a_c[k]),
            t_d_c=float(half.t_d_c[k]),
            m_ads_kg_s_m2=m_ads_ctrl,
            m_des_kg_s_m2=m_des_ctrl,
            params=ctrl_p,
            integral_ads_kg_m2=0.0,
            integral_des_kg_m2=0.0,
        )
        nat = mass_rates(
            loading_a=float(half.q_a[k]),
            loading_d=float(half.q_d[k]),
            h_a=h_a,
            h_d=h_d,
            t_a_c=float(half.t_a_c[k]),
            t_d_c=float(half.t_d_c[k]),
            t_cond_c=float(half.t_cond_c[k]),
            rh_amb=env.rh_amb,
            c_vac_kg_s_pa_m2=controls.c_vac_kg_s_pa_m2,
            config=config,
            equalize=False,
        )
        m_ads_nat[k] = nat.m_ads_kg_s_m2
        m_des_nat[k] = nat.m_des_kg_s_m2
        if nat.m_des_kg_s_m2 < 0.99 * nat.m_ads_kg_s_m2:
            limits.append("desorption")
        elif nat.m_ads_kg_s_m2 < 0.99 * nat.m_des_kg_s_m2:
            limits.append("absorption")
        else:
            limits.append("balanced")
        c_vac[k] = controls.c_vac_kg_s_pa_m2
        m_dot_f[k] = controls.m_dot_f_kg_s_m2
        rh_gap[k] = rh_outside_desorber(float(half.t_d_c[k]), float(half.t_cond_c[k]))

    if is_hydrogel(config):
        assert h is not None
        c_w = np.asarray(q, dtype=float)
        h_arr = np.asarray(h, dtype=float)
    else:
        c_w = np.asarray(q, dtype=float)
        h_arr = np.full(n, config.hydrogel_thickness_m)

    operating_role = np.array([tracked_phase] * n, dtype=object)
    return WaterInventorySeries(
        time_s=np.asarray(half.time_s, dtype=float) + t_offset_s,
        water_l_m2=water,
        phase=np.array([tracked_phase] * n, dtype=object),
        half_cycle_end_s=0.0,
        cycle_index=np.full(n, cycle_index, dtype=int),
        half_cycle=np.array([half_label] * n, dtype=object),
        m_ads_kg_s_m2=m_ads,
        m_des_kg_s_m2=m_des,
        m_eq_kg_s_m2=m_eq,
        m_ads_natural_kg_s_m2=m_ads_nat,
        m_des_natural_kg_s_m2=m_des_nat,
        mass_transfer_limit=np.array(limits, dtype=object),
        operating_flux_role=operating_role,
        c_w_mol_m3=c_w,
        h_m=h_arr,
        t_tracked_c=np.asarray(t_tracked, dtype=float),
        t_partner_c=np.asarray(t_partner, dtype=float),
        t_f_c=np.asarray(half.t_f_c, dtype=float),
        t_cond_c=np.asarray(half.t_cond_c, dtype=float),
        rh_vapor_gap=rh_gap,
        c_vac_kg_s_pa_m2=c_vac,
        m_dot_f_kg_s_m2=m_dot_f,
        collected_water_l_m2=np.zeros(n, dtype=float),
    )


def _concat_water_series(chunks: list[WaterInventorySeries], *, skip_first: int) -> WaterInventorySeries:
    if not chunks:
        raise ValueError("At least one inventory chunk is required.")

    def cat(attr: str) -> np.ndarray:
        parts = [getattr(chunks[0], attr)]
        for chunk in chunks[1:]:
            arr = getattr(chunk, attr)
            parts.append(arr[skip_first:])
        return np.concatenate(parts)

    return WaterInventorySeries(
        time_s=cat("time_s"),
        water_l_m2=cat("water_l_m2"),
        phase=cat("phase"),
        half_cycle_end_s=chunks[0].half_cycle_end_s,
        cycle_index=cat("cycle_index"),
        half_cycle=cat("half_cycle"),
        m_ads_kg_s_m2=cat("m_ads_kg_s_m2"),
        m_des_kg_s_m2=cat("m_des_kg_s_m2"),
        m_eq_kg_s_m2=cat("m_eq_kg_s_m2"),
        m_ads_natural_kg_s_m2=cat("m_ads_natural_kg_s_m2"),
        m_des_natural_kg_s_m2=cat("m_des_natural_kg_s_m2"),
        mass_transfer_limit=cat("mass_transfer_limit"),
        operating_flux_role=cat("operating_flux_role"),
        c_w_mol_m3=cat("c_w_mol_m3"),
        h_m=cat("h_m"),
        t_tracked_c=cat("t_tracked_c"),
        t_partner_c=cat("t_partner_c"),
        t_f_c=cat("t_f_c"),
        t_cond_c=cat("t_cond_c"),
        rh_vapor_gap=cat("rh_vapor_gap"),
        c_vac_kg_s_pa_m2=cat("c_vac_kg_s_pa_m2"),
        m_dot_f_kg_s_m2=cat("m_dot_f_kg_s_m2"),
        collected_water_l_m2=cat("collected_water_l_m2"),
    )


def water_inventory_series(
    cycle: CycleResult,
    *,
    config: DeviceConfig,
    profile: HalfCycleProfile | None = None,
    cycle_index: int = 0,
) -> WaterInventorySeries:
    """One physical bed: absorbs in half A, desorbs in half B (same gel as solar_lumped)."""
    ha = cycle.half_a
    hb = cycle.half_b
    abs_chunk = _tracked_half_series(
        ha,
        config=config,
        profile=profile,
        tracked_phase="absorption",
        half_label="A",
        cycle_index=cycle_index,
        t_offset_s=0.0,
    )
    des_chunk = _tracked_half_series(
        hb,
        config=config,
        profile=profile,
        tracked_phase="desorption",
        half_label="B",
        cycle_index=cycle_index,
        t_offset_s=float(ha.time_s[-1]) if len(ha.time_s) else 0.0,
    )
    out = _concat_water_series([abs_chunk, des_chunk], skip_first=1)
    yield_abs = cumulative_desorption_yield_l_m2(ha.time_s, ha.m_des_kg_s_m2)
    yield_des = cumulative_desorption_yield_l_m2(hb.time_s, hb.m_des_kg_s_m2) + yield_abs[-1]
    collected = np.concatenate([yield_abs, yield_des[1:]])
    return WaterInventorySeries(
        time_s=out.time_s,
        water_l_m2=out.water_l_m2,
        phase=out.phase,
        half_cycle_end_s=float(ha.time_s[-1]) if len(ha.time_s) else 0.0,
        cycle_index=out.cycle_index,
        half_cycle=out.half_cycle,
        m_ads_kg_s_m2=out.m_ads_kg_s_m2,
        m_des_kg_s_m2=out.m_des_kg_s_m2,
        m_eq_kg_s_m2=out.m_eq_kg_s_m2,
        m_ads_natural_kg_s_m2=out.m_ads_natural_kg_s_m2,
        m_des_natural_kg_s_m2=out.m_des_natural_kg_s_m2,
        mass_transfer_limit=out.mass_transfer_limit,
        operating_flux_role=out.operating_flux_role,
        c_w_mol_m3=out.c_w_mol_m3,
        h_m=out.h_m,
        t_tracked_c=out.t_tracked_c,
        t_partner_c=out.t_partner_c,
        t_f_c=out.t_f_c,
        t_cond_c=out.t_cond_c,
        rh_vapor_gap=out.rh_vapor_gap,
        c_vac_kg_s_pa_m2=out.c_vac_kg_s_pa_m2,
        m_dot_f_kg_s_m2=out.m_dot_f_kg_s_m2,
        collected_water_l_m2=collected,
    )


def _append_water_cycle(base: WaterInventorySeries, nxt: WaterInventorySeries) -> WaterInventorySeries:
    """Append a cycle after base, skipping duplicate boundary point and shifting time."""
    t0 = float(base.time_s[-1])
    shifted = WaterInventorySeries(
        time_s=nxt.time_s[1:] + t0,
        water_l_m2=nxt.water_l_m2[1:],
        phase=nxt.phase[1:],
        half_cycle_end_s=base.half_cycle_end_s,
        cycle_index=nxt.cycle_index[1:],
        half_cycle=nxt.half_cycle[1:],
        m_ads_kg_s_m2=nxt.m_ads_kg_s_m2[1:],
        m_des_kg_s_m2=nxt.m_des_kg_s_m2[1:],
        m_eq_kg_s_m2=nxt.m_eq_kg_s_m2[1:],
        m_ads_natural_kg_s_m2=nxt.m_ads_natural_kg_s_m2[1:],
        m_des_natural_kg_s_m2=nxt.m_des_natural_kg_s_m2[1:],
        mass_transfer_limit=nxt.mass_transfer_limit[1:],
        operating_flux_role=nxt.operating_flux_role[1:],
        c_w_mol_m3=nxt.c_w_mol_m3[1:],
        h_m=nxt.h_m[1:],
        t_tracked_c=nxt.t_tracked_c[1:],
        t_partner_c=nxt.t_partner_c[1:],
        t_f_c=nxt.t_f_c[1:],
        t_cond_c=nxt.t_cond_c[1:],
        rh_vapor_gap=nxt.rh_vapor_gap[1:],
        c_vac_kg_s_pa_m2=nxt.c_vac_kg_s_pa_m2[1:],
        m_dot_f_kg_s_m2=nxt.m_dot_f_kg_s_m2[1:],
        collected_water_l_m2=nxt.collected_water_l_m2[1:] + float(base.collected_water_l_m2[-1]),
    )
    return _concat_water_series([base, shifted], skip_first=0)


def water_inventory_daily_series(
    cycles: list[CycleResult],
    *,
    config: DeviceConfig,
    profile: HalfCycleProfile | None = None,
) -> WaterInventorySeries:
    if not cycles:
        raise ValueError("At least one cycle is required.")
    out = water_inventory_series(
        cycles[0],
        config=config,
        profile=profile,
        cycle_index=0,
    )
    for i, cycle in enumerate(cycles[1:], start=1):
        chunk = water_inventory_series(
            cycle,
            config=config,
            profile=profile,
            cycle_index=i,
        )
        out = _append_water_cycle(out, chunk)
    return out


def write_water_inventory_csv(path: Path, series: WaterInventorySeries, *, config: DeviceConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    col = inventory_column(config)
    fields = [
        "time_s",
        "time_h",
        "cycle_index",
        "half_cycle",
        "tracked_bed_phase",
        "operating_flux_role",
        "mass_transfer_limit",
        col,
        "collected_water_l_m2",
        "c_w_mol_m3",
        "h_m",
        "m_ads_kg_s_m2",
        "m_des_kg_s_m2",
        "m_eq_kg_s_m2",
        "m_ads_natural_kg_s_m2",
        "m_des_natural_kg_s_m2",
        "t_tracked_c",
        "t_partner_c",
        "t_f_c",
        "t_cond_c",
        "rh_vapor_gap",
        "c_vac_kg_s_pa_m2",
        "m_dot_f_kg_s_m2",
    ]
    if not is_hydrogel(config):
        fields = [f for f in fields if f not in ("c_w_mol_m3", "h_m")]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for k in range(len(series.time_s)):
            row: dict[str, object] = {
                "time_s": f"{float(series.time_s[k]):.3f}",
                "time_h": f"{float(series.time_s[k]) / 3600.0:.6f}",
                "cycle_index": int(series.cycle_index[k]),
                "half_cycle": str(series.half_cycle[k]),
                "tracked_bed_phase": str(series.phase[k]),
                "operating_flux_role": str(series.operating_flux_role[k]),
                "mass_transfer_limit": str(series.mass_transfer_limit[k]),
                col: f"{float(series.water_l_m2[k]):.6f}",
                "collected_water_l_m2": f"{float(series.collected_water_l_m2[k]):.6f}",
                "c_w_mol_m3": f"{float(series.c_w_mol_m3[k]):.3f}",
                "h_m": f"{float(series.h_m[k]):.6f}",
                "m_ads_kg_s_m2": f"{float(series.m_ads_kg_s_m2[k]):.9e}",
                "m_des_kg_s_m2": f"{float(series.m_des_kg_s_m2[k]):.9e}",
                "m_eq_kg_s_m2": f"{float(series.m_eq_kg_s_m2[k]):.9e}",
                "m_ads_natural_kg_s_m2": f"{float(series.m_ads_natural_kg_s_m2[k]):.9e}",
                "m_des_natural_kg_s_m2": f"{float(series.m_des_natural_kg_s_m2[k]):.9e}",
                "t_tracked_c": f"{float(series.t_tracked_c[k]):.4f}",
                "t_partner_c": f"{float(series.t_partner_c[k]):.4f}",
                "t_f_c": f"{float(series.t_f_c[k]):.4f}",
                "t_cond_c": f"{float(series.t_cond_c[k]):.4f}",
                "rh_vapor_gap": f"{float(series.rh_vapor_gap[k]):.6f}",
                "c_vac_kg_s_pa_m2": f"{float(series.c_vac_kg_s_pa_m2[k]):.9e}",
                "m_dot_f_kg_s_m2": f"{float(series.m_dot_f_kg_s_m2[k]):.9e}",
            }
            w.writerow([row[name] for name in fields])


def plot_water_inventory(
    path: Path,
    series: WaterInventorySeries,
    *,
    config: DeviceConfig,
    title: str | None = None,
    half_cycle_markers: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_h = series.time_s / 3600.0
    fig, (ax_inv, ax_yield) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax_inv.plot(time_h, series.water_l_m2, color="#4C72B0", linewidth=2)
    ax_yield.plot(time_h, series.collected_water_l_m2, color="#C44E52", linewidth=2)

    if half_cycle_markers:
        half_mark_h = series.half_cycle_end_s / 3600.0
        for ax in (ax_inv, ax_yield):
            ax.axvline(half_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
        cycle_period_h = 2.0 * series.half_cycle_end_s / 3600.0
        if cycle_period_h > 0.0 and time_h[-1] > cycle_period_h * 1.5:
            t_end_h = float(time_h[-1])
            t = cycle_period_h
            while t < t_end_h - 1e-9:
                for ax in (ax_inv, ax_yield):
                    ax.axvline(t, color="k", linewidth=0.5, linestyle="--", alpha=0.25)
                t += cycle_period_h

    ax_inv.set_ylabel(inventory_ylabel(config))
    ax_inv.grid(True, alpha=0.3)
    ax_yield.set_xlabel("Time (h)")
    ax_yield.set_ylabel("Collected water (L/m²)")
    ax_yield.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
HalfCycleLabel = Literal["A", "B"]


@dataclass(frozen=True, slots=True)
class DetailedSeries:
    time_s: np.ndarray
    cycle_index: np.ndarray
    half_cycle: np.ndarray
    half_cycle_end_s: float
    t_a_c: np.ndarray
    t_d_c: np.ndarray
    t_f_c: np.ndarray
    t_cond_c: np.ndarray
    t_amb_c: np.ndarray
    relative_humidity: np.ndarray
    h_amb_w_m2_k: np.ndarray
    t_wh_in_c: np.ndarray
    m_dot_wh_kg_s_m2: np.ndarray
    n_cycles: int = 1


# _default_env/_env_at_time already defined above (water_inventory section); same body,
# single definition.


def _half_boundary_series(
    half: HalfCycleResult,
    *,
    profile: HalfCycleProfile | None,
    half_label: HalfCycleLabel,
    cycle_index: int,
    t_offset_s: float,
) -> DetailedSeries:
    n = len(half.time_s)
    t_amb = np.zeros(n)
    rh = np.zeros(n)
    h_amb = np.zeros(n)
    t_wh = np.zeros(n)
    m_dot_wh = np.zeros(n)
    for k in range(n):
        env = _env_at_time(float(half.time_s[k]), profile)
        t_amb[k] = env.t_amb_c
        rh[k] = env.rh_amb
        h_amb[k] = env.h_amb_w_m2_k
        t_wh[k] = env.t_wh_in_c
        m_dot_wh[k] = env.m_dot_wh_kg_s_m2

    return DetailedSeries(
        time_s=np.asarray(half.time_s, dtype=float) + t_offset_s,
        cycle_index=np.full(n, cycle_index, dtype=int),
        half_cycle=np.array([half_label] * n, dtype=object),
        half_cycle_end_s=0.0,
        t_a_c=np.asarray(half.t_a_c, dtype=float),
        t_d_c=np.asarray(half.t_d_c, dtype=float),
        t_f_c=np.asarray(half.t_f_c, dtype=float),
        t_cond_c=np.asarray(half.t_cond_c, dtype=float),
        t_amb_c=t_amb,
        relative_humidity=rh,
        h_amb_w_m2_k=h_amb,
        t_wh_in_c=t_wh,
        m_dot_wh_kg_s_m2=m_dot_wh,
    )


def _concat_detailed_series(chunks: list[DetailedSeries], *, skip_first: int) -> DetailedSeries:
    if not chunks:
        raise ValueError("At least one detailed chunk is required.")

    def cat(attr: str) -> np.ndarray:
        parts = [getattr(chunks[0], attr)]
        for chunk in chunks[1:]:
            arr = getattr(chunk, attr)
            parts.append(arr[skip_first:])
        return np.concatenate(parts)

    return DetailedSeries(
        time_s=cat("time_s"),
        cycle_index=cat("cycle_index"),
        half_cycle=cat("half_cycle"),
        half_cycle_end_s=chunks[0].half_cycle_end_s,
        t_a_c=cat("t_a_c"),
        t_d_c=cat("t_d_c"),
        t_f_c=cat("t_f_c"),
        t_cond_c=cat("t_cond_c"),
        t_amb_c=cat("t_amb_c"),
        relative_humidity=cat("relative_humidity"),
        h_amb_w_m2_k=cat("h_amb_w_m2_k"),
        t_wh_in_c=cat("t_wh_in_c"),
        m_dot_wh_kg_s_m2=cat("m_dot_wh_kg_s_m2"),
        n_cycles=chunks[0].n_cycles,
    )


def detailed_series(
    cycle: CycleResult,
    *,
    config: DeviceConfig,
    profile: HalfCycleProfile | None = None,
    cycle_index: int = 0,
) -> DetailedSeries:
    """Build temperature and boundary trajectories for one full two-bed cycle."""
    del config  # reserved for future sorbent-specific diagnostics
    ha = cycle.half_a
    hb = cycle.half_b
    chunk_a = _half_boundary_series(
        ha,
        profile=profile,
        half_label="A",
        cycle_index=cycle_index,
        t_offset_s=0.0,
    )
    chunk_b = _half_boundary_series(
        hb,
        profile=profile,
        half_label="B",
        cycle_index=cycle_index,
        t_offset_s=float(ha.time_s[-1]) if len(ha.time_s) else 0.0,
    )
    out = _concat_detailed_series([chunk_a, chunk_b], skip_first=1)
    return DetailedSeries(
        time_s=out.time_s,
        cycle_index=out.cycle_index,
        half_cycle=out.half_cycle,
        half_cycle_end_s=float(ha.time_s[-1]) if len(ha.time_s) else 0.0,
        t_a_c=out.t_a_c,
        t_d_c=out.t_d_c,
        t_f_c=out.t_f_c,
        t_cond_c=out.t_cond_c,
        t_amb_c=out.t_amb_c,
        relative_humidity=out.relative_humidity,
        h_amb_w_m2_k=out.h_amb_w_m2_k,
        t_wh_in_c=out.t_wh_in_c,
        m_dot_wh_kg_s_m2=out.m_dot_wh_kg_s_m2,
        n_cycles=1,
    )


def _append_detailed_cycle(base: DetailedSeries, nxt: DetailedSeries) -> DetailedSeries:
    t0 = float(base.time_s[-1])
    shifted = DetailedSeries(
        time_s=nxt.time_s[1:] + t0,
        cycle_index=nxt.cycle_index[1:],
        half_cycle=nxt.half_cycle[1:],
        half_cycle_end_s=base.half_cycle_end_s,
        t_a_c=nxt.t_a_c[1:],
        t_d_c=nxt.t_d_c[1:],
        t_f_c=nxt.t_f_c[1:],
        t_cond_c=nxt.t_cond_c[1:],
        t_amb_c=nxt.t_amb_c[1:],
        relative_humidity=nxt.relative_humidity[1:],
        h_amb_w_m2_k=nxt.h_amb_w_m2_k[1:],
        t_wh_in_c=nxt.t_wh_in_c[1:],
        m_dot_wh_kg_s_m2=nxt.m_dot_wh_kg_s_m2[1:],
        n_cycles=base.n_cycles,
    )
    return _concat_detailed_series([base, shifted], skip_first=0)


def detailed_daily_series(
    cycles: list[CycleResult],
    *,
    config: DeviceConfig,
    profile: HalfCycleProfile | None = None,
) -> DetailedSeries:
    if not cycles:
        raise ValueError("At least one cycle is required.")
    out = detailed_series(cycles[0], config=config, profile=profile, cycle_index=0)
    for i, cycle in enumerate(cycles[1:], start=1):
        chunk = detailed_series(cycle, config=config, profile=profile, cycle_index=i)
        out = _append_detailed_cycle(out, chunk)
    return DetailedSeries(
        time_s=out.time_s,
        cycle_index=out.cycle_index,
        half_cycle=out.half_cycle,
        half_cycle_end_s=out.half_cycle_end_s,
        t_a_c=out.t_a_c,
        t_d_c=out.t_d_c,
        t_f_c=out.t_f_c,
        t_cond_c=out.t_cond_c,
        t_amb_c=out.t_amb_c,
        relative_humidity=out.relative_humidity,
        h_amb_w_m2_k=out.h_amb_w_m2_k,
        t_wh_in_c=out.t_wh_in_c,
        m_dot_wh_kg_s_m2=out.m_dot_wh_kg_s_m2,
        n_cycles=len(cycles),
    )


def write_detailed_csv(path: Path, series: DetailedSeries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time_s",
                "time_h",
                "cycle_index",
                "half_cycle",
                "t_a_c",
                "t_d_c",
                "t_f_c",
                "t_cond_c",
                "t_amb_c",
                "relative_humidity",
                "h_amb_w_m2_k",
                "t_wh_in_c",
                "m_dot_wh_kg_s_m2",
            ]
        )
        for k in range(len(series.time_s)):
            w.writerow(
                [
                    f"{float(series.time_s[k]):.3f}",
                    f"{float(series.time_s[k]) / 3600.0:.6f}",
                    int(series.cycle_index[k]),
                    series.half_cycle[k],
                    f"{float(series.t_a_c[k]):.4f}",
                    f"{float(series.t_d_c[k]):.4f}",
                    f"{float(series.t_f_c[k]):.4f}",
                    f"{float(series.t_cond_c[k]):.4f}",
                    f"{float(series.t_amb_c[k]):.4f}",
                    f"{float(series.relative_humidity[k]):.6f}",
                    f"{float(series.h_amb_w_m2_k[k]):.4f}",
                    f"{float(series.t_wh_in_c[k]):.4f}",
                    f"{float(series.m_dot_wh_kg_s_m2[k]):.6f}",
                ]
            )


def plot_detailed_diagnostics(
    path: Path,
    series: DetailedSeries,
    *,
    title: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_h = series.time_s / 3600.0
    phase_mark_h = series.half_cycle_end_s / 3600.0
    show_half_mark = series.n_cycles == 1 and series.half_cycle_end_s > 0.0

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax_t, ax_wx, ax_wh = axes

    ax_t.plot(time_h, series.t_a_c, color="#8b2000", linewidth=1.8, label="Contactor A")
    ax_t.plot(time_h, series.t_d_c, color="#b06000", linewidth=1.8, label="Contactor B")
    ax_t.plot(time_h, series.t_f_c, color="#6a3d9a", linewidth=1.4, linestyle="--", label="HTF loop")
    ax_t.plot(time_h, series.t_cond_c, color="#1a5a7a", linewidth=1.8, label="Condenser")
    ax_t.plot(time_h, series.t_amb_c, color="0.45", linewidth=1.2, linestyle=":", label="Ambient")
    ax_t.plot(
        time_h,
        series.t_wh_in_c,
        color="#d95f02",
        linewidth=1.2,
        linestyle="-.",
        label="Waste-heat inlet",
    )
    if show_half_mark:
        ax_t.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_t.set_ylabel("Temperature (°C)")
    ax_t.legend(loc="upper left", fontsize=7, ncol=2)
    ax_t.grid(True, alpha=0.3)

    ax_wx.plot(time_h, series.t_amb_c, color="#d95f02", linewidth=1.6, label="T_amb")
    if show_half_mark:
        ax_wx.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_wx.set_ylabel("Temperature (°C)", color="#d95f02")
    ax_wx.tick_params(axis="y", labelcolor="#d95f02")
    ax_wx.grid(True, alpha=0.3)

    ax_rh = ax_wx.twinx()
    ax_rh.plot(
        time_h,
        series.relative_humidity * 100.0,
        color="#1b9e77",
        linewidth=1.6,
        label="RH",
    )
    ax_rh.set_ylabel("Relative humidity (%)", color="#1b9e77")
    ax_rh.tick_params(axis="y", labelcolor="#1b9e77")
    ax_rh.set_ylim(0.0, 100.0)

    lines_l, labels_l = ax_wx.get_legend_handles_labels()
    lines_r, labels_r = ax_rh.get_legend_handles_labels()
    ax_wx.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", fontsize=8)

    ax_wh.plot(
        time_h,
        series.m_dot_wh_kg_s_m2,
        color="#e6ab02",
        linewidth=1.8,
        label="m_dot_wh",
    )
    if show_half_mark:
        ax_wh.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_wh.set_ylabel("Mass flow (kg/s/m²)", color="#e6ab02")
    ax_wh.tick_params(axis="y", labelcolor="#e6ab02")
    ax_wh.grid(True, alpha=0.3)

    ax_h = ax_wh.twinx()
    ax_h.plot(time_h, series.h_amb_w_m2_k, color="#7570b3", linewidth=1.4, label="h_amb")
    ax_h.set_ylabel("h_amb (W/m²K)", color="#7570b3")
    ax_h.tick_params(axis="y", labelcolor="#7570b3")

    lines_l, labels_l = ax_wh.get_legend_handles_labels()
    lines_r, labels_r = ax_h.get_legend_handles_labels()
    ax_wh.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", fontsize=8)

    ax_wh.set_xlabel("Time (h)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
@dataclass(frozen=True, slots=True)
class SimulationResult:
    mean_daily_yield_kg_m2: float
    mean_daily_yield_l_m2: float
    mean_thermal_efficiency: float
    specific_energy: SpecificEnergyBreakdown
    n_days: int

    @property
    def specific_energy_wh_kwh_per_l(self) -> float:
        return self.specific_energy.wh_kwh_per_l

    @property
    def specific_energy_parasitic_kwh_per_l(self) -> float:
        return self.specific_energy.parasitic_kwh_per_l

    @property
    def specific_energy_total_kwh_per_l(self) -> float:
        return self.specific_energy.total_kwh_per_l

    @property
    def n_cycles_per_day(self) -> int:
        return self.specific_energy.n_cycles_per_day


def simulate_daily(
    profile: HalfCycleProfile,
    config: DeviceConfig,
    *,
    n_cycles: int | None = None,
    parasitic_options: ParasiticLoadOptions | None = None,
    electric_heat_w_per_m2: float = 0.0,
) -> SimulationResult:
    yield_kg, eta, results = run_daily_operation(profile, config, n_cycles=n_cycles)
    h_fg = config.thermal_params().h_fg_j_per_kg
    energy = specific_energy_breakdown_from_daily_operation(
        yield_kg,
        thermal_efficiency=eta,
        cycle_results=results,
        h_fg_j_per_kg=h_fg,
        parasitic_options=parasitic_options,
        electric_heat_w_per_m2=electric_heat_w_per_m2,
    )
    return SimulationResult(
        mean_daily_yield_kg_m2=yield_kg,
        mean_daily_yield_l_m2=yield_kg * 1000.0,
        mean_thermal_efficiency=eta,
        specific_energy=energy,
        n_days=1,
    )
