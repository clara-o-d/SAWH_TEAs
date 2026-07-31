"""Simulation: device config, coupled thermal/mass dynamics, ODE integration,
detailed plotting, water-inventory accounting, site feasibility, and annual yield.

Consolidated from the former simulation/{device_config, coupled_dynamics, ode_system,
detailed_plots, water_inventory, site_feasibility, annual_yield}.py. Section headers
below mark each former module's boundary for traceability.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from solar_lumped.economics import LCOEconomicParams, lcow_from_daily_yield
from solar_lumped.physics import (
    CP_AL_J_KG_K,
    DEFAULT_MOF_NAME,
    DRY_COMPOSITE_DENSITY_KG_M3,
    EPS_ABS,
    EPS_ABS_IR_CASE2,
    EPS_AL,
    EPS_GEL,
    EPS_GLASS_IR_CASE2,
    FIN_AREA_RATIO,
    G_CHAMBER_M_S,
    H0_M,
    H_DES_J_PER_KG,
    H_FG_J_PER_KG,
    K_AIR_W_M_K,
    L_C_M,
    L_G_M,
    L_INS_M,
    RHO_AL_KG_M3,
    SALT_TO_POLYMER_RATIO_DEFAULT,
    STEFAN_BOLTZMANN_W_M2_K4,
    SaltProperties,
    SorbentKind,
    TAU_GLASS,
    TILT_DEG,
    VAPOR_GAP_TRANSPORT_MIN_M,
    DeviceThermalParams,
    MassTransferParams,
    MofProperties,
    ThermalState,
    clamp_temperature_c,
    clip_loading,
    condenser_h_conv_w_m2_k,
    desorption_water_activity,
    evaluate_mass_rates,
    get_mof,
    get_salt,
    hollands_vapor_gap_h_conv_w_m2_k,
    initial_loading,
    inventory_ylabel,
    m_des_kg_s_m2_from_dc_w,
    parallel_plate_emissivity,
    radiative_exchange_w_m2,
    salt_molarity_from_composite,
    solve_steady_thermal,
    u_gel_w_m2_k,
    water_in_sorbent_l_m2,
)
from solar_lumped.physics import _residuals as _thermal_residuals
from solar_lumped.weather import DailyWeatherProfile, PhaseProfile, day_weather_stats


# =============================================================================
# Device configuration dataclass
# =============================================================================

@dataclass(frozen=True, slots=True)
class DeviceConfig:
    sorbent: SorbentKind = "hydrogel"
    mof_name: str = DEFAULT_MOF_NAME
    salt_name: str = "LiCl"
    salt_to_polymer_ratio: float = SALT_TO_POLYMER_RATIO_DEFAULT
    hydrogel_thickness_m: float = H0_M
    vapor_gap_m: float = L_G_M
    insulation_gap_m: float = L_INS_M
    g_conv_m_s: float = G_CHAMBER_M_S
    hydrogel_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3
    fin_area_ratio: float = FIN_AREA_RATIO
    condenser_thickness_m: float = L_C_M
    condenser_rho_kg_m3: float = RHO_AL_KG_M3
    condenser_cp_j_kg_k: float = CP_AL_J_KG_K
    h_fg_j_per_kg: float = H_FG_J_PER_KG
    tilt_deg: float = TILT_DEG
    thermal: DeviceThermalParams | None = None
    # Override catalog salt formula weight (g/mol) for sensitivity sweeps.
    salt_formula_weight_g_mol: float | None = None
    # Scales MW_salt in gravimetric uptake only (DVS cap during absorption).
    salt_weight_factor: float = 1.0
    # Uniform surface/gel temperature at desorption start. None → algebraic steady
    # state (quasi_steady solves Eqs 1/3/4 algebraically each ODE step).
    segregated_initial_temp_c: float | None = None
    # Per-component desorption-start temperatures (T_gel, T_abs, T_glass, T_cond) in
    # °C. Takes precedence over ``segregated_initial_temp_c`` when set — e.g. to
    # match the first digitized Wilson data point.
    coupled_initial_temps_c: tuple[float, float, float, float] | None = None

    def desorption_surface_ic_c(self) -> tuple[float, float, float, float] | None:
        """Configured (T_gel, T_abs, T_glass, T_cond) at desorption start, if any."""
        if self.coupled_initial_temps_c is not None:
            return self.coupled_initial_temps_c
        if self.segregated_initial_temp_c is not None:
            t = self.segregated_initial_temp_c
            return (t, t, t, t)
        return None

    def salt(self) -> SaltProperties:
        return get_salt(self.salt_name)

    def mof(self) -> MofProperties:
        return get_mof(self.mof_name)

    def mass_params(self) -> MassTransferParams:
        if self.sorbent == "mof":
            props = self.mof()
            return MassTransferParams(
                g_conv_m_s=props.g_conv_m_s,
                h0_ref_m=self.hydrogel_thickness_m,
                vapor_gap_m=self.vapor_gap_m,
                tilt_deg=self.tilt_deg,
                c_s_mol_m3=0.0,
                ions_per_formula=1,
                rho_solution_kg_m3=1000.0,
                salt_name="MOF",
                formula_weight_g_mol=1.0,
                salt_to_polymer_ratio=1.0,
            )
        s = self.salt()
        fw = (
            self.salt_formula_weight_g_mol
            if self.salt_formula_weight_g_mol is not None
            else s.formula_weight_g_mol
        )
        return MassTransferParams(
            g_conv_m_s=self.g_conv_m_s,
            h0_ref_m=self.hydrogel_thickness_m,
            vapor_gap_m=self.vapor_gap_m,
            tilt_deg=self.tilt_deg,
            c_s_mol_m3=salt_molarity_from_composite(
                self.salt_to_polymer_ratio,
                self.hydrogel_density_kg_m3,
                fw,
            ),
            ions_per_formula=s.ions_per_formula,
            rho_solution_kg_m3=s.rho_solution_kg_m3,
            salt_name=s.name,
            formula_weight_g_mol=fw,
            salt_to_polymer_ratio=self.salt_to_polymer_ratio,
            salt_weight_factor=self.salt_weight_factor,
        )

    def thermal_params(self) -> DeviceThermalParams:
        if self.thermal is not None:
            return self.thermal
        if self.sorbent == "mof":
            h_des = self.mof().h_des_j_per_kg
        elif self.salt_name == "LiCl":
            # Wilson Table S3 COMSOL value (2320 kJ/kg), not the broader Díaz-Marín
            # literature range in salt_heat_of_desorption.csv (~2850 kJ/kg).
            h_des = H_DES_J_PER_KG
        else:
            h_des = self.salt().h_des_j_per_kg
        return DeviceThermalParams(
            insulation_gap_m=self.insulation_gap_m,
            vapor_gap_m=self.vapor_gap_m,
            eps_abs=EPS_ABS,
            tau_glass=TAU_GLASS,
            eps_gel=EPS_GEL,
            eps_al=EPS_AL,
            tilt_deg=self.tilt_deg,
            h_des_j_per_kg=h_des,
            # Case 2 ("selective surface", sawh_bayesopt.design_space.CASE_EPS_IR) is
            # this package's base case as of 2026-07: real absorber/glass IR
            # emissivities instead of Wilson's original blackbody/cavity
            # approximation (case1, still reachable via an explicit `thermal=`
            # override with eps_abs_ir=eps_glass_ir=1.0 -- see DeviceThermalParams).
            eps_abs_ir=EPS_ABS_IR_CASE2,
            eps_glass_ir=EPS_GLASS_IR_CASE2,
        )

    def condenser_thermal_mass_j_m2_k(self) -> float:
        return (
            self.condenser_rho_kg_m3
            * self.condenser_cp_j_kg_k
            * self.condenser_thickness_m
        )

    @classmethod
    def comsol_table_s3(cls, **overrides: object) -> DeviceConfig:
        """Wilson Table S3 / Note S1 COMSOL SAWH device defaults."""
        return cls(**overrides)  # type: ignore[arg-type]

    @classmethod
    def baseline(cls, **overrides: object) -> DeviceConfig:
        """Wilson Fig. 2 baseline device (Table S3, tilt 30°, fin area ratio 7.1)."""
        base = {
            "tilt_deg": TILT_DEG,
            "fin_area_ratio": FIN_AREA_RATIO,
        }
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

    @classmethod
    def atacama_field(cls, **overrides: object) -> DeviceConfig:
        """Wilson Atacama field-test geometry (Methods): tilt 25°, fin area ratio 5."""
        base = {
            "tilt_deg": 25.0,
            "fin_area_ratio": 5.0,
        }
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

# =============================================================================
# Coupled Wilson Eqs. 1-6 + condenser transient (Eq. 2) rate evaluation
# =============================================================================

CyclePhase = Literal["absorption", "desorption"]

_M_DES_BRACKET_MAX = 0.01  # kg/m²/s upper search bound for brentq bracket


@dataclass(frozen=True, slots=True)
class CoupledRates:
    dc_w_dt: float
    dH_dt: float
    dT_cond_dt: float
    t_gel_c: float
    m_des_kg_s_m2: float
    thermal: ThermalState


def _thermal_guess(thermal: ThermalState) -> tuple[float, float, float]:
    """(T_gel, T_abs, T_glass) warm-start tuple for the next ``solve_steady_thermal`` call."""
    return thermal.t_gel_c, thermal.t_abs_c, thermal.t_glass_c


def _m_des_calc(
    m_des: float,
    *,
    loading: float,
    h_m: float,
    t_cond_c: float,
    t_amb_c: float,
    q_solar_w_m2: float,
    h_amb: float,
    mass: MassTransferParams,
    thermal: DeviceThermalParams,
    vapor_gap_m: float,
    config: DeviceConfig,
    t_guess: tuple[float, float, float] | None,
) -> tuple[float, float, float, ThermalState]:
    state = solve_steady_thermal(
        t_cond_c=t_cond_c,
        t_amb_c=t_amb_c,
        q_solar_w_m2=q_solar_w_m2,
        m_des_kg_s_m2=max(0.0, m_des),
        h_amb=h_amb,
        params=thermal,
        h_m=h_m,
        t_guess=t_guess,
        vapor_gap_m=vapor_gap_m,
    )
    dc, dh, m_calc = evaluate_mass_rates(
        loading=loading,
        h_m=h_m,
        t_gel_c=state.t_gel_c,
        t_cond_c=t_cond_c,
        rh=0.0,
        phase="desorption",
        mass=mass,
        config=config,
        vapor_gap_m=vapor_gap_m,
    )
    if config.sorbent == "hydrogel":
        m_calc = m_des_kg_s_m2_from_dc_w(dc, h0_ref_m=mass.h0_ref_m)
    return m_calc, state.t_gel_c, dc, state


def evaluate_coupled_rates(
    *,
    c_w: float,
    h_m: float,
    t_cond_c: float,
    t_amb_c: float,
    rh: float,
    q_solar_w_m2: float,
    h_amb: float,
    phase: CyclePhase,
    mass: MassTransferParams,
    thermal: DeviceThermalParams,
    vapor_gap_m: float,
    condenser_thermal_mass_j_m2_k: float,
    fin_area_ratio: float,
    h_fg_j_per_kg: float,
    config: DeviceConfig,
    t_guess: tuple[float, float, float] | None = None,
    h_amb_cond: float | None = None,
) -> CoupledRates:
    """Return (dloading/dt, dH/dt, dT_cond/dt) with self-consistent T_gel and m_des.

    ``c_w`` stores hydrogel mol/m³ or MOF kg/kg loading depending on ``config.sorbent``.

    ``h_amb_cond`` sets the condenser-backing convection coefficient; when None the
    ambient ``h_amb`` (which also drives the absorber/glass) is reused. Providing a
    separate value models fan-forced condenser cooling decoupled from ambient wind.
    """
    gap_eff = max(vapor_gap_m - h_m, 0.0)
    q_sol = max(0.0, q_solar_w_m2)

    if phase == "absorption":
        # Note S1 Eq. S1: fast gel thermal storage → T_gel ≈ T_amb during open absorption
        t_gel = t_amb_c
        h_conv_g = hollands_vapor_gap_h_conv_w_m2_k(
            gap_eff, t_gel, t_cond_c, tilt_deg=thermal.tilt_deg
        ) if gap_eff > 0.0 else 0.0
        state = ThermalState(
            t_gel_c=t_gel,
            t_abs_c=t_amb_c,
            t_glass_c=t_amb_c,
            h_conv_g=h_conv_g,
            m_des_kg_s_m2=0.0,
        )
        dc, dh, _ = evaluate_mass_rates(
            loading=c_w,
            h_m=h_m,
            t_gel_c=t_gel,
            t_cond_c=None,
            rh=rh,
            phase="absorption",
            mass=mass,
            config=config,
            vapor_gap_m=vapor_gap_m,
        )
        return CoupledRates(
            dc_w_dt=dc,
            dH_dt=dh,
            dT_cond_dt=0.0,
            t_gel_c=t_gel,
            m_des_kg_s_m2=0.0,
            thermal=state,
        )

    t_cond = clamp_temperature_c(t_cond_c)

    # Root of m_calc(m) - m = 0 so Eqs. 1-4 and Eq. 5 agree (avoids fixed-point cycling).
    def _m_des_residual(m_des_guess: float) -> float:
        m_calc, _, _, _ = _m_des_calc(
            m_des_guess,
            loading=c_w,
            h_m=h_m,
            t_cond_c=t_cond,
            t_amb_c=t_amb_c,
            q_solar_w_m2=q_sol,
            h_amb=h_amb,
            mass=mass,
            thermal=thermal,
            vapor_gap_m=gap_eff,
            config=config,
            t_guess=t_guess,
        )
        if not math.isfinite(m_calc):
            return -m_des_guess
        return m_calc - m_des_guess

    m_at_zero, t_gel0, dc0, state0 = _m_des_calc(
        0.0,
        loading=c_w,
        h_m=h_m,
        t_cond_c=t_cond,
        t_amb_c=t_amb_c,
        q_solar_w_m2=q_sol,
        h_amb=h_amb,
        mass=mass,
        thermal=thermal,
        vapor_gap_m=gap_eff,
        config=config,
        t_guess=t_guess,
    )
    if not math.isfinite(m_at_zero) or not math.isfinite(t_gel0):
        state0 = solve_steady_thermal(
            t_cond_c=t_cond,
            t_amb_c=t_amb_c,
            q_solar_w_m2=q_sol,
            m_des_kg_s_m2=0.0,
            h_amb=h_amb,
            params=thermal,
            h_m=h_m,
            t_guess=t_guess,
            vapor_gap_m=gap_eff,
        )
        m_des, t_gel, dc, state = 0.0, state0.t_gel_c, 0.0, state0
    elif m_at_zero <= 0.0:
        m_des, t_gel, dc, state = 0.0, t_gel0, dc0, state0
    else:
        hi = max(m_at_zero * 2.0, 1e-8)
        while hi < _M_DES_BRACKET_MAX and _m_des_residual(hi) > 0.0:
            hi *= 2.0
        if _m_des_residual(hi) >= 0.0:
            m_des, t_gel, dc, state = 0.0, t_gel0, dc0, state0
        else:
            try:
                m_star = float(brentq(_m_des_residual, 0.0, hi, xtol=1e-14))
            except ValueError:
                m_des, t_gel, dc, state = 0.0, t_gel0, dc0, state0
            else:
                m_des, t_gel, dc, state = _m_des_calc(
                    m_star,
                    loading=c_w,
                    h_m=h_m,
                    t_cond_c=t_cond,
                    t_amb_c=t_amb_c,
                    q_solar_w_m2=q_sol,
                    h_amb=h_amb,
                    mass=mass,
                    thermal=thermal,
                    vapor_gap_m=gap_eff,
                    config=config,
                    t_guess=(state0.t_gel_c, state0.t_abs_c, state0.t_glass_c),
                )

    _, dh, _ = evaluate_mass_rates(
        loading=c_w,
        h_m=h_m,
        t_gel_c=t_gel,
        t_cond_c=t_cond,
        rh=rh,
        phase="desorption",
        mass=mass,
        config=config,
        vapor_gap_m=vapor_gap_m,
    )

    h_conv_g = state.h_conv_g
    h_amb_for_cond = h_amb if h_amb_cond is None else h_amb_cond
    h_conv_cond = condenser_h_conv_w_m2_k(h_amb_for_cond, fin_area_ratio=fin_area_ratio)
    eps_gc = parallel_plate_emissivity(thermal.eps_gel, thermal.eps_al)
    q_rad = radiative_exchange_w_m2(t_gel, t_cond, emissivity=eps_gc)
    tmass = max(condenser_thermal_mass_j_m2_k, 1.0)
    dT_cond = (
        h_conv_g * (t_gel - t_cond)
        - h_conv_cond * (t_cond - t_amb_c)
        + m_des * h_fg_j_per_kg
        + q_rad
    ) / tmass

    return CoupledRates(
        dc_w_dt=dc,
        dH_dt=dh,
        dT_cond_dt=dT_cond,
        t_gel_c=t_gel,
        m_des_kg_s_m2=m_des,
        thermal=state,
    )

# =============================================================================
# SciPy Radau integration for Wilson half-cycles (coupled Eqs. 1-6 + Eq. 2)
# =============================================================================

_ODE_RTOL = 1e-4
_ODE_ATOL = 1e-7


@dataclass
class PhaseResult:
    time_s: np.ndarray
    c_w: np.ndarray
    H: np.ndarray
    t_cond_c: np.ndarray | None
    t_gel_c: np.ndarray
    water_collected_kg_m2: float
    m_des_kg_s_m2: np.ndarray
    # Surface temperatures along the trajectory. Populated by transient solvers and
    # by quasi_steady (algebraic Eqs 1/3/4 each step, with k=0 pinned when ICs set).
    t_abs_c: np.ndarray | None = None
    t_glass_c: np.ndarray | None = None


def _profile_index(t: float, dt_s: float, n: int) -> int:
    return min(int(t / dt_s), n - 1)


def _integrate_absorption(
    c_w0: float,
    h0: float,
    profile: PhaseProfile,
    config: DeviceConfig,
) -> PhaseResult:
    mass = config.mass_params()
    thermal = config.thermal_params()
    n = len(profile.temperature_c)
    dt = profile.dt_s
    t_span = (0.0, dt * n)
    t_eval = np.linspace(0.0, t_span[1], n + 1)
    h_min = config.hydrogel_thickness_m
    # Gel cannot swell into the condenser; keep ≥7 mm effective gap (Wilson §2.2).
    h_max = max(
        config.vapor_gap_m - VAPOR_GAP_TRANSPORT_MIN_M,
        h_min + 1e-6,
    )
    t_guess: tuple[float, float, float] | None = None

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal t_guess
        i = _profile_index(t, dt, n)
        h_m = max(float(y[1]), h_min)
        rates = evaluate_coupled_rates(
            c_w=float(y[0]),
            h_m=h_m,
            t_cond_c=profile.temperature_c[i],
            t_amb_c=profile.temperature_c[i],
            rh=profile.relative_humidity[i],
            q_solar_w_m2=0.0,
            h_amb=profile.h_amb_w_m2_k[i],
            phase="absorption",
            mass=mass,
            thermal=thermal,
            vapor_gap_m=config.vapor_gap_m,
            condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
            fin_area_ratio=config.fin_area_ratio,
            h_fg_j_per_kg=config.h_fg_j_per_kg,
            config=config,
            t_guess=t_guess,
        )
        dh = rates.dH_dt if h_m > h_min + 1e-12 else max(0.0, rates.dH_dt)
        if h_m >= h_max and dh > 0.0:
            dh = 0.0
        t_guess = _thermal_guess(rates.thermal)
        return np.array([rates.dc_w_dt, dh])

    sol = solve_ivp(
        rhs,
        t_span,
        y0=np.array([c_w0, max(h0, h_min)]),
        method="Radau",
        t_eval=t_eval,
        max_step=dt,
        rtol=_ODE_RTOL,
        atol=_ODE_ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"Absorption integration failed: {sol.message}")

    t_gel_hist: list[float] = []
    guess: tuple[float, float, float] | None = None
    for k in range(len(sol.t)):
        i = _profile_index(float(sol.t[k]), dt, n)
        rates = evaluate_coupled_rates(
            c_w=float(sol.y[0, k]),
            h_m=max(float(sol.y[1, k]), h_min),
            t_cond_c=profile.temperature_c[i],
            t_amb_c=profile.temperature_c[i],
            rh=profile.relative_humidity[i],
            q_solar_w_m2=0.0,
            h_amb=profile.h_amb_w_m2_k[i],
            phase="absorption",
            mass=mass,
            thermal=thermal,
            vapor_gap_m=config.vapor_gap_m,
            condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
            fin_area_ratio=config.fin_area_ratio,
            h_fg_j_per_kg=config.h_fg_j_per_kg,
            config=config,
            t_guess=guess,
        )
        guess = _thermal_guess(rates.thermal)
        t_gel_hist.append(rates.t_gel_c)

    c_w_out = np.array([clip_loading(float(v), config=config) for v in sol.y[0]])
    h_out = np.clip(sol.y[1], h_min, h_max)
    return PhaseResult(
        time_s=sol.t,
        c_w=c_w_out,
        H=h_out,
        t_cond_c=None,
        t_gel_c=np.array(t_gel_hist),
        water_collected_kg_m2=0.0,
        m_des_kg_s_m2=np.zeros(len(sol.t)),
    )


def _integrate_desorption(
    c_w0: float,
    h0: float,
    profile: PhaseProfile,
    config: DeviceConfig,
    *,
    t_guess0: tuple[float, float, float] | None = None,
) -> PhaseResult:
    mass = config.mass_params()
    thermal = config.thermal_params()
    tmass = config.condenser_thermal_mass_j_m2_k()
    n = len(profile.temperature_c)
    dt = profile.dt_s
    t_span = (0.0, dt * n)
    t_eval = np.linspace(0.0, t_span[1], n + 1)
    h_min = config.hydrogel_thickness_m
    surface_ic = config.desorption_surface_ic_c()
    if surface_ic is not None:
        t_gel_ic, t_abs_ic, t_glass_ic, t_cond_ic = (
            clamp_temperature_c(t) for t in surface_ic
        )
        t_cond0 = t_cond_ic
        t_guess0 = (t_gel_ic, t_abs_ic, t_glass_ic)
    else:
        t_cond0 = clamp_temperature_c(profile.temperature_c[0])
        if t_guess0 is None:
            t_amb = t_cond0
            t_guess0 = (t_amb, t_amb, t_amb)
    t_guess: tuple[float, float, float] | None = t_guess0

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal t_guess
        i = _profile_index(t, dt, n)
        h_m = max(float(y[1]), h_min)
        rates = evaluate_coupled_rates(
            c_w=float(y[0]),
            h_m=h_m,
            t_cond_c=float(y[2]),
            t_amb_c=profile.temperature_c[i],
            rh=profile.relative_humidity[i],
            q_solar_w_m2=profile.solar_w_m2[i],
            h_amb=profile.h_amb_w_m2_k[i],
            phase="desorption",
            mass=mass,
            thermal=thermal,
            vapor_gap_m=config.vapor_gap_m,
            condenser_thermal_mass_j_m2_k=tmass,
            fin_area_ratio=config.fin_area_ratio,
            h_fg_j_per_kg=config.h_fg_j_per_kg,
            config=config,
            t_guess=t_guess,
            h_amb_cond=(
                profile.h_amb_cond_w_m2_k[i]
                if profile.h_amb_cond_w_m2_k is not None
                else None
            ),
        )
        dh = rates.dH_dt if h_m > h_min + 1e-12 else 0.0
        dc = min(0.0, rates.dc_w_dt)
        dh = min(0.0, dh)
        t_guess = _thermal_guess(rates.thermal)
        return np.array([dc, dh, rates.dT_cond_dt])

    sol = solve_ivp(
        rhs,
        t_span,
        y0=np.array([c_w0, max(h0, h_min), t_cond0]),
        method="Radau",
        t_eval=t_eval,
        max_step=dt,
        rtol=_ODE_RTOL,
        atol=_ODE_ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"Desorption integration failed: {sol.message}")

    t_gel_hist: list[float] = []
    t_abs_hist: list[float] = []
    t_glass_hist: list[float] = []
    t_cond_hist: list[float] = []
    m_des_hist: list[float] = []
    guess: tuple[float, float, float] | None = t_guess0
    for k in range(len(sol.t)):
        i = _profile_index(float(sol.t[k]), dt, n)
        rates = evaluate_coupled_rates(
            c_w=float(sol.y[0, k]),
            h_m=max(float(sol.y[1, k]), h_min),
            t_cond_c=float(sol.y[2, k]),
            t_amb_c=profile.temperature_c[i],
            rh=profile.relative_humidity[i],
            q_solar_w_m2=profile.solar_w_m2[i],
            h_amb=profile.h_amb_w_m2_k[i],
            phase="desorption",
            mass=mass,
            thermal=thermal,
            vapor_gap_m=config.vapor_gap_m,
            condenser_thermal_mass_j_m2_k=tmass,
            fin_area_ratio=config.fin_area_ratio,
            h_fg_j_per_kg=config.h_fg_j_per_kg,
            config=config,
            t_guess=guess,
            h_amb_cond=(
                profile.h_amb_cond_w_m2_k[i]
                if profile.h_amb_cond_w_m2_k is not None
                else None
            ),
        )
        guess = _thermal_guess(rates.thermal)
        if k == 0 and surface_ic is not None:
            t_gel_hist.append(t_gel_ic)
            t_abs_hist.append(t_abs_ic)
            t_glass_hist.append(t_glass_ic)
        else:
            t_gel_hist.append(rates.t_gel_c)
            t_abs_hist.append(rates.thermal.t_abs_c)
            t_glass_hist.append(rates.thermal.t_glass_c)
        t_cond_hist.append(float(sol.y[2, k]))
        m_des_hist.append(rates.m_des_kg_s_m2)

    water = float(cumulative_desorption_yield_l_m2(sol.t, m_des_hist)[-1])

    c_w_out = np.array([clip_loading(float(v), config=config) for v in sol.y[0]])
    h_out = np.maximum(sol.y[1], h_min)
    return PhaseResult(
        time_s=sol.t,
        c_w=c_w_out,
        H=h_out,
        t_cond_c=np.array(t_cond_hist),
        t_gel_c=np.array(t_gel_hist),
        water_collected_kg_m2=max(0.0, water),
        m_des_kg_s_m2=np.array(m_des_hist),
        t_abs_c=np.array(t_abs_hist),
        t_glass_c=np.array(t_glass_hist),
    )


def _h_rad_w_m2_k(t_hot_c: float, t_cold_c: float, emissivity: float) -> float:
    """Linearized radiative exchange coefficient σε(Tₕ²+T_c²)(Tₕ+T_c) [W/m²K]."""
    if emissivity <= 0.0:
        return 0.0
    th = t_hot_c + 273.15
    tc = t_cold_c + 273.15
    return emissivity * STEFAN_BOLTZMANN_W_M2_K4 * (th * th + tc * tc) * (th + tc)


def cycle_end_state(des_res: PhaseResult) -> tuple[float, float]:
    """Gel state (c_w, H) at end of a full absorption–desorption cycle."""
    return float(des_res.c_w[-1]), float(des_res.H[-1])


def warmup_cycle_state(
    profile: DailyWeatherProfile,
    config: DeviceConfig,
    *,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
) -> tuple[float, float]:
    """Run one full cycle; return gel state after desorption (for next-day IC)."""
    _, _, _, des_res = run_daily_cycle(
        profile,
        config,
        c_w_initial=c_w_initial,
        h_initial=h_initial,
    )
    return cycle_end_state(des_res)


def warmup_to_cyclic_state(
    profile: DailyWeatherProfile,
    config: DeviceConfig,
    *,
    n_cycles: int = 2,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
) -> tuple[float, float]:
    """Run repeated daily cycles until a periodic post-desorption (c_w, H) is reached."""
    cw, h = c_w_initial, h_initial
    for _ in range(max(1, n_cycles)):
        _, _, _, des_res = run_daily_cycle(
            profile,
            config,
            c_w_initial=cw,
            h_initial=h,
        )
        cw, h = cycle_end_state(des_res)
    return cw, h


def find_cyclic_state(
    profile: DailyWeatherProfile,
    config: DeviceConfig,
    *,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
    tol: float = 1e-6,
    max_rounds: int = 10,
    stall_ratio: float = 0.5,
    stall_rounds: int = 2,
    verbose: bool = True,
) -> tuple[float, float]:
    """Find the steady periodic post-desorption (c_w, H) state for a profile
    repeated indefinitely, without brute-force warmup cycling.

    Plain fixed-point iteration (``warmup_to_cyclic_state``) can need 100+
    cycles to converge at sites where the one-cycle map's slowest eigenvalue
    is close to 1 (e.g. profiles averaged from strongly seasonal weather).
    This instead accelerates convergence with restarted vector Aitken Δ²
    extrapolation: each round applies the real map twice, then extrapolates
    the fixed point from those two real evaluations (no derivative estimate,
    so no finite-difference noise), typically converging in ~3-6 rounds.

    Some (profile, config) pairs have no single fixed point: the one-cycle
    map bifurcates into a stable period-2 orbit (an alternating "wetter
    day / drier day" pattern under the repeated forcing), so ``rel_step``
    plateaus (or wobbles) at a small-but-nonzero value instead of shrinking
    toward ``tol``. This is detected as ``stall_rounds`` consecutive rounds
    where ``rel_step`` fails to shrink by at least ``stall_ratio`` relative
    to the previous round, and handled by returning the average of the two
    most recent extrapolated states rather than an arbitrary snapshot (which
    would otherwise depend on exactly which round happened to hit
    ``max_rounds``). The same averaging is applied as a fallback if
    ``max_rounds`` is exhausted without the detector firing (e.g. a noisier,
    not-quite-periodic wobble). Callers wanting the true alternating yields
    rather than the yield from this averaged state should run one cycle
    from each of the two branches directly.
    """
    if h_initial is None:
        h_initial = config.hydrogel_thickness_m
    if c_w_initial is None:
        c_w_initial = initial_loading(config)
    x = np.array([c_w_initial, h_initial], dtype=float)

    def step(state: np.ndarray) -> np.ndarray:
        _, _, _, des_res = run_daily_cycle(
            profile, config, c_w_initial=float(state[0]), h_initial=float(state[1])
        )
        return np.array(cycle_end_state(des_res))

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
    return float(x[0]), float(x[1])


def run_daily_cycle(
    profile: DailyWeatherProfile,
    config: DeviceConfig,
    *,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
    cyclic_initial: bool = False,
    cyclic_warmup_cycles: int = 2,
) -> tuple[float, float, PhaseResult, PhaseResult]:
    """Run absorption then desorption; return (yield kg/m2, eta_thermal, abs_res, des_res).

    If ``cyclic_initial`` is True, first find the true steady periodic state via
    Aitken Δ² extrapolation (``find_cyclic_state``, ~3-6 rounds; see its docstring)
    rather than a fixed number of warmup cycles, then simulate one reporting day
    from that end state. ``cyclic_warmup_cycles`` is passed through as
    ``max_rounds`` (floored at 3, since a round is 2 full daily cycles).
    """
    if cyclic_initial:
        cw, h = find_cyclic_state(
            profile,
            config,
            c_w_initial=c_w_initial,
            h_initial=h_initial,
            max_rounds=max(3, cyclic_warmup_cycles),
            verbose=False,
        )
        c_w_initial, h_initial = cw, h

    h0 = config.hydrogel_thickness_m
    if h_initial is None:
        h_initial = h0
    if c_w_initial is None:
        c_w_initial = initial_loading(config)

    abs_res = _integrate_absorption(c_w_initial, h_initial, profile.absorption, config)
    des_res = _integrate_desorption(
        float(abs_res.c_w[-1]),
        float(abs_res.H[-1]),
        profile.desorption,
        config,
    )
    yield_kg = max(0.0, des_res.water_collected_kg_m2)

    q_solar_int = sum(
        profile.desorption.solar_w_m2[i] * profile.desorption.dt_s
        for i in range(len(profile.desorption.solar_w_m2))
    )
    eta = (yield_kg * config.h_fg_j_per_kg / q_solar_int) if q_solar_int > 0 else 0.0
    return yield_kg, eta, abs_res, des_res

# =============================================================================
# Device temperatures and weather time series for a daily SAWH cycle
# =============================================================================

@dataclass(frozen=True, slots=True)
class DetailedSeries:
    time_s: np.ndarray
    phase: np.ndarray
    absorption_end_s: float
    t_abs_c: np.ndarray
    t_glass_c: np.ndarray
    t_cond_c: np.ndarray
    t_gel_c: np.ndarray
    t_amb_c: np.ndarray
    relative_humidity: np.ndarray
    solar_w_m2: np.ndarray
    h_amb_w_m2_k: np.ndarray


def _phase_weather(
    time_s: np.ndarray,
    profile: PhaseProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(profile.temperature_c)
    dt = profile.dt_s
    t_amb: list[float] = []
    rh: list[float] = []
    solar: list[float] = []
    h_amb: list[float] = []
    for t in time_s:
        i = _profile_index(float(t), dt, n)
        t_amb.append(profile.temperature_c[i])
        rh.append(profile.relative_humidity[i])
        solar.append(profile.solar_w_m2[i])
        h_amb.append(profile.h_amb_w_m2_k[i])
    return (
        np.array(t_amb),
        np.array(rh),
        np.array(solar),
        np.array(h_amb),
    )


def detailed_series(
    profile: DailyWeatherProfile,
    abs_res: PhaseResult,
    des_res: PhaseResult,
    config: DeviceConfig,
) -> DetailedSeries:
    """Build full-cycle device and weather trajectories."""

    def _absorption_device_temps() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mass = config.mass_params()
        thermal = config.thermal_params()
        abs_profile = profile.absorption
        n = len(abs_profile.temperature_c)
        dt = abs_profile.dt_s
        h_min = config.hydrogel_thickness_m
        t_abs: list[float] = []
        t_glass: list[float] = []
        t_cond: list[float] = []
        t_gel: list[float] = []
        guess: tuple[float, float, float] | None = None

        for k in range(len(abs_res.time_s)):
            i = _profile_index(float(abs_res.time_s[k]), dt, n)
            h_m = max(float(abs_res.H[k]), h_min)
            t_amb = abs_profile.temperature_c[i]
            rates = evaluate_coupled_rates(
                c_w=float(abs_res.c_w[k]),
                h_m=h_m,
                t_cond_c=t_amb,
                t_amb_c=t_amb,
                rh=abs_profile.relative_humidity[i],
                q_solar_w_m2=0.0,
                h_amb=abs_profile.h_amb_w_m2_k[i],
                phase="absorption",
                mass=mass,
                thermal=thermal,
                vapor_gap_m=config.vapor_gap_m,
                condenser_thermal_mass_j_m2_k=config.condenser_thermal_mass_j_m2_k(),
                fin_area_ratio=config.fin_area_ratio,
                h_fg_j_per_kg=config.h_fg_j_per_kg,
                config=config,
                t_guess=guess,
            )
            guess = _thermal_guess(rates.thermal)
            t_abs.append(rates.thermal.t_abs_c)
            t_glass.append(rates.thermal.t_glass_c)
            t_cond.append(t_amb)
            t_gel.append(rates.t_gel_c)

        return (
            np.array(t_abs),
            np.array(t_glass),
            np.array(t_cond),
            np.array(t_gel),
        )

    def _desorption_device_temps() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if des_res.t_cond_c is None:
            raise ValueError("Desorption result missing condenser temperature history.")

        if des_res.t_abs_c is not None and des_res.t_glass_c is not None:
            return des_res.t_abs_c, des_res.t_glass_c, des_res.t_cond_c, des_res.t_gel_c

        mass = config.mass_params()
        thermal = config.thermal_params()
        tmass = config.condenser_thermal_mass_j_m2_k()
        des_profile = profile.desorption
        n = len(des_profile.temperature_c)
        dt = des_profile.dt_s
        h_min = config.hydrogel_thickness_m
        t_abs: list[float] = []
        t_glass: list[float] = []
        guess: tuple[float, float, float] | None = None

        for k in range(len(des_res.time_s)):
            i = _profile_index(float(des_res.time_s[k]), dt, n)
            h_m = max(float(des_res.H[k]), h_min)
            rates = evaluate_coupled_rates(
                c_w=float(des_res.c_w[k]),
                h_m=h_m,
                t_cond_c=float(des_res.t_cond_c[k]),
                t_amb_c=des_profile.temperature_c[i],
                rh=des_profile.relative_humidity[i],
                q_solar_w_m2=des_profile.solar_w_m2[i],
                h_amb=des_profile.h_amb_w_m2_k[i],
                phase="desorption",
                mass=mass,
                thermal=thermal,
                vapor_gap_m=config.vapor_gap_m,
                condenser_thermal_mass_j_m2_k=tmass,
                fin_area_ratio=config.fin_area_ratio,
                h_fg_j_per_kg=config.h_fg_j_per_kg,
                config=config,
                t_guess=guess,
                h_amb_cond=(
                    des_profile.h_amb_cond_w_m2_k[i]
                    if des_profile.h_amb_cond_w_m2_k is not None
                    else None
                ),
            )
            guess = _thermal_guess(rates.thermal)
            t_abs.append(rates.thermal.t_abs_c)
            t_glass.append(rates.thermal.t_glass_c)

        return (
            np.array(t_abs),
            np.array(t_glass),
            des_res.t_cond_c,
            des_res.t_gel_c,
        )

    abs_t_abs, abs_t_glass, abs_t_cond, abs_t_gel = _absorption_device_temps()
    des_t_abs, des_t_glass, des_t_cond, des_t_gel = _desorption_device_temps()

    abs_weather = _phase_weather(abs_res.time_s, profile.absorption)
    des_weather = _phase_weather(des_res.time_s, profile.desorption)

    t_abs_end = float(abs_res.time_s[-1]) if len(abs_res.time_s) else 0.0
    time_s = np.concatenate([abs_res.time_s, t_abs_end + des_res.time_s[1:]])

    def _join(abs_arr: np.ndarray, des_arr: np.ndarray) -> np.ndarray:
        return np.concatenate([abs_arr, des_arr[1:]])

    phase = np.array(
        ["absorption"] * len(abs_res.time_s) + ["desorption"] * (len(des_res.time_s) - 1),
        dtype=object,
    )

    return DetailedSeries(
        time_s=time_s,
        phase=phase,
        absorption_end_s=t_abs_end,
        t_abs_c=_join(abs_t_abs, des_t_abs),
        t_glass_c=_join(abs_t_glass, des_t_glass),
        t_cond_c=_join(abs_t_cond, des_t_cond),
        t_gel_c=_join(abs_t_gel, des_t_gel),
        t_amb_c=_join(abs_weather[0], des_weather[0]),
        relative_humidity=_join(abs_weather[1], des_weather[1]),
        solar_w_m2=_join(abs_weather[2], des_weather[2]),
        h_amb_w_m2_k=_join(abs_weather[3], des_weather[3]),
    )


def write_detailed_csv(path: Path, series: DetailedSeries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time_s",
                "time_h",
                "phase",
                "t_abs_c",
                "t_glass_c",
                "t_cond_c",
                "t_gel_c",
                "t_amb_c",
                "relative_humidity",
                "solar_w_m2",
                "h_amb_w_m2_k",
            ]
        )
        for k in range(len(series.time_s)):
            w.writerow(
                [
                    f"{float(series.time_s[k]):.3f}",
                    f"{float(series.time_s[k]) / 3600.0:.6f}",
                    series.phase[k],
                    f"{float(series.t_abs_c[k]):.4f}",
                    f"{float(series.t_glass_c[k]):.4f}",
                    f"{float(series.t_cond_c[k]):.4f}",
                    f"{float(series.t_gel_c[k]):.4f}",
                    f"{float(series.t_amb_c[k]):.4f}",
                    f"{float(series.relative_humidity[k]):.6f}",
                    f"{float(series.solar_w_m2[k]):.2f}",
                    f"{float(series.h_amb_w_m2_k[k]):.4f}",
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
    phase_mark_h = series.absorption_end_s / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax_t, ax_wx, ax_sol = axes

    ax_t.plot(time_h, series.t_abs_c, color="#8b2000", linewidth=1.8, label="Absorber")
    ax_t.plot(time_h, series.t_glass_c, color="#b06000", linewidth=1.8, label="Glass")
    ax_t.plot(time_h, series.t_cond_c, color="#1a5a7a", linewidth=1.8, label="Condenser")
    ax_t.plot(time_h, series.t_gel_c, color="#6a3d9a", linewidth=1.4, linestyle="--", label="Gel")
    ax_t.plot(
        time_h,
        series.t_amb_c,
        color="0.45",
        linewidth=1.2,
        linestyle=":",
        label="Ambient (weather)",
    )
    ax_t.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_t.set_ylabel("Temperature (°C)")
    ax_t.legend(loc="upper left", fontsize=8, ncol=2)
    ax_t.grid(True, alpha=0.3)

    ax_wx.plot(time_h, series.t_amb_c, color="#d95f02", linewidth=1.6, label="T_amb")
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

    ax_sol.plot(time_h, series.solar_w_m2, color="#e6ab02", linewidth=1.8, label="Solar")
    ax_sol.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_sol.set_ylabel("Solar (W/m²)", color="#e6ab02")
    ax_sol.tick_params(axis="y", labelcolor="#e6ab02")
    ax_sol.grid(True, alpha=0.3)

    ax_h = ax_sol.twinx()
    ax_h.plot(time_h, series.h_amb_w_m2_k, color="#7570b3", linewidth=1.4, label="h_amb")
    ax_h.set_ylabel("h_amb (W/m²K)", color="#7570b3")
    ax_h.tick_params(axis="y", labelcolor="#7570b3")

    lines_l, labels_l = ax_sol.get_legend_handles_labels()
    lines_r, labels_r = ax_h.get_legend_handles_labels()
    ax_sol.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", fontsize=8)

    ax_sol.set_xlabel("Time (h)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# =============================================================================
# Water-in-sorbent inventory time series from a daily absorption-desorption cycle
# =============================================================================

@dataclass(frozen=True, slots=True)
class WaterInventorySeries:
    time_s: np.ndarray
    water_l_m2: np.ndarray
    phase: np.ndarray
    absorption_end_s: float
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


def water_inventory_series(
    abs_res: PhaseResult,
    des_res: PhaseResult,
    *,
    config: DeviceConfig,
) -> WaterInventorySeries:
    """Concatenate absorption and desorption phases into one sorbent water trajectory."""
    w_abs = np.array(
        [
            water_in_sorbent_l_m2(float(c), float(h), config=config)
            for c, h in zip(abs_res.c_w, abs_res.H)
        ]
    )
    w_des = np.array(
        [
            water_in_sorbent_l_m2(float(c), float(h), config=config)
            for c, h in zip(des_res.c_w, des_res.H)
        ]
    )
    t_abs_end = float(abs_res.time_s[-1]) if len(abs_res.time_s) else 0.0
    time_s = np.concatenate([abs_res.time_s, t_abs_end + des_res.time_s[1:]])
    water_l_m2 = np.concatenate([w_abs, w_des[1:]])
    phase = np.array(
        ["absorption"] * len(w_abs) + ["desorption"] * (len(w_des) - 1),
        dtype=object,
    )
    collected_des = cumulative_desorption_yield_l_m2(
        des_res.time_s, des_res.m_des_kg_s_m2
    )
    collected_water_l_m2 = np.zeros(len(time_s), dtype=float)
    collected_water_l_m2[len(w_abs) - 1 :] = collected_des
    return WaterInventorySeries(
        time_s=time_s,
        water_l_m2=water_l_m2,
        phase=phase,
        absorption_end_s=t_abs_end,
        collected_water_l_m2=collected_water_l_m2,
    )


def write_water_inventory_csv(path: Path, series: WaterInventorySeries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "time_h", "phase", "water_in_sorbent_l_m2", "collected_water_l_m2"])
        for t, ph, w_l, y_l in zip(
            series.time_s,
            series.phase,
            series.water_l_m2,
            series.collected_water_l_m2,
        ):
            w.writerow(
                [
                    f"{float(t):.3f}",
                    f"{float(t) / 3600.0:.6f}",
                    ph,
                    f"{float(w_l):.6f}",
                    f"{float(y_l):.6f}",
                ]
            )


def plot_water_inventory(
    path: Path,
    series: WaterInventorySeries,
    *,
    config: DeviceConfig,
    title: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_h = series.time_s / 3600.0
    phase_mark_h = series.absorption_end_s / 3600.0
    ylabel = inventory_ylabel(config)
    fig, (ax_inv, ax_yield) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax_inv.plot(time_h, series.water_l_m2, color="#4C72B0", linewidth=2)
    ax_inv.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_inv.set_ylabel(ylabel)
    ax_inv.grid(True, alpha=0.3)

    ax_yield.plot(time_h, series.collected_water_l_m2, color="#C44E52", linewidth=2)
    ax_yield.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_yield.set_xlabel("Time (h)")
    ax_yield.set_ylabel("Collected water (L/m²)")
    ax_yield.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# =============================================================================
# Site-level salt feasibility and LCOW simulation helpers for global maps
# =============================================================================

FAIL_LCO: float = 1e30


@dataclass(slots=True)
class SaltSimResult:
    feasible: bool
    lcow: float
    yield_kg_m2: float
    eta_thermal: float
    gel_temperature_c: float
    desorption_aw: float
    failure_reason: str = ""


def profile_diagnostics(profile: DailyWeatherProfile) -> dict[str, float]:
    """Extract absorption/desorption extrema from a daily weather profile."""
    rh_abs = max(profile.absorption.relative_humidity)
    rh_des = max(profile.desorption.relative_humidity)
    rh_high = max(rh_abs, rh_des)
    rh_low = min(
        min(profile.absorption.relative_humidity),
        min(profile.desorption.relative_humidity),
    )
    temp_high = max(max(profile.desorption.temperature_c), max(profile.absorption.temperature_c))
    temp_low = min(min(profile.desorption.temperature_c), min(profile.absorption.temperature_c))
    solar_max = max(profile.desorption.solar_w_m2)
    return {
        "rh_high": float(rh_high),
        "rh_low": float(rh_low),
        "temp_high_c": float(temp_high),
        "temp_low_c": float(temp_low),
        "solar_irradiance_w_per_m2": float(solar_max),
    }


def passive_gel_temperature_c(profile: DailyWeatherProfile, config: DeviceConfig) -> float:
    """Passive sun-only gel temperature at peak desorption conditions."""
    des = profile.desorption
    i_peak = int(np.argmax(des.solar_w_m2))
    thermal = config.thermal_params()
    state = solve_steady_thermal(
        t_cond_c=des.temperature_c[i_peak],
        t_amb_c=des.temperature_c[i_peak],
        q_solar_w_m2=des.solar_w_m2[i_peak],
        m_des_kg_s_m2=0.0,
        h_amb=des.h_amb_w_m2_k[i_peak],
        params=thermal,
        h_m=config.hydrogel_thickness_m,
        vapor_gap_m=config.vapor_gap_m,
    )
    return float(state.t_gel_c)


def salt_climate_feasible(
    salt: SaltProperties,
    rh_abs: float,
    t_cond_c: float,
    t_gel_c: float,
) -> tuple[bool, str]:
    """Check DRH window on absorption RH and desorption water activity."""
    if not (salt.rh_min <= rh_abs <= salt.rh_max):
        return False, f"absorption RH {rh_abs:.3f} outside [{salt.rh_min}, {salt.rh_max}]"
    aw_des = desorption_water_activity(t_cond_c, t_gel_c)
    if not math.isfinite(aw_des):
        return False, "desorption a_w undefined"
    if not (salt.rh_min <= aw_des <= salt.rh_max):
        return False, f"desorption a_w {aw_des:.3f} outside [{salt.rh_min}, {salt.rh_max}]"
    return True, ""


def simulate_salt_lcow(
    profile: DailyWeatherProfile,
    config: DeviceConfig,
    econ: LCOEconomicParams | None = None,
    *,
    rh_abs: float | None = None,
    skip_feasibility: bool = False,
    cyclic_initial: bool = True,
    cyclic_warmup_cycles: int = 1,
    verbose: bool = True,
) -> SaltSimResult:
    """Run one cyclic daily simulation and return LCOW plus diagnostics."""
    econ = econ or LCOEconomicParams()
    salt = get_salt(config.salt_name)
    diag = profile_diagnostics(profile)
    rh_uptake = rh_abs if rh_abs is not None else diag["rh_high"]
    t_cond = diag["temp_high_c"]
    t_gel_passive = passive_gel_temperature_c(profile, config)

    if not skip_feasibility:
        ok, reason = salt_climate_feasible(salt, rh_uptake, t_cond, t_gel_passive)
        if not ok:
            return SaltSimResult(
                feasible=False,
                lcow=FAIL_LCO,
                yield_kg_m2=float("nan"),
                eta_thermal=float("nan"),
                gel_temperature_c=t_gel_passive,
                desorption_aw=desorption_water_activity(t_cond, t_gel_passive),
                failure_reason=reason,
            )

    if verbose:
        if cyclic_initial:
            msg = (
                f"running ODE ({cyclic_warmup_cycles} warmup day(s) + 1 reporting day, "
                f"~30–90s/salt)…"
            )
        else:
            msg = "running ODE (1 day, ~30s/salt)…"
        print(msg, end="", flush=True)

    try:
        yield_kg, eta, _, des_res = run_daily_cycle(
            profile,
            config,
            cyclic_initial=cyclic_initial,
            cyclic_warmup_cycles=cyclic_warmup_cycles,
        )
    except Exception as exc:
        return SaltSimResult(
            feasible=False,
            lcow=FAIL_LCO,
            yield_kg_m2=float("nan"),
            eta_thermal=float("nan"),
            gel_temperature_c=t_gel_passive,
            desorption_aw=desorption_water_activity(t_cond, t_gel_passive),
            failure_reason=str(exc).split("\n", 1)[0][:240],
        )

    if not math.isfinite(yield_kg) or yield_kg <= 0.0:
        t_gel = float(np.mean(des_res.t_gel_c)) if len(des_res.t_gel_c) else t_gel_passive
        t_cond_mean = float(np.mean(des_res.t_cond_c)) if des_res.t_cond_c is not None else t_cond
        return SaltSimResult(
            feasible=False,
            lcow=FAIL_LCO,
            yield_kg_m2=max(0.0, yield_kg),
            eta_thermal=eta,
            gel_temperature_c=t_gel,
            desorption_aw=desorption_water_activity(t_cond_mean, t_gel),
            failure_reason="zero or invalid yield",
        )

    lcow = lcow_from_daily_yield(
        yield_kg,
        salt_name=config.salt_name,
        salt_to_polymer_ratio=config.salt_to_polymer_ratio,
        hydrogel_thickness_m=config.hydrogel_thickness_m,
        econ=econ,
    )
    t_gel = float(np.mean(des_res.t_gel_c))
    t_cond_mean = float(np.mean(des_res.t_cond_c)) if des_res.t_cond_c is not None else t_cond
    aw_des = desorption_water_activity(t_cond_mean, t_gel)

    if not math.isfinite(lcow) or lcow <= 0.0:
        return SaltSimResult(
            feasible=False,
            lcow=FAIL_LCO,
            yield_kg_m2=yield_kg,
            eta_thermal=eta,
            gel_temperature_c=t_gel,
            desorption_aw=aw_des,
            failure_reason="invalid LCOW",
        )

    return SaltSimResult(
        feasible=True,
        lcow=lcow,
        yield_kg_m2=yield_kg,
        eta_thermal=eta,
        gel_temperature_c=t_gel,
        desorption_aw=aw_des,
    )

# =============================================================================
# Annual yield aggregation over real weather days
# =============================================================================

@dataclass(frozen=True, slots=True)
class SimulationResult:
    mean_daily_yield_kg_m2: float
    mean_daily_yield_l_m2: float
    mean_thermal_efficiency: float
    n_days: int
    daily_yields_kg_m2: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DailySimulationRecord:
    date: date
    day_of_year: int
    rh_avg_frac: float
    rh_peak_frac: float
    temp_avg_c: float
    temp_peak_c: float
    solar_avg_w_m2: float
    solar_peak_w_m2: float
    daily_yield_kg_m2: float
    daily_yield_l_m2: float
    eta_thermal: float
    water_uptake_l_m2: float
    water_release_l_m2: float
    t_abs_peak_c: float
    t_glass_peak_c: float
    t_cond_peak_c: float
    t_gel_peak_c: float


DAILY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "date",
    "day_of_year",
    "rh_avg_frac",
    "rh_peak_frac",
    "temp_avg_c",
    "temp_peak_c",
    "solar_avg_w_m2",
    "solar_peak_w_m2",
    "daily_yield_kg_m2",
    "daily_yield_l_m2",
    "eta_thermal",
    "water_uptake_l_m2",
    "water_release_l_m2",
    "t_abs_peak_c",
    "t_glass_peak_c",
    "t_cond_peak_c",
    "t_gel_peak_c",
)


def simulate_annual_year(
    day_items: list[tuple[date, DailyWeatherProfile, pd.DataFrame]],
    config: DeviceConfig,
    *,
    warmup_cycles: int = 2,
    save_daily_timeseries: bool = False,
    timeseries_dir: Path | None = None,
    progress_callback: Callable[[int, int, date], None] | None = None,
) -> list[DailySimulationRecord]:
    """Simulate a sequential year, warming up on Jan 1 weather before recording."""
    if not day_items:
        return []

    def _record_from_day(
        day_key: date,
        profile: DailyWeatherProfile,
        day_df: pd.DataFrame,
        *,
        c_w_initial: float | None,
        h_initial: float | None,
    ) -> tuple[DailySimulationRecord, tuple[float, float], DetailedSeries, WaterInventorySeries]:
        yield_kg, eta, abs_res, des_res = run_daily_cycle(
            profile,
            config,
            c_w_initial=c_w_initial,
            h_initial=h_initial,
        )
        detailed = detailed_series(profile, abs_res, des_res, config)
        inventory = water_inventory_series(abs_res, des_res, config=config)
        weather = day_weather_stats(day_df)

        abs_mask = inventory.time_s <= inventory.absorption_end_s + 1e-9
        water_abs = inventory.water_l_m2[abs_mask]
        water_uptake_l_m2 = (
            max(0.0, float(water_abs.max()) - float(water_abs[0])) if len(water_abs) else 0.0
        )

        record = DailySimulationRecord(
            date=day_key,
            day_of_year=day_key.timetuple().tm_yday,
            rh_avg_frac=weather.get("rh_avg_frac", 0.0),
            rh_peak_frac=weather.get("rh_peak_frac", 0.0),
            temp_avg_c=weather.get("temp_avg_c", 0.0),
            temp_peak_c=weather.get("temp_peak_c", 0.0),
            solar_avg_w_m2=weather.get("solar_avg_w_m2", 0.0),
            solar_peak_w_m2=weather.get("solar_peak_w_m2", 0.0),
            daily_yield_kg_m2=float(yield_kg),
            daily_yield_l_m2=float(yield_kg),
            eta_thermal=float(eta),
            water_uptake_l_m2=water_uptake_l_m2,
            water_release_l_m2=float(yield_kg),
            t_abs_peak_c=float(detailed.t_abs_c.max()),
            t_glass_peak_c=float(detailed.t_glass_c.max()),
            t_cond_peak_c=float(detailed.t_cond_c.max()),
            t_gel_peak_c=float(detailed.t_gel_c.max()),
        )
        return record, cycle_end_state(des_res), detailed, inventory

    jan1 = date(day_items[0][0].year, 1, 1)
    warmup_profile = next(
        (prof for day_key, prof, _ in day_items if day_key == jan1),
        day_items[0][1],
    )
    cw, h = None, None
    for _ in range(max(0, warmup_cycles)):
        _, _, _, des_res = run_daily_cycle(warmup_profile, config, c_w_initial=cw, h_initial=h)
        cw, h = cycle_end_state(des_res)

    records: list[DailySimulationRecord] = []
    n_days = len(day_items)
    for i, (day_key, profile, day_df) in enumerate(day_items):
        record, (cw, h), detailed, inventory = _record_from_day(
            day_key,
            profile,
            day_df,
            c_w_initial=cw,
            h_initial=h,
        )
        records.append(record)

        if save_daily_timeseries and timeseries_dir is not None:
            day_tag = day_key.isoformat()
            write_detailed_csv(
                timeseries_dir / f"{day_tag}_diagnostics.csv",
                detailed,
            )
            write_water_inventory_csv(
                timeseries_dir / f"{day_tag}_water_inventory.csv",
                inventory,
            )

        if progress_callback is not None:
            progress_callback(i + 1, n_days, day_key)

    return records


def write_daily_summary_csv(
    path: Path,
    records: list[DailySimulationRecord],
) -> None:
    """Write one row per simulated day."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DAILY_SUMMARY_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "date": rec.date.isoformat(),
                    "day_of_year": rec.day_of_year,
                    "rh_avg_frac": f"{rec.rh_avg_frac:.6f}",
                    "rh_peak_frac": f"{rec.rh_peak_frac:.6f}",
                    "temp_avg_c": f"{rec.temp_avg_c:.4f}",
                    "temp_peak_c": f"{rec.temp_peak_c:.4f}",
                    "solar_avg_w_m2": f"{rec.solar_avg_w_m2:.2f}",
                    "solar_peak_w_m2": f"{rec.solar_peak_w_m2:.2f}",
                    "daily_yield_kg_m2": f"{rec.daily_yield_kg_m2:.6f}",
                    "daily_yield_l_m2": f"{rec.daily_yield_l_m2:.6f}",
                    "eta_thermal": f"{rec.eta_thermal:.6f}",
                    "water_uptake_l_m2": f"{rec.water_uptake_l_m2:.6f}",
                    "water_release_l_m2": f"{rec.water_release_l_m2:.6f}",
                    "t_abs_peak_c": f"{rec.t_abs_peak_c:.4f}",
                    "t_glass_peak_c": f"{rec.t_glass_peak_c:.4f}",
                    "t_cond_peak_c": f"{rec.t_cond_peak_c:.4f}",
                    "t_gel_peak_c": f"{rec.t_gel_peak_c:.4f}",
                }
            )


def aggregate_yields(
    day_profiles: list[tuple[date, DailyWeatherProfile]] | list[DailyWeatherProfile],
    config: DeviceConfig,
    *,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
    warmup: bool = False,
) -> SimulationResult:
    yields: list[float] = []
    etas: list[float] = []
    cw, h = c_w_initial, h_initial
    for i, item in enumerate(day_profiles):
        prof = item[1] if isinstance(item, tuple) else item
        y, eta, _, des_res = run_daily_cycle(prof, config, c_w_initial=cw, h_initial=h)
        cw, h = cycle_end_state(des_res)
        if warmup and i == 0:
            continue
        if y >= 0.0:
            yields.append(y)
            etas.append(eta)
    if not yields:
        return SimulationResult(0.0, 0.0, 0.0, 0, tuple())
    mean_y = sum(yields) / len(yields)
    mean_eta = sum(etas) / len(etas)
    return SimulationResult(
        mean_daily_yield_kg_m2=mean_y,
        mean_daily_yield_l_m2=mean_y,
        mean_thermal_efficiency=mean_eta,
        n_days=len(yields),
        daily_yields_kg_m2=tuple(yields),
    )
