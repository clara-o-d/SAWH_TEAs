"""Simulation: system config, coupled thermal/mass dynamics, ODE integration, detailed
plotting, water-inventory accounting, site feasibility, and annual yield."""

from __future__ import annotations

import csv
import dataclasses
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

from solar_lumped._parameters_xlsx import physics_value as _pv
from solar_lumped.complex_model import ComplexOptions
from solar_lumped.physics import (
    concentration_ratio_absorption,
    _absorption_effective_water_activity,
    dc_w_dt_from_m_des,
    dh_dt_from_dc_w,
    equilibrium_c_w_absorption,
    equilibrium_t_gel_desorption_c,
    FABRICATION_EQUILIBRIUM_RH,
    CP_AL_J_KG_K,
    DRY_COMPOSITE_DENSITY_KG_M3,
    EPS_ABS,
    EPS_AL,
    EPS_GEL,
    FIN_AREA_RATIO,
    G_CHAMBER_M_S,
    H0_M,
    H_DES_J_PER_KG,
    H_FG_J_PER_KG,
    L_C_M,
    L_G_M,
    L_INS_M,
    RHO_AL_KG_M3,
    SALT_LOADING_DEFAULT,
    SaltProperties,
    TAU_GLASS,
    TILT_DEG,
    GEL_CONDENSER_CLEARANCE_M,
    WATER_MOLAR_MASS_KG_MOL,
    SystemThermalParams,
    MassTransferParams,
    ThermalState,
    clamp_temperature_c,
    # Private, deliberately: this is the exact a_w the desorption ODE differences against
    # c_r, ZSR branches included. Re-deriving it here would let the plot drift.
    _mass_transfer_driving_force,
    concentration_ratio_desorption,
    condenser_h_conv_w_m2_k,
    h_amb_density_factor,
    pressure_from_elevation_m,
    dc_w_dt,
    deliquescence_rh,
    evaluate_mass_rates,
    get_salt,
    dilution_ceiling_c_w,
    drh_floor_c_w,
    vapor_gap_h_conv_w_m2_k,
    hydrate_floor_c_w,
    initial_loading,
    inventory_ylabel,
    m_des_kg_s_m2_from_dc_w,
    parallel_plate_emissivity,
    radiative_exchange_w_m2,
    salt_molarity_from_composite,
    ISOSTERIC_H_DES_SALTS,
    brine_salt_fraction_from_c_w,
    solve_steady_thermal,
    water_in_sorbent_l_m2,
)
from solar_lumped.weather import DailyWeatherProfile, PhaseProfile, day_weather_stats


# --- System configuration dataclass ---

@dataclass(frozen=True, slots=True)
class SystemConfig:
    # "isosteric" derives h_des(xi, T) from the salt's own isotherm by
    # Clausius-Clapeyron (physics.isosteric_h_des_j_per_kg); "constant" keeps the
    # tabulated scalar. Only LiCl/CaCl2 have temperature-dependent isotherms, so
    # every other salt falls back to the constant regardless of this setting.
    h_des_mode: str = "isosteric"
    salt_name: str = "LiCl"
    salt_loading: float = SALT_LOADING_DEFAULT
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
    thermal: SystemThermalParams | None = None
    # Override catalog salt formula weight (g/mol) for sensitivity sweeps.
    salt_formula_weight_g_mol: float | None = None
    # Scales MW_salt in the gravimetric-uptake bookkeeping (dry composite mass).
    salt_weight_factor: float = _pv("Salt weight factor (dry-mass MW scaling)")
    # Uniform surface/gel temperature at desorption start. None → algebraic steady
    # state (quasi_steady solves Eqs 1/3/4 algebraically each ODE step).
    segregated_initial_temp_c: float | None = None
    # Per-component desorption-start temperatures (T_gel, T_abs, T_glass, T_cond) in °C;
    # takes precedence over ``segregated_initial_temp_c``.
    coupled_initial_temps_c: tuple[float, float, float, float] | None = None
    # Complex-fidelity option set (A1/B1/B2/B3/B4/B8). None (default) runs the simple
    # Wilson model, which is what gpu_sweep's JAX path and every existing script keep
    # doing -- see solar_lumped/complex_model.py.
    complex: ComplexOptions | None = None
    # False (default): T_cond is a third ODE state (Eq. 2), radiatively/convectively
    # coupled to the gel and ambient. True: the idealized limit of a condenser with
    # infinite cooling capacity, T_cond == T_amb at every instant -- an upper bound
    # on desorption driving force, not a physical design.
    condenser_tracks_ambient: bool = False
    # Which physical limit stops desorption. "hydrate" (default): the crystal-hydrate
    # water content, n*c_s -- the driest the gel can get before dehydration chemistry
    # this model doesn't have takes over. "drh": the equilibrium c_w at the salt's
    # deliquescence RH, i.e. stop where the brine saturates and the activity
    # correlations stop describing a real solution. "drh" is the wetter, more
    # conservative bound and yields less. See physics.hydrate_floor_c_w / drh_floor_c_w.
    c_w_floor_mode: Literal["hydrate", "drh"] = "hydrate"
    # Third ideal case, alongside perfect optics and condenser_tracks_ambient. True:
    # the g -> infinity limit -- absorption and desorption are instantaneous and the
    # gel is on its equilibrium isotherm at every instant (a_w == RH absorbing,
    # a_w == c_r(T_gel, T_cond) desorbing). Desorption becomes energy-limited rather
    # than transport-limited: m_des is whatever Eqs. 1-4 can pay h_des for. An upper
    # bound on sorption kinetics, not a physical design. See
    # physics.mass_transfer_g_m_s.
    instant_equilibrium: bool = False
    # Site elevation (m). Thins the air: D_air rises as 1/p (more desorption) while h_amb
    # falls as rho^n (a hotter condenser, less desorption), so the net is a competition
    # rather than a free gain. 0.0 -- sea level -- reproduces the previous behaviour
    # bit-for-bit. Real sites should set it; site_pressure_pa() is what physics consumes.
    site_elevation_m: float = 0.0

    def site_pressure_pa(self) -> float:
        """ISA ambient pressure for this site's elevation."""
        return pressure_from_elevation_m(self.site_elevation_m)

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

    def hydrogel_floor_thickness_m(self) -> float:
        """Thinnest the gel can get: the thickness at the hydrate floor ``c_w_min``.

        The floor used to be H₀ itself, which is the *as-cast* thickness at the
        fabrication equilibrium (~20% RH) -- but desorption drives the gel well below
        that, and a hydrogel that has lost water is thinner than as-cast. Pinning it at
        H₀ made the gel shed water at constant volume for up to half the desorption
        window, and left dH/dt bounded in a different coordinate than dc_w/dt (which
        physics.dc_w_dt already clamps at c_w_min/c_w_max), so the two states could
        disagree about whether the gel was still drying.

        No new parameter: Eq. 6 is dH/dt = dc_w/dt·(MW·H₀/ρ_sol), i.e. the gel's volume
        changes by the volume of water it gains or loses. Integrating from (H₀, c_w,fab)
        -- which run_daily_cycle pairs by construction -- and evaluating at the hydrate
        floor gives H₀·[1 + (c_w_min − c_w,fab)·MW/ρ_sol]. For baseline LiCl that is
        2.373 mm, 59.3% of H₀.

        This is a bound on how thin, not a prediction: it extrapolates a relation
        linearized about the fabrication state, and a real PAM network resists collapse
        before the brine-volume argument says it must. The 1e-4 backstop is numerical
        only -- U_gel = k/H divides by this -- and is unreachable for any real brine,
        where the bracket stays above ~0.44 even at c_w_min = 0.

        Capped at H₀: for a salt whose hydrate floor sits *above* the fabrication water
        content (CaCl₂ at these loadings), the relation would put the floor above the
        as-cast thickness, and ``max(h0, h_min)`` in the integrators would silently
        inflate the initial gel. Such a salt simply cannot dry below its fabrication
        state, so H₀ is the floor and behaviour is unchanged from before this bound
        existed.
        """
        mass = self.mass_params()
        shrink = 1.0 + (
            (mass.c_w_min_mol_m3 - initial_loading(self))
            * WATER_MOLAR_MASS_KG_MOL
            / mass.rho_solution_kg_m3
        )
        return min(self.hydrogel_thickness_m, max(1e-4, self.hydrogel_thickness_m * shrink))

    def mass_params(self) -> MassTransferParams:
        s = self.salt()
        fw = (
            self.salt_formula_weight_g_mol
            if self.salt_formula_weight_g_mol is not None
            else s.formula_weight_g_mol
        )
        # Complex mode (B8) always routes through ZSR, including the pure-salt
        # corners where the mixing rule reduces to a single isotherm anyway.
        #
        # That is deliberate symmetry with the JAX backend, which has no choice but
        # to read every water activity off the tabulated inversion. Letting the CPU
        # take a closed-form shortcut here reintroduces a real divergence: the
        # closed-form LiCl isotherm returns NaN above its validity limit (xi > 0.56)
        # and silently stalls desorption, while the table clamps at saturation and
        # keeps going. The simple model never gets the gel dry enough to notice, but
        # an evacuated double-glazed absorber runs ~10 C hotter and crosses it --
        # which showed up as a 35% backend disagreement. Same inversion on both
        # paths, no shortcut, no divergence.
        blend_weights = None
        if self.complex is not None:
            blend_weights = self.complex.blend_weights
            if self.salt_formula_weight_g_mol is None:
                # The salt inventory is fixed at fabrication, so the blend's
                # effective formula weight is pinned at the fabrication
                # equilibrium RH (see zsr_effective_formula_weight_g_mol).
                from solar_lumped.complex_model import zsr_effective_formula_weight_g_mol

                fw_blend = zsr_effective_formula_weight_g_mol(
                    blend_weights, reference_rh=FABRICATION_EQUILIBRIUM_RH
                )
                if math.isfinite(fw_blend):
                    fw = fw_blend
        c_s = salt_molarity_from_composite(
            self.salt_loading, self.hydrogel_density_kg_m3, fw
        )
        return MassTransferParams(
            p_atm_pa=self.site_pressure_pa(),
            g_conv_m_s=self.g_conv_m_s,
            h0_ref_m=self.hydrogel_thickness_m,
            vapor_gap_m=self.vapor_gap_m,
            tilt_deg=self.tilt_deg,
            c_s_mol_m3=c_s,
            ions_per_formula=s.ions_per_formula,
            rho_solution_kg_m3=s.rho_solution_kg_m3,
            c_w_min_mol_m3=(
                hydrate_floor_c_w(
                    c_s_mol_m3=c_s, salt_name=s.name, blend_weights=blend_weights
                )
                if self.c_w_floor_mode == "hydrate"
                else drh_floor_c_w(
                    c_s_mol_m3=c_s, salt_name=s.name, formula_weight_g_mol=fw,
                    blend_weights=blend_weights,
                )
            ),
            c_w_max_mol_m3=dilution_ceiling_c_w(
                c_s_mol_m3=c_s, salt_name=s.name, formula_weight_g_mol=fw,
                blend_weights=blend_weights,
            ),
            salt_name=s.name,
            formula_weight_g_mol=fw,
            salt_loading=self.salt_loading,
            salt_weight_factor=self.salt_weight_factor,
            blend_weights=blend_weights,
            instant_equilibrium=self.instant_equilibrium,
        )

    def thermal_params(self) -> SystemThermalParams:
        if self.thermal is not None:
            # site_elevation_m is the single source of truth for pressure, so an explicit
            # thermal override gets it applied rather than returned untouched. site_sweep's
            # build_system_config ALWAYS passes a thermal, so returning it as-is ran every
            # site in the global sweep at sea level no matter what elevation was set.
            return dataclasses.replace(self.thermal, p_atm_pa=self.site_pressure_pa())
        if self.salt_name == "LiCl":
            # Wilson Table S3 COMSOL value (2320 kJ/kg), not the broader Díaz-Marín
            # literature range in salt_heat_of_desorption.csv (~2850 kJ/kg).
            h_des = H_DES_J_PER_KG
        else:
            h_des = self.salt().h_des_j_per_kg
        # Isosteric h_des(xi, T) needs a Conde-backed isotherm; anything else keeps the
        # tabulated constant. Blends are excluded: ZSR gives a mixture a_w, and the
        # per-salt Clausius-Clapeyron slope is not defined on it.
        # Restricted to the simple path on purpose. Complex mode routes a_w through the
        # ZSR mixture (even for a single-weight "blend"), and the per-salt
        # Clausius-Clapeyron slope is not defined on a mixture activity -- taking it from
        # self.salt_name there would silently pair LiCl's slope with, say, MgCl2's a_w.
        h_des_salt = (
            self.salt_name
            if self.h_des_mode == "isosteric"
            and self.complex is None
            and self.salt_name in ISOSTERIC_H_DES_SALTS
            else None
        )
        if self.complex is not None:
            cx = self.complex
            # B1 puts eps_abs_ir under continuous optimizer control (it is a priced
            # coating choice, not a fixed case flag); B2 sets the stack's
            # transmittance and pane count.
            return SystemThermalParams(
                p_atm_pa=self.site_pressure_pa(),
                insulation_gap_m=self.insulation_gap_m,
                vapor_gap_m=self.vapor_gap_m,
                eps_abs=EPS_ABS,
                tau_glass=cx.tau_glass,
                eps_gel=EPS_GEL,
                eps_al=EPS_AL,
                tilt_deg=self.tilt_deg,
                h_des_j_per_kg=h_des,
                h_des_salt_name=h_des_salt,
                eps_abs_ir=cx.eps_abs_ir,
                has_glass=cx.has_glass,
                n_glazing_panes=cx.n_glazing_panes,
                evacuated_gap=cx.evacuated_gap,
            )
        # eps_abs_ir/eps_glass_ir are left unset -- SystemThermalParams' own default
        # is Case 2 (selective surface). Pass thermal=SystemThermalParams(eps_abs_ir=1.0,
        # eps_glass_ir=1.0) for Case 1's original Wilson blackbody/cavity approximation.
        return SystemThermalParams(
            p_atm_pa=self.site_pressure_pa(),
            insulation_gap_m=self.insulation_gap_m,
            vapor_gap_m=self.vapor_gap_m,
            eps_abs=EPS_ABS,
            tau_glass=TAU_GLASS,
            eps_gel=EPS_GEL,
            eps_al=EPS_AL,
            tilt_deg=self.tilt_deg,
            h_des_j_per_kg=h_des,
            h_des_salt_name=h_des_salt,
        )

    def condenser_thermal_mass_j_m2_k(self) -> float:
        return (
            self.condenser_rho_kg_m3
            * self.condenser_cp_j_kg_k
            * self.condenser_thickness_m
        )

    @classmethod
    def comsol_table_s3(cls, **overrides: object) -> SystemConfig:
        """Wilson Table S3 / Note S1 COMSOL SAWH system defaults.

        Pins ``h_des_mode="constant"``: Table S3 is a recreation target, and its
        2320 kJ/kg is the number COMSOL was actually run with."""
        base: dict[str, object] = {"h_des_mode": "constant"}
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

    @classmethod
    def baseline(cls, **overrides: object) -> SystemConfig:
        """Wilson Fig. 2 baseline system (Table S3, tilt 30°, fin area ratio 7.1)."""
        base = {
            "tilt_deg": TILT_DEG,
            "fin_area_ratio": FIN_AREA_RATIO,
        }
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

    @classmethod
    def atacama_field(cls, **overrides: object) -> SystemConfig:
        """Wilson Atacama field-test geometry (Methods): tilt 25°, fin area ratio 5."""
        base = {
            "tilt_deg": 25.0,
            "fin_area_ratio": 5.0,
        }
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

# --- Coupled Wilson Eqs. 1-6 + condenser transient (Eq. 2) rate evaluation ---

CyclePhase = Literal["absorption", "desorption"]

_M_DES_BRACKET_MAX = _pv("Desorption mass-flux bracket upper")  # kg/m²/s brentq bracket

# How instant_equilibrium is imposed. True: as the equilibrium constraint itself
# (_instant_equilibrium_desorption) -- the default, and the only non-stiff route. False:
# the legacy penalty, g scaled by _INSTANT_EQUILIBRIUM_G_SCALE until Eq. 5's residual is
# negligible. The penalty route is kept reachable for exactly one reason: it is the
# independent reference the constraint route is pinned against
# (tests/test_local_equilibrium.py). Not a config field -- production has no reason to
# pick the stiff route.
_INSTANT_EQUILIBRIUM_USE_CONSTRAINT = True


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
    thermal: SystemThermalParams,
    vapor_gap_m: float,
    config: SystemConfig,
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
        brine_salt_fraction=(
            None
            if thermal.h_des_salt_name is None
            else brine_salt_fraction_from_c_w(
                loading,
                c_s_mol_m3=mass.c_s_mol_m3,
                h0_ref_m=mass.h0_ref_m,
                formula_weight_g_mol=mass.formula_weight_g_mol,
                salt_weight_factor=mass.salt_weight_factor,
            )
        ),
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
    m_calc = m_des_kg_s_m2_from_dc_w(dc, h0_ref_m=mass.h0_ref_m)
    return m_calc, state.t_gel_c, dc, state


def _instant_equilibrium_desorption(
    *,
    c_w: float,
    h_m: float,
    t_cond: float,
    t_amb_c: float,
    q_sol: float,
    h_amb: float,
    mass: MassTransferParams,
    thermal: SystemThermalParams,
    gap_eff: float,
    config: SystemConfig,
    t_gel0: float,
    state0: ThermalState,
) -> tuple[float, float, float, ThermalState]:
    """(m_des, T_gel, dc_w/dt, thermal state) for g -> infinity, as a constraint.

    The ideal-kinetics limit is not "a very large g" -- it is the statement that the gel
    surface sits at equilibrium with the condenser. So pin T_gel there
    (``equilibrium_t_gel_desorption_c``) and let m_des be whatever the steady thermal
    balance needs to hold it: desorption becomes energy-limited by construction, which is
    what the ideal case is meant to represent.

    Numerically this is the point of the exercise. T_gel falls monotonically as latent
    load rises, so ``T_gel(m) - T_eq`` is a well-scaled monotone root in kelvin. The
    penalty formulation instead solved ``m_calc(m) - m`` with a slope of order g, which is
    near-vertical -- stiff for the ODE downstream, and ill-conditioned enough here to have
    needed the "brentq returned the jump, not the root" fallback below.
    """
    if c_w <= mass.c_w_min_mol_m3:
        # At the hydrate floor there is no more removable water, whatever the energy.
        return 0.0, t_gel0, 0.0, state0

    t_eq = equilibrium_t_gel_desorption_c(c_w, t_cond_c=t_cond, params=mass, h_m=h_m)
    if not math.isfinite(t_eq):
        return 0.0, t_gel0, 0.0, state0

    def _t_gel_gap(m_des_guess: float) -> float:
        _m, t_gel, _dc, _state = _m_des_calc(
            m_des_guess,
            loading=c_w, h_m=h_m, t_cond_c=t_cond, t_amb_c=t_amb_c, q_solar_w_m2=q_sol,
            h_amb=h_amb, mass=mass, thermal=thermal, vapor_gap_m=gap_eff, config=config,
            t_guess=(state0.t_gel_c, state0.t_abs_c, state0.t_glass_c),
        )
        if not math.isfinite(t_gel):
            return float("nan")
        return t_gel - t_eq

    gap_at_zero = _t_gel_gap(0.0)
    if not math.isfinite(gap_at_zero) or gap_at_zero <= 0.0:
        # Even carrying no latent load the gel cannot reach equilibrium: the energy is not
        # there, so nothing desorbs. Same physical branch as the penalty path's
        # m_at_zero <= 0, reached without consulting a rate law.
        return 0.0, t_gel0, 0.0, state0

    hi = 1e-6
    while hi < _M_DES_BRACKET_MAX:
        gap_hi = _t_gel_gap(hi)
        if not math.isfinite(gap_hi):
            return 0.0, t_gel0, 0.0, state0
        if gap_hi < 0.0:
            break
        hi *= 2.0
    else:
        # Monotone and still above equilibrium at the physical ceiling: take the ceiling
        # rather than extrapolate past a bracket that does not exist.
        hi = _M_DES_BRACKET_MAX

    try:
        m_star = float(brentq(_t_gel_gap, 0.0, hi, xtol=1e-16, rtol=1e-12))
    except ValueError:
        return 0.0, t_gel0, 0.0, state0

    _m, t_gel, _dc, state = _m_des_calc(
        m_star,
        loading=c_w, h_m=h_m, t_cond_c=t_cond, t_amb_c=t_amb_c, q_solar_w_m2=q_sol,
        h_amb=h_amb, mass=mass, thermal=thermal, vapor_gap_m=gap_eff, config=config,
        t_guess=(state0.t_gel_c, state0.t_abs_c, state0.t_glass_c),
    )
    # dc from the flux, not from Eq. 5: under the constraint Eq. 5's driving force is zero
    # by definition, so it can no longer be the thing that sets the rate.
    dc = dc_w_dt_from_m_des(m_star, h0_ref_m=mass.h0_ref_m)
    return m_star, t_gel, dc, state


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
    thermal: SystemThermalParams,
    vapor_gap_m: float,
    condenser_thermal_mass_j_m2_k: float,
    fin_area_ratio: float,
    h_fg_j_per_kg: float,
    config: SystemConfig,
    t_guess: tuple[float, float, float] | None = None,
    h_amb_cond: float | None = None,
) -> CoupledRates:
    """(dloading/dt, dH/dt, dT_cond/dt) with self-consistent T_gel and m_des. ``c_w`` is
    the hydrogel brine concentration in mol/m³; ``h_amb_cond`` models fan-forced
    condenser cooling decoupled from ambient wind (None reuses ``h_amb``)."""
    gap_eff = max(vapor_gap_m - h_m, 0.0)
    q_sol = max(0.0, q_solar_w_m2)

    if phase == "absorption":
        # Note S1 Eq. S1: fast gel thermal storage → T_gel ≈ T_amb during open absorption
        t_gel = t_amb_c
        h_conv_g = vapor_gap_h_conv_w_m2_k(
            gap_eff, t_gel, t_cond_c, tilt_deg=thermal.tilt_deg, p_atm_pa=thermal.p_atm_pa
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
    elif config.instant_equilibrium and _INSTANT_EQUILIBRIUM_USE_CONSTRAINT:
        # g -> infinity is imposed as the equilibrium constraint, not as a large g. Eq. 5's
        # rate law plays no part in setting the rate here; see
        # _instant_equilibrium_desorption.
        m_des, t_gel, dc, state = _instant_equilibrium_desorption(
            c_w=c_w, h_m=h_m, t_cond=t_cond, t_amb_c=t_amb_c, q_sol=q_sol, h_amb=h_amb,
            mass=mass, thermal=thermal, gap_eff=gap_eff, config=config,
            t_gel0=t_gel0, state0=state0,
        )
    else:
        # Cap at the documented search bound. Without it, instant_equilibrium's
        # scaled g makes m_at_zero ~1e3 kg/s/m2 (measured ~1e4 when the scale was 1e6),
        # and brentq on a bracket seven
        # orders wide -- flat at -m over all but its first 1e-4 -- returns a point
        # nowhere near the real root, which then feeds the ODE a nonsense rate.
        hi = min(max(m_at_zero * 2.0, 1e-8), _M_DES_BRACKET_MAX)
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
                # brentq only guarantees a SIGN CHANGE in [0, hi], and with
                # instant_equilibrium's scaled g this residual has a jump: m_calc is
                # enormous below the crossing and plummets past it. Bisection then returns
                # the jump rather than a root, and m_des here -- which is m_calc at that
                # point, not m_star -- came back at 2.4e3 kg/s/m2 against a 1e-2 bracket.
                # A "root" the bracket cannot contain is not a root. Feeding it to the ODE
                # integrates a yield the gel inventory cannot support (446 L/m2 against a
                # 3.9 L/m2 c_w drop, which is what _integrate_desorption's conservation
                # guard then trips on), so take the same no-desorption fallback the other
                # failure branches here use.
                if not math.isfinite(m_des) or m_des > hi * (1.0 + 1e-6):
                    # dc = 0 too, NOT dc0. The sibling fallbacks above can keep dc0 because
                    # they only fire when the gel is not desorbing anyway (m_at_zero <= 0),
                    # so dc0 is already consistent with m_des = 0. Here dc0 is strongly
                    # negative, and keeping it drains the gel while crediting no yield --
                    # water vanishes. At 4000 m that opened a 5% gap between the yield
                    # integral and the c_w inventory and tripped _integrate_desorption's
                    # conservation guard. Freezing both is the self-consistent state.
                    m_des, t_gel, dc, state = 0.0, t_gel0, 0.0, state0

    if config.instant_equilibrium and _INSTANT_EQUILIBRIUM_USE_CONSTRAINT:
        # Eq. 6 shares Eq. 5's driving force, which the constraint sets to zero -- so
        # reading dH/dt off the rate law here would freeze the gel's thickness while its
        # water drained. Use the identity dH/dt = dc_w/dt * MW * H0 / rho_sol instead, the
        # same ratio dH_dt itself carries, so H and c_w stay on one trajectory.
        dh = dh_dt_from_dc_w(
            dc, rho_solution_kg_m3=mass.rho_solution_kg_m3, h0_ref_m=mass.h0_ref_m
        )
        if c_w <= mass.c_w_min_mol_m3:
            dh = 0.0
        dh = min(dh, 0.0)
    else:
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
    # B4: the design's fan speed is the source of truth for condenser-side
    # convection; the profile channel is only an override for weather that carries
    # one. Deriving it here too keeps the CPU symmetric with the JAX backend, which
    # has only the design to work from (jax_daily_cycle._h_amb_cond_for).
    if h_amb_cond is None and config.complex is not None:
        h_amb_cond = config.complex.condenser_h_amb_w_m2_k()
    h_amb_for_cond = h_amb if h_amb_cond is None else h_amb_cond
    # Same thinner-air derate _residuals applies to the absorber side. The condenser is
    # where it bites hardest: it rejects the latent load to ambient, so weaker convection
    # means a hotter condenser and a smaller desorption driving force -- the offset that
    # stops elevation's D_air gain from being one-sided. A fan-forced condenser is not
    # exempt: the fan sets air SPEED, and thin air still carries less heat per m/s.
    h_amb_for_cond = h_amb_for_cond * h_amb_density_factor(t_amb_c, p_atm_pa=thermal.p_atm_pa)
    # Complex mode (B3) supplies fin geometry, which derates the added fin area by
    # its efficiency; without it this stays Wilson's ideal A_r * h_amb.
    cx = config.complex
    h_conv_cond = condenser_h_conv_w_m2_k(
        h_amb_for_cond,
        fin_area_ratio=fin_area_ratio,
        fin_thickness_m=cx.fin_thickness_m if cx is not None else None,
        fin_height_m=cx.fin_height_m if cx is not None else None,
    )
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

# --- SciPy LSODA integration for Wilson half-cycles (coupled Eqs. 1-6 + Eq. 2) ---

_ODE_RTOL = _pv("ODE relative tolerance")
_ODE_ATOL = _pv("ODE absolute tolerance")
WATER_BALANCE_TOL: float = _pv("Desorption water-balance closure tolerance")
INSTANT_EQUILIBRIUM_RESIDUAL_TOL: float = _pv(
    "Instant-equilibrium residual driving-force tolerance"
)


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
    # Raw RHS rate at each output point, kept only for the conservation_drift_series
    # self-consistency check below -- not used by anything upstream.
    dT_cond_dt: np.ndarray | None = None
    # Cumulative collected water along the trajectory, the integrated W state. Desorption
    # only; None from absorption, which collects nothing. Plots read this rather than
    # re-trapezoiding m_des, so the curve and water_collected_kg_m2 agree exactly.
    water_cumulative_kg_m2: np.ndarray | None = None


def _profile_index(t: float, dt_s: float, n: int) -> int:
    return min(int(t / dt_s), n - 1)


def _absorb_at_equilibrium(
    c_w0: float,
    h0: float,
    profile: PhaseProfile,
    config: SystemConfig,
    *,
    mass: MassTransferParams,
    h_min: float,
    h_max: float,
    t_eval: np.ndarray,
) -> PhaseResult:
    """Absorption under the local-equilibrium closure -- no ODE at all.

    T_gel is T_amb here (Note S1 Eq. S1) so nothing on this half-cycle is heat-limited,
    and the weather is piecewise-constant on dt (which is why the ODE path pins
    max_step=dt). Instant kinetics on piecewise-constant forcing therefore means the gel
    is at the equilibrium loading for the *current* interval, always: the trajectory is a
    staircase and the exact solution is one isotherm inversion per interval.

    Which removes the last stiff term in the ideal case. The penalty route integrated a
    relaxation of rate ~g toward this same staircase, which needed >16,384 explicit steps
    per day in the JAX backend and silently truncated when it ran out.

    H follows c_w by Eq. 6's identity dH = dc_w * MW * H0 / rho_sol, accumulated interval
    by interval so the swelling clamps bite where they would have during integration
    rather than only at the end.
    """
    n = len(profile.temperature_c)
    ratio = WATER_MOLAR_MASS_KG_MOL * mass.h0_ref_m / mass.rho_solution_kg_m3
    c_w = float(c_w0)
    h_m = min(max(float(h0), h_min), h_max)

    c_hist, h_hist, t_gel_hist = [c_w], [h_m], [float(profile.temperature_c[0])]
    for k in range(1, len(t_eval)):
        i = _profile_index(float(t_eval[k]), profile.dt_s, n)
        t_amb = float(profile.temperature_c[i])
        c_eq = equilibrium_c_w_absorption(
            rh=float(profile.relative_humidity[i]), t_gel_c=t_amb, params=mass, h_m=h_m
        )
        if math.isfinite(c_eq):
            h_m = min(max(h_m + (c_eq - c_w) * ratio, h_min), h_max)
            c_w = float(c_eq)
        c_hist.append(c_w)
        h_hist.append(h_m)
        t_gel_hist.append(t_amb)

    # The invariant this path asserts is its own definition: it ends ON the isotherm.
    # (The ODE path cannot check that directly -- hence its rate-ratio guard -- because it
    # only approaches equilibrium.) Cheap and exact, so it is checked rather than argued.
    i_last = _profile_index(float(t_eval[-1]), profile.dt_s, n)
    residual = abs(
        concentration_ratio_absorption(float(profile.relative_humidity[i_last]))
        - _absorption_effective_water_activity(
            c_w, t_gel_c=float(profile.temperature_c[i_last]), params=mass, h_m=h_m
        )
    )
    at_bound = c_w <= mass.c_w_min_mol_m3 * (1.0 + 1e-9) or c_w >= mass.c_w_max_mol_m3 * (1.0 - 1e-9)
    if residual > 1e-6 and not at_bound:
        raise RuntimeError(
            f"Local-equilibrium absorption did not land on the isotherm: |c_r - a_w| "
            f"{residual:.3e} at c_w {c_w:.1f} mol/m3 (bounds "
            f"{mass.c_w_min_mol_m3:.1f}-{mass.c_w_max_mol_m3:.1f})"
        )

    return PhaseResult(
        time_s=np.asarray(t_eval, dtype=float),
        c_w=np.asarray(c_hist, dtype=float),
        H=np.asarray(h_hist, dtype=float),
        t_cond_c=None,
        t_gel_c=np.asarray(t_gel_hist, dtype=float),
        water_collected_kg_m2=0.0,
        m_des_kg_s_m2=np.zeros(len(t_eval)),
    )


def _integrate_absorption(
    c_w0: float,
    h0: float,
    profile: PhaseProfile,
    config: SystemConfig,
) -> PhaseResult:
    mass = config.mass_params()
    thermal = config.thermal_params()
    n = len(profile.temperature_c)
    dt = profile.dt_s
    t_span = (0.0, dt * n)
    t_eval = np.linspace(0.0, t_span[1], n + 1)
    h_min = config.hydrogel_floor_thickness_m()
    # Geometry, not transport: the gel cannot swell into the condenser. The clearance is
    # a numerical one (the correlations' k/L diverges at contact), NOT a physical setback
    # -- near-contact is meant to be punished by the k/L heat leak the thermal balance
    # already carries, not by this ceiling. It previously reused Wilson's 7 mm transport
    # floor, which is ~70% of a narrow vapor gap and left this constant setting
    # night-time uptake directly. Mirrored in gpu_sweep/jax_daily_cycle.py.
    h_max = max(
        config.vapor_gap_m - GEL_CONDENSER_CLEARANCE_M,
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

    if config.instant_equilibrium and _INSTANT_EQUILIBRIUM_USE_CONSTRAINT:
        return _absorb_at_equilibrium(
            c_w0, h0, profile, config, mass=mass, h_min=h_min, h_max=h_max, t_eval=t_eval
        )

    # LSODA, matching _integrate_desorption -- one integrator for the whole cycle.
    # Absorption has no thermal Newton solve (T_gel is just T_amb) so it never had
    # desorption's non-differentiable RHS, but there is no reason for the two halves
    # to disagree on how they are solved. max_step=dt is load-bearing: the weather
    # profile is piecewise-constant on dt and _profile_index samples it, so a step
    # spanning a profile change would miss it.
    sol = solve_ivp(
        rhs,
        t_span,
        y0=np.array([c_w0, max(h0, h_min)]),
        method="LSODA",
        t_eval=t_eval,
        max_step=dt,
        rtol=_ODE_RTOL,
        atol=_ODE_ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"Absorption integration failed: {sol.message}")

    t_gel_hist: list[float] = []
    dc_w_dt_hist: list[float] = []
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
        dc_w_dt_hist.append(rates.dc_w_dt)

    # Absorption's counterpart to _integrate_desorption's water-balance guard. There is
    # no independent yield integral on this half-cycle (nothing is condensed), so the
    # invariant is the weaker but still binding one: the trapezoid of the dc_w/dt the
    # solver claims it integrated must match the c_w it actually returned. Costs no extra
    # RHS evaluations -- the loop above already computes every term.
    #
    # This half was never the one that broke: T_gel is just T_amb here, so there is no
    # thermal Newton solve and none of desorption's non-differentiable kink. That is an
    # argument, not a check, and it is the same argument that would have sounded fine for
    # desorption before Radau was tried on it -- hence the check. Measured closure on a
    # sound integration is <=2.5e-4 across RH 0-0.95, H0 1-10 mm and the ZSR blends, two
    # orders inside WATER_BALANCE_TOL.
    #
    # instant_equilibrium gets a different invariant rather than no invariant. It scales g
    # by _INSTANT_EQUILIBRIUM_G_SCALE, and absorption *starts* off-equilibrium (gel at the fabrication RH, air at
    # the profile RH), so dc_w/dt spikes and decays entirely inside the first output
    # interval; the trapezoid on the reporting grid cannot resolve that and reads ~2.6e3x
    # high while the integration itself is fine. What that mode asserts instead is its own
    # definition -- the gel ends at equilibrium, so the residual driving force is zero.
    # That needs no closed-form target for the equilibrium c_w, which is convenient
    # because the binding ceiling still varies with RH (brine isotherm vs the dilution
    # ceiling / hydrate floor).
    # Desorption needs neither special case: it starts from where absorption ended, i.e.
    # already at equilibrium, so it has no opening spike.
    if config.instant_equilibrium:
        residual, opening = abs(dc_w_dt_hist[-1]), abs(dc_w_dt_hist[0])
        if residual > INSTANT_EQUILIBRIUM_RESIDUAL_TOL * max(opening, 1e-30):
            raise RuntimeError(
                f"Instant-equilibrium absorption did not reach equilibrium: |dc_w/dt| "
                f"{residual:.3e} mol/m3/s at end vs {opening:.3e} at start "
                f"(ratio {residual / max(opening, 1e-30):.3e}, "
                f"tol {INSTANT_EQUILIBRIUM_RESIDUAL_TOL:.0e}); "
                f"c_w {sol.y[0, 0]:.1f} -> {sol.y[0, -1]:.1f} mol/m3"
            )
    else:
        state_change = float(sol.y[0, -1] - sol.y[0, 0])
        rhs_integral = float(np.trapezoid(dc_w_dt_hist, sol.t))
        # The denominator needs an absolute floor on the SCALE OF THE LOADING, not 1e-12
        # mol/m3, which is ~1e-16 relative and below float dust. Now that the fabrication
        # loading and the absorption equilibrium come from one function, a gel cast at the
        # ambient RH starts exactly at equilibrium and legitimately does not move; a purely
        # relative tolerance then divides by zero-change and trips on the trapezoid's last
        # bits. A real imbalance is a fraction of the loading, so scale the floor to it.
        scale = max(abs(state_change), 1e-9 * max(abs(float(sol.y[0, 0])), 1.0))
        if abs(rhs_integral - state_change) > WATER_BALANCE_TOL * scale:
            raise RuntimeError(
                f"Absorption integration did not conserve water: dc_w/dt integral "
                f"{rhs_integral:.1f} mol/m3 vs c_w state change {state_change:.1f} mol/m3 "
                f"(c_w {sol.y[0, 0]:.1f} -> {sol.y[0, -1]:.1f} mol/m3)"
            )

    c_w_out = np.asarray(sol.y[0], dtype=float)
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
    config: SystemConfig,
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
    h_min = config.hydrogel_floor_thickness_m()
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
    ambient_condenser = config.condenser_tracks_ambient

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal t_guess
        i = _profile_index(t, dt, n)
        h_m = max(float(y[1]), h_min)
        t_cond_c = profile.temperature_c[i] if ambient_condenser else float(y[2])
        rates = evaluate_coupled_rates(
            c_w=float(y[0]),
            h_m=h_m,
            t_cond_c=t_cond_c,
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
        # Cumulative collected water as a state, dW/dt = m_des, mirroring the JAX backend's
        # y = [c_w, H, T_cond, W]. Derived from the CLIPPED dc rather than from
        # rates.m_des_kg_s_m2, so W integrates exactly the rate c_w integrates and the two
        # cannot disagree by construction.
        dw = m_des_kg_s_m2_from_dc_w(dc, h0_ref_m=mass.h0_ref_m)
        if ambient_condenser:
            return np.array([dc, dh, dw])
        return np.array([dc, dh, rates.dT_cond_dt, dw])

    # W starts at 0: it accumulates this phase's yield only.
    w_idx = 2 if ambient_condenser else 3
    y0 = (
        np.array([c_w0, max(h0, h_min), 0.0])
        if ambient_condenser
        else np.array([c_w0, max(h0, h_min), t_cond0, 0.0])
    )
    # LSODA, not Radau. The obstacle is a step discontinuity in the RHS itself, at the
    # hydrate floor: dc_w_dt returns exactly 0.0 once c_w reaches params.c_w_min_mol_m3
    # (physics.py, the `c_w <= c_w_min and rate < 0` clip), so crossing the floor jumps
    # dc_w/dt from a finite negative value to zero with nothing in between -- measured
    # -0.1998 -> 0.0 across 0.01 mol/m3, i.e. 1e-6 relative. A Jacobian there is
    # meaningless: the finite difference reports -0.2 or 0 purely by which side the
    # perturbation lands on. Implicit methods diverge or stall; the 2-pane evacuated case,
    # whose trajectory reaches the floor exactly, fails on Radau with "required step size
    # is less than spacing between numbers".
    #
    # This is not about T_gel. An earlier version of this comment blamed the thermal
    # Newton solve clamping T_gel at TEMP_CLAMP_HI_C, which was true when that clamp sat
    # at 120 C; it is now 373.9 C (the Saul-Wagner critical-point ceiling) and never
    # binds -- the hottest configuration here reaches 158 C. The salt isotherm caps that
    # *are* crossed (LiCl BET at 155.5 C, Conde at 100 C for CaCl2) are only C1 breaks:
    # dc_w/dt stays continuous across them and its slope bends ~9%, which implicit
    # methods handle fine. Radau reproduces LSODA to the last digit on any case whose
    # trajectory stays off the floor -- test_implicit_and_explicit_solvers_agree_off_the_
    # hydrate_floor pins that, and is the control that isolates the floor as the cause.
    #
    # Everything that does not build a Jacobian agrees with the JAX backend to <=0.4%:
    # LSODA (it stays in non-stiff Adams mode here), RK45, DOP853, and diffrax's Tsit5.
    # LSODA switching to its own BDF mode would hit the same wall, which is what the
    # water-balance guard below exists to catch rather than assume away.
    sol = solve_ivp(
        rhs,
        t_span,
        y0=y0,
        method="LSODA",
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
    dT_cond_hist: list[float] = []
    guess: tuple[float, float, float] | None = t_guess0
    for k in range(len(sol.t)):
        i = _profile_index(float(sol.t[k]), dt, n)
        t_cond_c = profile.temperature_c[i] if ambient_condenser else float(sol.y[2, k])
        rates = evaluate_coupled_rates(
            c_w=float(sol.y[0, k]),
            h_m=max(float(sol.y[1, k]), h_min),
            t_cond_c=t_cond_c,
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
        t_cond_hist.append(t_cond_c)
        m_des_hist.append(rates.m_des_kg_s_m2)
        dT_cond_hist.append(rates.dT_cond_dt)

    # Yield is the integrated W state, not a trapezoid over the sampled m_des history.
    # The trapezoid was a second, coarser quadrature of the same flux, and it diverged from
    # the c_w trajectory by ~5% on fast trajectories -- high-elevation sites (thin air =>
    # more diffusivity and a better-insulated collector) desorb quickly enough that the
    # sampled-history quadrature broke the 1% conservation tolerance below, on runs whose
    # trajectories were fine. LSODA controls error on W as it does on any other state, so
    # that error source is gone rather than tolerated.
    water = float(sol.y[w_idx, -1])

    # The guard's role has changed with W. dW/dt is derived from the same clipped dc that
    # c_w integrates, so this is now an invariant assertion (both states, one rate) rather
    # than the cross-check between two independent quadratures it used to be: it still
    # catches a thrown state or a basis mismatch, but it can no longer detect a quadrature
    # error, because there is no second quadrature left to be wrong.
    #
    # Retained history, since it is what motivated the guard: Radau does not report failure
    # on this RHS -- the floor discontinuity documented above gives the implicit step a
    # Jacobian inconsistent with its residual, and it could throw the state (observed:
    # T_cond -> 1799 C, c_w -> -131000) while still returning success=True.
    inventory_drop = float(sol.y[0, 0] - sol.y[0, -1]) * mass.h0_ref_m * WATER_MOLAR_MASS_KG_MOL
    if abs(water - inventory_drop) > WATER_BALANCE_TOL * max(abs(inventory_drop), 1e-12):
        t_cond_note = "" if ambient_condenser else f", T_cond -> {sol.y[2, -1]:.1f} C"
        raise RuntimeError(
            f"Desorption integration did not conserve water: yield {water:.4f} L/m2 "
            f"vs c_w inventory drop {inventory_drop:.4f} L/m2 "
            f"(c_w {sol.y[0, 0]:.1f} -> {sol.y[0, -1]:.1f} mol/m3{t_cond_note})"
        )

    c_w_out = np.asarray(sol.y[0], dtype=float)
    h_out = np.maximum(sol.y[1], h_min)
    return PhaseResult(
        water_cumulative_kg_m2=np.asarray(sol.y[w_idx], dtype=float),
        time_s=sol.t,
        c_w=c_w_out,
        H=h_out,
        t_cond_c=np.array(t_cond_hist),
        t_gel_c=np.array(t_gel_hist),
        water_collected_kg_m2=max(0.0, water),
        m_des_kg_s_m2=np.array(m_des_hist),
        t_abs_c=np.array(t_abs_hist),
        t_glass_c=np.array(t_glass_hist),
        dT_cond_dt=np.array(dT_cond_hist),
    )


def cycle_end_state(des_res: PhaseResult) -> tuple[float, float]:
    """Gel state (c_w, H) at end of a full absorption–desorption cycle."""
    return float(des_res.c_w[-1]), float(des_res.H[-1])


def find_cyclic_state(
    profile: DailyWeatherProfile,
    config: SystemConfig,
    *,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
    tol: float = 1e-6,
    max_rounds: int = 10,
    stall_ratio: float = 0.5,
    stall_rounds: int = 2,
    verbose: bool = True,
) -> tuple[float, float]:
    """Steady periodic post-desorption (c_w, H) state for an indefinitely repeated profile,
    via restarted vector Aitken Δ² extrapolation (~3-6 rounds) instead of the 100+ cycles
    plain fixed-point iteration can need when the one-cycle map's slowest eigenvalue ≈ 1.

    Some (profile, config) pairs have no fixed point but a stable period-2 orbit, so
    ``rel_step`` plateaus instead of reaching ``tol``. That is detected as ``stall_rounds``
    rounds without a ``stall_ratio`` shrink and answered by averaging the two most recent
    extrapolated states (also the fallback when ``max_rounds`` runs out). Callers wanting
    the true alternating yields should run one cycle from each branch."""
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
    config: SystemConfig,
    *,
    c_w_initial: float | None = None,
    h_initial: float | None = None,
    cyclic_initial: bool = False,
    cyclic_warmup_cycles: int = 2,
) -> tuple[float, float, PhaseResult, PhaseResult]:
    """Run absorption then desorption; return (yield kg/m2, eta_thermal, abs_res, des_res).
    With ``cyclic_initial``, first find the steady periodic state via ``find_cyclic_state``
    and report one day from there; ``cyclic_warmup_cycles`` becomes its ``max_rounds``
    (floored at 3, since a round is 2 full daily cycles)."""
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

# --- System temperatures and weather time series for a daily SAWH cycle ---

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
    # Desorption-only (NaN during absorption): the condenser/gel vapour-pressure ratio
    # driving desorption, the brine activity at the current gel loading, their difference
    # (the Eq. 5 driving force), and the salt's DRH at the gel temperature -- desorption
    # stalls once c_r drops to DRH, since a_w cannot fall below it.
    c_r: np.ndarray
    a_w: np.ndarray
    driving_force: np.ndarray
    drh: np.ndarray


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
    config: SystemConfig,
) -> DetailedSeries:
    """Build full-cycle system and weather trajectories."""

    def _absorption_system_temps() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mass = config.mass_params()
        thermal = config.thermal_params()
        abs_profile = profile.absorption
        n = len(abs_profile.temperature_c)
        dt = abs_profile.dt_s
        h_min = config.hydrogel_floor_thickness_m()
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

    def _desorption_system_temps() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        h_min = config.hydrogel_floor_thickness_m()
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

    abs_t_abs, abs_t_glass, abs_t_cond, abs_t_gel = _absorption_system_temps()
    des_t_abs, des_t_glass, des_t_cond, des_t_gel = _desorption_system_temps()

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

    des_c_r = np.array(
        [
            concentration_ratio_desorption(float(tg), float(tc))
            for tg, tc in zip(des_t_gel, des_t_cond)
        ]
    )
    des_drh = np.array(
        [deliquescence_rh(config.salt().name, float(tg)) for tg in des_t_gel]
    )
    des_mass = config.mass_params()
    des_driving = np.array(
        [
            _mass_transfer_driving_force(
                float(c),
                t_gel_c=float(tg),
                c_r=float(cr),
                params=des_mass,
                h_m=max(float(h), config.hydrogel_floor_thickness_m()),
                phase="desorption",
            )
            for c, h, tg, cr in zip(des_res.c_w, des_res.H, des_t_gel, des_c_r)
        ]
    )
    nan_abs = np.full(len(abs_res.time_s), np.nan)

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
        c_r=_join(nan_abs, des_c_r),
        a_w=_join(nan_abs, des_c_r - des_driving),
        driving_force=_join(nan_abs, des_driving),
        drh=_join(nan_abs, des_drh),
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
                "c_r",
                "a_w",
                "driving_force",
                "drh",
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
                    f"{float(series.c_r[k]):.6f}",
                    f"{float(series.a_w[k]):.6f}",
                    f"{float(series.driving_force[k]):.6f}",
                    f"{float(series.drh[k]):.6f}",
                ]
            )


def plot_detailed_diagnostics(
    path: Path,
    series: DetailedSeries,
    *,
    title: str | None = None,
    show_solar: bool = True,
    grid: bool = True,
) -> None:
    """Three stacked panels: temperatures with the desorption driving force, weather
    (T_amb + RH), and forcing (solar + h_amb).

    ``show_solar=False`` drops the third panel, and ``grid=False`` drops the gridlines. Both
    default to the historical behaviour, because site_drh_diagnostics.py and
    run_physics_checks.py also render through here and their figures should not move.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    time_h = series.time_s / 3600.0
    phase_mark_h = series.absorption_end_s / 3600.0

    def _grid(ax) -> None:
        """ax.grid(False, alpha=...) turns the grid ON -- matplotlib treats any kwarg as
        'you want the grid', overriding visible=False. So style kwargs only when enabling."""
        if grid:
            ax.grid(True, alpha=0.3)
        else:
            ax.grid(False)

    n_panels = 3 if show_solar else 2
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(8, 8 if show_solar else 5.6), sharex=True,
    )
    ax_t, ax_wx = axes[0], axes[1]
    ax_sol = axes[2] if show_solar else None

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
    _grid(ax_t)

    # c_r vs a_w vs DRH on an RH axis: the shaded c_r - a_w gap is the Eq. 5 driving
    # force, and desorption stalls where c_r meets DRH.
    ax_cr = ax_t.twinx()
    _grid(ax_cr)
    ax_cr.plot(time_h, series.c_r * 100.0, color="#1b9e77", linewidth=1.6, label="c_r")
    ax_cr.plot(
        time_h,
        series.a_w * 100.0,
        color="#1b9e77",
        linewidth=1.4,
        linestyle="--",
        label="a_w (gel brine)",
    )
    ax_cr.fill_between(
        time_h,
        series.a_w * 100.0,
        series.c_r * 100.0,
        color="#1b9e77",
        alpha=0.15,
        linewidth=0,
        label="driving force (c_r − a_w)",
    )
    ax_cr.plot(
        time_h,
        series.drh * 100.0,
        color="#1b9e77",
        linewidth=1.4,
        linestyle=":",
        label="DRH (at T_gel)",
    )
    ax_cr.set_ylabel("RH (%)", color="#1b9e77")
    ax_cr.tick_params(axis="y", labelcolor="#1b9e77")
    # Autoscaled, not 0-100: c_r and DRH both sit in the low-RH corner, and the gap
    # between them is the whole point.
    ax_cr.set_ylim(bottom=0.0)

    lines_l, labels_l = ax_t.get_legend_handles_labels()
    lines_r, labels_r = ax_cr.get_legend_handles_labels()
    ax_t.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", fontsize=8, ncol=2)

    ax_wx.plot(time_h, series.t_amb_c, color="#d95f02", linewidth=1.6, label="T_amb")
    ax_wx.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_wx.set_ylabel("Temperature (°C)", color="#d95f02")
    ax_wx.tick_params(axis="y", labelcolor="#d95f02")
    _grid(ax_wx)

    ax_rh = ax_wx.twinx()
    _grid(ax_rh)
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

    if ax_sol is not None:
        ax_sol.plot(time_h, series.solar_w_m2, color="#e6ab02", linewidth=1.8, label="Solar")
        ax_sol.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
        ax_sol.set_ylabel("Solar (W/m²)", color="#e6ab02")
        ax_sol.tick_params(axis="y", labelcolor="#e6ab02")
        _grid(ax_sol)

        ax_h = ax_sol.twinx()
        _grid(ax_h)
        ax_h.plot(time_h, series.h_amb_w_m2_k, color="#7570b3", linewidth=1.4, label="h_amb")
        ax_h.set_ylabel("h_amb (W/m²K)", color="#7570b3")
        ax_h.tick_params(axis="y", labelcolor="#7570b3")

        lines_l, labels_l = ax_sol.get_legend_handles_labels()
        lines_r, labels_r = ax_h.get_legend_handles_labels()
        ax_sol.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time (h)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# --- Water-in-sorbent inventory time series from a daily absorption-desorption cycle ---

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
    config: SystemConfig,
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
    # The integrated W state where available, so the plotted curve ends exactly on the
    # reported yield; the trapezoid only as a fallback for hand-built PhaseResults.
    collected_des = (
        des_res.water_cumulative_kg_m2
        if des_res.water_cumulative_kg_m2 is not None
        else cumulative_desorption_yield_l_m2(des_res.time_s, des_res.m_des_kg_s_m2)
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


# --- Per-point mass/enthalpy drift: LSODA's dense output vs. the RHS it integrated.
# Growing drift here is the error indicator solve_ivp doesn't otherwise expose -- it
# is not corrected/normalized, only reported. Absorption has no independent output
# flux to check the sorbent uptake against (the ambient air reservoir isn't tracked),
# so this only covers desorption, where m_des and dT_cond_dt are known boundary flows.

@dataclass(frozen=True, slots=True)
class ConservationDriftSeries:
    time_s: np.ndarray
    mass_l_m2: np.ndarray
    mass_drift_l_m2: np.ndarray
    enthalpy_j_m2: np.ndarray | None
    enthalpy_drift_j_m2: np.ndarray | None


def conservation_drift_series(
    des_res: PhaseResult, *, config: SystemConfig
) -> ConservationDriftSeries:
    """Sorbent water mass and condenser enthalpy at each desorption output point,
    compared to what the integrated m_des/dT_cond_dt flows predict."""
    mass_l_m2 = np.array(
        [
            water_in_sorbent_l_m2(float(c), float(h), config=config)
            for c, h in zip(des_res.c_w, des_res.H)
        ]
    )
    predicted_loss = cumulative_desorption_yield_l_m2(des_res.time_s, des_res.m_des_kg_s_m2)
    mass_drift_l_m2 = (mass_l_m2[0] - mass_l_m2) - predicted_loss

    enthalpy_j_m2 = None
    enthalpy_drift_j_m2 = None
    if des_res.dT_cond_dt is not None and des_res.t_cond_c is not None:
        tmass = config.condenser_thermal_mass_j_m2_k()
        enthalpy_j_m2 = tmass * des_res.t_cond_c
        predicted_denthalpy = cumulative_desorption_yield_l_m2(
            des_res.time_s, tmass * des_res.dT_cond_dt
        )
        enthalpy_drift_j_m2 = (enthalpy_j_m2 - enthalpy_j_m2[0]) - predicted_denthalpy

    return ConservationDriftSeries(
        time_s=des_res.time_s,
        mass_l_m2=mass_l_m2,
        mass_drift_l_m2=mass_drift_l_m2,
        enthalpy_j_m2=enthalpy_j_m2,
        enthalpy_drift_j_m2=enthalpy_drift_j_m2,
    )


# --- How hard the desorption one-way clamp is working.
#
# evaluate_mass_rates pins dc_w/dt <= 0 during desorption (physics.py). That clamp is
# the correct quasi-steady limit, not a hack: the vapor gap holds only ~0.9 L/m2 x
# 23 g/m3 ~ 0.9 g/m2 of vapor against daily yields of 1-4 kg/m2, and the model tracks
# no gas-phase inventory, so unclamped reverse flow would draw on a reservoir that does
# not exist and book the water with no counter-entry.
#
# What the clamp hides is that it binds at all -- and it binds at both ends of the
# sealed window, where T_gel -> T_cond drives c_r -> 1 above any reachable a_w. Those
# are hours the device is sealed at zero rate that could have been absorption hours, so
# a widening window looks free to a schedule optimizer. ``discarded_l_m2`` is the
# unbounded demand for that reverse flow: an upper bound on the re-absorption the clamp
# refuses, since the real ceiling is the un-drained condensate film (Wilson's 131 g gel
# loss vs. 130 mL collected puts that at ~13 g/m2, i.e. tens of times smaller).
#
# Reported as an uncertainty band, never subtracted -- same rule as the drift series.

@dataclass(frozen=True, slots=True)
class ReversalDiagnostics:
    time_s: np.ndarray
    driving: np.ndarray  # c_r - a_w at each point; > 0 means the gel would re-absorb
    reversed_time_fraction: float  # share of the sealed window pinned at zero rate
    discarded_l_m2: float  # upper bound on re-absorption the clamp discarded


def reversal_diagnostics(
    des_res: PhaseResult, *, config: SystemConfig
) -> ReversalDiagnostics:
    """Where and how much the desorption ``dc_w/dt <= 0`` clamp binds.

    Recomputed from the stored trajectory rather than tallied inside the RHS: LSODA
    evaluates the RHS at rejected and intermediate stages too, so an accumulator there
    would count evaluations that never made it into the solution.
    """
    mass = config.mass_params()
    h_min = config.hydrogel_floor_thickness_m()
    assert des_res.t_cond_c is not None, "desorption result must carry condenser temps"

    driving = np.zeros(len(des_res.time_s), dtype=float)
    dc_discarded = np.zeros(len(des_res.time_s), dtype=float)
    for k, (c, h, tg, tc) in enumerate(
        zip(des_res.c_w, des_res.H, des_res.t_gel_c, des_res.t_cond_c)
    ):
        h_m = max(float(h), h_min)
        c_r = concentration_ratio_desorption(float(tg), float(tc))
        driving[k] = _mass_transfer_driving_force(
            float(c), t_gel_c=float(tg), c_r=c_r, params=mass, h_m=h_m, phase="desorption"
        )
        dc = dc_w_dt(
            float(c),
            t_gel_c=float(tg),
            c_r=c_r,
            params=mass,
            h_m=h_m,
            phase="desorption",
            t_cond_c=float(tc),
        )
        # Keyed on dc, not on driving: below the 7 mm transport cutoff g is zero, so the
        # driving force can be positive while there is no rate for the clamp to bite on.
        #
        # Negated through the model's own flux conversion rather than multiplying by
        # MW*H by hand: forward yield is booked on the H0 basis (Note S1's dH/dt ~ 0
        # limit), so using the live thickness here would inflate this by up to h/H0 and
        # make the bound incommensurable with the yield it qualifies.
        dc_discarded[k] = m_des_kg_s_m2_from_dc_w(-dc, h0_ref_m=mass.h0_ref_m)

    return ReversalDiagnostics(
        time_s=des_res.time_s,
        driving=driving,
        reversed_time_fraction=float((dc_discarded > 0.0).mean()),
        discarded_l_m2=float(
            cumulative_desorption_yield_l_m2(des_res.time_s, dc_discarded)[-1]
        ),
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
    config: SystemConfig,
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

# --- Annual yield aggregation over real weather days ---

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
    # Desorption dc_w/dt <= 0 clamp, per day (see reversal_diagnostics). Carried on
    # every row rather than a run-level summary because the clamp binds seasonally --
    # a year's mean would hide the winter days where the sealed window overruns worst.
    clamp_reversed_frac: float
    clamp_discarded_l_m2: float


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
    "clamp_reversed_frac",
    "clamp_discarded_l_m2",
)


def simulate_annual_year(
    day_items: list[tuple[date, DailyWeatherProfile, pd.DataFrame]],
    config: SystemConfig,
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
        reversal = reversal_diagnostics(des_res, config=config)
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
            clamp_reversed_frac=reversal.reversed_time_fraction,
            clamp_discarded_l_m2=reversal.discarded_l_m2,
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
                    "clamp_reversed_frac": f"{rec.clamp_reversed_frac:.6f}",
                    "clamp_discarded_l_m2": f"{rec.clamp_discarded_l_m2:.6f}",
                }
            )
