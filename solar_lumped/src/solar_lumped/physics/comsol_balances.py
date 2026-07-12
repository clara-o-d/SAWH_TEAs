"""COMSOL lumped algebraic glass / absorber balances."""

from __future__ import annotations

import numpy as np
from scipy.optimize import root

from solar_lumped.physics import comsol_lumped as cl
from solar_lumped.physics.salt_properties import clamp_temperature_c


def _emit_w_m2(t_c: float, eps: float) -> float:
    t_k = t_c + 273.15
    return eps * cl.STEFAN_BOLTZMANN_W_M2_K4 * t_k**4


def _glass_absorber_residuals(
    x: np.ndarray,
    *,
    t_gel_c: float,
    q_solar_w_m2: float,
    solar_on: bool,
    h_front: float,
    tint_c: float,
    optics: dict[str, float | bool],
) -> np.ndarray:
    t_ads, t_glass = float(x[0]), float(x[1])
    eps_ads = float(optics["eps_ads"])
    eps_glass_ir = float(optics["eps_glass_ir"])
    reflect_glass = float(optics["reflect_glass"])
    solar_ads = float(optics["solar_ads_frac"])
    solar_glass = float(optics["solar_glass_frac"])

    glass_cond = cl.K_AIR_W_M_K / cl.L_GLASS_GAP_M * (t_ads - t_glass)
    ads_emit = _emit_w_m2(t_ads, eps_ads)
    ads_reflect = reflect_glass * ads_emit
    glass_emit = _emit_w_m2(t_glass, eps_glass_ir)
    glass_conv = h_front * (t_glass - tint_c)
    q_in = q_solar_w_m2 if solar_on else 0.0
    stage_cond = cl.stage_conductance_w_m2_k() * (t_ads - t_gel_c)

    r_glass = (
        solar_glass * q_in
        + glass_cond
        - glass_conv
        - glass_emit
        + ads_emit
        - ads_reflect
    )
    r_ads = (
        -glass_cond
        + solar_ads * q_in
        - ads_emit
        + ads_reflect
        + cl.GLASS_EMIT_BACK_FRAC * glass_emit
        - stage_cond
    )
    return np.array([r_glass, r_ads], dtype=float)


def _no_glass_absorber_residual(
    t_ads: float,
    *,
    t_gel_c: float,
    q_solar_w_m2: float,
    solar_on: bool,
    h_front: float,
    tint_c: float,
    optics: dict[str, float | bool],
) -> float:
    eps_ads = float(optics["eps_ads"])
    solar_ads = float(optics["solar_ads_frac"])
    q_in = q_solar_w_m2 if solar_on else 0.0
    ads_emit = _emit_w_m2(t_ads, eps_ads)
    stage_cond = cl.stage_conductance_w_m2_k() * (t_ads - t_gel_c)
    return solar_ads * q_in - h_front * (t_ads - tint_c) - ads_emit - stage_cond


def solve_glass_absorber_temps(
    *,
    t_gel_c: float,
    q_solar_w_m2: float,
    solar_on: bool,
    h_front: float = cl.H_FRONT_W_M2_K,
    tint_c: float = cl.T_INT_C,
    optics: dict[str, float | bool] | None = None,
    t_guess: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Solve COMSOL algebraic T_ads, T_glass (°C)."""
    opt = optics or {
        "solar_ads_frac": cl.SOLAR_ADS_FRAC,
        "solar_glass_frac": cl.SOLAR_GLASS_FRAC,
        "eps_ads": cl.EPS_ADS,
        "eps_glass_ir": cl.EPS_GLASS_IR,
        "reflect_glass": cl.REFLECT_GLASS,
        "has_glass": True,
    }
    if not bool(opt.get("has_glass", True)):
        t0 = clamp_temperature_c(tint_c + 10.0) if t_guess is None else t_guess[0]
        sol = root(
            lambda t: _no_glass_absorber_residual(
                float(t[0]),
                t_gel_c=t_gel_c,
                q_solar_w_m2=q_solar_w_m2,
                solar_on=solar_on,
                h_front=h_front,
                tint_c=tint_c,
                optics=opt,
            ),
            x0=np.array([t0]),
            method="hybr",
            tol=1e-8,
        )
        t_ads = clamp_temperature_c(float(sol.x[0]) if sol.success else t0)
        return t_ads, tint_c

    if t_guess is None:
        t0 = clamp_temperature_c(tint_c + 5.0)
        t_guess = (t0, tint_c)
    def residual(x: np.ndarray) -> np.ndarray:
        return _glass_absorber_residuals(
            x,
            t_gel_c=t_gel_c,
            q_solar_w_m2=q_solar_w_m2,
            solar_on=solar_on,
            h_front=h_front,
            tint_c=tint_c,
            optics=opt,
        )

    sol = root(
        residual,
        x0=np.array([t_guess[0], t_guess[1]]),
        method="hybr",
        tol=1e-8,
    )
    if sol.success:
        return clamp_temperature_c(float(sol.x[0])), clamp_temperature_c(float(sol.x[1]))
    return t_guess[0], t_guess[1]


def gel_heating_w_m2(
    *,
    t_gel_c: float,
    t_cond_c: float,
    t_ads_c: float,
    h_m: float,
    vapor_gap_m: float,
    qdes_w_m2: float,
    phase: str,
    h_cond: float = cl.H_COND_W_M2_K,
    tint_c: float = cl.T_INT_C,
) -> float:
    gap = max(vapor_gap_m - h_m, 0.0)
    if phase == "desorption":
        hconv = cl.comsol_h_conv_vapor_gap_w_m2_k(gap, t_gel_c, t_cond_c)
        gap_conv = hconv * (t_gel_c - t_cond_c)
        stage_cond = cl.stage_conductance_w_m2_k() * (t_ads_c - t_gel_c)
        return -qdes_w_m2 - stage_cond + gap_conv
    if phase == "cooling":
        return h_cond * (t_gel_c - tint_c)
    return 0.0


def cond_heating_w_m2(
    *,
    t_gel_c: float,
    t_cond_c: float,
    h_m: float,
    vapor_gap_m: float,
    qcond_w_m2: float,
    phase: str,
    h_cond: float = cl.H_COND_W_M2_K,
    tint_c: float = cl.T_INT_C,
) -> float:
    gap = max(vapor_gap_m - h_m, 0.0)
    if phase == "desorption":
        hconv = cl.comsol_h_conv_vapor_gap_w_m2_k(gap, t_gel_c, t_cond_c)
        gap_conv = hconv * (t_gel_c - t_cond_c)
        t_back = t_cond_c - (cl.L_COND_M / cl.K_AL_W_M_K) * (
            hconv * (t_gel_c - t_cond_c) + qcond_w_m2
        )
        cond_conv = h_cond * (t_back - tint_c)
        return -qcond_w_m2 - gap_conv + cond_conv
    if phase == "cooling":
        return h_cond * (t_cond_c - tint_c)
    return 0.0
