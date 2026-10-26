"""Shared multi-site definitions: the scenario list, the parameters.xlsx baseline design
(``BASELINE_COMBO``), and the per-instance SystemConfig. The ``DEFAULT_*`` geometry ranges
set gpu_sweep/run_bayesopt_sweep.py's design box. For a single-day CPU run of one design,
use ``system.py --weather-mode real --day YYYY-MM-DD``."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from solar_lumped.physics import (
    EPS_ABS,
    EPS_ABS_IR_CASE2,
    EPS_GLASS_IR_CASE2,
    FIN_AREA_RATIO,
    H0_M,
    L_G_M,
    L_INS_M,
    SALT_LOADING_DEFAULT,
    SystemThermalParams,
    TAU_GLASS,
    # Re-exported as gps.TILT_DEG by gpu_sweep's CLI defaults -- not unused.
    TILT_DEG,  # noqa: F401
)
from solar_lumped.simulation import SystemConfig

# Baselines: table_s3.H0_M=4mm, L_G_M=40mm, EPS_ABS=0.95, TAU_GLASS=0.9, FIN_AREA_RATIO=7.1.
DEFAULT_HYDROGEL_THICKNESS_MM: tuple[float, ...] = (1.0, 3.25, 5.5, 7.75, 10.0)
DEFAULT_FIN_AREA_RATIO: tuple[float, ...] = (3.0, 5.275, 7.55, 9.825, 12.0)
DEFAULT_VAPOR_GAP_MM: tuple[float, ...] = (20.0, 30.0, 40.0, 50.0, 60.0)


@dataclass(frozen=True, slots=True)
class Combo:
    hydrogel_thickness_mm: float
    fin_area_ratio: float
    vapor_gap_mm: float


# --- Scenarios: optics and relaxed physical limits, each at the baseline design --------
BASELINE_COMBO = Combo(
    hydrogel_thickness_mm=H0_M * 1e3,
    fin_area_ratio=FIN_AREA_RATIO,
    vapor_gap_mm=L_G_M * 1e3,
)
BASELINE_INSULATION_GAP_MM: float = L_INS_M * 1e3
BASELINE_SALT_LOADING: float = SALT_LOADING_DEFAULT


@dataclass(frozen=True, slots=True)
class Scenario:
    """Absorber/glass optics, plus which physical limits are relaxed.

    ``instant_equilibrium`` (g -> infinity) and ``condenser_ambient`` (T_cond == T_amb)
    each select a *code path* in the JAX step, not a per-instance number, so they are
    uniform across a compiled batch.
    """

    eps_abs: float
    tau_glass: float
    eps_abs_ir: float
    eps_glass_ir: float
    instant_equilibrium: bool = False
    condenser_ambient: bool = False


# Wilson & Diaz-Marin's original blackbody/cavity radiative exchange (eps_IR = 1).
_WILSON = Scenario(eps_abs=EPS_ABS, tau_glass=TAU_GLASS, eps_abs_ir=1.0, eps_glass_ir=1.0)
# "Reasonable improvements": a real selective absorber behind ordinary glass.
_IMPROVED = Scenario(
    eps_abs=EPS_ABS, tau_glass=TAU_GLASS,
    eps_abs_ir=EPS_ABS_IR_CASE2, eps_glass_ir=EPS_GLASS_IR_CASE2,
)
# "Optical material limits": perfect absorber, perfect glass, no IR loss.
_LIMITS = Scenario(eps_abs=1.0, tau_glass=1.0, eps_abs_ir=0.0, eps_glass_ir=0.0)

SCENARIOS: dict[str, Scenario] = {
    "wilson": _WILSON,
    "improved": _IMPROVED,
    "optical_limits": _LIMITS,
    # Instantaneous sorption (g -> infinity): desorption becomes energy-limited rather
    # than transport-limited. Only meaningful on top of the two improved optics cases.
    "improved_instant_g": dataclasses.replace(_IMPROVED, instant_equilibrium=True),
    "optical_limits_instant_g": dataclasses.replace(_LIMITS, instant_equilibrium=True),
    # Perfect condenser (T_cond == T_amb): infinite cooling capacity, so the condenser
    # never warms under its own latent load.
    "improved_perfect_cond": dataclasses.replace(_IMPROVED, condenser_ambient=True),
    "optical_limits_perfect_cond": dataclasses.replace(_LIMITS, condenser_ambient=True),
    # All three limits at once -- the idealized-device upper bound.
    "optical_limits_instant_g_perfect_cond": dataclasses.replace(
        _LIMITS, instant_equilibrium=True, condenser_ambient=True,
    ),
}


def build_system_config(
    combo: Combo,
    *,
    salt: str,
    salt_loading: float,
    insulation_gap_mm: float,
    tilt_deg: float,
    eps_abs: float,
    tau_glass: float,
    eps_abs_ir: float | None = EPS_ABS_IR_CASE2,
    eps_glass_ir: float | None = EPS_GLASS_IR_CASE2,
    # Site elevation, from the weather frame (weather.site_elevation_m). Thins the air:
    # more vapor diffusivity and a better-insulated collector, so high sites gain. The
    # 0.0 default keeps every non-site caller (tests, single designs) at sea level.
    site_elevation_m: float = 0.0,
) -> SystemConfig:
    thermal = SystemThermalParams(
        insulation_gap_m=insulation_gap_mm * 1e-3,
        vapor_gap_m=combo.vapor_gap_mm * 1e-3,
        eps_abs=eps_abs,
        tau_glass=tau_glass,
        tilt_deg=tilt_deg,
        eps_abs_ir=eps_abs_ir,
        eps_glass_ir=eps_glass_ir,
    )
    return SystemConfig(
        salt_name=salt,
        salt_loading=salt_loading,
        hydrogel_thickness_m=combo.hydrogel_thickness_mm * 1e-3,
        vapor_gap_m=combo.vapor_gap_mm * 1e-3,
        insulation_gap_m=insulation_gap_mm * 1e-3,
        fin_area_ratio=combo.fin_area_ratio,
        tilt_deg=tilt_deg,
        thermal=thermal,
        site_elevation_m=site_elevation_m,
    )
