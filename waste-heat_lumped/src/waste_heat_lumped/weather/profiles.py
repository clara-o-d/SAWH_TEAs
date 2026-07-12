"""Weather profiles for fluid-heated daily-cycle SAWH."""

from __future__ import annotations

from dataclasses import dataclass

from waste_heat_lumped.physics import device_defaults as dd
from waste_heat_lumped.physics.salt_properties import (
    FABRICATION_EQUILIBRIUM_RH,
    equilibrium_c_w_from_dvs_at_rh,
)

PHASE_DT_S = 100.0
PHASE_HOURS = 12.0
STEPS_PER_PHASE = int(round(PHASE_HOURS * 3600.0 / PHASE_DT_S))


@dataclass(frozen=True, slots=True)
class PhaseProfile:
    """One half-cycle (12 h) ambient boundary conditions."""

    temperature_c: tuple[float, ...]
    relative_humidity: tuple[float, ...]
    h_amb_w_m2_k: tuple[float, ...]
    dt_s: float = PHASE_DT_S


@dataclass(frozen=True, slots=True)
class DailyWeatherProfile:
    absorption: PhaseProfile
    desorption: PhaseProfile


def _constant_phase(
    *,
    n: int = STEPS_PER_PHASE,
    t_amb_c: float,
    rh: float,
    h_amb: float,
    dt_s: float = PHASE_DT_S,
) -> PhaseProfile:
    return PhaseProfile(
        temperature_c=(t_amb_c,) * n,
        relative_humidity=(rh,) * n,
        h_amb_w_m2_k=(h_amb,) * n,
        dt_s=dt_s,
    )


def datacenter_baseline_profile(
    *,
    t_amb_c: float = dd.T_AMB_C,
    rh: float = dd.RH_AMB,
    h_amb: float = dd.H_AMB_W_M2_K,
    dt_s: float = PHASE_DT_S,
) -> DailyWeatherProfile:
    """Fixed 12 h absorption + 12 h desorption at data-center return-air conditions."""
    abs_prof = _constant_phase(t_amb_c=t_amb_c, rh=rh, h_amb=h_amb, dt_s=dt_s)
    des_prof = _constant_phase(t_amb_c=t_amb_c, rh=rh, h_amb=h_amb, dt_s=dt_s)
    return DailyWeatherProfile(absorption=abs_prof, desorption=des_prof)


def baseline_profile(
    *,
    temperature_c: float = dd.T_AMB_C,
    relative_humidity: float = dd.RH_AMB,
    h_amb_w_m2_k: float = dd.H_AMB_W_M2_K,
) -> DailyWeatherProfile:
    return datacenter_baseline_profile(
        t_amb_c=temperature_c,
        rh=relative_humidity,
        h_amb=h_amb_w_m2_k,
    )


def baseline_initial_c_w(
    *,
    equilibrium_rh: float = FABRICATION_EQUILIBRIUM_RH,
    h0_m: float = dd.H0_M,
) -> float:
    return equilibrium_c_w_from_dvs_at_rh(equilibrium_rh, h0_m)
