"""Site-level salt feasibility and LCOW simulation helpers for global maps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from solar_lumped.economics.lcow import lcow_from_daily_yield
from solar_lumped.economics.params import LCOEconomicParams
from solar_lumped.physics.device_balances import solve_steady_thermal
from solar_lumped.physics.salt_properties import (
    SaltProperties,
    desorption_water_activity,
    get_salt,
)
from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.simulation.ode_system import run_daily_cycle
from solar_lumped.weather.profiles import DailyWeatherProfile

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
