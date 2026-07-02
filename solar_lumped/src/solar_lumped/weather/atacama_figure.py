"""Load Wilson Fig. 4 Atacama field-test weather (24 h from 18:00)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from solar_lumped.weather.profiles import (
    PHASE_DT_S,
    STEPS_PER_PHASE,
    DailyWeatherProfile,
    PhaseProfile,
)

_PACKAGE_DIR = Path(__file__).resolve().parent
ATACAMA_RH_CSV = _PACKAGE_DIR / "Atacama_RH.csv"
ATACAMA_TEMP_CSV = _PACKAGE_DIR / "Atacama_Temp.csv"
ATACAMA_SOLAR_CSV = _PACKAGE_DIR / "solar_kW_m2.csv"

# 24 h timeline origin: 18:00 (6 pm) on absorption night, matching paper figure.
CYCLE_ORIGIN_HOUR = 18.0
ABSORPTION_HOURS = 12.0
DESORPTION_HOURS = 12.0

# Atacama field protocol (Methods): install at 8 a.m., ~8 h desorption in sun.
ATACAMA_INSTALL_HOUR_FROM_ORIGIN = 14.0  # 18:00 + 14 h = 08:00
ATACAMA_FIELD_DESORPTION_HOURS = 8.0
ATACAMA_FIELD_DESORPTION_STEPS = int(
    round(ATACAMA_FIELD_DESORPTION_HOURS * 3600.0 / PHASE_DT_S)
)


def _load_figure_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    hours: list[float] = []
    values: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            hours.append(float(parts[0].strip()))
            values.append(float(parts[1].strip()))
    if not hours:
        raise ValueError(f"No data in {path}")
    h = np.array(hours, dtype=float)
    v = np.array(values, dtype=float)
    order = np.argsort(h)
    return h[order], v[order]


def _phase_hour_grid(
    phase_start_h: float,
    phase_hours: float,
    n_steps: int,
) -> np.ndarray:
    """Hours from 18:00 origin at cell-centers for a phase."""
    dt_h = phase_hours / n_steps
    return phase_start_h + (np.arange(n_steps, dtype=float) + 0.5) * dt_h


def _interp_clamped(
    h_data: np.ndarray,
    v_data: np.ndarray,
    h_query: np.ndarray,
) -> np.ndarray:
    h_min, h_max = float(h_data.min()), float(h_data.max())
    hq = np.clip(h_query, h_min, h_max)
    return np.interp(hq, h_data, v_data)


def _atacama_h_amb_w_m2_k(hours_from_6pm: float) -> float:
    """Paper Atacama: h=1 W/m²K overnight; h=10 from 08:00 (hour 14 on Fig. 4 axis).

    The Fig. 4 timeline starts at 18:00; field desorption begins at hour 14 (= 08:00).
    Previously this used hour 20 (14:00 wall time), leaving morning desorption at h=1
    and overheating the condenser.
    """
    return 10.0 if hours_from_6pm >= ATACAMA_INSTALL_HOUR_FROM_ORIGIN else 1.0


def _build_atacama_profile(
    *,
    desorption_start_h: float,
    desorption_hours: float,
    desorption_steps: int,
) -> DailyWeatherProfile:
    h_rh, rh = _load_figure_csv(ATACAMA_RH_CSV)
    h_t, temp_c = _load_figure_csv(ATACAMA_TEMP_CSV)
    h_s, solar_kw = _load_figure_csv(ATACAMA_SOLAR_CSV)

    abs_h = _phase_hour_grid(0.0, ABSORPTION_HOURS, STEPS_PER_PHASE)
    des_h = _phase_hour_grid(desorption_start_h, desorption_hours, desorption_steps)

    abs_rh = _interp_clamped(h_rh, rh, abs_h)
    des_rh = _interp_clamped(h_rh, rh, des_h)
    abs_t = _interp_clamped(h_t, temp_c, abs_h)
    des_t = _interp_clamped(h_t, temp_c, des_h)
    des_solar = _interp_clamped(h_s, solar_kw, des_h) * 1000.0  # kW/m² → W/m²

    abs_hamb = tuple(_atacama_h_amb_w_m2_k(float(h)) for h in abs_h)
    des_hamb = tuple(_atacama_h_amb_w_m2_k(float(h)) for h in des_h)

    return DailyWeatherProfile(
        absorption=PhaseProfile(
            temperature_c=tuple(float(x) for x in abs_t),
            relative_humidity=tuple(float(x) for x in abs_rh),
            solar_w_m2=(0.0,) * STEPS_PER_PHASE,
            h_amb_w_m2_k=abs_hamb,
            dt_s=PHASE_DT_S,
        ),
        desorption=PhaseProfile(
            temperature_c=tuple(float(x) for x in des_t),
            relative_humidity=tuple(float(x) for x in des_rh),
            solar_w_m2=tuple(max(0.0, float(x)) for x in des_solar),
            h_amb_w_m2_k=des_hamb,
            dt_s=PHASE_DT_S,
        ),
    )


def atacama_figure_profile() -> DailyWeatherProfile:
    """Fig. 4 symmetric 12 h + 12 h replay (legacy)."""
    return _build_atacama_profile(
        desorption_start_h=ABSORPTION_HOURS,
        desorption_hours=DESORPTION_HOURS,
        desorption_steps=STEPS_PER_PHASE,
    )


def atacama_field_profile() -> DailyWeatherProfile:
    """Atacama field validation: 12 h open absorption, install 8 a.m., 8 h desorption."""
    return _build_atacama_profile(
        desorption_start_h=ATACAMA_INSTALL_HOUR_FROM_ORIGIN,
        desorption_hours=ATACAMA_FIELD_DESORPTION_HOURS,
        desorption_steps=ATACAMA_FIELD_DESORPTION_STEPS,
    )
