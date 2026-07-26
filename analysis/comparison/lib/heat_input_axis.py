"""Map a normalized ``heat_input_frac in [0, 1]`` to each config's physical heat input.

``passive`` is driven by solar irradiance (W/m^2); the three waste-heat
configs are driven by a source/loop temperature (deg C). Keeping the
normalized fraction as the common sweep axis lets ``grid_heatmap.py`` treat
"heat input" uniformly across configs while every output CSV still records
the real physical quantity (never just the dimensionless fraction).
"""

from __future__ import annotations

from dataclasses import dataclass

SOLAR_W_M2_RANGE: tuple[float, float] = (200.0, 900.0)
SOURCE_TEMP_C_RANGE: tuple[float, float] = (40.0, 75.0)


@dataclass(frozen=True, slots=True)
class HeatInputMapping:
    physical_value: float
    unit: str
    param_name: str


def map_heat_input_frac(config_id: str, heat_input_frac: float) -> HeatInputMapping:
    """Linearly map ``heat_input_frac`` to the physical heat-input value for ``config_id``."""
    frac = float(heat_input_frac)
    if config_id == "passive":
        lo, hi = SOLAR_W_M2_RANGE
        value = lo + frac * (hi - lo)
        return HeatInputMapping(physical_value=value, unit="W/m^2", param_name="solar_w_m2")
    if config_id in ("single_loop", "multi_loop", "multi_noloop"):
        lo, hi = SOURCE_TEMP_C_RANGE
        value = lo + frac * (hi - lo)
        param_name = "t_f_c" if config_id == "single_loop" else "t_wh_in_c"
        return HeatInputMapping(physical_value=value, unit="degC", param_name=param_name)
    raise ValueError(f"Unknown config_id: {config_id!r}")
