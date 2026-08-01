"""Map a normalized ``heat_input_frac in [0, 1]`` to each config's physical heat input.

``passive`` is driven by solar irradiance (W/m^2) and ``waste_heat`` by the source
temperature (deg C). The normalized fraction is the common sweep axis, while every
output CSV still records the real physical quantity.
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
    if config_id == "waste_heat":
        lo, hi = SOURCE_TEMP_C_RANGE
        value = lo + frac * (hi - lo)
        return HeatInputMapping(physical_value=value, unit="degC", param_name="t_wh_in_c")
    raise ValueError(f"Unknown config_id: {config_id!r}")
