"""Shared baseline scenario constants for cross-package comparison.

These are the ambient/economic conditions applied uniformly to all four
device configs so that comparisons are apples-to-apples. Individual scripts
may override any of these via CLI flags.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Ambient conditions (identical across all four configs) ---
T_AMB_C: float = 32.0
RH_AMB: float = 0.45
H_AMB_W_M2_K: float = 15.0

# --- Hydrogel / sorbent material (identical across all four configs) ---
SALT_NAME: str = "LiCl"
SALT_TO_POLYMER_RATIO: float = 4.0
HYDROGEL_THICKNESS_M: float = 0.004

# --- Economics ---
DEVICE_LIFETIME_YEARS: int = 20
DISCOUNT_RATE: float = 0.08
ELECTRICITY_PRICE_USD_PER_KWH: float = 0.10

# Water price baseline: $5.00/m3, matching the baseline already used by every
# per-package parameter_sweep.py (`_BASELINE_WATER_PRICE_USD_PER_M3 = 5.0` in
# solar_lumped, waste_heat_lumped, waste_heat_cycle_lumped, and
# waste_heat_cycle_lumped_no_loop) — chosen over the global_tap tariff-table
# 15 m3-tier median (~$9-10/m3 across countries; e.g. Afghanistan's own 15 m3
# tariff is $7.07/m3, not the global median) specifically for continuity with
# the per-package sweeps/heatmaps prior agents already produced, so numbers
# in this comparison directory are directly comparable to those artifacts.
WATER_PRICE_USD_PER_M3: float = 5.0


@dataclass(frozen=True, slots=True)
class Scenario:
    t_amb_c: float = T_AMB_C
    rh_amb: float = RH_AMB
    h_amb_w_m2_k: float = H_AMB_W_M2_K
    salt_name: str = SALT_NAME
    salt_to_polymer_ratio: float = SALT_TO_POLYMER_RATIO
    hydrogel_thickness_m: float = HYDROGEL_THICKNESS_M
    device_lifetime_years: int = DEVICE_LIFETIME_YEARS
    discount_rate: float = DISCOUNT_RATE
    electricity_price_usd_per_kwh: float = ELECTRICITY_PRICE_USD_PER_KWH
    water_price_usd_per_m3: float = WATER_PRICE_USD_PER_M3


BASELINE_SCENARIO = Scenario()
