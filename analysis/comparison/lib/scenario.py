"""Shared baseline scenario constants for cross-package comparison.

Ambient/economic conditions applied uniformly to both device configs so comparisons
are apples-to-apples. Individual scripts may override any of these via CLI flags.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Ambient conditions (identical across both configs) ---
T_AMB_C: float = 32.0
RH_AMB: float = 0.45
H_AMB_W_M2_K: float = 10.0

# --- Hydrogel / sorbent material (identical across both configs) ---
SALT_NAME: str = "LiCl"
SALT_TO_POLYMER_RATIO: float = 4.0
HYDROGEL_THICKNESS_M: float = 0.004

# --- Economics ---
DEVICE_LIFETIME_YEARS: int = 20
DISCOUNT_RATE: float = 0.03
ELECTRICITY_PRICE_USD_PER_KWH: float = 0.10

# $5.00/m3 matches every per-package parameter_sweep.py's own baseline, chosen over the
# benchmark/tariff 15 m3-tier median (~$9-10/m3) for continuity with the sweeps/heatmaps
# already produced, so numbers here are directly comparable to those artifacts.
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
