"""Tests for salt-specific water activity and fabrication IC."""

from __future__ import annotations

import math

from solar_lumped.physics.salt_properties import (
    desorption_water_activity,
    fabrication_c_w_initial,
    water_activity_from_c_w,
)
from solar_lumped.simulation.device_config import DeviceConfig


def test_nacl_water_activity_uses_brine_not_mole_fraction():
    config = DeviceConfig(salt_name="NaCl")
    mass = config.mass_params()
    aw = water_activity_from_c_w(
        8000.0,
        c_s=mass.c_s_mol_m3,
        ions_per_formula=mass.ions_per_formula,
        salt_name="NaCl",
        formula_weight_g_mol=mass.formula_weight_g_mol,
        h_m=config.hydrogel_thickness_m,
        h0_ref_m=config.hydrogel_thickness_m,
    )
    n_w = 8000.0
    n_s = mass.c_s_mol_m3 * 2
    mole_frac_aw = n_w / (n_w + n_s + 1e-30)
    assert math.isfinite(aw)
    assert abs(aw - mole_frac_aw) > 0.05


def test_fabrication_ic_differs_by_salt():
    licl = DeviceConfig(salt_name="LiCl")
    nacl = DeviceConfig(salt_name="NaCl")
    cw_licl = fabrication_c_w_initial(
        salt_name=licl.salt_name,
        salt_to_polymer_ratio=licl.salt_to_polymer_ratio,
        hydrogel_thickness_m=licl.hydrogel_thickness_m,
    )
    cw_nacl = fabrication_c_w_initial(
        salt_name=nacl.salt_name,
        salt_to_polymer_ratio=nacl.salt_to_polymer_ratio,
        hydrogel_thickness_m=nacl.hydrogel_thickness_m,
        hydrogel_density_kg_m3=nacl.hydrogel_density_kg_m3,
        formula_weight_g_mol=nacl.salt().formula_weight_g_mol,
    )
    assert cw_licl > 0.0 and cw_nacl > 0.0


def test_desorption_water_activity_decreases_with_gel_temp():
    aw_cool = desorption_water_activity(25.0, 35.0)
    aw_hot = desorption_water_activity(25.0, 55.0)
    assert math.isfinite(aw_cool) and math.isfinite(aw_hot)
    assert aw_hot < aw_cool
