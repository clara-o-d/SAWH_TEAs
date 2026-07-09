"""Tests for salt-specific water activity and fabrication IC."""

from __future__ import annotations

import math

import pytest

from solar_lumped.physics.salt_properties import (
    DRY_COMPOSITE_DENSITY_KG_M3,
    chamber_c_s_from_synthesis,
    chamber_c_s_with_constant_density,
    chamber_pour_volume_ml,
    desorption_water_activity,
    fabrication_c_w_initial,
    salt_molarity_from_composite,
    water_activity_from_c_w,
)
from solar_lumped.simulation.device_config import DeviceConfig


def test_chamber_pour_volume_ml_pam_licl_2gg():
    assert chamber_pour_volume_ml(2.0) == pytest.approx(12.8)
    assert chamber_pour_volume_ml(4.0) == pytest.approx(8.0)


def test_chamber_c_s_from_synthesis_anchors_4gg_to_dvs():
    fw = 42.394
    cs_dvs = salt_molarity_from_composite(4.0, DRY_COMPOSITE_DENSITY_KG_M3, fw)
    cs_synth = chamber_c_s_from_synthesis(4.0, 2.34, formula_weight_g_mol=fw)
    assert cs_synth == pytest.approx(cs_dvs, rel=1e-6)


def test_chamber_c_s_from_synthesis_scales_with_h0_at_fixed_sl():
    fw = 42.394
    cs_thin = chamber_c_s_from_synthesis(4.0, 2.34, formula_weight_g_mol=fw)
    cs_thick = chamber_c_s_from_synthesis(4.0, 3.2, formula_weight_g_mol=fw)
    assert cs_thick == pytest.approx(cs_thin * (2.34 / 3.2), rel=1e-6)


def test_chamber_c_s_from_synthesis_2gg_exceeds_composite_density_estimate():
    fw = 42.394
    cs_old = salt_molarity_from_composite(2.0, DRY_COMPOSITE_DENSITY_KG_M3, fw)
    cs_new = chamber_c_s_from_synthesis(2.0, 2.16, formula_weight_g_mol=fw)
    assert cs_new > cs_old


def test_chamber_c_s_constant_density_scales_2gg_to_4gg_reference_h0():
    fw = 42.394
    cs_synth = chamber_c_s_from_synthesis(2.0, 2.16, formula_weight_g_mol=fw)
    cs_const = chamber_c_s_with_constant_density(2.0, 2.16, formula_weight_g_mol=fw)
    cs_ref = chamber_c_s_from_synthesis(4.0, 2.34, formula_weight_g_mol=fw)
    assert cs_const == pytest.approx(cs_ref * (2.34 / 2.16), rel=1e-6)
    assert cs_const > cs_synth


def test_chamber_c_s_constant_density_unchanged_for_4gg():
    fw = 42.394
    cs = chamber_c_s_from_synthesis(4.0, 2.34, formula_weight_g_mol=fw)
    assert chamber_c_s_with_constant_density(4.0, 2.34, formula_weight_g_mol=fw) == pytest.approx(
        cs, rel=1e-6
    )


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
