"""Complex-fidelity mode: ZSR mixing, glazing stack, fin efficiency, and cost deltas.

The load-bearing property is the first test: ZSR must reduce to each salt's own
binary isotherm at the pure-salt corners. Everything else in B8 rests on that, and
it is what licenses the single-salt fast path.
"""

from __future__ import annotations

import math

import pytest

from solar_lumped.complex_model import (
    CASE2_COATING_PREMIUM_USD_PER_M2,
    ZSR_SALTS,
    ComplexOptions,
    absorber_coating_cost_usd_per_m2,
    blend_fully_dissolved_window,
    blend_water_activity_window,
    clamp_reference_rh,
    complex_system_capex_usd_per_m2,
    fin_efficiency,
    fin_material_cost_usd_per_m2,
    forced_cooling_capex_usd_per_m2,
    glazing_cost_usd_per_m2,
    zsr_blend_price_usd_per_kg,
    zsr_brine_state,
    zsr_water_activity_at_brine_fraction,
)
from solar_lumped.economics import C_SYSTEM_USD, complex_system_cost_usd
from solar_lumped.physics import (
    SystemThermalParams,
    condenser_h_conv_w_m2_k,
    equilibrate_salt_mf,
    get_salt,
    solve_steady_thermal,
    thermal_residual_norm,
)


# --- B8: ZSR mixing rule ---


# Tolerance of the tabulated molality curve (_MOLALITY_GRID_POINTS linear interp),
# measured at ~5e-7. Orders of magnitude below the isotherm fits' own accuracy, and
# the price of not root-solving inside the ODE right-hand side.
ZSR_INTERP_TOL = 1e-5


@pytest.mark.parametrize("index,salt", list(enumerate(ZSR_SALTS)))
@pytest.mark.parametrize("rh", [0.40, 0.55, 0.70, 0.85])
def test_zsr_reduces_to_binary_isotherm_at_pure_corners(index: int, salt: str, rh: float) -> None:
    """A one-hot blend must return that salt's own equilibrium brine fraction."""
    weights = tuple(1.0 if i == index else 0.0 for i in range(len(ZSR_SALTS)))
    f_zsr, _ions, mw = zsr_brine_state(weights, rh, 25.0)
    f_binary = equilibrate_salt_mf(salt, rh, 25.0)
    assert f_zsr == pytest.approx(f_binary, abs=ZSR_INTERP_TOL)
    assert mw == pytest.approx(get_salt(salt).formula_weight_g_mol, rel=1e-12)


def test_zsr_blend_is_bracketed_by_its_members() -> None:
    """A blend's salt fraction sits between the pure endpoints at the same a_w."""
    f_licl, _, _ = zsr_brine_state((1.0, 0.0, 0.0), 0.55, 25.0)
    f_cacl2, _, _ = zsr_brine_state((0.0, 1.0, 0.0), 0.55, 25.0)
    f_mix, _, _ = zsr_brine_state((0.5, 0.5, 0.0), 0.55, 25.0)
    assert min(f_licl, f_cacl2) < f_mix < max(f_licl, f_cacl2)


def test_operating_window_reaches_as_dry_as_the_best_member() -> None:
    """Weaker salts precipitate; the strongest keeps a brine, so it sets the floor."""
    for weights in [(1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.4, 0.3, 0.3)]:
        lo, _ = blend_water_activity_window(weights)
        assert lo == pytest.approx(get_salt("LiCl").rh_min, abs=1e-3)
    assert blend_water_activity_window((0.0, 1.0, 0.0))[0] == pytest.approx(
        get_salt("CaCl2").rh_min, abs=1e-3
    )


def test_fully_dissolved_window_is_the_intersection() -> None:
    """Fabrication references the state where all the salt is actually in solution."""
    # The binding constraint is whichever active salt deliquesces last, so assert the
    # max over active salts rather than naming one: the winner depends on the derived
    # DRHs and so on the solubility data (CaCl2 and MgCl2 sit within ~0.01 of each other).
    for weights in ((1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.4, 0.3, 0.3)):
        expected = max(
            get_salt(salt).rh_min
            for salt, w in zip(ZSR_SALTS, weights)
            if w > 0.0
        )
        assert blend_fully_dissolved_window(weights)[0] == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("weights", [(0.999, 0.001, 0.0), (0.5, 0.5, 0.0), (0.4, 0.3, 0.3)])
def test_brine_fraction_is_continuous_and_monotone_across_deliquescence_points(
    weights: tuple[float, ...],
) -> None:
    """The saturation plateau is what keeps f_b(a_w) invertible.

    Dropping a precipitated salt to zero instead would put a step at each member's
    DRH -- non-monotone, so the tabulated inverse silently returns garbage, and a
    trace component would abolish an otherwise strong brine.
    """
    import numpy as np

    lo, hi = blend_water_activity_window(weights)
    aw = np.linspace(lo, hi, 400)
    fb = np.array([zsr_brine_state(weights, float(a), 25.0)[0] for a in aw])
    assert np.isfinite(fb).all()
    diffs = np.diff(fb)
    assert (diffs <= 1e-12).all(), "f_b must be non-increasing in a_w"
    assert np.abs(diffs).max() < 0.01, "no step discontinuity at a deliquescence point"


@pytest.mark.parametrize("weights", [(1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.3, 0.2)])
def test_zsr_inverse_round_trips(weights: tuple[float, ...]) -> None:
    """a_w -> f_b -> a_w, over the blend's own validated window."""
    lo, hi = blend_water_activity_window(weights)
    for k in range(1, 10):
        aw = lo + (hi - lo) * k / 10.0
        f_b, _, _ = zsr_brine_state(weights, aw, 25.0)
        assert math.isfinite(f_b)
        assert zsr_water_activity_at_brine_fraction(weights, f_b, 25.0) == pytest.approx(
            aw, abs=1e-5
        )


def test_precipitated_member_stops_contributing_but_does_not_kill_the_brine() -> None:
    """At 20% RH MgCl2 (DRH 0.33) is solid, yet a LiCl-bearing blend still has one."""
    f_mix, _, _ = zsr_brine_state((0.5, 0.0, 0.5), 0.20, 25.0)
    f_licl, _, _ = zsr_brine_state((1.0, 0.0, 0.0), 0.20, 25.0)
    assert math.isfinite(f_mix) and math.isfinite(f_licl)
    # Half the molality weight is a salt held at saturation, so the mixture is
    # more concentrated than half-strength LiCl alone but still a real brine.
    assert 0.0 < f_mix < 1.0


def test_clamp_reference_rh_pulls_fabrication_into_the_window() -> None:
    """Wilson casts at 20% RH; a CaCl2 blend cannot, so it is cast wetter."""
    assert clamp_reference_rh((1.0, 0.0, 0.0), 0.20) == pytest.approx(0.20, abs=1e-6)
    assert clamp_reference_rh((0.5, 0.5, 0.0), 0.20) > get_salt("CaCl2").rh_min


def test_blend_price_is_mass_weighted_between_member_prices() -> None:
    """ZSR weights are molality weights; pricing converts to mass shares first."""
    p_licl = zsr_blend_price_usd_per_kg((1.0, 0.0, 0.0), reference_rh=0.35)
    p_cacl2 = zsr_blend_price_usd_per_kg((0.0, 1.0, 0.0), reference_rh=0.35)
    p_mix = zsr_blend_price_usd_per_kg((0.5, 0.5, 0.0), reference_rh=0.35)
    assert p_licl == pytest.approx(get_salt("LiCl").price_usd_per_kg, rel=1e-9)
    assert p_cacl2 == pytest.approx(get_salt("CaCl2").price_usd_per_kg, rel=1e-9)
    assert min(p_licl, p_cacl2) < p_mix < max(p_licl, p_cacl2)


# --- B2: glazing stack ---


def test_single_pane_default_reproduces_wilson_residuals() -> None:
    assert _solve(SystemThermalParams()).t_glass_outer_c is None
    assert _residual_norm(SystemThermalParams()) < 1e-6


def _solve(params: SystemThermalParams):
    """Wilson's 600 W/m2 / 25 C desorption operating point, one gel thickness."""
    return solve_steady_thermal(
        t_cond_c=25.0, t_amb_c=25.0, q_solar_w_m2=600.0, m_des_kg_s_m2=1e-5,
        h_amb=10.0, params=params, h_m=0.004,
    )


def _residual_norm(params: SystemThermalParams) -> float:
    return thermal_residual_norm(
        t_cond_c=25.0, t_amb_c=25.0, q_solar_w_m2=600.0, m_des_kg_s_m2=1e-5,
        h_amb=10.0, params=params, h_m=0.004,
    )


def _two_pane(**overrides) -> SystemThermalParams:
    tau = SystemThermalParams().tau_glass
    return SystemThermalParams(n_glazing_panes=2, tau_glass=tau**2, **overrides)


def test_second_pane_adds_a_solved_outer_temperature_and_raises_gel_temp() -> None:
    """Two panes lose 19% of transmittance yet still run hotter -- that is B2's trade."""
    one = _solve(SystemThermalParams())
    two = _solve(_two_pane())
    assert two.t_glass_outer_c is not None
    # Physical cascade: absorber hottest, then inner pane, outer pane, ambient.
    assert two.t_abs_c > two.t_glass_c > two.t_glass_outer_c > 25.0
    assert two.t_gel_c > one.t_gel_c
    assert _residual_norm(_two_pane()) < 1e-6


def test_evacuated_gap_beats_air_gap() -> None:
    assert _solve(_two_pane(evacuated_gap=True)).t_gel_c > _solve(_two_pane()).t_gel_c


# --- B3: fin efficiency and mass cost ---


def test_fin_efficiency_is_bounded_and_falls_with_length() -> None:
    short = fin_efficiency(10.0, fin_thickness_m=1e-3, fin_height_m=0.010)
    long = fin_efficiency(10.0, fin_thickness_m=1e-3, fin_height_m=0.100)
    assert 0.0 < long < short <= 1.0


def test_finned_condenser_never_beats_the_ideal_fin() -> None:
    ideal = condenser_h_conv_w_m2_k(10.0, fin_area_ratio=7.0)
    real = condenser_h_conv_w_m2_k(
        10.0, fin_area_ratio=7.0, fin_thickness_m=1e-3, fin_height_m=0.025
    )
    assert real < ideal
    # The bare base plate is never derated, only the added fin area.
    assert real > 10.0


def test_fin_cost_scales_with_added_area() -> None:
    c3 = fin_material_cost_usd_per_m2(3.0, fin_thickness_m=1e-3)
    c12 = fin_material_cost_usd_per_m2(12.0, fin_thickness_m=1e-3)
    assert fin_material_cost_usd_per_m2(1.0, fin_thickness_m=1e-3) == pytest.approx(0.0)
    assert c12 == pytest.approx(c3 * (11.0 / 2.0), rel=1e-9)


# --- B1 / B4 cost correlations ---


def test_coating_cost_decreases_with_emissivity() -> None:
    assert absorber_coating_cost_usd_per_m2(0.05) > absorber_coating_cost_usd_per_m2(0.95)
    # Clamped outside the anchor table rather than extrapolated to nonsense.
    assert absorber_coating_cost_usd_per_m2(0.0) == pytest.approx(
        absorber_coating_cost_usd_per_m2(0.05)
    )


def test_case2_selective_surface_costs_a_ten_dollar_premium_over_paint() -> None:
    """The measured material premium; a re-fit that breaks it should fail here."""
    premium = absorber_coating_cost_usd_per_m2(0.05) - absorber_coating_cost_usd_per_m2(0.95)
    assert premium == pytest.approx(CASE2_COATING_PREMIUM_USD_PER_M2, abs=0.01)


def test_forced_cooling_matches_wilson_table_s2_at_his_design_point() -> None:
    """0.5 m/s is Wilson's own fan array: $34.50 fans + $20.00 PV."""
    assert forced_cooling_capex_usd_per_m2(0.5) == pytest.approx(54.50, rel=1e-9)
    assert forced_cooling_capex_usd_per_m2(0.0) == 0.0


def test_glazing_cost_scales_per_pane() -> None:
    assert glazing_cost_usd_per_m2(0) == 0.0
    assert glazing_cost_usd_per_m2(2) == pytest.approx(2 * glazing_cost_usd_per_m2(1))
    assert glazing_cost_usd_per_m2(1, evacuated=True) > glazing_cost_usd_per_m2(1)


# --- Cost integration: defaults must be a no-op against the flat Wilson BOM ---


def test_default_complex_options_cost_exactly_the_simple_bom() -> None:
    from solar_lumped.physics import FIN_AREA_RATIO

    assert complex_system_capex_usd_per_m2(
        ComplexOptions(), fin_area_ratio=FIN_AREA_RATIO
    ) == pytest.approx(0.0, abs=1e-9)
    assert complex_system_cost_usd(
        ComplexOptions(), fin_area_ratio=FIN_AREA_RATIO
    ) == pytest.approx(C_SYSTEM_USD, abs=1e-9)
    assert complex_system_cost_usd(None, fin_area_ratio=FIN_AREA_RATIO) == C_SYSTEM_USD


def test_complex_options_rejects_malformed_input() -> None:
    with pytest.raises(ValueError):
        ComplexOptions(blend_weights=(0.5, 0.5))  # wrong length
    with pytest.raises(ValueError):
        ComplexOptions(n_glazing_panes=3)


def test_single_salt_corners_take_the_fast_path() -> None:
    assert not ComplexOptions().is_blend
    assert ComplexOptions().dominant_salt == "LiCl"
    assert not ComplexOptions(blend_weights=(0.0, 1.0, 0.0)).is_blend
    assert ComplexOptions(blend_weights=(0.0, 1.0, 0.0)).dominant_salt == "CaCl2"
    assert ComplexOptions(blend_weights=(0.6, 0.4, 0.0)).is_blend
