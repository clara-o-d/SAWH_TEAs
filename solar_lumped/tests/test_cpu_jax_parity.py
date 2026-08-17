"""CPU vs JAX backend parity for complex mode.

Both backends must answer the same question the same way -- the CPU path is the
single-site/reference implementation, the JAX path is what the global sweep runs,
and a design that scores differently on the two is worse than useless to an
optimizer that will happily exploit whichever one is wrong.

Skipped unless diffrax/jax are importable (solar_lumped/.venv_gpu has them; the
default env deliberately does not, so the CPU path needs no GPU stack).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("diffrax")
pytest.importorskip("jax")

_GPU_SWEEP = Path(__file__).resolve().parents[1] / "gpu_sweep"
if str(_GPU_SWEEP) not in sys.path:
    sys.path.insert(0, str(_GPU_SWEEP))

from solar_lumped.complex_model import ComplexOptions  # noqa: E402
from solar_lumped.physics import WATER_MOLAR_MASS_KG_MOL, initial_loading  # noqa: E402
from solar_lumped.simulation import SystemConfig, run_daily_cycle  # noqa: E402
from solar_lumped.weather import baseline_profile  # noqa: E402

# The two paths differ by construction in ways that cannot be driven to zero: the
# JAX side reads water activity off a tabulated ZSR inversion while the CPU side
# interpolates the same table at a different point, and the ODE solvers differ
# (Radau vs Tsit5). 0.5% bounds every configuration measured; the simple path holds
# to <0.03% (gpu_sweep/FINDINGS.md) and is asserted tighter below.
COMPLEX_PARITY_TOL = 0.005
SIMPLE_PARITY_TOL = 0.0005


def _jax_daily_yield(config: SystemConfig, profile, *, complex_mode: bool) -> float:
    import jax_daily_cycle as jdc

    system = jdc.build_system_arrays(
        [config], complex_mode=complex_mode,
        instant_equilibrium=config.instant_equilibrium,
    )
    dt, n_abs, n_des = jdc.year_padding([[profile]])
    step = jdc.make_year_step_fn(
        system, dt, n_abs, n_des, complex_mode=complex_mode,
        condenser_tracks_ambient=config.condenser_tracks_ambient,
        # Same gate as SystemConfig.thermal_params(): isosteric h_des on the simple
        # LiCl path only, so the backends cannot silently disagree on it.
        h_des_isosteric=(config.h_des_mode == "isosteric" and config.complex is None),
        instant_equilibrium=config.instant_equilibrium,
    )
    weather = jdc.build_day_weather([profile], n_abs, n_des)
    water, _eta, _c_w, _h = step(
        np.array([initial_loading(config)]),
        np.array([config.hydrogel_thickness_m]),
        weather,
    )
    return float(np.asarray(water)[0])


COMPLEX_CASES = {
    "default": ComplexOptions(),
    "B1_black_paint": ComplexOptions(eps_abs_ir=0.95),
    "B1_mid_selective": ComplexOptions(eps_abs_ir=0.30),
    "B2_uncovered": ComplexOptions(n_glazing_panes=0),
    "B2_two_pane": ComplexOptions(n_glazing_panes=2),
    # Runs the gel ~10 C hotter than anything the simple model reaches, which is
    # what first exposed the closed-form LiCl isotherm's NaN above xi=0.56.
    "B2_two_pane_evacuated": ComplexOptions(n_glazing_panes=2, evacuated_gap=True),
    "B3_thin_tall_fins": ComplexOptions(fin_thickness_m=3e-4, fin_height_m=0.06),
    "B4_forced_air": ComplexOptions(condenser_air_speed_m_s=1.5),
    "B8_binary_blend": ComplexOptions(blend_weights=(0.5, 0.5, 0.0)),
    "B8_ternary_blend": ComplexOptions(blend_weights=(0.5, 0.3, 0.2)),
    "B8_pure_cacl2": ComplexOptions(blend_weights=(0.0, 1.0, 0.0)),
    "B8_pure_mgcl2": ComplexOptions(blend_weights=(0.0, 0.0, 1.0)),
    "all_features": ComplexOptions(
        eps_abs_ir=0.2, n_glazing_panes=2, condenser_air_speed_m_s=1.0,
        blend_weights=(0.6, 0.25, 0.15),
    ),
}


def test_simple_mode_backends_agree() -> None:
    """The pre-existing simple path must not have moved."""
    profile = baseline_profile()
    config = SystemConfig.baseline()
    cpu, _eta, _a, _d = run_daily_cycle(profile, config)
    jax_yield = _jax_daily_yield(config, profile, complex_mode=False)
    assert jax_yield == pytest.approx(cpu, rel=SIMPLE_PARITY_TOL)


def test_condenser_ambient_mode_backends_agree() -> None:
    """T_cond == T_amb must change the answer identically on both backends, not just
    parse without error on one of them."""
    profile = baseline_profile()
    ode_config = SystemConfig.baseline()
    ambient_config = SystemConfig.baseline(condenser_tracks_ambient=True)

    cpu_ode, _eta, _a, ode_des = run_daily_cycle(profile, ode_config)
    cpu_ambient, _eta, _a, ambient_des = run_daily_cycle(profile, ambient_config)
    assert cpu_ambient != pytest.approx(cpu_ode, rel=SIMPLE_PARITY_TOL)
    # baseline_profile() holds ambient temperature constant, so pinning is verifiable
    # without re-deriving the ODE's time-to-index mapping.
    assert np.all(ambient_des.t_cond_c == profile.desorption.temperature_c[0])

    jax_ambient = _jax_daily_yield(ambient_config, profile, complex_mode=False)
    assert jax_ambient == pytest.approx(cpu_ambient, rel=SIMPLE_PARITY_TOL)


def test_instant_equilibrium_backends_agree() -> None:
    """The g -> infinity ideal case must idealize by the same factor on both backends."""
    profile = baseline_profile()
    finite_config = SystemConfig.baseline()
    ideal_config = SystemConfig.baseline(instant_equilibrium=True)

    cpu_finite, _eta, _a, _d = run_daily_cycle(profile, finite_config)
    cpu_ideal, _eta, _a, _d = run_daily_cycle(profile, ideal_config)
    assert cpu_ideal > cpu_finite

    jax_ideal = _jax_daily_yield(ideal_config, profile, complex_mode=False)
    assert jax_ideal == pytest.approx(cpu_ideal, rel=SIMPLE_PARITY_TOL)

    # Stacked with the ideal condenser the gel runs all the way down onto the hard c_w
    # floor, where dc_w/dt steps to zero -- the one point in this model where LSODA and
    # Tsit5 genuinely disagree. Measured 8e-4, so it gets the complex-mode tolerance.
    both_config = SystemConfig.baseline(
        instant_equilibrium=True, condenser_tracks_ambient=True
    )
    cpu_both, _eta, _a, both_des = run_daily_cycle(profile, both_config)
    assert float(both_des.c_w[-1]) == pytest.approx(
        both_config.mass_params().c_w_min_mol_m3, rel=1e-3
    )
    jax_both = _jax_daily_yield(both_config, profile, complex_mode=False)
    assert jax_both == pytest.approx(cpu_both, rel=COMPLEX_PARITY_TOL)


@pytest.mark.parametrize("name", sorted(COMPLEX_CASES))
def test_complex_mode_backends_agree(name: str) -> None:
    profile = baseline_profile()
    config = SystemConfig.baseline(complex=COMPLEX_CASES[name])
    cpu, _eta, _a, des = run_daily_cycle(profile, config)
    jax_yield = _jax_daily_yield(config, profile, complex_mode=True)
    # "cpu > 0" passed happily on a trajectory that had run T_cond to 1799 C and c_w
    # to -131000; what makes the comparison meaningful is that the reference actually
    # conserved water. _integrate_desorption raises on that now, so this is the
    # backstop for the clipped public trajectory rather than the primary guard.
    inventory_drop = (
        float(des.c_w[0] - des.c_w[-1]) * config.hydrogel_thickness_m * WATER_MOLAR_MASS_KG_MOL
    )
    assert cpu == pytest.approx(inventory_drop, rel=0.01), (
        f"{name}: CPU reference lost water -- yield {cpu:.4f} L/m2 vs inventory drop "
        f"{inventory_drop:.4f} L/m2; parity against it would be meaningless"
    )
    assert jax_yield == pytest.approx(cpu, rel=COMPLEX_PARITY_TOL)


def test_forced_cooling_coefficient_has_one_definition() -> None:
    """B4 must not drift between backends: both read ComplexOptions directly."""
    import jax_daily_cycle as jdc
    from solar_lumped.physics import H_AMB_W_M2_K

    passive = ComplexOptions()
    assert passive.condenser_h_amb_w_m2_k() is None
    assert jdc._h_amb_cond_for(passive) == pytest.approx(H_AMB_W_M2_K)

    forced = ComplexOptions(condenser_air_speed_m_s=1.5)
    assert jdc._h_amb_cond_for(forced) == pytest.approx(forced.condenser_h_amb_w_m2_k())
    # Fans are a floor on ambient convection, never a downgrade from it.
    assert forced.condenser_h_amb_w_m2_k() >= H_AMB_W_M2_K
