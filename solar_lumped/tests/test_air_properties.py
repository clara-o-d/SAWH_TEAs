"""Air transport properties against their published sources.

Both correlations replaced fixed workbook constants, so nothing else in the suite would
notice if a coefficient were mistyped -- the yields would just be quietly wrong. These
check them against the papers' own numbers instead of against previous outputs:

- Tsilingiris (2008) humid air, cross-checked against that paper's Table 4 saturated
  mixture fits, which are an independent route to the same properties;
- Marrero & Mason (1972) H2O-air diffusivity, checked against its Table 13 row.

Two unit captions in Tsilingiris are wrong (see physics.py) and the wrong reading is
catastrophic rather than subtle, so each is pinned by its own test.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from solar_lumped.physics import (
    P_ATM_SEA_LEVEL_PA,
    d_h2o_air_m2_s,
    h_amb_density_factor,
    humid_air_props,
    pressure_from_elevation_m,
    vapor_gap_water_mole_fraction,
    water_vapor_pressure_pa,
)


def _poly(coeffs: tuple[float, ...], x: float) -> float:
    return sum(c * x**i for i, c in enumerate(coeffs))


# Tsilingiris Table 4: independent polynomial fits of the SATURATED (RH = 100%) mixture,
# t in degC. SK2 carries the .tex's corrected exponent (the printed -1.788037411E-2 drives
# k to -178 W/m K); even corrected, the k/alpha/Pr rows drift above ~65 C, which is why
# the comparison below stops there and physics.py uses Eq. (29) rather than these.
_SD = (1.293393662, -5.538444326e-3, 3.860201577e-5, -5.2536065e-7)
_SV = (1.715747771e-5, 4.722402075e-8, -3.663027156e-10, 1.873236686e-12, -8.050218737e-14)
_SK = (2.40073953e-2, 7.278410162e-5, -1.788037411e-7, -1.351703529e-9, -3.322412767e-11)
_SA = (1.847185729e-5, 1.161914598e-7, 2.373056947e-10, -5.769352751e-12, -6.369279936e-14)


def _saturated_x_v(t_c: float) -> float:
    """RH = 100% at this temperature -- the condition Table 4 was fitted at, which is NOT
    the vapor gap's condition (see test_vapor_fraction_is_pinned_by_the_condenser)."""
    return water_vapor_pressure_pa(t_c) / P_ATM_SEA_LEVEL_PA


@pytest.mark.parametrize("t_c", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
def test_mixture_route_reproduces_tsilingiris_table_4(t_c: float) -> None:
    """Eqs. (12)/(21)/(29)/(34)/(35) vs the paper's own saturated-mixture fits.

    Two independent routes to the same numbers: agreement means the mixing rules, the
    constituent correlations and the unit factors are all right together. Density and
    thermal conductivity pin k directly; alpha folds in cp and rho as well.
    """
    k, nu, alpha = humid_air_props(t_c, x_v=_saturated_x_v(t_c))
    assert k == pytest.approx(_poly(_SK, t_c), rel=0.01)
    assert alpha == pytest.approx(_poly(_SA, t_c), rel=0.01)
    # nu is mu/rho and Table 4 fits them separately, so recombine before comparing.
    assert nu == pytest.approx(_poly(_SV, t_c) / _poly(_SD, t_c), rel=0.01)


def test_water_vapor_viscosity_uses_the_corrected_decade() -> None:
    """Tsilingiris Eq. (41) is captioned Ns/m2 x 10^-6 but needs 10^-7.

    Getting this wrong does not degrade gracefully: at 100 C the printed decade puts the
    saturated mixture viscosity ~10x high (+905% measured). Anchored against Table 4 at
    100 C, where the vapor mole fraction is near unity and mu_v dominates the mixture --
    the one place the decade cannot hide.
    """
    _, nu, _ = humid_air_props(100.0, x_v=_saturated_x_v(100.0))
    expected = _poly(_SV, 100.0) / _poly(_SD, 100.0)
    # 4%, not the 1% used below 60 C: at 100 C Table 4's own rows have drifted (its
    # density is 2% off the mixture route and its k 13%), so this is as tight as an
    # independent comparison can honestly be here. The decade error is a factor of ~10 --
    # the assertion that matters is the bound, and no plausible tolerance hides it.
    assert nu == pytest.approx(expected, rel=0.04)
    assert nu / expected < 2.0, "mu_v is off by a decade -- Eq. (41) needs 1e-7, not 1e-6"


def test_dry_air_conductivity_needs_no_millifactor() -> None:
    """Tsilingiris Eq. (39) is captioned W/m K x 10^-3 but already returns W/m K: the
    paper's own Fig. 3 anchor is 0.0240 at 0 C, which is what it gives unscaled. (The
    anchor is quoted at T = 273 K, so 273.15 K lands a touch above it.) The error this
    guards against is a factor of 1000, so a loose window is the honest one."""
    assert humid_air_props(0.0)[0] == pytest.approx(0.0240, abs=1e-4)


def test_marrero_mason_diffusivity_matches_table_13() -> None:
    """D = 1.87e-10 T^2.072 / P[atm] from Table 13's air-H2O row (10^5 A = 0.187,
    s = 2.072, no Sutherland term). Checked as an absolute value at 25 C -- 0.2505 cm2/s,
    inside the +/-5-10% Table 11 assigns this pair -- and as the exponent, recovered from
    the ratio at two temperatures so a mistyped A cannot mask a mistyped s."""
    assert d_h2o_air_m2_s(25.0) == pytest.approx(2.5054e-5, rel=1e-3)

    lo, hi = 300.0, 400.0
    ratio = d_h2o_air_m2_s(hi - 273.15) / d_h2o_air_m2_s(lo - 273.15)
    assert math.log(ratio) / math.log(hi / lo) == pytest.approx(2.072, rel=1e-6)


def test_diffusivity_clamped_to_the_papers_stated_range() -> None:
    """Marrero & Mason warn specifically against downward extrapolation (their Eqs.
    4.3-1/2 are unusable where London dispersion dominates), and the fit stops at 450 K.
    Solver iterates go far outside both bounds, so the clamp must hold rather than
    silently extrapolating a power law over hundreds of kelvin."""
    assert d_h2o_air_m2_s(-100.0) == d_h2o_air_m2_s(8.85)
    assert d_h2o_air_m2_s(500.0) == d_h2o_air_m2_s(176.85)


def test_vapor_fraction_is_pinned_by_the_condenser() -> None:
    """x_v follows p_sat(T_cond), not saturation at the film temperature.

    The condensing surface sets the gap's vapor partial pressure. Reading RH = 100% at the
    film temperature instead is the easy mistake and it is not small -- it inflates x_v by
    ~6x at a 65 C film over a 30 C condenser, and alpha by 12% rather than 2%.
    """
    x_v = vapor_gap_water_mole_fraction(30.0)
    assert x_v == pytest.approx(0.0419, abs=5e-4)
    assert x_v < _saturated_x_v(65.0) / 5.0

    alpha_dry = humid_air_props(65.0)[2]
    alpha_gap = humid_air_props(65.0, x_v=x_v)[2]
    alpha_wrong = humid_air_props(65.0, x_v=_saturated_x_v(65.0))[2]
    assert 0.015 < 1.0 - alpha_gap / alpha_dry < 0.03
    assert 1.0 - alpha_wrong / alpha_dry > 0.10

    # Saul-Wagner is NaN over ice, so a frozen condenser reads as a dry gap rather than
    # poisoning every property downstream with a NaN.
    assert vapor_gap_water_mole_fraction(-5.0) == 0.0


def test_properties_move_the_right_way_over_the_device_range() -> None:
    """Signs and magnitudes across 25-90 C, the band the gaps actually run in. These are
    what make the change worth having: the retired constants held k at its 63 C value
    while nu and rho sat at ~20 C, so the old Ra mixed two temperatures."""
    props = [humid_air_props(t) for t in (25.0, 45.0, 65.0, 85.0)]
    for i in range(1, len(props)):
        assert props[i][0] > props[i - 1][0]  # k rises
        assert props[i][1] > props[i - 1][1]  # nu rises
        assert props[i][2] > props[i - 1][2]  # alpha rises

    k25, nu25, a25 = props[0]
    k65, nu65, a65 = props[2]
    assert k65 / k25 == pytest.approx(1.11, abs=0.02)
    assert nu65 / nu25 == pytest.approx(1.25, abs=0.03)
    assert a65 / a25 == pytest.approx(1.26, abs=0.03)

    # Pr = nu/alpha stays in air's narrow band -- a coefficient typo in any one of k, mu,
    # cp or rho would show up here even if that property's own trend looked plausible.
    for k, nu, alpha in props:
        assert 0.70 < nu / alpha < 0.72


def test_air_property_clamp_bounds_wild_solver_iterates() -> None:
    """solve_steady_thermal probes deeply inverted (T_gel, T_cond) pairs -- ~16% of calls,
    up to a few hundred kelvin inverted. Property evaluation is clamped into the
    water-vapor fits' 0-120 C range so those iterates cannot return negative k or blow up
    a Rayleigh number, while converged states stay well inside."""
    for t in (-273.0, -50.0, 500.0, 1e4):
        k, nu, alpha = humid_air_props(t)
        assert k > 0.0 and nu > 0.0 and alpha > 0.0
    assert humid_air_props(-50.0) == humid_air_props(0.0)
    assert humid_air_props(500.0) == humid_air_props(120.0)


# --- CPU/JAX parity on the property functions themselves ---------------------------
#
# test_cpu_jax_parity.py compares end-to-end yields, where a property-level divergence
# could hide inside the 0.5% complex-mode tolerance. These compare the functions directly,
# so any drift between the two implementations is caught at its source.

# importorskip at module scope would skip this whole file, the source-checked tests above
# included, in the default env -- jax lives only in solar_lumped/.venv_gpu.
_GPU_SWEEP = Path(__file__).resolve().parents[1] / "gpu_sweep"
if str(_GPU_SWEEP) not in sys.path:
    sys.path.insert(0, str(_GPU_SWEEP))

requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="jax not installed (see .venv_gpu)"
)


@requires_jax
@pytest.mark.parametrize("t_film_c", [5.0, 25.0, 45.0, 65.0, 85.0, 110.0])
@pytest.mark.parametrize("t_cond_c", [10.0, 30.0, 50.0])
def test_backends_agree_on_air_properties(t_film_c: float, t_cond_c: float) -> None:
    import jax_physics as jp

    x_v_cpu = vapor_gap_water_mole_fraction(t_cond_c)
    x_v_jax = float(jp.vapor_gap_water_mole_fraction(t_cond_c))
    assert x_v_jax == pytest.approx(x_v_cpu, rel=1e-12)

    cpu = humid_air_props(t_film_c, x_v=x_v_cpu)
    jax_vals = [float(v) for v in jp.humid_air_props(t_film_c, x_v_jax)]
    for got, want in zip(jax_vals, cpu):
        assert got == pytest.approx(want, rel=1e-12)

    # Dry air: the CPU path early-returns past the mixing rules while JAX runs them with
    # x_v = 0 (one traced path, no branch). The two must still land on the same number.
    cpu_dry = humid_air_props(t_film_c)
    jax_dry = [float(v) for v in jp.humid_air_props(t_film_c)]
    for got, want in zip(jax_dry, cpu_dry):
        assert got == pytest.approx(want, rel=1e-12)

    assert float(jp.d_h2o_air_m2_s(t_film_c)) == pytest.approx(d_h2o_air_m2_s(t_film_c), rel=1e-12)


@requires_jax
def test_backends_agree_on_gap_conductance_and_mass_transfer() -> None:
    """The three consumers, at a representative desorption point and at an inverted
    iterate (condenser hotter), which takes the other Nu branch."""
    import jax_physics as jp

    from solar_lumped.physics import (
        mass_transfer_g_from_h_conv_m_s,
        rayleigh_vapor_gap,
        vapor_gap_h_conv_w_m2_k,
    )

    for gap, t_gel, t_cond in ((0.04, 76.0, 28.8), (0.04, 20.0, 40.0), (0.02, 95.0, 45.0)):
        x_v = vapor_gap_water_mole_fraction(t_cond)
        ra_cpu = rayleigh_vapor_gap(gap, t_gel, t_cond, x_v=x_v)
        assert float(jp._rayleigh_vapor_gap(gap, t_gel, t_cond, x_v)) == pytest.approx(
            ra_cpu, rel=1e-12
        )

        h_cpu = vapor_gap_h_conv_w_m2_k(gap, t_gel, t_cond, tilt_deg=30.0)
        h_jax = float(jp.vapor_gap_h_conv_w_m2_k(gap, t_gel, t_cond, tilt_deg=30.0))
        assert h_jax == pytest.approx(h_cpu, rel=1e-12)

        g_cpu = mass_transfer_g_from_h_conv_m_s(h_cpu, t_gel_c=t_gel, t_cond_c=t_cond)
        assert float(jp.mass_transfer_g_from_h_conv_m_s(h_cpu, t_gel, t_cond)) == pytest.approx(
            g_cpu, rel=1e-12
        )

    # Glazing gaps: each evaluated at its own film temperature, so the hotter
    # absorber-glass cavity must NOT get the same k as the cooler pane-pane cavity.
    q_ag = float(jp.conduction_air_gap_w_m2(105.0, 45.0, 0.02))
    q_io = float(jp.conduction_air_gap_w_m2(45.0, 30.0, 0.02))
    k_ag = q_ag * 0.02 / (105.0 - 45.0)
    k_io = q_io * 0.02 / (45.0 - 30.0)
    assert k_ag > k_io
    assert k_ag == pytest.approx(humid_air_props(75.0)[0], rel=1e-12)
    assert k_io == pytest.approx(humid_air_props(37.5)[0], rel=1e-12)


# --- Site elevation --------------------------------------------------------------------


def test_sea_level_is_the_exact_identity() -> None:
    """The whole elevation mechanism must be a no-op at 0 m, or every previously published
    sea-level result silently moves. Exact equality, not approx."""
    assert pressure_from_elevation_m(0.0) == P_ATM_SEA_LEVEL_PA
    assert h_amb_density_factor(25.0) == 1.0
    assert humid_air_props(65.0, x_v=0.03) == humid_air_props(
        65.0, x_v=0.03, p_atm_pa=P_ATM_SEA_LEVEL_PA
    )


def test_isa_pressure_matches_known_elevations() -> None:
    """ISA troposphere branch against standard-atmosphere table values."""
    for elev, kpa in ((1000.0, 89.87), (2400.0, 75.63), (5000.0, 54.02)):
        assert pressure_from_elevation_m(elev) / 1000.0 == pytest.approx(kpa, abs=0.05)
    # Below sea level clamps rather than extrapolating (Dead Sea, Turfan Depression).
    assert pressure_from_elevation_m(-400.0) == pressure_from_elevation_m(0.0)


def test_elevation_pushes_the_gap_properties_the_right_way() -> None:
    """D_air ~ 1/p (up), x_v ~ 1/p (up, so a wetter gap), rho ~ p (down, so nu and alpha up
    and Ra ~ p^2 down). These are the competing effects that make the net yield change a
    balance rather than a straight gain."""
    p_hi, p_lo = pressure_from_elevation_m(0.0), pressure_from_elevation_m(2400.0)

    assert d_h2o_air_m2_s(65.0, p_atm_pa=p_lo) / d_h2o_air_m2_s(65.0, p_atm_pa=p_hi) == (
        pytest.approx(p_hi / p_lo, rel=1e-12)
    )
    x_hi = vapor_gap_water_mole_fraction(30.0, p_atm_pa=p_hi)
    x_lo = vapor_gap_water_mole_fraction(30.0, p_atm_pa=p_lo)
    assert x_lo > x_hi

    _, nu_hi, al_hi = humid_air_props(65.0, x_v=x_hi, p_atm_pa=p_hi)
    _, nu_lo, al_lo = humid_air_props(65.0, x_v=x_lo, p_atm_pa=p_lo)
    assert nu_lo > nu_hi and al_lo > al_hi
    # Ra ~ 1/(nu*alpha), so thinner air suppresses it -- weaker gap convection.
    assert (nu_lo * al_lo) / (nu_hi * al_hi) == pytest.approx((p_hi / p_lo) ** 2, rel=0.02)


def test_h_amb_density_factor_is_the_offsetting_penalty() -> None:
    """h_amb ~ rho^0.5, so altitude cuts convective cooling: 14% at 2400 m, 27% at 5000 m.
    Without this the elevation gain in D_air would be one-sided."""
    assert h_amb_density_factor(25.0, p_atm_pa=pressure_from_elevation_m(2400.0)) == (
        pytest.approx(0.864, abs=0.005)
    )
    assert h_amb_density_factor(25.0, p_atm_pa=pressure_from_elevation_m(5000.0)) == (
        pytest.approx(0.730, abs=0.005)
    )
    # Density, not bare pressure: a hot day is thin air too, at any elevation -- and a
    # cold one is denser than the 25 C reference, so it convects BETTER (factor > 1).
    assert h_amb_density_factor(45.0) < 1.0 < h_amb_density_factor(5.0)
    # The exponent is a knob, and n = 0 recovers the previous no-scaling behaviour.
    p = pressure_from_elevation_m(3000.0)
    assert h_amb_density_factor(25.0, p_atm_pa=p, exponent=0.0) == 1.0
    assert h_amb_density_factor(25.0, p_atm_pa=p, exponent=0.8) < h_amb_density_factor(
        25.0, p_atm_pa=p, exponent=0.5
    )


def test_config_elevation_reaches_both_params_objects() -> None:
    """A site pressure that stops at the config would silently leave the gaps at sea level
    -- and the mass side is the one that carries the D_air gain."""
    import dataclasses

    from solar_lumped.simulation import SystemConfig

    cfg = dataclasses.replace(SystemConfig.baseline(), site_elevation_m=2400.0)
    want = pressure_from_elevation_m(2400.0)
    assert cfg.site_pressure_pa() == want
    assert cfg.thermal_params().p_atm_pa == want
    assert cfg.mass_params().p_atm_pa == want
    assert SystemConfig.baseline().thermal_params().p_atm_pa == P_ATM_SEA_LEVEL_PA


@requires_jax
@pytest.mark.parametrize("elevation_m", [0.0, 1200.0, 2400.0, 5000.0])
def test_backends_agree_at_elevation(elevation_m: float) -> None:
    """Pressure has to reach the same places in both backends. A p_atm_pa that reached the
    JAX vapor gap but not its glazing gaps, or vice versa, would show up here."""
    import jax_physics as jp

    p_atm = pressure_from_elevation_m(elevation_m)
    assert jp.P_ATM_SEA_LEVEL_PA == P_ATM_SEA_LEVEL_PA  # same workbook row on both sides

    x_cpu = vapor_gap_water_mole_fraction(30.0, p_atm_pa=p_atm)
    assert float(jp.vapor_gap_water_mole_fraction(30.0, p_atm)) == pytest.approx(x_cpu, rel=1e-12)

    cpu = humid_air_props(65.0, x_v=x_cpu, p_atm_pa=p_atm)
    for got, want in zip([float(v) for v in jp.humid_air_props(65.0, x_cpu, p_atm)], cpu):
        assert got == pytest.approx(want, rel=1e-12)

    assert float(jp.d_h2o_air_m2_s(65.0, p_atm)) == pytest.approx(
        d_h2o_air_m2_s(65.0, p_atm_pa=p_atm), rel=1e-12
    )
    assert float(jp.h_amb_density_factor(30.0, p_atm)) == pytest.approx(
        h_amb_density_factor(30.0, p_atm_pa=p_atm), rel=1e-12
    )


# --- Elevation wiring: weather feed -> config ------------------------------------------


def _frame(elevation_m: float | None):
    """Minimal weather frame, with or without the elevation column."""
    import pandas as pd

    cols = {"temperature_2m": [20.0, 21.0], "latitude": [0.0, 0.0]}
    if elevation_m is not None:
        cols["elevation_m"] = [elevation_m, elevation_m]
    return pd.DataFrame(cols)


def test_site_elevation_read_from_weather_frame() -> None:
    from solar_lumped.weather import site_elevation_m

    assert site_elevation_m(_frame(2400.0)) == 2400.0
    # Frames without the column (hand-built, or from a cache predating it) fall back to
    # sea level rather than raising -- that is the exact no-op, so old paths keep working.
    assert site_elevation_m(_frame(None)) == 0.0
    assert site_elevation_m(_frame(2400.0).iloc[0:0]) == 0.0


def test_openmeteo_response_elevation_lands_on_the_frame() -> None:
    """The column has to come off the response body, since nothing requests it as a
    variable -- if Open-Meteo's key were misread every site would silently be sea level."""
    from solar_lumped.weather import WeatherClient, site_elevation_m

    data = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [10.0, 11.0],
        },
        "utc_offset_seconds": 0,
        "elevation": 2410.0,
    }
    df = WeatherClient._series_to_dataframe(data, "hourly", 0.0, 0.0)
    assert site_elevation_m(df) == 2410.0
    # A response with no elevation key must not crash the fetch.
    del data["elevation"]
    assert site_elevation_m(WeatherClient._series_to_dataframe(data, "hourly", 0.0, 0.0)) == 0.0


def test_sweep_and_cli_builders_carry_elevation() -> None:
    """Both build_system_config functions -- the sweep's and the CLI's -- have to pass it
    through, or the elevation reaches the CSV row and not the physics."""
    from solar_lumped.site_sweep import Combo, build_system_config as sweep_build
    from solar_lumped.system import build_system_config as cli_build

    cfg = sweep_build(
        Combo(hydrogel_thickness_mm=4.0, fin_area_ratio=7.1, vapor_gap_mm=40.0),
        salt="LiCl", salt_loading=4.0, insulation_gap_mm=5.0, tilt_deg=30.0,
        eps_abs=0.95, tau_glass=0.9, site_elevation_m=2400.0,
    )
    assert cfg.site_elevation_m == 2400.0
    assert cfg.thermal_params().p_atm_pa == pressure_from_elevation_m(2400.0)
    assert cli_build(site_elevation_m=1500.0).mass_params().p_atm_pa == (
        pressure_from_elevation_m(1500.0)
    )
    # Default stays sea level for every caller that does not pass one.
    assert cli_build().site_elevation_m == 0.0


def test_sweep_csv_schema_records_elevation() -> None:
    """A sweep row without elevation is ambiguous now that elevation changes yield."""
    from solar_lumped.site_sweep import _CSV_COLUMNS

    assert "elevation_m" in _CSV_COLUMNS


# --- Cumulative water as an ODE state -------------------------------------------------


def test_yield_comes_from_the_integrated_state_not_a_trapezoid() -> None:
    """Desorption carries W (cumulative water) as an ODE state, so the reported yield and
    the c_w inventory drop are the same integration rather than two quadratures of it.

    The trapezoid this replaced diverged ~5% from the trajectory on fast desorption and
    tripped the conservation guard on runs that were actually fine. The invariant now holds
    far tighter than the 1% guard, which is what makes it worth asserting directly.
    """
    from solar_lumped.physics import WATER_MOLAR_MASS_KG_MOL
    from solar_lumped.simulation import SystemConfig, run_daily_cycle
    from solar_lumped.weather import baseline_profile

    cfg = SystemConfig.baseline()
    yield_kg, _eta, _abs_res, des = run_daily_cycle(baseline_profile(), cfg)

    assert des.water_cumulative_kg_m2 is not None
    w = des.water_cumulative_kg_m2
    assert w[0] == 0.0
    # Monotone: the gel cannot un-collect water.
    assert all(w[i] >= w[i - 1] - 1e-12 for i in range(1, len(w)))
    assert float(w[-1]) == pytest.approx(yield_kg, rel=1e-12)

    drop = float(des.c_w[0] - des.c_w[-1]) * cfg.mass_params().h0_ref_m * WATER_MOLAR_MASS_KG_MOL
    assert float(w[-1]) == pytest.approx(drop, rel=1e-4)


def test_high_elevation_desorption_integrates_without_tripping_the_guard() -> None:
    """A 4000 m site desorbs fast enough that the retired trapezoid broke the 1%
    conservation tolerance. Thin air is exactly the regime this fix was for, so the
    high-elevation case is the one worth pinning."""
    import dataclasses

    from solar_lumped.simulation import SystemConfig, run_daily_cycle
    from solar_lumped.weather import baseline_profile

    profile = baseline_profile(solar_w_m2=1050.0, temperature_c=8.0, relative_humidity=0.35)
    sea = run_daily_cycle(profile, SystemConfig.baseline())[0]
    high = run_daily_cycle(
        profile, dataclasses.replace(SystemConfig.baseline(), site_elevation_m=4000.0)
    )[0]
    assert high > sea > 0.0
