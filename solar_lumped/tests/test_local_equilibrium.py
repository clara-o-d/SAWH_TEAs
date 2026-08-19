"""The g -> infinity ideal case is imposed as an equilibrium constraint, not as a large g.

Both are meant to be the same limit, so the constraint route has to reproduce what the
penalty route converges to -- that is what this file pins. The penalty is the older,
independently-derived implementation (Eq. 5's rate law with g scaled until the residual
driving force is negligible), which makes it a real reference rather than a restatement.

Why it matters that the constraint route exists at all: the penalty makes dc_w/dt stiff
in proportion to the scale factor, so the JAX backend's explicit solver needed >16,384
steps per day and silently truncated. The constraint has no fast timescale to resolve.
"""

from __future__ import annotations

import numpy as np
import pytest

from solar_lumped import physics, simulation
from solar_lumped.physics import equilibrium_t_gel_desorption_c
from solar_lumped.simulation import SystemConfig, run_daily_cycle
from solar_lumped.weather import baseline_profile

# The penalty answer still carries its own residual: 1e5 sits 0.12% from 1e6, so 1e6 is
# itself ~0.01-0.1% from the true limit. 0.5% pins the physics while leaving the
# reference's own convergence error room.
PENALTY_PIN_TOL = 5e-3


def _penalty_yield(profile, config, monkeypatch, scale: float) -> float:
    """Daily yield via the legacy stiff route, at a chosen g scale factor."""
    monkeypatch.setattr(simulation, "_INSTANT_EQUILIBRIUM_USE_CONSTRAINT", False)
    monkeypatch.setattr(physics, "_INSTANT_EQUILIBRIUM_G_SCALE", scale)
    water, _eta, _abs_res, _des_res = run_daily_cycle(profile, config)
    monkeypatch.undo()
    return float(water)


@pytest.mark.parametrize("condenser_ambient", [False, True])
def test_constraint_reproduces_the_g_1e6_penalty_limit(monkeypatch, condenser_ambient) -> None:
    profile = baseline_profile()
    config = SystemConfig.baseline(
        instant_equilibrium=True, condenser_tracks_ambient=condenser_ambient
    )
    penalty = _penalty_yield(profile, config, monkeypatch, 1e6)
    constraint, _eta, _a, _d = run_daily_cycle(profile, config)
    assert penalty > 0.0
    assert float(constraint) == pytest.approx(penalty, rel=PENALTY_PIN_TOL)


def test_the_penalty_scale_stops_mattering_under_the_constraint(monkeypatch) -> None:
    """Every penalty scale from 1e4 up brackets the constraint answer within the ODE's own
    tolerance. That is the useful statement: the constraint route removes a knob whose
    residual was of the same order as the solver noise, so there is nothing left to tune.

    Deliberately NOT asserted: that the constraint sits closer to 1e6 than to 1e4. The
    whole 1e4-1e6 spread is ~0.02% here, i.e. below rtol=1e-4, so any ordering inside it
    is noise and a test asserting one would fail at random.
    """
    profile = baseline_profile()
    config = SystemConfig.baseline(instant_equilibrium=True)
    constraint, _eta, _a, _d = run_daily_cycle(profile, config)
    for scale in (1e4, 1e5, 1e6):
        penalty = _penalty_yield(profile, config, monkeypatch, scale)
        assert float(constraint) == pytest.approx(penalty, rel=PENALTY_PIN_TOL), scale


def test_desorption_ends_on_the_isotherm() -> None:
    """The defining property of the constraint route: the gel's surface vapour pressure
    tracks the condenser's, so T_gel equals the isotherm temperature at that loading."""
    profile = baseline_profile()
    config = SystemConfig.baseline(instant_equilibrium=True)
    _water, _eta, _abs_res, des = run_daily_cycle(profile, config)
    mass = config.mass_params()
    floor = mass.c_w_min_mol_m3
    checked = 0
    for c_w, h_m, t_gel, t_cond in zip(des.c_w, des.H, des.t_gel_c, des.t_cond_c):
        if float(c_w) <= floor * 1.001:
            continue  # at the hydrate floor the constraint is inactive by design
        t_eq = equilibrium_t_gel_desorption_c(
            float(c_w), t_cond_c=float(t_cond), params=mass, h_m=float(h_m)
        )
        if not np.isfinite(t_eq):
            continue
        # Only where desorption is actually running: with no energy the gel sits BELOW
        # equilibrium and m_des is zero, the complementary branch.
        if float(t_gel) > t_eq + 1.0:
            continue
        assert float(t_gel) == pytest.approx(t_eq, abs=0.5)
        checked += 1
    assert checked > 0, "no desorbing samples found to check the constraint on"


def test_no_step_cap_pathology_in_the_equilibrium_temperature() -> None:
    """T_eq is monotone in loading -- drier gel needs a hotter surface to reach the same
    condenser pressure. A non-monotone T_eq would mean the bracketed root can jump."""
    config = SystemConfig.baseline(instant_equilibrium=True)
    mass = config.mass_params()
    loadings = np.linspace(mass.c_w_min_mol_m3 * 1.05, mass.c_w_max_mol_m3 * 0.95, 12)
    t_eqs = [
        equilibrium_t_gel_desorption_c(
            float(c), t_cond_c=30.0, params=mass, h_m=config.hydrogel_thickness_m
        )
        for c in loadings
    ]
    assert all(np.isfinite(t_eqs))
    assert all(a > b for a, b in zip(t_eqs, t_eqs[1:])), t_eqs


def test_equilibrium_temperature_is_found_for_libr() -> None:
    """LiBr's brine activity correlation is undefined above ~200 C, and
    _mass_transfer_driving_force maps a non-finite a_w to 0.0 -- so a two-endpoint bracket
    test on [T_cond, T_cond+200] saw fa*fb == 0 and reported "no root", which the callers
    read as "cannot desorb". Every LiBr desorption step took that branch and a whole day's
    yield came out as exactly 0.0, with nothing raised.
    """
    config = SystemConfig.baseline(salt_name="LiBr", instant_equilibrium=True)
    mass = config.mass_params()
    # A loading LiBr actually reaches after a humid night (~13% of the way up its range).
    c_w = mass.c_w_min_mol_m3 + 0.13 * (mass.c_w_max_mol_m3 - mass.c_w_min_mol_m3)
    t_eq = equilibrium_t_gel_desorption_c(
        c_w, t_cond_c=20.0, params=mass, h_m=config.hydrogel_thickness_m
    )
    assert np.isfinite(t_eq), "no equilibrium temperature found for LiBr"
    assert 20.0 < t_eq < 60.0, t_eq


def test_libr_instant_equilibrium_produces_yield() -> None:
    """The end-to-end version of the above: the zero was only visible as a yield."""
    profile = baseline_profile()
    ideal = SystemConfig.baseline(salt_name="LiBr", instant_equilibrium=True)
    finite = SystemConfig.baseline(salt_name="LiBr")
    water_ideal, _eta, _a, _d = run_daily_cycle(profile, ideal)
    water_finite, _eta, _a, _d = run_daily_cycle(profile, finite)
    assert float(water_ideal) > 0.0
    # Instant kinetics is an upper bound on the real thing, for LiBr as for LiCl.
    assert float(water_ideal) > float(water_finite)
