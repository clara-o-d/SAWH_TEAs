"""Complex-fidelity option set: ZSR salt blends, glazing stacks, selective absorber
coatings, finned/forced condensers, and shifted cycle schedules.

The simple model (everything else in this package, and all of ``gpu_sweep``) is
unchanged and stays the default: every entry point here is reached only when a
``ComplexOptions`` instance is attached to ``SystemConfig.complex``. ``None`` (the
default) reproduces Wilson et al. 2025 exactly, so the JAX fast path -- which is
LiCl-hardcoded and never sees this module -- remains valid.

Feature map (labels match the design discussion):

* **A1** cycle schedule  -- ``seal_offset_h`` / ``open_offset_h`` shift the
  desorption window off the raw GHI day/night split (applied in ``weather.py``).
* **B1** selective absorber -- continuous ``eps_abs_ir`` priced by
  :func:`absorber_coating_cost_usd_per_m2`.
* **B2** glazing stack -- ``n_glazing_panes`` (0/1/2) and ``evacuated_gap``, feeding
  both the Eq. 3/4 residuals and :func:`glazing_cost_usd_per_m2`.
* **B3** finned condenser -- ``fin_thickness_m`` / ``fin_height_m`` give a real fin
  efficiency and a mass-proportional cost, so ``fin_area_ratio`` stops being free.
* **B4** forced condenser cooling -- ``condenser_air_speed_m_s`` priced at Wilson's
  own Table S2 fan + PV line items.
* **B8** ZSR salt blend -- molality-additive Zdanovskii mixing over
  :data:`ZSR_SALTS`, replacing the single-salt water-activity path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from solar_lumped._parameters_xlsx import physics_value as _pv

# =============================================================================
# Option set
# =============================================================================

# LiCl + CaCl2 + MgCl2. NaCl is excluded deliberately: its deliquescence RH is 0.75
# (derived from solubility, see physics.deliquescence_rh), so it never deliquesces at
# any realistic SAWH site and would only waste a simplex dimension.
ZSR_SALTS: tuple[str, ...] = ("LiCl", "CaCl2", "MgCl2")

# Below this weight a salt is treated as absent, so a nominally 3-component blend
# that the optimizer has driven into a corner takes the fast single-salt path.
_SINGLE_SALT_TOL: float = 1e-9

_TAU_GLASS_PER_PANE: float = _pv("Glass transmittance (tau_glass)")
_K_AL_W_M_K: float = 167.0  # Table S3 k_Al, for the fin-efficiency parameter m

# The simple model's base case (Case 2) is already a *selective* absorber at
# eps_abs_ir = 0.05 -- not the black spray paint Wilson's Methods describe. Complex
# mode therefore defaults to the same value, so ComplexOptions() is a true no-op,
# and nets its coating cost against that same baseline. One consequence worth
# knowing: the simple model has been quietly assuming a premium sputtered-cermet
# surface for free, and B1 is what finally puts a price on it.
_EPS_ABS_IR_BASELINE: float = _pv("Absorber IR emissivity (eps_abs_ir)")
_EPS_ABS_IR_PAINT: float = 0.95  # commodity black paint, the cheap end of the curve


@dataclass(frozen=True, slots=True)
class ComplexOptions:
    """Complex-fidelity knobs. Every default reproduces the simple model.

    Attached to ``SystemConfig.complex``; ``None`` there means "run the simple
    model", which is what ``gpu_sweep`` and every existing script continue to do.
    """

    # --- A1: cycle schedule (hours; + delays, - advances) ---
    # Applied against the GHI-derived sunrise/sunset in weather.profile_from_day_df.
    seal_offset_h: float = 0.0
    open_offset_h: float = 0.0

    # --- B1: absorber selective coating ---
    eps_abs_ir: float = _EPS_ABS_IR_BASELINE

    # --- B2: glazing stack ---
    n_glazing_panes: int = 1
    tau_per_pane: float = _TAU_GLASS_PER_PANE
    evacuated_gap: bool = False

    # --- B3: condenser fin geometry (drives efficiency *and* aluminum mass) ---
    fin_thickness_m: float = 1.0e-3
    fin_height_m: float = 0.025

    # --- B4: forced condenser cooling ---
    condenser_air_speed_m_s: float = 0.0

    # --- B8: ZSR blend weights over ZSR_SALTS (molality weights, renormalized) ---
    blend_weights: tuple[float, ...] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.blend_weights) != len(ZSR_SALTS):
            raise ValueError(
                f"blend_weights must have {len(ZSR_SALTS)} entries (one per "
                f"{ZSR_SALTS}), got {len(self.blend_weights)}"
            )
        if self.n_glazing_panes not in (0, 1, 2):
            raise ValueError(f"n_glazing_panes must be 0, 1, or 2, got {self.n_glazing_panes}")

    @property
    def is_blend(self) -> bool:
        """True when more than one salt carries meaningful weight.

        A single-salt "blend" is not run through ZSR at all: the mixing rule
        reduces exactly to that salt's own isotherm (verified in the tests), and
        the closed-form path avoids a root solve nested three deep inside the ODE.
        """
        w = normalized_blend_weights(self.blend_weights)
        return int(np.count_nonzero(w > _SINGLE_SALT_TOL)) > 1

    @property
    def dominant_salt(self) -> str:
        return ZSR_SALTS[int(np.argmax(normalized_blend_weights(self.blend_weights)))]

    @property
    def tau_glass(self) -> float:
        """Stack transmittance: each pane multiplies, zero panes is a bare absorber."""
        return float(self.tau_per_pane ** self.n_glazing_panes)

    @property
    def has_glass(self) -> bool:
        return self.n_glazing_panes > 0

    def condenser_h_amb_w_m2_k(self) -> float | None:
        """B4 condenser-side convection coefficient from this design's fan speed.

        A floor, not a replacement: Wilson's fans hold ~0.5 m/s regardless of
        ambient wind (Note S2), so a windier site still gets the better of the two.
        ``None`` when passive, meaning "share the absorber's h_amb".

        Both backends call this, so forced cooling cannot drift between them.
        """
        from solar_lumped.physics import H_AMB_W_M2_K, wind_to_h_amb_w_m2_k

        if self.condenser_air_speed_m_s <= 0.0:
            return None
        return float(max(wind_to_h_amb_w_m2_k(self.condenser_air_speed_m_s), H_AMB_W_M2_K))


def normalized_blend_weights(weights: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Clip negatives and sum-normalize onto the simplex."""
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(w.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("blend_weights must be non-negative and not all zero")
    return w / total


# =============================================================================
# B8 -- Zdanovskii-Stokes-Robinson (ZSR) mixing rule
# =============================================================================
#
# At a fixed water activity, each salt contributes its pure-binary molality scaled
# by its blend weight:
#
#     m_i(a_w) = w_i * m_i,binary(a_w, T)
#
# so the mixture is molality-additive at constant a_w. Ported from
# electrolyte_optimization/src/optimization/zsr_mixing.py::zsr_brine_state, but
# resting on this package's own isotherms (physics.equilibrate_salt_mf) rather
# than that repo's duplicate brine_equilibrium module.


# CaCl2 and MgCl2 invert their isotherms with their own bracketed root solves, so
# evaluating ZSR pointwise puts a root solve inside a root solve inside the ODE
# right-hand side -- measured at ~54 s per simulated day, i.e. hours per BayesOpt
# evaluation. Instead each salt's molality curve is tabulated once per temperature
# bucket and interpolated.
# ponytail: linear interp on a 512-point grid, ~1e-6 relative on a smooth monotone
# curve; swap in a spline only if the blend optimum ever looks grid-sensitive.
_MOLALITY_GRID_POINTS: int = 512
_T_BUCKET_C: float = 0.5  # temperature quantization for the tabulated curve


@lru_cache(maxsize=4096)
def _binary_molality_curve(salt_name: str, t_bucket_c: float) -> tuple[np.ndarray, np.ndarray]:
    """Tabulated (a_w, molality) for a pure binary brine over its validated window."""
    from solar_lumped.physics import equilibrate_salt_mf, get_salt

    rec = get_salt(salt_name)
    aw = np.linspace(rec.rh_min, rec.rh_max, _MOLALITY_GRID_POINTS)
    mw = rec.formula_weight_g_mol
    m = np.empty_like(aw)
    for i, a in enumerate(aw):
        f_b = equilibrate_salt_mf(salt_name, float(a), t_bucket_c)
        m[i] = (
            1000.0 * f_b / (mw * (1.0 - f_b))
            if (math.isfinite(f_b) and 0.0 < f_b < 1.0)
            else np.nan
        )
    ok = np.isfinite(m)
    return aw[ok], m[ok]


def _binary_molality_at_aw(salt_name: str, water_activity: float, temperature_c: float) -> float:
    """Molality (mol salt / kg water) of a pure binary brine at this water activity.

    Below the salt's deliquescence point the molality is held at its saturation
    value rather than dropped or extrapolated: the solution is saturated with
    respect to that salt and the excess has precipitated, so its *dissolved*
    concentration stops changing. Two things depend on this being a plateau and not
    a step -- f_b(a_w) stays continuous and monotone, which is what licenses the
    tabulated inverse, and a trace component can no longer abolish (or
    discontinuously join) an otherwise strong brine.

    Above rh_max the fit has no support at all, so that end really is NaN.
    """
    aw_grid, m_grid = _binary_molality_curve(
        salt_name, round(float(temperature_c) / _T_BUCKET_C) * _T_BUCKET_C
    )
    a = float(water_activity)
    if aw_grid.size == 0 or a > aw_grid[-1]:
        return float("nan")
    if a < aw_grid[0]:
        return float(m_grid[0])  # saturated: excess salt is solid
    return float(np.interp(a, aw_grid, m_grid))


def zsr_brine_state(
    blend_weights: tuple[float, ...] | np.ndarray,
    water_activity: float,
    temperature_c: float,
) -> tuple[float, float, float]:
    """Mixed-brine state at a given water activity.

    Returns ``(brine_salt_mass_fraction, effective_ions_per_formula,
    effective_formula_weight_g_mol)``, or three NaNs when no component can hold a
    brine at this water activity.

    Below a component's own deliquescence point that component **crystallizes out**
    and simply stops contributing to the liquid phase; the remaining salts carry
    the brine on. Failing the whole mixture instead (the obvious reading of
    ``equilibrate_salt_mf`` returning NaN) is wrong, and badly so: a trace of CaCl2
    would abolish an otherwise pure LiCl brine below 31% RH, putting a cliff at the
    pure-LiCl corner that scales with nothing physical.

    ponytail: the precipitated solid is treated as inert -- it stops binding water,
    but its effect on the gel's mechanics and on re-dissolution kinetics is not
    modeled. Fine for equilibrium water activity, which is all Eq. 5 needs.
    """
    w = normalized_blend_weights(blend_weights)
    from solar_lumped.physics import get_salt

    molality = np.zeros(len(ZSR_SALTS), dtype=float)
    formula_weight = np.zeros(len(ZSR_SALTS), dtype=float)
    ions = np.zeros(len(ZSR_SALTS), dtype=float)
    for i, name in enumerate(ZSR_SALTS):
        rec = get_salt(name)
        formula_weight[i] = rec.formula_weight_g_mol
        ions[i] = float(rec.ions_per_formula)
        if w[i] <= 1e-15:
            continue
        m_ref = _binary_molality_at_aw(name, water_activity, temperature_c)
        if not math.isfinite(m_ref):
            continue  # crystallized at this a_w; contributes no dissolved salt
        molality[i] = w[i] * m_ref

    total_molality = float(molality.sum())
    if total_molality <= 0.0 or not math.isfinite(total_molality):
        return float("nan"), float("nan"), float("nan")

    dissolved_g_per_kg_water = float(np.sum(molality * formula_weight))
    f_b = dissolved_g_per_kg_water / (dissolved_g_per_kg_water + 1000.0)
    eff_ions = float(np.sum(molality * ions) / total_molality)
    eff_mw = dissolved_g_per_kg_water / total_molality
    if not (math.isfinite(f_b) and 0.0 < f_b < 1.0 and math.isfinite(eff_ions) and math.isfinite(eff_mw)):
        return float("nan"), float("nan"), float("nan")
    return float(f_b), float(eff_ions), float(eff_mw)


# Inset from the window edges: the isotherm fits are evaluated *at* rh_min/rh_max,
# so stepping a hair inside keeps both bracket endpoints finite.
_AW_WINDOW_INSET: float = 1e-4


def blend_water_activity_window(
    blend_weights: tuple[float, ...] | np.ndarray,
) -> tuple[float, float]:
    """Water-activity range over which the blend holds *some* liquid brine.

    The lower edge is the most hygroscopic member's deliquescence point, not the
    least: below a given salt's DRH that salt precipitates (see
    :func:`zsr_brine_state`) while the stronger ones keep a brine. So a blend
    reaches as dry as its best component, with progressively less dissolved salt.
    The upper edge is the *smallest* rh_max, past which the isotherm fits stop.
    """
    from solar_lumped.physics import get_salt

    w = normalized_blend_weights(blend_weights)
    active = [get_salt(n) for n, wi in zip(ZSR_SALTS, w) if wi > 1e-15]
    lo = min(rec.rh_min for rec in active) + _AW_WINDOW_INSET
    hi = min(rec.rh_max for rec in active) - _AW_WINDOW_INSET
    return float(lo), float(hi)


def zsr_water_activity_at_brine_fraction(
    blend_weights: tuple[float, ...] | np.ndarray,
    brine_salt_fraction: float,
    temperature_c: float,
) -> float:
    """Invert :func:`zsr_brine_state`: water activity that yields this salt fraction.

    ZSR is naturally posed as a_w -> composition, but the ODE carries water
    concentration and needs composition -> a_w. Rather than root-solve that
    inversion on every right-hand-side call, the blend's f_b(a_w) curve is
    tabulated once per (blend, temperature bucket) and inverted by interpolation:
    f_b decreases monotonically in a_w (every binary molality does, so their ZSR
    sum does), so the reversed table is a valid interpolant.

    Returns NaN when the target fraction lies outside the blend's window, which
    callers treat as an infeasible operating point.
    """
    f_target = float(brine_salt_fraction)
    if not (math.isfinite(f_target) and 0.0 < f_target < 1.0):
        return float("nan")
    aw_grid, fb_grid = _blend_fb_curve(
        tuple(float(w) for w in normalized_blend_weights(blend_weights)),
        round(float(temperature_c) / _T_BUCKET_C) * _T_BUCKET_C,
    )
    if aw_grid.size < 2:
        return float("nan")
    # fb decreases with aw, so reverse both for np.interp's ascending-x contract.
    # np.interp edge-clamps outside the range, which is what we want and NOT a
    # convenience: past the concentrated end the brine is saturated (see the plateau
    # in _binary_molality_at_aw) and its activity stops falling, so the edge value is
    # the physical answer. Returning NaN there instead would halt desorption at the
    # very point the gel is driest -- and would disagree with the JAX backend, which
    # reads the same inversion off an edge-clamped table.
    return float(np.interp(f_target, fb_grid[::-1], aw_grid[::-1]))


@lru_cache(maxsize=4096)
def _blend_fb_curve(
    blend_weights: tuple[float, ...], t_bucket_c: float
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulated (a_w, brine salt fraction) for one blend over its validated window.

    Cached on the exact weight tuple, which is constant for the whole of any single
    ODE integration -- so one design pays for one table.
    """
    lo, hi = blend_water_activity_window(blend_weights)
    if not (hi > lo):
        return np.empty(0), np.empty(0)
    aw = np.linspace(lo, hi, _MOLALITY_GRID_POINTS)
    fb = np.array(
        [zsr_brine_state(blend_weights, float(a), t_bucket_c)[0] for a in aw], dtype=float
    )
    ok = np.isfinite(fb)
    return aw[ok], fb[ok]


# --- Host-side (T, f_b) -> a_w table, shared by the CPU and JAX backends ---
#
# The JAX path cannot root-solve inside a jitted ODE right-hand side, but it does
# not have to: blend weights are constant for a whole instance, so the inversion is
# tabulated once on the host in numpy and becomes a bilinear lookup on device. The
# CPU path uses the same builder, so the two backends are inverting *the same*
# numbers rather than two implementations that have to be kept in agreement.

ZSR_TABLE_T_GRID_C: tuple[float, float, int] = (0.0, 100.0, 41)  # lo, hi, count
ZSR_TABLE_FB_POINTS: int = 256


@lru_cache(maxsize=1024)
def zsr_inverse_table(
    blend_weights: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tabulate a_w over a regular (temperature, brine salt fraction) grid.

    Returns ``(t_grid_c, fb_grid, aw_table)`` with ``aw_table[i, j]`` the water
    activity at ``t_grid_c[i]`` and ``fb_grid[j]``. ``fb_grid`` is a single common
    axis across temperatures -- each temperature's native f_b range is resampled
    onto it -- so device-side lookup is a plain bilinear interpolation instead of a
    ragged search. Entries outside a temperature's reachable f_b range are filled
    by edge-clamping, matching what the CPU path's clamped inversion returns.
    """
    t_lo, t_hi, n_t = ZSR_TABLE_T_GRID_C
    t_grid = np.linspace(t_lo, t_hi, n_t)
    aw_lo, aw_hi = blend_water_activity_window(blend_weights)

    # Common f_b axis spanning every temperature's reachable range.
    fb_bounds = []
    per_t_curves = []
    for t in t_grid:
        aw = np.linspace(aw_lo, aw_hi, ZSR_TABLE_FB_POINTS)
        fb = np.array([zsr_brine_state(blend_weights, float(a), float(t))[0] for a in aw])
        ok = np.isfinite(fb)
        per_t_curves.append((aw[ok], fb[ok]))
        if ok.any():
            fb_bounds.append((fb[ok].min(), fb[ok].max()))
    if not fb_bounds:
        raise ValueError(f"blend {blend_weights} has no brine state anywhere on the T grid")
    fb_grid = np.linspace(
        min(b[0] for b in fb_bounds), max(b[1] for b in fb_bounds), ZSR_TABLE_FB_POINTS
    )

    aw_table = np.empty((n_t, ZSR_TABLE_FB_POINTS), dtype=float)
    for i, (aw_i, fb_i) in enumerate(per_t_curves):
        if aw_i.size < 2:
            aw_table[i, :] = np.nan
            continue
        # f_b decreases with a_w, so reverse for np.interp's ascending-x contract;
        # np.interp edge-clamps outside the range, which is the behaviour we want.
        aw_table[i, :] = np.interp(fb_grid, fb_i[::-1], aw_i[::-1])
    # Any all-NaN temperature row (no brine at all) inherits its nearest valid row.
    bad = ~np.isfinite(aw_table).all(axis=1)
    if bad.any():
        good_idx = np.flatnonzero(~bad)
        for i in np.flatnonzero(bad):
            aw_table[i, :] = aw_table[good_idx[np.argmin(np.abs(good_idx - i))], :]
    return t_grid, fb_grid, aw_table


def blend_fully_dissolved_window(
    blend_weights: tuple[float, ...] | np.ndarray,
) -> tuple[float, float]:
    """Water-activity range over which *every* member is still in solution.

    Distinct from :func:`blend_water_activity_window`, which is the wider range
    where *some* brine survives. The as-built gel is cast with all its salt
    dissolved, so this narrower window -- the intersection of the members' ranges --
    is the right basis for fabrication-referenced quantities (salt inventory,
    effective formula weight, blend price).
    """
    from solar_lumped.physics import get_salt

    w = normalized_blend_weights(blend_weights)
    active = [get_salt(n) for n, wi in zip(ZSR_SALTS, w) if wi > 1e-15]
    lo = max(rec.rh_min for rec in active) + _AW_WINDOW_INSET
    hi = min(rec.rh_max for rec in active) - _AW_WINDOW_INSET
    return float(lo), float(hi)


def clamp_reference_rh(
    blend_weights: tuple[float, ...] | np.ndarray, reference_rh: float
) -> float:
    """Pull a fabrication reference RH into the blend's fully-dissolved window.

    Wilson casts his PAM-LiCl at 20% RH, but CaCl2 (DRH 0.35) and MgCl2 (0.34)
    have no brine there -- a gel loaded with them simply could not be cast in
    equilibrium with 20% RH air, it would be cast wetter. Clamping to the
    fully-dissolved window says exactly that. Using the wider operating window here
    instead would reference the inventory to a state where most of the salt has
    already crystallized, which silently mis-sizes c_s and the initial loading.
    """
    lo, hi = blend_fully_dissolved_window(blend_weights)
    return float(min(max(float(reference_rh), lo), hi))


def zsr_effective_formula_weight_g_mol(
    blend_weights: tuple[float, ...] | np.ndarray,
    *,
    reference_rh: float,
    temperature_c: float = 25.0,
) -> float:
    """Blend formula weight for the fixed composite salt inventory (c_s).

    The gel's salt inventory is fixed at fabrication, but ZSR's effective formula
    weight drifts with water activity (the molality mix re-proportions). One number
    is needed for ``salt_molarity_from_composite``, so it is pinned at the
    fabrication equilibrium RH, matching how the rest of the model treats c_s as a
    constant of the as-built gel.
    """
    rh = clamp_reference_rh(blend_weights, reference_rh)
    _f_b, _ions, eff_mw = zsr_brine_state(blend_weights, rh, temperature_c)
    return float(eff_mw)


def zsr_blend_price_usd_per_kg(
    blend_weights: tuple[float, ...] | np.ndarray,
    *,
    reference_rh: float,
    temperature_c: float = 25.0,
) -> float:
    """Mass-weighted blend salt price (USD/kg of salt actually loaded).

    ZSR weights are *molality* weights, so they are converted to mass shares at the
    fabrication reference state before pricing -- a mole of MgCl2 (95.2 g/mol) is
    not a mole of LiCl (42.4 g/mol), and pricing on molality weights directly (as
    the original electrolyte_optimization port did) understates the heavy salts.
    """
    from solar_lumped.physics import get_salt

    w = normalized_blend_weights(blend_weights)
    rh = clamp_reference_rh(blend_weights, reference_rh)
    mass_share = np.zeros(len(ZSR_SALTS), dtype=float)
    for i, name in enumerate(ZSR_SALTS):
        if w[i] <= 1e-15:
            continue
        m_ref = _binary_molality_at_aw(name, rh, temperature_c)
        if not math.isfinite(m_ref):
            return float("nan")
        mass_share[i] = w[i] * m_ref * get_salt(name).formula_weight_g_mol

    total = float(mass_share.sum())
    if total <= 0.0 or not math.isfinite(total):
        return float("nan")
    prices = np.array([get_salt(n).price_usd_per_kg for n in ZSR_SALTS], dtype=float)
    return float(np.sum(mass_share * prices) / total)


# =============================================================================
# B3 -- fin efficiency (physics side; the mass cost lives in economics.py)
# =============================================================================


def fin_efficiency(
    h_amb_w_m2_k: float,
    *,
    fin_thickness_m: float,
    fin_height_m: float,
    k_fin_w_m_k: float = _K_AL_W_M_K,
) -> float:
    """Straight-fin efficiency eta_f = tanh(mL)/(mL), m = sqrt(2h / (k t)).

    Wilson assumes an ideally efficient fin (h_cond = A_r * h_amb), which makes
    added fin area free in both physics and cost. This is the physics half of the
    correction: past roughly A_r ~ 8 on natural convection, extra area stops
    paying for itself.
    """
    h = max(float(h_amb_w_m2_k), 0.0)
    t = max(float(fin_thickness_m), 1e-6)
    length = max(float(fin_height_m), 1e-6)
    if h <= 0.0:
        return 1.0
    m = math.sqrt(2.0 * h / (k_fin_w_m_k * t))
    ml = m * length
    if ml < 1e-9:
        return 1.0
    return float(math.tanh(ml) / ml)


# =============================================================================
# Cost correlations (B1 / B2 / B3 / B4)
# =============================================================================

# --- B1: absorber coating, USD/m^2 as a function of IR emissivity ---
#
# PLACEHOLDER ANCHORS -- Clara is fitting the real correlation. Replace the tuples
# below (and nothing else) when that lands; the interpolation and every caller
# stay as they are. Ordered by ascending eps_abs_ir, i.e. best/most expensive
# selective surface first, commodity black paint last.
# Anchored so the Case 2 selective surface (eps_abs_ir = 0.05) costs exactly
# $10/m^2 more than Wilson's black spray paint -- the measured material premium.
# The intermediate points keep the curve's shape between those two endpoints.
ABSORBER_COATING_COST_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.05, 10.60),  # sputtered cermet / TiNOX-class selective surface (Case 2)
    (0.10, 7.88),   # black chrome
    (0.30, 3.12),   # mid-grade selective paint
    (0.60, 1.25),   # semi-selective coating
    (0.95, 0.60),   # commercial black spray paint (Wilson's own absorber)
)

# The premium the anchors above encode, asserted so a future re-fit that breaks it
# is caught rather than silently shifting every B1 result.
CASE2_COATING_PREMIUM_USD_PER_M2: float = 10.00


def absorber_coating_cost_usd_per_m2(eps_abs_ir: float) -> float:
    """B1 absorber coating cost. Log-linear interpolation over the anchor table.

    Interpolated in log-cost because the anchors span ~50x over a narrow
    emissivity range; linear interpolation there would badly undercost the
    selective end.
    """
    eps = np.array([a for a, _ in ABSORBER_COATING_COST_ANCHORS], dtype=float)
    cost = np.array([c for _, c in ABSORBER_COATING_COST_ANCHORS], dtype=float)
    x = float(np.clip(eps_abs_ir, eps[0], eps[-1]))
    return float(np.exp(np.interp(x, eps, np.log(cost))))


# --- B2: glazing stack, USD/m^2 ---
GLAZING_COST_PER_PANE_USD_PER_M2: float = 12.00  # low-iron float, installed
EVACUATED_GAP_PREMIUM_USD_PER_M2: float = 90.00  # sealed evacuated panel


def glazing_cost_usd_per_m2(n_panes: int, *, evacuated: bool = False) -> float:
    """B2 glazing cost. An evacuated gap needs at least one pane to seal against."""
    cost = GLAZING_COST_PER_PANE_USD_PER_M2 * max(int(n_panes), 0)
    if evacuated and n_panes > 0:
        cost += EVACUATED_GAP_PREMIUM_USD_PER_M2
    return float(cost)


# --- B3: condenser fin aluminum, USD/m^2 ---
ALUMINUM_PRICE_USD_PER_KG: float = 2.20  # Table S2 basis: $29.87 / 13.55 kg
_RHO_AL_KG_M3: float = _pv("Aluminum density (rho_Al)")


def fin_material_cost_usd_per_m2(
    fin_area_ratio: float,
    *,
    fin_thickness_m: float,
    price_usd_per_kg: float = ALUMINUM_PRICE_USD_PER_KG,
) -> float:
    """B3 fin aluminum cost per m^2 of condenser base area.

    Area beyond the bare base plate (A_r - 1) is fin surface; at thickness t that
    is ``rho * t * (A_r - 1)`` kg per m^2 of base. This is what stops the optimizer
    from pinning ``fin_area_ratio`` at its upper bound, since Wilson's flat
    $29.87/m^2 condenser line is charged regardless of A_r.
    """
    extra_area = max(float(fin_area_ratio) - 1.0, 0.0)
    mass_kg_per_m2 = _RHO_AL_KG_M3 * max(float(fin_thickness_m), 0.0) * extra_area
    return float(mass_kg_per_m2 * price_usd_per_kg)


# --- B4: forced condenser cooling, USD/m^2 (Wilson Table S2 line items) ---
# "Condenser fans, 50 qty, $34.50" + "Electronics, PV cells, 50 qty, $20.00" buy
# the ~0.5 m/s the Atacama system blows over its condenser (Note S2).
CONDENSER_FAN_COST_USD_PER_M2: float = 34.50
CONDENSER_PV_COST_USD_PER_M2: float = 20.00
WILSON_FAN_AIR_SPEED_M_S: float = 0.5
FAN_LIFETIME_YEARS: float = 5.0


def forced_cooling_capex_usd_per_m2(air_speed_m_s: float) -> float:
    """B4 fan + PV capex, scaled linearly off Wilson's own 0.5 m/s design point."""
    v = max(float(air_speed_m_s), 0.0)
    if v <= 0.0:
        return 0.0
    scale = v / WILSON_FAN_AIR_SPEED_M_S
    return float(scale * (CONDENSER_FAN_COST_USD_PER_M2 + CONDENSER_PV_COST_USD_PER_M2))


def forced_cooling_annual_replacement_usd_per_m2(air_speed_m_s: float) -> float:
    """Annualized fan replacement. The PV share is not replaced on this interval."""
    v = max(float(air_speed_m_s), 0.0)
    if v <= 0.0:
        return 0.0
    scale = v / WILSON_FAN_AIR_SPEED_M_S
    return float(scale * CONDENSER_FAN_COST_USD_PER_M2 / FAN_LIFETIME_YEARS)


_FIN_AREA_RATIO_DEFAULT: float = _pv("Condenser fin area ratio (A_r)")


def complex_system_capex_usd_per_m2(options: ComplexOptions, *, fin_area_ratio: float) -> float:
    """Complex-mode CAPEX *added on top of* the flat simple-model BOM (USD/m^2).

    Wilson's BOM already buys a black-painted absorber, one glass cover, and a
    condenser at the default fin ratio, so each of those baselines is netted out
    rather than double-charged: this returns only the delta against his build.
    It is therefore signed -- a cheaper design (no glazing, fewer fins) returns a
    negative number, which is the point.
    """
    return float(
        (absorber_coating_cost_usd_per_m2(options.eps_abs_ir)
         - absorber_coating_cost_usd_per_m2(_EPS_ABS_IR_BASELINE))
        + (glazing_cost_usd_per_m2(options.n_glazing_panes, evacuated=options.evacuated_gap)
           - glazing_cost_usd_per_m2(1))
        + (fin_material_cost_usd_per_m2(fin_area_ratio, fin_thickness_m=options.fin_thickness_m)
           - fin_material_cost_usd_per_m2(_FIN_AREA_RATIO_DEFAULT, fin_thickness_m=options.fin_thickness_m))
        + forced_cooling_capex_usd_per_m2(options.condenser_air_speed_m_s)
    )
