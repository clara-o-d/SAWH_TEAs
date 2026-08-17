"""System design-variable bounds, normalization, and space-filling sampling. Bounds come
from solar_lumped's documented ranges; see docs/design_notes.md for provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import qmc

from solar_lumped._parameters_xlsx import physics_bounds as _bounds
from solar_lumped.physics import EPS_ABS_IR_CASE2, EPS_GLASS_IR_CASE2
from solar_lumped.physics import VAPOR_GAP_TRANSPORT_MIN_M as _VAPOR_GAP_TRANSPORT_MIN_M

VAR_ORDER: tuple[str, ...] = (
    "hydrogel_thickness_m",
    "vapor_gap_m",
    "insulation_gap_m",
    "fin_area_ratio",
    "tilt_deg",
    "salt_loading",
)

# --- Complex-fidelity dimensions (solar_lumped.complex_model) ---
#
# Appended after VAR_ORDER, so a simple-mode vector is a strict prefix of a
# complex-mode one and the two share every bound, hash, and sampler code path.
#
# Screening verdict behind this list -- a dimension earns a slot only if
# d(LCOW)/dx changes sign inside its box, otherwise the optimizer just pins a
# bound and the GP dimension is wasted:
#
#   hydrogel_thickness_m   KEEP  capacity vs. sensible load + conduction + sorbent cost
#   salt_loading  KEEP  uptake gain saturates while salt cost keeps climbing
#   insulation_gap_m       KEEP  conduction falls as 1/L until Hollands convection starts
#   fin_area_ratio         KEEP  was a free lever (flat BOM line); B3 now prices the
#                                aluminum, and the measured optimum moved below A_r=3
#   tilt_deg               KEEP  only once POA is on -- complex mode enables it, so tilt
#                                drives solar gain, not just the Hollands cos(theta)
#   vapor_gap_m            WEAK  physics optimum flattens past ~40 mm and gap height is
#                                still free (system height is not costed -- that was B5,
#                                out of scope here). First candidate to drop if the GP
#                                struggles at this dimensionality.
#   eps_abs_ir             B1    interior optimum near 0.1-0.3 at the $10/m2 premium
#   glazing_config         B2    ordinal 0..3, non-monotone: pane 2 costs 19% of tau
#   condenser_air_speed    B4    fans buy condenser cooling for ~$54.50/m2 of capex
#   seal/open_offset_h     A1    two-sided, and interacts strongly with thickness
#   blend_u, blend_v       B8    stick-breaking coords onto the ZSR simplex
COMPLEX_VAR_ORDER: tuple[str, ...] = (
    "eps_abs_ir",
    "glazing_config",
    "condenser_air_speed_m_s",
    "seal_offset_h",
    "open_offset_h",
    "blend_u",
    "blend_v",
)

FULL_VAR_ORDER: tuple[str, ...] = VAR_ORDER + COMPLEX_VAR_ORDER

# glazing_config packs B2's pane count and evacuated flag into one ordinal axis,
# ordered by increasing cost and increasing thermal isolation, so the GP sees a
# meaningful ordering instead of two correlated binaries.
GLAZING_CONFIGS: tuple[tuple[int, bool], ...] = (
    (0, False),  # uncovered absorber
    (1, False),  # Wilson's single borosilicate cover
    (2, False),  # double glazed
    (2, True),   # double glazed, evacuated gap
)


def var_order(complex_mode: bool) -> tuple[str, ...]:
    return FULL_VAR_ORDER if complex_mode else VAR_ORDER

# Wilson's ~7mm transport floor on the effective vapor gap; used only to avoid spending
# initial samples on designs the physics already returns near-zero yield for.
VAPOR_GAP_TRANSPORT_MIN_M: float = _VAPOR_GAP_TRANSPORT_MIN_M

# Absorber/glass IR emissivity pairs for the modified Eqs. 3/4 radiative term, matching
# run_gpu_sweep.py's --eps-abs-ir/--eps-glass-ir: case2 (selective surface) is the base
# case, case1 (1.0, 1.0) is Wilson's blackbody/cavity approximation, case3 the idealized
# optical limit. case1 uses explicit floats -- numerically identical to the old (None,
# None), so historical case1 cache.jsonl entries stay valid.
CASE_EPS_IR: dict[str, tuple[float | None, float | None]] = {
    "case1": (1.0, 1.0),
    "case2": (EPS_ABS_IR_CASE2, EPS_GLASS_IR_CASE2),
    "case3": (0.0, 0.0),
}


# Box bounds come from the "Lower (for Sweeps)" / "Upper (for Sweeps)" columns of
# solar_lumped/docs/parameters.xlsx, the repo-wide parameter source. Nothing here is a
# second opinion on those ranges.
_HYDROGEL_THICKNESS_BOUNDS_M = _bounds("Hydrogel reference thickness (H0)", mm_to_m=True)
_VAPOR_GAP_BOUNDS_M = _bounds("Vapor gap (L_g)", mm_to_m=True)
_INSULATION_GAP_BOUNDS_M = _bounds("Insulation gap (L_ins)", mm_to_m=True)
_FIN_AREA_RATIO_BOUNDS = _bounds("Condenser fin area ratio (A_r)")
_TILT_BOUNDS_DEG = _bounds("Tilt angle (theta)")
_SALT_LOADING_BOUNDS = _bounds("Salt loading (SL)")
_EPS_ABS_IR_BOUNDS = _bounds("Absorber IR emissivity (eps_abs_ir)")
_CONDENSER_AIR_SPEED_BOUNDS_M_S = _bounds("Condenser forced-air speed")
_SEAL_OPEN_OFFSET_BOUNDS_H = _bounds("Seal / open offset from sunrise-sunset")


@dataclass(frozen=True, slots=True)
class DesignBounds:
    """(low, high) box bounds. 6 rows in simple mode, 13 with ``complex_mode=True``.

    No condenser_thickness_m: LCOW charges a flat condenser BOM cost and the JAX fast
    path hardcodes condenser thermal mass at the Table S3 constant, which is already
    SystemConfig's default.
    """

    hydrogel_thickness_m: tuple[float, float] = _HYDROGEL_THICKNESS_BOUNDS_M
    vapor_gap_m: tuple[float, float] = _VAPOR_GAP_BOUNDS_M
    insulation_gap_m: tuple[float, float] = _INSULATION_GAP_BOUNDS_M
    fin_area_ratio: tuple[float, float] = _FIN_AREA_RATIO_BOUNDS
    tilt_deg: tuple[float, float] = _TILT_BOUNDS_DEG
    salt_loading: tuple[float, float] = _SALT_LOADING_BOUNDS

    # --- complex mode only (appended in COMPLEX_VAR_ORDER when complex_mode) ---
    complex_mode: bool = False
    eps_abs_ir: tuple[float, float] = _EPS_ABS_IR_BOUNDS
    # Continuous stand-in for the GLAZING_CONFIGS index; rounded on use. Not a workbook
    # row -- the range is just the length of that tuple.
    glazing_config: tuple[float, float] = (0.0, float(len(GLAZING_CONFIGS) - 1))
    condenser_air_speed_m_s: tuple[float, float] = _CONDENSER_AIR_SPEED_BOUNDS_M_S
    seal_offset_h: tuple[float, float] = _SEAL_OPEN_OFFSET_BOUNDS_H
    open_offset_h: tuple[float, float] = _SEAL_OPEN_OFFSET_BOUNDS_H
    # Stick-breaking coordinates onto the ZSR simplex (see blend_weights_from_uv).
    blend_u: tuple[float, float] = (0.0, 1.0)
    blend_v: tuple[float, float] = (0.0, 1.0)

    def names(self) -> tuple[str, ...]:
        return var_order(self.complex_mode)

    def as_array(self) -> np.ndarray:
        """(n, 2) array of (low, high), in this bounds object's variable order."""
        return np.array([getattr(self, name) for name in self.names()], dtype=float)


def blend_weights_from_uv(u: float, v: float) -> tuple[float, float, float]:
    """Stick-breaking map [0,1]^2 -> the 3-salt ZSR simplex, in ZSR_SALTS order.

    Two coordinates for a 2-simplex, so the GP never sees the redundant third
    weight (three normalized dims would carry a sum-to-one ridge the kernel cannot
    represent). ``u`` is LiCl's share outright; ``v`` splits what remains between
    CaCl2 and MgCl2. The map is onto and axis-aligned, which is what a per-
    dimension length-scale kernel wants.
    """
    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    rest = 1.0 - u
    return (u, rest * v, rest * (1.0 - v))


def to_complex_options(x: np.ndarray):
    """Build a ``ComplexOptions`` from the complex tail of a FULL_VAR_ORDER vector."""
    from solar_lumped.complex_model import ComplexOptions

    vals = dict(zip(FULL_VAR_ORDER, (float(v) for v in np.asarray(x, dtype=float).reshape(-1))))
    n_panes, evacuated = GLAZING_CONFIGS[
        int(np.clip(round(vals["glazing_config"]), 0, len(GLAZING_CONFIGS) - 1))
    ]
    return ComplexOptions(
        seal_offset_h=vals["seal_offset_h"],
        open_offset_h=vals["open_offset_h"],
        eps_abs_ir=vals["eps_abs_ir"],
        n_glazing_panes=n_panes,
        evacuated_gap=evacuated,
        condenser_air_speed_m_s=vals["condenser_air_speed_m_s"],
        blend_weights=blend_weights_from_uv(vals["blend_u"], vals["blend_v"]),
    )


def to_system_config_kwargs(
    x: np.ndarray,
    *,
    case: str = "case2",
    complex_mode: bool = False,
    condenser_tracks_ambient: bool = False,
    instant_equilibrium: bool = False,
) -> dict[str, Any]:
    """Map a design vector to SystemConfig field names.

    ``case`` picks the IR emissivity pair (CASE_EPS_IR) in simple mode. Every case
    gets an explicit "thermal" kwarg with SystemThermalParams re-derived per point,
    since insulation_gap_m/vapor_gap_m/tilt_deg are all swept dims.

    In complex mode the vector carries COMPLEX_VAR_ORDER as well, ``case`` is
    ignored (B1 optimizes eps_abs_ir directly, which is what the case flag used to
    stand in for), and no "thermal" kwarg is set -- SystemConfig.thermal_params()
    derives it from the ComplexOptions so B2's pane count and stack transmittance
    stay consistent with the cost model.

    ``condenser_tracks_ambient`` is orthogonal to both: False (default) keeps
    solar_lumped's Eq. 2 condenser ODE, True pins T_cond to T_amb (the infinite-
    cooling-capacity limit) in either fidelity mode. ``instant_equilibrium`` is
    orthogonal in the same way: the g -> infinity sorption limit, either fidelity.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    names = var_order(complex_mode)
    if x.shape[0] != len(names):
        raise ValueError(f"Expected a length-{len(names)} design vector, got shape {x.shape}")
    kwargs: dict[str, Any] = dict(zip(VAR_ORDER, (float(v) for v in x[: len(VAR_ORDER)])))
    kwargs["condenser_tracks_ambient"] = condenser_tracks_ambient
    kwargs["instant_equilibrium"] = instant_equilibrium

    if complex_mode:
        kwargs["complex"] = to_complex_options(x)
        return kwargs

    if case not in CASE_EPS_IR:
        raise ValueError(f"Unknown case {case!r}, expected one of {sorted(CASE_EPS_IR)}")
    from solar_lumped.physics import EPS_ABS, TAU_GLASS
    from solar_lumped.physics import SystemThermalParams

    eps_abs_ir, eps_glass_ir = CASE_EPS_IR[case]
    kwargs["thermal"] = SystemThermalParams(
        insulation_gap_m=kwargs["insulation_gap_m"],
        vapor_gap_m=kwargs["vapor_gap_m"],
        eps_abs=EPS_ABS,
        tau_glass=TAU_GLASS,
        tilt_deg=kwargs["tilt_deg"],
        eps_abs_ir=eps_abs_ir,
        eps_glass_ir=eps_glass_ir,
    )
    return kwargs


def to_unit_cube(x: np.ndarray, bounds: DesignBounds) -> np.ndarray:
    """Raw design vector(s) -> the unit cube. Accepts (n_dims,) or (n, n_dims)."""
    x = np.asarray(x, dtype=float)
    lo, hi = bounds.as_array()[:, 0], bounds.as_array()[:, 1]
    return (x - lo) / (hi - lo)


def from_unit_cube(u: np.ndarray, bounds: DesignBounds) -> np.ndarray:
    """Unit cube -> raw design vector(s). Accepts (n_dims,) or (n, n_dims)."""
    u = np.asarray(u, dtype=float)
    lo, hi = bounds.as_array()[:, 0], bounds.as_array()[:, 1]
    return lo + u * (hi - lo)


def is_gap_degenerate(x: np.ndarray, *, margin_m: float = VAPOR_GAP_TRANSPORT_MIN_M) -> bool:
    """True if the effective gap (vapor_gap_m - hydrogel_thickness_m) leaves under
    ``margin_m`` of headroom over Wilson's transport floor before the gel even swells.
    Not a hard infeasibility -- just a signal not to spend a sample there.

    Reads the first two VAR_ORDER slots directly, so it works on simple and complex
    vectors alike (complex dims are appended after them)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return (float(x[VAR_ORDER.index("vapor_gap_m")])
            - float(x[VAR_ORDER.index("hydrogel_thickness_m")])) < margin_m


def latin_hypercube_design(
    n: int,
    bounds: DesignBounds,
    *,
    seed: int,
    reject_gap_degenerate: bool = True,
    max_resample_rounds: int = 20,
) -> np.ndarray:
    """n Latin-hypercube design vectors within ``bounds``, shape (n, n_dims). With
    ``reject_gap_degenerate``, degenerate rows are resampled (never dropped) up to
    ``max_resample_rounds`` times, so the result always has exactly n rows."""
    sampler = qmc.LatinHypercube(d=bounds.as_array().shape[0], seed=seed)
    u = sampler.random(n)
    x = from_unit_cube(u, bounds)
    if not reject_gap_degenerate:
        return x

    bad = np.array([is_gap_degenerate(row) for row in x])
    rounds = 0
    while bad.any() and rounds < max_resample_rounds:
        n_bad = int(bad.sum())
        u_replacement = sampler.random(n_bad)
        x[bad] = from_unit_cube(u_replacement, bounds)
        bad = np.array([is_gap_degenerate(row) for row in x])
        rounds += 1
    return x
