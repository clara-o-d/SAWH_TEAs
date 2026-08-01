"""Device design-variable bounds, normalization, and space-filling sampling. Bounds come
from solar_lumped's documented ranges; see docs/design_notes.md for provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import qmc

from solar_lumped.physics import EPS_ABS_IR_CASE2, EPS_GLASS_IR_CASE2
from solar_lumped.physics import VAPOR_GAP_TRANSPORT_MIN_M as _VAPOR_GAP_TRANSPORT_MIN_M

VAR_ORDER: tuple[str, ...] = (
    "hydrogel_thickness_m",
    "vapor_gap_m",
    "insulation_gap_m",
    "fin_area_ratio",
    "tilt_deg",
    "salt_to_polymer_ratio",
)

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


@dataclass(frozen=True, slots=True)
class DesignBounds:
    """(low, high) box bounds for the 6 v1 design variables. No condenser_thickness_m:
    LCOW charges a flat condenser BOM cost and the JAX fast path hardcodes condenser
    thermal mass at the Table S3 constant, which is already DeviceConfig's default."""

    hydrogel_thickness_m: tuple[float, float] = (0.001, 0.010)
    vapor_gap_m: tuple[float, float] = (0.007, 0.060)
    insulation_gap_m: tuple[float, float] = (0.001, 0.020)
    fin_area_ratio: tuple[float, float] = (3.0, 12.0)
    tilt_deg: tuple[float, float] = (0.0, 60.0)
    salt_to_polymer_ratio: tuple[float, float] = (1.0, 8.0)

    def as_array(self) -> np.ndarray:
        """(6, 2) array of (low, high), in VAR_ORDER."""
        return np.array([getattr(self, name) for name in VAR_ORDER], dtype=float)


def bounds_array(bounds: DesignBounds) -> np.ndarray:
    return bounds.as_array()


def to_device_config_kwargs(x: np.ndarray, *, case: str = "case2") -> dict[str, Any]:
    """Map a VAR_ORDER-ordered vector to DeviceConfig field names. ``case`` picks the IR
    emissivity pair (CASE_EPS_IR). Every case gets an explicit "thermal" kwarg with
    DeviceThermalParams re-derived per point, since insulation_gap_m/vapor_gap_m/tilt_deg
    are all swept dims."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] != len(VAR_ORDER):
        raise ValueError(f"Expected a length-{len(VAR_ORDER)} design vector, got shape {x.shape}")
    if case not in CASE_EPS_IR:
        raise ValueError(f"Unknown case {case!r}, expected one of {sorted(CASE_EPS_IR)}")
    kwargs: dict[str, Any] = dict(zip(VAR_ORDER, (float(v) for v in x)))
    from solar_lumped.physics import EPS_ABS, TAU_GLASS
    from solar_lumped.physics import DeviceThermalParams

    eps_abs_ir, eps_glass_ir = CASE_EPS_IR[case]
    kwargs["thermal"] = DeviceThermalParams(
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
    """Raw design vector(s) -> [0, 1]^6. Accepts (6,) or (n, 6)."""
    x = np.asarray(x, dtype=float)
    lo, hi = bounds.as_array()[:, 0], bounds.as_array()[:, 1]
    return (x - lo) / (hi - lo)


def from_unit_cube(u: np.ndarray, bounds: DesignBounds) -> np.ndarray:
    """[0, 1]^6 -> raw design vector(s). Accepts (6,) or (n, 6)."""
    u = np.asarray(u, dtype=float)
    lo, hi = bounds.as_array()[:, 0], bounds.as_array()[:, 1]
    return lo + u * (hi - lo)


def is_gap_degenerate(x: np.ndarray, *, margin_m: float = VAPOR_GAP_TRANSPORT_MIN_M) -> bool:
    """True if the effective gap (vapor_gap_m - hydrogel_thickness_m) leaves under
    ``margin_m`` of headroom over Wilson's transport floor before the gel even swells.
    Not a hard infeasibility -- just a signal not to spend a sample there."""
    kwargs = to_device_config_kwargs(x)
    return (float(kwargs["vapor_gap_m"]) - float(kwargs["hydrogel_thickness_m"])) < margin_m


def latin_hypercube_design(
    n: int,
    bounds: DesignBounds,
    *,
    seed: int,
    reject_gap_degenerate: bool = True,
    max_resample_rounds: int = 20,
) -> np.ndarray:
    """n Latin-hypercube design vectors within ``bounds``, shape (n, 6). With
    ``reject_gap_degenerate``, degenerate rows are resampled (never dropped) up to
    ``max_resample_rounds`` times, so the result always has exactly n rows."""
    sampler = qmc.LatinHypercube(d=len(VAR_ORDER), seed=seed)
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
