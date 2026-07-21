"""Device design-variable bounds, normalization, and space-filling sampling.

Bounds are reused from what solar_lumped has already explored/documented as
sensible ranges (scripts/parameter_sweep.py::make_sweep_params,
data/economics/lcow_economic_params.csv's hydrogel_thickness_min/max_m), not
invented -- see docs/design_notes.md for the per-variable provenance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

VAR_ORDER: tuple[str, ...] = (
    "hydrogel_thickness_m",
    "vapor_gap_m",
    "insulation_gap_m",
    "fin_area_ratio",
    "tilt_deg",
    "salt_to_polymer_ratio",
)

# Wilson's ~7mm thermobuoyancy/transport floor on the *effective* vapor gap
# (table_s3.VAPOR_GAP_TRANSPORT_MIN_M); used only to avoid wasting expensive
# initial samples on designs the physics already handles gracefully as
# near-zero yield.
VAPOR_GAP_TRANSPORT_MIN_M: float = 0.007


@dataclass(frozen=True, slots=True)
class DesignBounds:
    """(low, high) box bounds for each of the 6 v1 design variables.

    No condenser_thickness_m dimension: economics/lcow.py charges a flat
    condenser BOM cost regardless of thickness (a free cost-side lever with
    no downside), and the JAX gpu_sweep fast path this package evaluates
    against (see evaluator.py) hardcodes condenser thermal mass at Table
    S3's constant rather than taking it as a per-instance input -- it was
    never a real physics knob on that path. DeviceConfig's own default
    already matches that constant, so simply not setting it is correct.
    """

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


def to_device_config_kwargs(x: np.ndarray) -> dict[str, float]:
    """Map an unnamed VAR_ORDER-ordered vector to DeviceConfig field names."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] != len(VAR_ORDER):
        raise ValueError(f"Expected a length-{len(VAR_ORDER)} design vector, got shape {x.shape}")
    return dict(zip(VAR_ORDER, (float(v) for v in x)))


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
    """True if the nominal vapor gap leaves less than ``margin_m`` of headroom
    over the hydrogel thickness -- i.e. the effective gap (vapor_gap_m -
    hydrogel_thickness_m) would already sit at/below Wilson's transport floor
    even before the gel swells further during absorption. Not a hard
    infeasibility (the physics degrades to near-zero yield gracefully, it
    doesn't raise), just a signal to avoid spending an expensive sample there.
    """
    kwargs = to_device_config_kwargs(x)
    return (kwargs["vapor_gap_m"] - kwargs["hydrogel_thickness_m"]) < margin_m


def latin_hypercube_design(
    n: int,
    bounds: DesignBounds,
    *,
    seed: int,
    reject_gap_degenerate: bool = True,
    max_resample_rounds: int = 20,
) -> np.ndarray:
    """n Latin-hypercube-sampled design vectors within ``bounds``, shape (n, 6).

    When ``reject_gap_degenerate``, degenerate rows (see is_gap_degenerate)
    are resampled (not dropped) up to ``max_resample_rounds`` times so the
    returned array always has exactly n rows.
    """
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
