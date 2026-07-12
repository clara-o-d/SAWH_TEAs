"""Numeric helpers (ported from electrolyte_optimization)."""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.optimize import brentq


def find_root_bracketed(
    f: Callable[[float], float],
    x_min: float,
    x_max: float,
    *,
    scan: bool = False,
    n_intervals: int = 20,
    maxiter: int = 200,
) -> float:
    """Find a root of ``f`` on ``[x_min, x_max]``; return nan if no bracket exists."""
    if scan:
        step = (x_max - x_min) / n_intervals
        x_prev, f_prev = x_min, f(x_min)
        if not math.isfinite(f_prev):
            return float("nan")
        for i in range(1, n_intervals + 1):
            x_curr = x_min + i * step
            f_curr = f(x_curr)
            if math.isfinite(f_curr) and f_prev * f_curr < 0.0:
                return float(brentq(f, x_prev, x_curr, maxiter=maxiter))
            if math.isfinite(f_curr):
                x_prev, f_prev = x_curr, f_curr
        return float("nan")

    fa, fb = f(x_min), f(x_max)
    if not (fa * fb < 0) or math.isnan(fa) or math.isnan(fb):
        return float("nan")
    return float(brentq(f, x_min, x_max, maxiter=maxiter))
