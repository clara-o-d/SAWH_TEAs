"""Cached wrapper around solar_lumped/gpu_sweep's JAX/diffrax fast path for
combined_lcow(design_vector).

Evaluates through solar_lumped/gpu_sweep's JAX daily-cycle + Aitken pipeline
(jax_daily_cycle.py), not solar_lumped's CPU ode_system.py directly --
gpu_sweep/FINDINGS.md Results 6/7 show the two agree to <0.03% and the JAX
path is ~8x faster even single-threaded on a CPU with no GPU, purely from
compiling the daily-cycle function once and reusing it across every Aitken
round and every (design, site, month) instance, instead of re-dispatching
scipy's Radau/root/brentq calls from Python one at a time. Every uncached
design in a round is therefore evaluated together: all of its (site, month)
instances, across every uncached design in the batch, are stacked into one
jax.vmap-compiled call rather than one process per design.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from math import floor, log10
from pathlib import Path
from typing import Literal

import numpy as np

from sawh_bayesopt import design_space
from sawh_bayesopt.sites import MonthlyProfiles, SiteSpec

CombineRule = Literal["mean", "worst_case"]

# Finite stand-in for solar_lumped's FAIL_LCO (1e30) once a failure is
# clamped into the surrogate-facing combined_lcow -- 1e30 would wreck GP
# kernel hyperparameter fitting, but this is still comfortably worse than
# any real design (LCOW here is typically single/low-double-digit USD/m^3).
PENALTY_LCOW_USD_PER_M3: float = 1.0e4

# Fixed-round-count Aitken convergence, as validated in gpu_sweep/FINDINGS.md
# Result 7 (<0.03% worst-per-month accuracy cost vs. the adaptive
# per-instance result) and used as run_gpu_sweep.py's own default.
JAX_AITKEN_MAX_ROUNDS: int = 8

# .../sawh_bayesopt/src/sawh_bayesopt/evaluator.py -> .../SAWH_TEAs
_SAWH_TEAS_ROOT = Path(__file__).resolve().parents[3]
_GPU_SWEEP_DIR = _SAWH_TEAS_ROOT / "solar_lumped" / "gpu_sweep"


@dataclasses.dataclass(frozen=True, slots=True)
class SiteResult:
    site_name: str
    lcow: float
    feasible: bool
    failure_reason: str
    yield_kg_m2: float
    eta_thermal: float


@dataclasses.dataclass(frozen=True, slots=True)
class DesignEvalResult:
    design_vector: tuple[float, ...]
    site_results: tuple[SiteResult, ...]
    combined_lcow: float
    # Wall-clock of the whole batched jax.vmap call this design was evaluated
    # in, not this design's own share of it -- every design in the same
    # evaluate_batch() call gets the same value.
    wall_time_s: float

    def site(self, name: str) -> SiteResult:
        for r in self.site_results:
            if r.site_name == name:
                return r
        raise KeyError(name)


def _round_sig(v: float, sig: int = 6) -> float:
    if v == 0.0 or not math.isfinite(v):
        return v
    d = sig - int(floor(log10(abs(v)))) - 1
    return round(v, d)


def design_vector_hash(
    x: np.ndarray,
    *,
    sites: tuple[str, ...],
    resolution: str = "monthly",
) -> str:
    """Stable cache key: sig-fig-rounded design vector + site set + resolution,
    so float jitter from LHS/EI proposals doesn't create spurious cache misses
    for effectively-identical points."""
    rounded = tuple(_round_sig(float(v)) for v in np.asarray(x, dtype=float).reshape(-1))
    payload = {"x": rounded, "sites": sorted(sites), "resolution": resolution}
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _load_jax_daily_cycle():
    """Import solar_lumped/gpu_sweep's jax_daily_cycle module.

    gpu_sweep/ isn't an installed package (it's meant to be run from a
    checkout on a GPU node, see gpu_sweep/SHERLOCK_GPU_RUNBOOK.md), so it's
    reached the same way gpu_sweep/run_gpu_sweep.py reaches it: sys.path-
    inserting the gpu_sweep/ directory itself. Done lazily, only when an
    evaluation actually needs real physics, so unit tests that never hit
    real physics (empty weather profiles, monkeypatched internals) don't
    require jax/diffrax to be installed.
    """
    if str(_GPU_SWEEP_DIR) not in sys.path:
        sys.path.insert(0, str(_GPU_SWEEP_DIR))
    import jax_daily_cycle

    return jax_daily_cycle


def combine_site_lcows(
    site_results: tuple[SiteResult, ...],
    *,
    combine_rule: CombineRule = "mean",
    penalty: float = PENALTY_LCOW_USD_PER_M3,
) -> float:
    from solar_lumped.economics.lcow import FAIL_LCO

    vals = [penalty if (not r.feasible or r.lcow >= 0.99 * FAIL_LCO) else r.lcow for r in site_results]
    if combine_rule == "mean":
        return float(sum(vals) / len(vals))
    if combine_rule == "worst_case":
        return float(max(vals))
    raise ValueError(f"Unknown combine_rule {combine_rule!r}")


def _result_to_jsonable(result: DesignEvalResult) -> dict:
    return dataclasses.asdict(result)


def _result_from_jsonable(d: dict) -> DesignEvalResult:
    site_results = tuple(SiteResult(**sr) for sr in d["site_results"])
    return DesignEvalResult(
        design_vector=tuple(d["design_vector"]),
        site_results=site_results,
        combined_lcow=d["combined_lcow"],
        wall_time_s=d["wall_time_s"],
    )


class EvalCache:
    """Append-only jsonl ledger of completed evaluations, keyed by design-vector
    hash, so an interrupted/crashed optimization run can resume without
    re-paying for already-completed evaluations."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._by_key: dict[str, DesignEvalResult] = {}
        if self.path.is_file():
            with self.path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._by_key[rec["key"]] = _result_from_jsonable(rec["result"])

    def __len__(self) -> int:
        return len(self._by_key)

    def get_or_none(self, key: str) -> DesignEvalResult | None:
        return self._by_key.get(key)

    def put(self, key: str, result: DesignEvalResult) -> None:
        self._by_key[key] = result
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "result": _result_to_jsonable(result)}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def all_results(self) -> list[DesignEvalResult]:
        return list(self._by_key.values())


def _run_jax_batch(
    flat_profiles: list,
    flat_configs: list,
    flat_weights: list[int],
    owner: list[tuple[int, int]],
    *,
    initial_loading,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], str | None]:
    """Run one jax.vmap-batched daily-cycle call over every flattened
    (design, site, month) instance and reduce it to a day-weighted mean
    yield/eta per (design, site) pair. Returns (yield_by_pair, eta_by_pair,
    error) -- error is set (and the two dicts left empty) if the batched
    jax/diffrax call itself raised.
    """
    if not flat_profiles:
        return {}, {}, None

    try:
        jdc = _load_jax_daily_cycle()
        batch, dt, n_abs_max, n_des_max = jdc.build_batch_arrays(flat_profiles, flat_configs)
        batched_fn = jdc.make_batched_daily_cycle_fn(batch, dt, n_abs_max, n_des_max)
        cw0 = np.array([initial_loading(c) for c in flat_configs])
        h0 = np.array([c.hydrogel_thickness_m for c in flat_configs])
        cw_conv, h_conv = jdc.find_cyclic_state_batched(
            batched_fn, c_w_initial=cw0, h_initial=h0, max_rounds=JAX_AITKEN_MAX_ROUNDS,
        )
        water, eta, _, _ = batched_fn(cw_conv, h_conv)
        water = np.asarray(water)
        eta = np.asarray(eta)
    except Exception as exc:  # noqa: BLE001 -- the batched jax/diffrax call can raise
        return {}, {}, str(exc).split("\n", 1)[0][:240]

    yield_sums: dict[tuple[int, int], float] = {}
    eta_sums: dict[tuple[int, int], float] = {}
    weight_sums: dict[tuple[int, int], float] = {}
    for pair, w, y, e in zip(owner, flat_weights, water, eta):
        yield_sums[pair] = yield_sums.get(pair, 0.0) + float(y) * w
        eta_sums[pair] = eta_sums.get(pair, 0.0) + float(e) * w
        weight_sums[pair] = weight_sums.get(pair, 0.0) + w

    yield_by_pair = {pair: yield_sums[pair] / w for pair, w in weight_sums.items()}
    eta_by_pair = {pair: eta_sums[pair] / w for pair, w in weight_sums.items()}
    return yield_by_pair, eta_by_pair, None


def evaluate_batch(
    xs: list[np.ndarray],
    *,
    cache: EvalCache,
    sites: tuple[SiteSpec, ...],
    site_profiles: dict[str, MonthlyProfiles],
    econ,
    combine_rule: CombineRule = "mean",
    resolution: str = "monthly",
) -> list[DesignEvalResult]:
    """Evaluate every x in *xs*, skipping any already present in *cache*.

    Every uncached design's (site, month) instances -- across every uncached
    design in *xs* -- are stacked into one jax.vmap-batched daily-cycle call
    (see jax_daily_cycle.py::make_batched_daily_cycle_fn) instead of
    dispatching one process per design; that cross-design batching is what
    actually delivers the JAX path's speedup, not just a faster single call.
    """
    from solar_lumped.economics.lcow import FAIL_LCO, lcow_from_daily_yield
    from solar_lumped.physics.sorbent import initial_loading
    from solar_lumped.simulation.device_config import DeviceConfig

    site_names = tuple(s.name for s in sites)
    keys = [design_vector_hash(x, sites=site_names, resolution=resolution) for x in xs]
    results: list[DesignEvalResult | None] = [None] * len(xs)

    to_run = [i for i, key in enumerate(keys) if cache.get_or_none(key) is None]
    for i, key in enumerate(keys):
        cached = cache.get_or_none(key)
        if cached is not None:
            results[i] = cached

    if not to_run:
        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    configs = {i: DeviceConfig(**design_space.to_device_config_kwargs(xs[i])) for i in to_run}

    # Flatten every (design, site, month) instance that actually has weather
    # data into one cross-product batch. (design, site) pairs with no
    # profiles at all are handled directly below without needing jax.
    flat_profiles = []
    flat_configs = []
    flat_weights: list[int] = []
    owner: list[tuple[int, int]] = []  # (design index into xs, site index)
    no_weather: set[tuple[int, int]] = set()
    for i in to_run:
        for si, spec in enumerate(sites):
            profiles = site_profiles[spec.name]
            if not profiles:
                no_weather.add((i, si))
                continue
            for _month, profile, n_days in profiles:
                flat_profiles.append(profile)
                flat_configs.append(configs[i])
                flat_weights.append(n_days)
                owner.append((i, si))

    t0 = time.perf_counter()
    yield_by_pair, eta_by_pair, batch_error = _run_jax_batch(
        flat_profiles, flat_configs, flat_weights, owner, initial_loading=initial_loading,
    )
    wall = time.perf_counter() - t0

    new_results: dict[int, DesignEvalResult] = {}
    for i in to_run:
        site_results = []
        for si, spec in enumerate(sites):
            if (i, si) in no_weather:
                site_results.append(
                    SiteResult(spec.name, FAIL_LCO, False, "no weather profiles", float("nan"), float("nan"))
                )
                continue
            if batch_error is not None:
                site_results.append(
                    SiteResult(spec.name, FAIL_LCO, False, batch_error, float("nan"), float("nan"))
                )
                continue

            mean_yield = yield_by_pair[(i, si)]
            mean_eta = eta_by_pair[(i, si)]
            if not math.isfinite(mean_yield) or mean_yield <= 0.0:
                site_results.append(
                    SiteResult(spec.name, FAIL_LCO, False, "zero or invalid yield", mean_yield, mean_eta)
                )
                continue

            cfg = configs[i]
            lcow = lcow_from_daily_yield(
                mean_yield,
                salt_name=cfg.salt_name,
                salt_to_polymer_ratio=cfg.salt_to_polymer_ratio,
                hydrogel_thickness_m=cfg.hydrogel_thickness_m,
                econ=econ,
            )
            if not math.isfinite(lcow) or lcow >= 0.99 * FAIL_LCO:
                site_results.append(SiteResult(spec.name, FAIL_LCO, False, "invalid LCOW", mean_yield, mean_eta))
                continue

            site_results.append(SiteResult(spec.name, lcow, True, "", mean_yield, mean_eta))

        combined = combine_site_lcows(tuple(site_results), combine_rule=combine_rule)
        new_results[i] = DesignEvalResult(
            design_vector=tuple(float(v) for v in np.asarray(xs[i], dtype=float).reshape(-1)),
            site_results=tuple(site_results),
            combined_lcow=combined,
            wall_time_s=wall,
        )

    for i in to_run:
        results[i] = new_results[i]
        cache.put(keys[i], new_results[i])

    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]
