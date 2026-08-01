"""Cached combined_lcow(design_vector) over gpu_sweep's JAX daily-cycle + Aitken pipeline.

Agrees with solar_lumped's CPU path to <0.03% and is ~8x faster (FINDINGS.md 6/7): every
uncached design's (site, month) instances stack into one jax.vmap-compiled call."""

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
from sawh_bayesopt.sites import DailyProfiles, SiteSpec

CombineRule = Literal["mean", "worst_case"]

# Finite stand-in for FAIL_LCO (1e30), which would wreck GP hyperparameter fitting;
# still far worse than any real design (LCOW is typically single/low-double-digit).
PENALTY_LCOW_USD_PER_M3: float = 1.0e4

# Fixed-round-count Aitken convergence: <0.03% worst-per-month cost vs. adaptive
# (FINDINGS.md Result 7), and run_gpu_sweep.py's own default.
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
    # Wall-clock of the whole batched jax.vmap call, not this design's share -- every
    # design in the same evaluate_batch() gets the same value.
    wall_time_s: float

    @property
    def is_feasible(self) -> bool:
        """True iff every site succeeded. A property so it always tracks site_results and
        old cache records get it free. combined_lcow can't substitute: with
        combine_rule="mean" a partial failure lands between a real value and the penalty."""
        return all(r.feasible for r in self.site_results)

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
    resolution: str = "annual",
    case: str = "case2",
) -> str:
    """Stable cache key: sig-fig-rounded design vector + sites + resolution (+ case when
    non-default), so LHS/EI float jitter doesn't cause spurious misses. The ``case``
    default must stay in sync with evaluate_batch's, and "case1" is omitted from the
    payload so pre-case-awareness cache.jsonl records stay valid."""
    rounded = tuple(_round_sig(float(v)) for v in np.asarray(x, dtype=float).reshape(-1))
    payload = {"x": rounded, "sites": sorted(sites), "resolution": resolution}
    if case != "case1":
        payload["case"] = case
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _load_jax_daily_cycle():
    """Import gpu_sweep's jax_daily_cycle by sys.path-inserting gpu_sweep/ (it isn't an
    installed package). Lazy, so tests that never touch real physics need no jax/diffrax."""
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
    from solar_lumped.economics import FAIL_LCO

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
    """Append-only jsonl ledger of completed evaluations keyed by design-vector hash, so an
    interrupted run resumes without re-paying for finished evaluations."""

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


def _run_jax_year(
    instance_profiles: list,
    instance_configs: list,
    owner: list[tuple[int, int]],
    *,
    initial_loading,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], str | None]:
    """Run a full 365-day year for every (design, site) instance and reduce to mean daily
    yield/eta per pair. Returns (yield, eta, error); on a raised jax/diffrax call, error is
    set and both dicts are empty.

    The batch axis is (design, site): every instance advances through the year in lockstep,
    one vmapped step per calendar day, so days stay sequential (each warm-starts from the
    previous day's end state) while designs and sites run in parallel."""
    if not instance_profiles:
        return {}, {}, None

    try:
        jdc = _load_jax_daily_cycle()
        dt, n_abs_max, n_des_max = jdc.year_padding(instance_profiles)
        device = jdc.build_device_arrays(instance_configs)
        step_fn = jdc.make_year_step_fn(device, dt, n_abs_max, n_des_max)

        # Instances can disagree on year length (leap years, gaps in the weather record);
        # truncate to the shortest so every day is a full batch.
        n_days = min(len(p) for p in instance_profiles)
        day_weathers = [
            jdc.build_day_weather([p[d] for p in instance_profiles], n_abs_max, n_des_max)
            for d in range(n_days)
        ]
        water, eta = jdc.run_year_batched(
            step_fn, day_weathers,
            c_w_initial=np.array([initial_loading(c) for c in instance_configs]),
            h_initial=np.array([c.hydrogel_thickness_m for c in instance_configs]),
            aitken_max_rounds=JAX_AITKEN_MAX_ROUNDS,
        )
    except Exception as exc:  # noqa: BLE001 -- the batched jax/diffrax call can raise
        return {}, {}, str(exc).split("\n", 1)[0][:240]

    yield_by_pair = {pair: float(y) for pair, y in zip(owner, water)}
    eta_by_pair = {pair: float(e) for pair, e in zip(owner, eta)}
    return yield_by_pair, eta_by_pair, None


def evaluate_batch(
    xs: list[np.ndarray],
    *,
    cache: EvalCache,
    sites: tuple[SiteSpec, ...],
    site_profiles: dict[str, DailyProfiles],
    econ,
    combine_rule: CombineRule = "mean",
    resolution: str = "annual",
    case: str = "case2",
) -> list[DesignEvalResult]:
    """Evaluate every x in *xs* not already in *cache*. Each (design, site) instance runs
    all 365 real days, batched across designs and sites. ``case`` picks the IR emissivity
    variant (design_space.CASE_EPS_IR); "case2" matches solar_lumped base-case physics."""
    from solar_lumped.economics import FAIL_LCO, lcow_from_daily_yield
    from solar_lumped.physics import initial_loading
    from solar_lumped.simulation import DeviceConfig

    site_names = tuple(s.name for s in sites)
    keys = [design_vector_hash(x, sites=site_names, resolution=resolution, case=case) for x in xs]
    results: list[DesignEvalResult | None] = [None] * len(xs)

    to_run = [i for i, key in enumerate(keys) if cache.get_or_none(key) is None]
    for i, key in enumerate(keys):
        cached = cache.get_or_none(key)
        if cached is not None:
            results[i] = cached

    if not to_run:
        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    configs = {i: DeviceConfig(**design_space.to_device_config_kwargs(xs[i], case=case)) for i in to_run}

    # One instance per (design, site); each carries that site's full list of day profiles.
    instance_profiles: list[list] = []
    instance_configs = []
    owner: list[tuple[int, int]] = []  # (design index into xs, site index)
    no_weather: set[tuple[int, int]] = set()
    for i in to_run:
        for si, spec in enumerate(sites):
            profiles = site_profiles[spec.name]
            if not profiles:
                no_weather.add((i, si))
                continue
            instance_profiles.append([prof for _doy, prof in profiles])
            instance_configs.append(configs[i])
            owner.append((i, si))

    t0 = time.perf_counter()
    yield_by_pair, eta_by_pair, batch_error = _run_jax_year(
        instance_profiles, instance_configs, owner, initial_loading=initial_loading,
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
