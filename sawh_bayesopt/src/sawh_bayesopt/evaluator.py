"""Cached combined_lcow(design_vector), over either physics backend.

"jax" is gpu_sweep's daily-cycle + Aitken pipeline: every uncached design's (site,
day) instances stack into one jax.vmap-compiled call, which is what makes global
sweeps affordable. "cpu" is solar_lumped's own ODE path -- sequential, no GPU stack,
and the reference when a single site is being studied closely.

Both backends implement simple *and* complex fidelity, and agree to <0.03% simple /
<0.5% complex (FINDINGS.md 6/7, solar_lumped/tests/test_cpu_jax_parity.py)."""

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

# Which physics backend evaluates a design. Both implement simple *and* complex
# fidelity and agree to <0.5% on every configuration
# (solar_lumped/tests/test_cpu_jax_parity.py):
#   "jax" -- gpu_sweep's vmapped/jitted path. Batches every (design, site) instance
#            into one call, so it is the backend for global sweeps. Needs jax+diffrax.
#   "cpu" -- solar_lumped's own ODE path. Sequential, no GPU stack required, and the
#            reference implementation when a single site is being studied closely.
Backend = Literal["jax", "cpu"]

# Finite stand-in for FAIL_LCO (1e30), which would wreck GP hyperparameter fitting;
# still far worse than any real design (LCOW is typically single/low-double-digit).
PENALTY_LCOW_USD_PER_M3: float = 1.0e4

# Fixed-round-count Aitken convergence: <0.03% worst-per-month cost vs. adaptive
# (FINDINGS.md Result 7), and run_gpu_sweep.py's own default.
JAX_AITKEN_MAX_ROUNDS: int = 8

# Broken code, not a bad design. The two solver calls below catch Exception so one
# unphysical design can't kill a 130-evaluation batch -- but that same catch turns a typo
# in the physics into "every design is infeasible", and the run then reports the 1e4
# penalty as a completed optimization. That is exactly how a missing _M_DES_BRACKET_MAX
# import in jax_physics.py produced a full sweep of penalties that looked like a
# successful run. These never mean "infeasible", so they propagate and stop the run.
_BUG_EXCEPTIONS = (NameError, AttributeError, ImportError, IndentationError)

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
        """True iff the site succeeded. A property so it always tracks site_results and
        old cache records get it free. combined_lcow can't substitute: it maps a failure
        onto the penalty, which is indistinguishable from a real but terrible LCOW."""
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
    case: str = "case2",
) -> str:
    """Stable cache key: sig-fig-rounded design vector + sites (+ case when non-default),
    so LHS/EI float jitter doesn't cause spurious misses. The ``case`` default must stay
    in sync with evaluate_batch's, and "case1" is omitted from the payload so
    pre-case-awareness cache.jsonl records stay valid.

    The "annual" resolution tag is a fixed objective-version marker: every evaluation
    walks the real 365-day year, so pre-refactor mean-day entries (tagged "monthly")
    can never collide with these.
    """
    rounded = tuple(_round_sig(float(v)) for v in np.asarray(x, dtype=float).reshape(-1))
    payload = {"x": rounded, "sites": sorted(sites), "resolution": "annual"}
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


def site_lcow_or_penalty(
    site_results: tuple[SiteResult, ...],
    *,
    penalty: float = PENALTY_LCOW_USD_PER_M3,
) -> float:
    """The one site's LCOW, or the penalty if it failed.

    Optimization is single-site: a design is scored where it will be built, not against
    an average of climates it will never see. Multi-site studies run one optimization
    per site (see gpu_sweep/run_bayesopt_sweep.py, which loops the land grid) and
    compare the resulting per-site optima.
    """
    from solar_lumped.economics import FAIL_LCO

    if len(site_results) != 1:
        raise ValueError(f"single-site optimization expects exactly 1 site, got {len(site_results)}")
    r = site_results[0]
    return float(penalty if (not r.feasible or r.lcow >= 0.99 * FAIL_LCO) else r.lcow)


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
    complex_mode: bool = False,
    condenser_tracks_ambient: bool = False,
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
        system = jdc.build_system_arrays(instance_configs, complex_mode=complex_mode)
        step_fn = jdc.make_year_step_fn(
            system, dt, n_abs_max, n_des_max, complex_mode=complex_mode,
            condenser_tracks_ambient=condenser_tracks_ambient,
        )

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
    except _BUG_EXCEPTIONS:
        raise
    except Exception as exc:  # noqa: BLE001 -- the batched jax/diffrax call can raise
        return {}, {}, str(exc).split("\n", 1)[0][:240]

    yield_by_pair = {pair: float(y) for pair, y in zip(owner, water)}
    eta_by_pair = {pair: float(e) for pair, e in zip(owner, eta)}
    return yield_by_pair, eta_by_pair, None


def fetch_site_inputs(cfg) -> tuple[dict, dict | None]:
    """(site_profiles, site_frames) for a run config.

    Complex mode rebuilds profiles per design point -- A1's schedule offsets, B4's
    condenser air, and POA tilt all live *in* the profile -- so it needs the raw
    frames and the usual fetch-once-per-site reuse does not apply.
    """
    from sawh_bayesopt.sites import fetch_daily_profiles, fetch_site_frame

    if cfg.complex_mode:
        return {}, {
            s.name: fetch_site_frame(s, cache_dir=cfg.weather_cache_dir) for s in cfg.sites
        }
    return {
        s.name: fetch_daily_profiles(s, cache_dir=cfg.weather_cache_dir) for s in cfg.sites
    }, None


def evaluate_for_config(xs, *, cfg, cache, econ, site_profiles=None, site_frames=None):
    """evaluate_batch with every mode argument taken from ``cfg``.

    The loop, the verification pass, and the baseline comparison must all evaluate
    designs identically -- each one that assembled these kwargs by hand was a place
    to silently drop complex_mode or backend and score a 13-dim design against
    6-dim physics. Callers that already fetched weather can pass it through.
    """
    if site_profiles is None and site_frames is None:
        site_profiles, site_frames = fetch_site_inputs(cfg)
    return evaluate_batch(
        xs,
        cache=cache,
        sites=cfg.sites,
        site_profiles=site_profiles,
        econ=econ,
        case=cfg.case,
        complex_mode=cfg.complex_mode,
        condenser_tracks_ambient=cfg.condenser_tracks_ambient,
        site_frames=site_frames,
        backend=cfg.backend,
    )


def _profiles_for_design(df, config) -> DailyProfiles:
    """Rebuild a site's per-day profiles under one design's profile-level variables.

    Complex mode puts three design variables inside the weather profile itself: A1's
    seal/open offsets move the day/night split, B4's forced condenser air fills the
    ``h_amb_cond`` channel, and POA transposition makes ``tilt_deg`` drive solar gain
    instead of only the Hollands ``cos(theta)``. So profiles become design-dependent
    and the fetch-once-per-site reuse no longer applies -- this rebuilds them from
    the site's cached DataFrame, which is the expensive-but-correct option.
    """
    from solar_lumped.weather import real_weather_days_from_df

    cx = config.complex
    days = real_weather_days_from_df(
        df,
        seal_offset_h=cx.seal_offset_h,
        open_offset_h=cx.open_offset_h,
        condenser_air_speed_m_s=cx.condenser_air_speed_m_s,
        poa_tilt_deg=config.tilt_deg,
    )
    return [(d.timetuple().tm_yday, prof) for d, prof, _group in days]


def _run_cpu_year(
    instance_profiles: list,
    instance_configs: list,
    owner: list[tuple[int, int]],
    *,
    initial_loading,
    complex_mode: bool = False,
    condenser_tracks_ambient: bool = False,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], str | None]:
    """CPU equivalent of :func:`_run_jax_year`. ``condenser_tracks_ambient`` is unused here
    (already baked into each ``instance_configs`` entry) -- kept for signature parity with
    :func:`_run_jax_year`, which needs it explicitly to build the JAX vector field.

    The JAX fast path is LiCl-hardcoded (``water_activity_licl_from_c_w``,
    ``_XI_MAX_LICL``) and knows nothing about glazing stacks or ZSR blends, so
    complex mode runs solar_lumped's own ODE path instead. That costs the vmap
    batching: instances run sequentially, days chained the same way -- steady
    periodic state on day 1, then each day warm-starting from the last.
    """
    from solar_lumped.simulation import find_cyclic_state, run_daily_cycle

    yield_by_pair: dict[tuple[int, int], float] = {}
    eta_by_pair: dict[tuple[int, int], float] = {}
    for profiles, config, pair in zip(instance_profiles, instance_configs, owner):
        if not profiles:
            continue
        try:
            c_w, h = find_cyclic_state(
                profiles[0], config, max_rounds=JAX_AITKEN_MAX_ROUNDS, verbose=False
            )
            yields, etas = [], []
            for prof in profiles:
                y, eta, _abs_res, des_res = run_daily_cycle(
                    prof, config, c_w_initial=c_w, h_initial=h
                )
                yields.append(y)
                etas.append(eta)
                c_w, h = float(des_res.c_w[-1]), float(des_res.H[-1])
        except _BUG_EXCEPTIONS:
            raise
        except Exception as exc:  # noqa: BLE001 -- one bad design must not kill the batch
            return {}, {}, str(exc).split("\n", 1)[0][:240]
        yield_by_pair[pair] = float(np.mean(yields))
        eta_by_pair[pair] = float(np.mean(etas))
    return yield_by_pair, eta_by_pair, None


def evaluate_batch(
    xs: list[np.ndarray],
    *,
    cache: EvalCache,
    sites: tuple[SiteSpec, ...],
    site_profiles: dict[str, DailyProfiles],
    econ,
    case: str = "case2",
    complex_mode: bool = False,
    condenser_tracks_ambient: bool = False,
    site_frames: dict[str, object] | None = None,
    backend: Backend = "jax",
) -> list[DesignEvalResult]:
    """Evaluate every x in *xs* not already in *cache*. Each (design, site) instance runs
    all 365 real days, batched across designs and sites. ``case`` picks the IR emissivity
    variant (design_space.CASE_EPS_IR); "case2" matches solar_lumped base-case physics."""
    from solar_lumped.economics import FAIL_LCO, lcow_from_daily_yield
    from solar_lumped.physics import initial_loading
    from solar_lumped.simulation import SystemConfig

    site_names = tuple(s.name for s in sites)
    # Complex results are not interchangeable with simple ones at the same design
    # vector, so the mode joins the cache key rather than silently colliding.
    key_case = f"{case}+complex" if complex_mode else case
    if condenser_tracks_ambient:
        key_case = f"{key_case}+ambient_cond"
    if backend != "jax":
        key_case = f"{key_case}+{backend}"
    keys = [design_vector_hash(x, sites=site_names, case=key_case) for x in xs]
    results: list[DesignEvalResult | None] = [None] * len(xs)

    to_run = [i for i, key in enumerate(keys) if cache.get_or_none(key) is None]
    for i, key in enumerate(keys):
        cached = cache.get_or_none(key)
        if cached is not None:
            results[i] = cached

    if not to_run:
        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    configs = {
        i: SystemConfig(
            **design_space.to_system_config_kwargs(
                xs[i], case=case, complex_mode=complex_mode,
                condenser_tracks_ambient=condenser_tracks_ambient,
            )
        )
        for i in to_run
    }

    # One instance per (design, site); each carries that site's full list of day profiles.
    instance_profiles: list[list] = []
    instance_configs = []
    owner: list[tuple[int, int]] = []  # (design index into xs, site index)
    no_weather: set[tuple[int, int]] = set()
    for i in to_run:
        for si, spec in enumerate(sites):
            # Complex mode's A1 schedule shift, B4 condenser air, and POA tilt are all
            # design variables that live in the *weather profile*, so profiles cannot be
            # built once per site and shared -- they are rebuilt per design point.
            if complex_mode:
                if site_frames is None:
                    raise ValueError("complex_mode requires site_frames (profiles are per-design)")
                profiles = _profiles_for_design(site_frames[spec.name], configs[i])
            else:
                profiles = site_profiles[spec.name]
            if not profiles:
                no_weather.add((i, si))
                continue
            instance_profiles.append([prof for _doy, prof in profiles])
            instance_configs.append(configs[i])
            owner.append((i, si))

    run_year = _run_cpu_year if backend == "cpu" else _run_jax_year
    t0 = time.perf_counter()
    yield_by_pair, eta_by_pair, batch_error = run_year(
        instance_profiles, instance_configs, owner,
        initial_loading=initial_loading, complex_mode=complex_mode,
        condenser_tracks_ambient=condenser_tracks_ambient,
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
                # Complex mode prices the design itself: B1/B2/B3/B4 move the BOM and
                # B8 sets the blended salt price, so LCOW stops being yield-only.
                complex_options=cfg.complex,
                fin_area_ratio=cfg.fin_area_ratio if cfg.complex is not None else None,
            )
            if not math.isfinite(lcow) or lcow >= 0.99 * FAIL_LCO:
                site_results.append(SiteResult(spec.name, FAIL_LCO, False, "invalid LCOW", mean_yield, mean_eta))
                continue

            site_results.append(SiteResult(spec.name, lcow, True, "", mean_yield, mean_eta))

        combined = site_lcow_or_penalty(tuple(site_results))
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
