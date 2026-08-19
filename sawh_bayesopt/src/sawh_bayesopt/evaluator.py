"""Cached combined_lcow(design_vector), over either physics backend.

"jax" is gpu_sweep's daily-cycle + Aitken pipeline: every uncached design's (site,
day) instances stack into one jax.vmap-compiled call, which is what makes global
sweeps affordable. "cpu" is solar_lumped's own ODE path -- sequential, no GPU stack,
and single-site only -- the reference for studying one location closely.

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
#   "cpu" -- solar_lumped's own ODE path. Single-site only (one simulation at a time in
#            one location). Sequential, no GPU stack required, and the
#            reference implementation when a single site is being studied closely.
Backend = Literal["jax", "cpu"]

# Finite stand-in for FAIL_LCO (1e30), which would wreck GP hyperparameter fitting;
# still far worse than any real design (LCOW is typically single/low-double-digit).
PENALTY_LCOW_USD_PER_M3: float = 1.0e4

# Fixed-round-count Aitken convergence: <0.03% worst-per-month cost vs. adaptive
# (FINDINGS.md Result 7), and run_gpu_sweep.py's own default.
JAX_AITKEN_MAX_ROUNDS: int = 8

# A year is ~366 sequential day-steps at ~10s each on an A100 (the batch axis is where
# the parallelism lives, so widening it barely moves this), and run_year_batched reaches
# disk only when the whole year returns. Without a day counter an hour-long call is
# indistinguishable from a hung one -- which is exactly how two Sherlock runs got stared
# at. 30 days is ~12 lines per call: enough to read a s/day rate off, not enough to bury
# the round-level progress.
JAX_YEAR_PROGRESS_EVERY: int = 30

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
    day_stride: int = 1,
) -> str:
    """Stable cache key: sig-fig-rounded design vector + sites (+ case when non-default),
    so LHS/EI float jitter doesn't cause spurious misses. The ``case`` default must stay
    in sync with evaluate_batch's, and "case1" is omitted from the payload so
    pre-case-awareness cache.jsonl records stay valid.

    The resolution tag is a fixed objective-version marker: pre-refactor mean-day entries
    (tagged "monthly") can never collide with a full-year walk. ``day_stride`` > 1 joins
    that tag because it is a *different objective*, not a cheaper route to the same number
    -- a mean over every 5th day, with the sorbent state chain stepping 5 days at a time.
    Stride 1 is omitted, so every existing cache.jsonl entry stays valid.
    """
    rounded = tuple(_round_sig(float(v)) for v in np.asarray(x, dtype=float).reshape(-1))
    # "annual+elev" retires every pre-elevation entry. Site pressure changes yield by
    # ~+2.4%/1000 m, and h_amb's density factor moves even sea-level sites (it tracks
    # ambient temperature as well as pressure), so those results are not comparable.
    resolution = "annual+elev" if day_stride == 1 else f"annual+elev+stride{int(day_stride)}"
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
    owner: list[int],
    *,
    initial_loading,
    complex_mode: bool = False,
    condenser_tracks_ambient: bool = False,
    instant_equilibrium: bool = False,
) -> tuple[dict[int, float], dict[int, float], str | None]:
    """Run a full 365-day year for every instance and reduce to mean daily yield/eta,
    keyed by the caller's ``owner`` index. Returns (yield, eta, error); on a raised
    jax/diffrax call, error is set and both dicts are empty.

    Every instance advances through the year in lockstep, one vmapped step per calendar
    day, so days stay sequential (each warm-starts from the previous day's end state)
    while instances run in parallel. Days are the sequential axis and instances the
    parallel one, which is why widening the batch is nearly free -- see
    :func:`evaluate_requests`."""
    if not instance_profiles:
        return {}, {}, None

    try:
        jdc = _load_jax_daily_cycle()
        dt, n_abs_max, n_des_max = jdc.year_padding(instance_profiles)
        system = jdc.build_system_arrays(
            instance_configs, complex_mode=complex_mode,
            instant_equilibrium=instant_equilibrium,
        )
        step_fn = jdc.make_year_step_fn(
            system, dt, n_abs_max, n_des_max, complex_mode=complex_mode,
            condenser_tracks_ambient=condenser_tracks_ambient,
            instant_equilibrium=instant_equilibrium,
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
            progress_every=JAX_YEAR_PROGRESS_EVERY,
        )
    except _BUG_EXCEPTIONS:
        raise
    except Exception as exc:  # noqa: BLE001 -- the batched jax/diffrax call can raise
        return {}, {}, str(exc).split("\n", 1)[0][:240]

    return (
        {i: float(y) for i, y in zip(owner, water)},
        {i: float(e) for i, e in zip(owner, eta)},
        None,
    )


def fetch_site_inputs(cfg) -> tuple[dict, dict[str, float]]:
    """(frames, elevations), each keyed by site name, for a run config.

    Neither fidelity can share profiles across a batch any more: A1's schedule offsets
    move the day/night split and POA transposition puts tilt in the profile, and all three
    are optimized in both modes, so profiles are rebuilt per design point from these
    frames (see :func:`_profiles_for_design`).

    Elevations ride along rather than being a separate fetch because they must not be
    forgotten: leaving them out silently runs the site at sea level, which changes every
    air property in the gaps and the h_amb density derate. Off the same cached frames, so
    it costs no extra request.
    """
    from sawh_bayesopt.sites import fetch_site_frame
    from solar_lumped.weather import site_elevation_m

    frames = {s.name: fetch_site_frame(s, cache_dir=cfg.weather_cache_dir) for s in cfg.sites}
    return frames, {name: site_elevation_m(df) for name, df in frames.items()}


def evaluate_for_config(
    requests: list[tuple[SiteSpec, np.ndarray]],
    *,
    cfg,
    caches: dict[str, EvalCache],
    econ,
    site_inputs: tuple[dict, dict[str, float]] | None = None,
) -> list[DesignEvalResult]:
    """:func:`evaluate_requests` with every mode argument taken from ``cfg``.

    The loop, the verification pass, and the baseline comparison must all evaluate designs
    identically -- each place that assembled these kwargs by hand was a place to silently
    drop complex_mode or backend and score a 13-dim design against 5-dim physics, or (as
    ``site_elevations`` did) run the verification neighbours at sea level while the loop
    they are compared against ran at real site pressure.

    ``site_inputs`` is ``fetch_site_inputs``' (frames, elevations); pass it when the caller
    already has them, which also keeps a many-site pass from re-reading every frame.
    """
    frames, elevations = fetch_site_inputs(cfg) if site_inputs is None else site_inputs
    return evaluate_requests(
        requests,
        caches=caches,
        econ=econ,
        case=cfg.case,
        complex_mode=cfg.complex_mode,
        condenser_tracks_ambient=cfg.condenser_tracks_ambient,
        instant_equilibrium=cfg.instant_equilibrium,
        site_frames=frames,
        site_elevations=elevations,
        backend=cfg.backend,
        day_stride=cfg.day_stride,
    )


def _profiles_for_design(df, x, complex_mode: bool, day_stride: int = 1) -> DailyProfiles:
    """Rebuild a site's per-day profiles under one design's profile-level variables.

    Design variables live inside the weather profile itself: A1's seal/open offsets move
    the day/night split and POA transposition makes ``tilt_deg`` drive solar gain rather
    than only the Hollands ``cos(theta)`` (both fidelities), and in complex mode B4's
    forced condenser air fills the ``h_amb_cond`` channel. So profiles are
    design-dependent and the fetch-once-per-site reuse does not apply -- this rebuilds
    them from the site's cached DataFrame, which is the expensive-but-correct option.

    ``day_stride`` > 1 keeps every Nth calendar day, which is the one lever that shortens
    the year walk -- the sequential day loop is ~100% of an evaluation's cost. It changes
    the objective (see :func:`design_vector_hash`): the mean is over the sampled days, and
    each sampled day warm-starts from the previous *sampled* day's end state, so the
    sorbent's seasonal history advances in N-day jumps. Cheap because the cyclic state
    re-equilibrates within about a day, not free.

    ponytail: one full pandas re-split per (design, site), ~365 groupby days of work that
    only the day/night mask actually depends on. If the simple-mode loop gets weather-
    bound, the upgrade is a re-split that reuses the already-parsed day frames.
    """
    from solar_lumped.weather import real_weather_days_from_df

    days = real_weather_days_from_df(
        df, stride=int(day_stride),
        **design_space.to_profile_kwargs(x, complex_mode=complex_mode),
    )
    return [(d.timetuple().tm_yday, prof) for d, prof, _group in days]


def _run_cpu_year(
    instance_profiles: list,
    instance_configs: list,
    owner: list[int],
    *,
    initial_loading,
    complex_mode: bool = False,
    condenser_tracks_ambient: bool = False,
    instant_equilibrium: bool = False,
) -> tuple[dict[int, float], dict[int, float], str | None]:
    """CPU equivalent of :func:`_run_jax_year`. ``condenser_tracks_ambient`` and
    ``instant_equilibrium`` are unused here (already baked into each ``instance_configs``
    entry) -- kept for signature parity with
    :func:`_run_jax_year`, which needs it explicitly to build the JAX vector field.

    The JAX fast path is LiCl-hardcoded (``water_activity_licl_from_c_w``,
    ``_XI_MAX_LICL``) and knows nothing about glazing stacks or ZSR blends, so
    complex mode runs solar_lumped's own ODE path instead. That costs the vmap
    batching: instances run sequentially, days chained the same way -- steady
    periodic state on day 1, then each day warm-starting from the last.
    """
    from solar_lumped.simulation import find_cyclic_state, run_daily_cycle

    yield_by_instance: dict[int, float] = {}
    eta_by_instance: dict[int, float] = {}
    for profiles, config, i in zip(instance_profiles, instance_configs, owner):
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
        yield_by_instance[i] = float(np.mean(yields))
        eta_by_instance[i] = float(np.mean(etas))
    return yield_by_instance, eta_by_instance, None


def evaluate_batch(
    xs: list[np.ndarray],
    *,
    cache: EvalCache,
    sites: tuple[SiteSpec, ...],
    econ,
    case: str = "case2",
    complex_mode: bool = False,
    condenser_tracks_ambient: bool = False,
    instant_equilibrium: bool = False,
    site_frames: dict[str, object] | None = None,
    site_elevations: dict[str, float] | None = None,
    backend: Backend = "jax",
    day_stride: int = 1,
) -> list[DesignEvalResult]:
    """Every x in *xs* at the one site in *sites*, in one batched call.

    Single-site wrapper over :func:`evaluate_requests`; optimization is single-site (see
    :func:`site_lcow_or_penalty`), so there is no cross product to take. Sweeping many
    sites means many requests, not many sites per design -- see
    ``bayesopt.run_bayesopt_sites``.
    """
    if len(sites) != 1:
        raise ValueError(f"evaluate_batch is single-site, got {len(sites)}; use evaluate_requests")
    site = sites[0]
    return evaluate_requests(
        [(site, x) for x in xs],
        caches={site.name: cache},
        econ=econ,
        case=case,
        complex_mode=complex_mode,
        condenser_tracks_ambient=condenser_tracks_ambient,
        instant_equilibrium=instant_equilibrium,
        site_frames=site_frames,
        site_elevations=site_elevations,
        backend=backend,
        day_stride=day_stride,
    )


def evaluate_requests(
    requests: list[tuple[SiteSpec, np.ndarray]],
    *,
    # site name -> that site's cache.jsonl. One per site, because each site owns a run
    # directory; a request whose site has no cache here is an error, not a cache miss.
    caches: dict[str, EvalCache],
    econ,
    case: str = "case2",
    complex_mode: bool = False,
    condenser_tracks_ambient: bool = False,
    instant_equilibrium: bool = False,
    # name -> raw year weather frame. Required: profiles are rebuilt per design point in
    # both fidelities, since the schedule offsets and POA tilt are optimized dims that
    # live in the profile. Both come from fetch_site_inputs, which resolves them together
    # so a caller cannot take the frames and forget the elevations.
    site_frames: dict[str, object] | None = None,
    # name -> elevation (m). None runs every site at sea level, which changes every gap
    # air property and the h_amb density derate -- only test paths should want that.
    site_elevations: dict[str, float] | None = None,
    backend: Backend = "jax",
    # 1 (default) walks every calendar day. >1 keeps every Nth, which shortens the
    # sequential day loop -- the only part of an evaluation that costs anything -- at the
    # price of a different objective. It joins the cache key for that reason.
    day_stride: int = 1,
) -> list[DesignEvalResult]:
    """Evaluate one (site, design) request per element, returning results in that order.

    **The batch axis is the request list.** Every uncached request becomes one instance in
    a single vmapped call, so 400 requests cost barely more than 8: a year is ~366
    *sequential* day-steps whatever the width (measured on an A100: 1 design 60.1 min,
    8 designs 68.2 min -- 8x the work for +13% time), which makes width nearly free and
    the number of calls the thing that matters. Batch as wide as the caller can.

    Requests are pairs, not a cross product: each site is optimized on its own LCOW, and
    the lockstep sweep gives every site its own designs. ``case`` picks the IR emissivity
    variant (design_space.CASE_EPS_IR); "case2" matches solar_lumped base-case physics.
    """
    from solar_lumped.economics import FAIL_LCO, lcow_from_daily_yield
    from solar_lumped.physics import initial_loading
    from solar_lumped.simulation import SystemConfig

    # Complex results are not interchangeable with simple ones at the same design
    # vector, so the mode joins the cache key rather than silently colliding.
    key_case = f"{case}+complex" if complex_mode else case
    if condenser_tracks_ambient:
        key_case = f"{key_case}+ambient_cond"
    if instant_equilibrium:
        key_case = f"{key_case}+instant_eq"
    if backend != "jax":
        key_case = f"{key_case}+{backend}"

    missing = {spec.name for spec, _x in requests} - set(caches)
    if missing:
        raise ValueError(f"no cache for site(s) {sorted(missing)}")
    if backend == "cpu" and len({spec.name for spec, _x in requests}) > 1:
        # The CPU path is the single-location physics reference: one simulation at a
        # time, one site. Multi-site/global sweeping is the JAX backend's job.
        raise ValueError("cpu backend is single-site only")

    keys = [
        design_vector_hash(x, sites=(spec.name,), case=key_case, day_stride=day_stride)
        for spec, x in requests
    ]
    results: list[DesignEvalResult | None] = [
        caches[spec.name].get_or_none(key) for (spec, _x), key in zip(requests, keys)
    ]

    # Deduplicated by (site, key): the same design can legitimately appear twice in one
    # request list (a verification neighbour that lands on an evaluated point), and paying
    # for it twice inside one call would be silent waste.
    to_run: dict[tuple[str, str], int] = {}
    for i, ((spec, _x), key) in enumerate(zip(requests, keys)):
        if results[i] is None:
            to_run.setdefault((spec.name, key), i)
    if not to_run:
        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    if site_frames is None:
        raise ValueError("evaluate_requests needs site_frames (profiles are per-design)")

    # One instance per uncached request. Profiles are per (design, site) because the
    # schedule offsets and POA tilt reshape them; the config is per (design, site) too,
    # because elevation is a site property and a shared config would run every site at
    # whichever elevation happened to be on it.
    instance_profiles: list[list] = []
    instance_configs: list = []
    owner: list[int] = []          # request index for each instance
    configs: dict[int, object] = {}  # request index -> its SystemConfig
    no_weather: set[int] = set()
    for i in to_run.values():
        spec, x = requests[i]
        configs[i] = SystemConfig(
            **design_space.to_system_config_kwargs(
                x, case=case, complex_mode=complex_mode,
                condenser_tracks_ambient=condenser_tracks_ambient,
                instant_equilibrium=instant_equilibrium,
            )
        )
        profiles = _profiles_for_design(site_frames[spec.name], x, complex_mode, day_stride)
        if not profiles:
            no_weather.add(i)
            continue
        instance_profiles.append([prof for _doy, prof in profiles])
        elev = 0.0 if site_elevations is None else site_elevations.get(spec.name, 0.0)
        instance_configs.append(dataclasses.replace(configs[i], site_elevation_m=elev))
        owner.append(i)

    run_year = _run_cpu_year if backend == "cpu" else _run_jax_year
    t0 = time.perf_counter()
    yield_by_instance, eta_by_instance, batch_error = run_year(
        instance_profiles, instance_configs, owner,
        initial_loading=initial_loading, complex_mode=complex_mode,
        condenser_tracks_ambient=condenser_tracks_ambient,
        instant_equilibrium=instant_equilibrium,
    )
    wall = time.perf_counter() - t0

    computed: dict[tuple[str, str], DesignEvalResult] = {}
    for i in to_run.values():
        spec, x = requests[i]
        cfg = configs[i]
        if i in no_weather:
            site_result = SiteResult(spec.name, FAIL_LCO, False, "no weather profiles", float("nan"), float("nan"))
        elif batch_error is not None:
            site_result = SiteResult(spec.name, FAIL_LCO, False, batch_error, float("nan"), float("nan"))
        else:
            mean_yield = yield_by_instance[i]
            mean_eta = eta_by_instance[i]
            if not math.isfinite(mean_yield) or mean_yield <= 0.0:
                site_result = SiteResult(spec.name, FAIL_LCO, False, "zero or invalid yield", mean_yield, mean_eta)
            else:
                lcow = lcow_from_daily_yield(
                    mean_yield,
                    salt_name=cfg.salt_name,
                    salt_loading=cfg.salt_loading,
                    hydrogel_thickness_m=cfg.hydrogel_thickness_m,
                    econ=econ,
                    # Complex mode prices the design itself: B1/B2/B3/B4 move the BOM and
                    # B8 sets the blended salt price, so LCOW stops being yield-only.
                    complex_options=cfg.complex,
                    fin_area_ratio=cfg.fin_area_ratio if cfg.complex is not None else None,
                )
                if not math.isfinite(lcow) or lcow >= 0.99 * FAIL_LCO:
                    site_result = SiteResult(spec.name, FAIL_LCO, False, "invalid LCOW", mean_yield, mean_eta)
                else:
                    site_result = SiteResult(spec.name, lcow, True, "", mean_yield, mean_eta)

        result = DesignEvalResult(
            design_vector=tuple(float(v) for v in np.asarray(x, dtype=float).reshape(-1)),
            site_results=(site_result,),
            combined_lcow=site_lcow_or_penalty((site_result,)),
            wall_time_s=wall,
        )
        computed[(spec.name, keys[i])] = result
        caches[spec.name].put(keys[i], result)

    for i, ((spec, _x), key) in enumerate(zip(requests, keys)):
        if results[i] is None:
            results[i] = computed[(spec.name, key)]

    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]
