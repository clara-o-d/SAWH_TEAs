#!/usr/bin/env python3
"""The one parameter sweep. A JSON config says which model, which parameters, which sweep
method and which metrics; this runs it and writes a CSV that ``plot_tornado.py`` reads.

Config::

    {
      "model":   "solar" | "waste_heat",
      "method":  "oat" | "grid" | "lhs",
      "n_levels": 5,                       // points per parameter (oat, grid)
      "n_samples": 200,                    // lhs only
      "seed": 0,                           // lhs only
      "site":    {"lat": 42.36, "lon": -71.09, "year": 2024},   // omit for the baseline profile
      "metrics": ["lcow_usd_per_m3", "daily_yield_kg_m2"],      // omit for all available
      "parameters": {
        "hydrogel_thickness_mm": {"lo": 2.0, "hi": 6.0},
        "vapor_gap_mm":          {"lo": 20.0, "hi": 60.0, "baseline": 40.0}
      },
      "output": "sweep.csv"
    }

``oat`` moves one parameter at a time about the baseline and writes long-format rows
(``sweep_param`` / ``param_value``), which is what the elasticity tornado wants. ``grid``
is full-factorial and ``lhs`` is a Latin hypercube; both write wide-format rows, one
column per parameter, which the OAT-pair and regression tornadoes want.

Run: python parameter_sweep.py config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_METHODS = ("oat", "grid", "lhs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--output", type=Path, default=None, help="Override the config's output path")
    ap.add_argument("--dry-run", action="store_true", help="List the points without simulating")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    model_name = cfg.get("model", "solar")
    method = cfg.get("method", "oat")
    if method not in _METHODS:
        sys.exit(f"method must be one of {_METHODS}, got {method!r}")

    model = _MODELS.get(model_name)
    if model is None:
        sys.exit(f"model must be one of {sorted(_MODELS)}, got {model_name!r}")
    model = model()

    params = cfg.get("parameters") or {}
    if not params:
        sys.exit("config needs a non-empty 'parameters' object")
    unknown = [p for p in params if p not in model.knobs]
    if unknown:
        sys.exit(f"unknown parameter(s) for model {model_name!r}: {', '.join(unknown)}. "
                 f"Available: {', '.join(sorted(model.knobs))}")
    spec = {name: _resolve(name, p, model) for name, p in params.items()}

    points = _build_points(method, spec, cfg)
    metrics = cfg.get("metrics") or model.metric_names
    out = args.output or Path(cfg.get("output", f"sweep_{model_name}_{method}.csv"))

    print(f"{model_name} / {method}: {len(points)} point(s) over {len(spec)} parameter(s)")
    if args.dry_run:
        for label, overrides in points[:20]:
            print(f"  {label}: {overrides}")
        if len(points) > 20:
            print(f"  ... {len(points) - 20} more")
        return 0

    site = cfg.get("site")
    rows = []
    for k, (label, overrides) in enumerate(points, start=1):
        values = model.run(overrides, site=site)
        row = {"sweep_param": label[0], "param_label": label[1], "param_value": label[2]} \
            if method == "oat" else dict(overrides)
        row.update({m: values.get(m, float("nan")) for m in metrics})
        rows.append(row)
        print(f"  [{k}/{len(points)}] {label[0] if method == 'oat' else 'point'} "
              f"-> " + "  ".join(f"{m}={values.get(m, float('nan')):.4g}" for m in metrics), flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")
    return 0


def _resolve(name: str, p: dict, model) -> dict:
    """Fill in a parameter's baseline from the model when the config omits it."""
    if "lo" not in p or "hi" not in p:
        sys.exit(f"parameter {name!r} needs 'lo' and 'hi'")
    return {"lo": float(p["lo"]), "hi": float(p["hi"]),
            "baseline": float(p.get("baseline", model.knobs[name]))}


def _build_points(method: str, spec: dict, cfg: dict) -> list:
    """(label, overrides) per simulation. label is (param, pretty, value) for OAT."""
    baseline = {n: s["baseline"] for n, s in spec.items()}
    if method == "oat":
        n = int(cfg.get("n_levels", 5))
        points = [(("baseline", "baseline", 0.0), dict(baseline))]
        for name, s in spec.items():
            for v in np.linspace(s["lo"], s["hi"], n):
                points.append(((name, name, float(v)), {**baseline, name: float(v)}))
        return points
    if method == "grid":
        n = int(cfg.get("n_levels", 5))
        names = list(spec)
        axes = [np.linspace(spec[x]["lo"], spec[x]["hi"], n) for x in names]
        return [(None, {x: float(v) for x, v in zip(names, combo)})
                for combo in _product(axes)]
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n_samples = int(cfg.get("n_samples", 200))
    names = list(spec)
    # Latin hypercube: one stratified sample per parameter, independently shuffled.
    strata = (np.arange(n_samples)[:, None] + rng.random((n_samples, len(names)))) / n_samples
    for j in range(len(names)):
        rng.shuffle(strata[:, j])
    lo = np.array([spec[x]["lo"] for x in names])
    hi = np.array([spec[x]["hi"] for x in names])
    scaled = lo + strata * (hi - lo)
    return [(None, {x: float(v) for x, v in zip(names, row)}) for row in scaled]


def _product(axes):
    import itertools

    return itertools.product(*axes)


class SolarModel:
    """Passive solar hydrogel system (``solar_lumped``)."""

    metric_names = ("daily_yield_kg_m2", "thermal_efficiency", "lcow_usd_per_m3",
                    "npv_usd_per_m2", "payback_years_simple", "payback_years_discounted")

    def __init__(self) -> None:
        from solar_lumped.system import default_solar_sim_args

        self._args_factory = default_solar_sim_args
        base = default_solar_sim_args()
        # Sweepable knobs -> their baseline, taken from the CLI defaults so the two agree.
        self.knobs = {
            "hydrogel_thickness_mm": base.hydrogel_thickness_mm,
            "vapor_gap_mm": base.vapor_gap_mm,
            "insulation_gap_mm": base.insulation_gap_mm,
            "salt_loading": base.salt_loading,
            "tilt_deg": 30.0,
            "fin_area_ratio": 7.1,
        }

    def run(self, overrides: dict, *, site: dict | None) -> dict:
        from solar_lumped.system import resolve_solar_sim_arguments, run_solar_simulation
        from solar_lumped.economics import npv_from_daily_yield

        args = self._args_factory()
        for k, v in overrides.items():
            setattr(args, k, v)
        if site:
            args.weather_mode = "real"
            args.lat, args.lon = float(site["lat"]), float(site["lon"])
            args.year = int(site.get("year", 2024))
        else:
            args.weather_mode = "baseline"
        resolve_solar_sim_arguments(args, argparse.ArgumentParser())
        r = run_solar_simulation(args)

        npv = npv_from_daily_yield(
            r.daily_yield_kg_per_m2, float(site.get("water_price_usd_per_m3", 5.0)) if site else 5.0,
            salt_name=r.config.salt_name, salt_loading=r.config.salt_loading,
            hydrogel_thickness_m=r.config.hydrogel_thickness_m, econ=r.econ, cycles_per_day=1.0,
        )
        nan = float("nan")
        return {
            "daily_yield_kg_m2": r.daily_yield_kg_per_m2,
            "thermal_efficiency": r.thermal_efficiency,
            "lcow_usd_per_m3": r.lcow_usd_per_m3,
            "npv_usd_per_m2": npv.npv_usd_per_m2 if npv else nan,
            "payback_years_simple": npv.payback_years_simple if npv else nan,
            "payback_years_discounted": npv.payback_years_discounted if npv else nan,
        }


class WasteHeatModel:
    """Two-bed waste-heat system with direct coupling (``waste_heat``)."""

    metric_names = ("daily_yield_kg_m2", "thermal_efficiency", "n_cycles_per_day",
                    "lcow_usd_per_m3", "npv_usd_per_m2", "payback_years_simple")

    def __init__(self) -> None:
        from waste_heat.simulation import SystemConfig

        base = SystemConfig()
        self.knobs = {
            "hydrogel_thickness_mm": base.hydrogel_thickness_m * 1e3,
            "vapor_gap_mm": base.vapor_gap_m * 1e3,
            "salt_loading": base.salt_loading,
            "tau_half_min": base.tau_half_s / 60.0,
            "rh_desorber_switch": base.rh_desorber_switch,
            "tilt_deg": base.tilt_deg,
        }
        # knob -> (SystemConfig field, scale from knob units to SI)
        self._fields = {
            "hydrogel_thickness_mm": ("hydrogel_thickness_m", 1e-3),
            "vapor_gap_mm": ("vapor_gap_m", 1e-3),
            "salt_loading": ("salt_loading", 1.0),
            "tau_half_min": ("tau_half_s", 60.0),
            "rh_desorber_switch": ("rh_desorber_switch", 1.0),
            "tilt_deg": ("tilt_deg", 1.0),
        }

    def run(self, overrides: dict, *, site: dict | None) -> dict:
        from waste_heat.economics import LCOEconomicParams, lcow_from_daily_yield, npv_from_daily_yield
        from waste_heat.simulation import SystemConfig, simulate_daily
        from waste_heat.weather import datacenter_baseline_profile

        kwargs = {}
        for knob, value in overrides.items():
            field, scale = self._fields[knob]
            kwargs[field] = value * scale
        cfg = SystemConfig(**kwargs)
        profile = datacenter_baseline_profile(tau_half_s=cfg.tau_half_s)
        econ = LCOEconomicParams()

        r = simulate_daily(profile, cfg)
        cycles = float(r.n_cycles_per_day)
        common = dict(salt_name=cfg.salt_name, salt_loading=cfg.salt_loading,
                      hydrogel_thickness_m=cfg.hydrogel_thickness_m, econ=econ, cycles_per_day=cycles)
        # Per cycle, not per day: mean_daily_yield_kg_m2 is already summed over all
        # n_cycles_per_day cycles, and the economics multiply back up by cycles_per_day.
        # Passing the daily total here counted the water `cycles` times over (18x on the
        # datacenter baseline), understating LCOW by the same factor. cycles_per_day still
        # has to be the true count -- it drives the per-cycle energy term.
        yield_per_cycle = r.mean_daily_yield_kg_m2 / cycles if cycles > 0 else 0.0
        lcow = lcow_from_daily_yield(yield_per_cycle, **common)
        npv = npv_from_daily_yield(yield_per_cycle, 5.0, **common)
        nan = float("nan")
        return {
            "daily_yield_kg_m2": r.mean_daily_yield_kg_m2,
            "thermal_efficiency": r.mean_thermal_efficiency,
            "n_cycles_per_day": cycles,
            "lcow_usd_per_m3": lcow,
            "npv_usd_per_m2": npv.npv_usd_per_m2 if npv else nan,
            "payback_years_simple": npv.payback_years_simple if npv else nan,
        }


_MODELS = {"solar": SolarModel, "waste_heat": WasteHeatModel}


if __name__ == "__main__":
    raise SystemExit(main())
