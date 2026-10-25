#!/usr/bin/env python3
"""Monte Carlo over the parameters.xlsx empirical parameters, running OUR model (the
package defaults) on Wilson et al. (2025)'s two measured test days: the Atacama field
test (Fig. 4) and the Cambridge rooftop test (Fig. 3).

What is Wilson's and what is ours:
  - Wilson's: the weather (their digitized Atacama and Cambridge records, via
    weather.atacama_field_profile and figure3's Cambridge CSVs) and their operation --
    tilt (Atacama 25 deg, Cambridge 35 deg), fin area ratio (Atacama 5, Cambridge 7.1),
    and phase timing (12 h absorption; desorption 8:09 a.m.-4 p.m. Atacama, 10 h
    Cambridge).
  - Wilson's hardware where it differs from our base case: the black-spray-paint
    absorber, IR emissivity 0.95 (Wilson main text; our default is a 0.05 selective
    coating), so the temperatures are comparable to his thermocouples.
  - Not Wilson's: Atacama's h_amb is a constant 10 W/m2K rather than the paper's fitted 1 -> 10
    step at 2 p.m. -- see ATACAMA_H_AMB_W_M2_K.
  - Ours: everything else at its package default -- isosteric h_des, the modified
    (real-emissivity) radiative exchange, the algebraic desorption-start temperatures,
    and the steady periodic gel state (cyclic warm-up, as --weather-mode atacama-replay
    / cambridge-replay do) rather than a cold start from the fabrication state.

The condenser emissivity runs as two scenarios on the SAME draws: dry metal and
condensate-wetted. Within each scenario eps_Al is drawn from that scenario's range at
the draw's own quantile, so dry and wet differ by the condenser surface only.

Each draw is a Latin hypercube point over every sampled parameter's ``Lower/Upper (for
Sweeps)`` bounds in parameters.xlsx, run on both sites x both scenarios. Output is
every run's daily yield in outputs/samples.csv, and every run's desorption-phase time
series (absorber, glass, gel and condenser temperatures, cumulative water) on the
site's weather grid in outputs/timeseries.npz -- enough for P10-P90 bands on the Fig. 3
/ Fig. 4 recreations. plot_monte_carlo.py reads both without re-running solves.

Not sampled -- see the NOT_SAMPLED table below for why.

Examples::

  python run_monte_carlo.py --n 8            # pilot
  python run_monte_carlo.py --n 256 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import openpyxl
from scipy.stats import qmc

import solar_lumped.physics as physics
from solar_lumped import _parameters_xlsx
from solar_lumped._parameters_xlsx import physics_bounds, physics_value
from solar_lumped.simulation import SystemConfig, run_daily_cycle
from solar_lumped.weather import DailyWeatherProfile, atacama_field_profile

_WILSON_SCRIPTS = Path(__file__).resolve().parents[1] / "paper_recreation" / "wilson" / "scripts"
sys.path.insert(0, str(_WILSON_SCRIPTS))
import figure3  # noqa: E402  (Cambridge digitized weather + absorber area)

_OUT_DIR = Path(__file__).resolve().parent / "outputs"

# (key, parameters.xlsx Physics row, log-uniform). Log-uniform where the bounds span
# ~an order of magnitude or more, so the low decade isn't sampled 10x less than the high.
SAMPLED: tuple[tuple[str, str, bool], ...] = (
    ("eps_abs", "Absorber emissivity (eps_abs)", False),
    ("tau_glass", "Glass transmittance (tau_glass)", False),
    ("eps_glass_ir", "Glass IR emissivity (eps_glass_ir)", False),
    ("g_chamber", "Chamber convection coefficient, absorption (g_chamber)", False),
    ("eps_gel", "Gel emissivity (eps_gel)", False),
    ("k_gel", "Hydrogel thermal conductivity (k_gel)", False),
    ("rho_al", "Aluminum density (rho_Al)", False),
    ("cp_al", "Aluminum specific heat (cp_Al)", False),
    ("l_c", "Condenser aluminum plate thickness (L_c)", False),
    ("rabs", "Absorber-to-gel constant resistance (Rabs)", True),
    ("rho_dry", "Dry composite density (rho_dry)", False),
)
# h_amb is sampled as a multiplier on each site's h_amb (10 W/m2K at both). +/-25% is
# Wilson Methods' h_amb = 10 +/- 2.5 (the h_amb row's Source); the row's own 1-12.5
# bounds mix that range with the Atacama fit.
H_AMB_SCALE_BOUNDS: tuple[float, float] = (0.75, 1.25)

# Atacama runs at a constant h_amb, not Wilson's fitted 1 -> 10 W/m2K step at 2 p.m.
# (weather.atacama_field_profile). Our model applies one h_amb to glass and finned
# condenser, so at h = 1 the condenser rejects heat almost only by radiation and runs
# ~25 C above the thermocouples (glass ~20 C, absorber ~25 C) until the step. The paper
# itself argues against h = 1 there: Methods put ~0.5 m/s of fan flow on the condenser
# backing, and the test wind was 1-15 mph. At a constant 10 the nominal run tracks all
# three thermocouples to a few C through 2 p.m.
ATACAMA_H_AMB_W_M2_K: float = 10.0

# Condenser (Al) emissivity scenarios -- (lower, upper, nominal). Both inside the
# eps_Al row's 0.04-0.90 bounds; sources are that row's Bound Source. Dry: polished to
# heavily oxidized bare aluminum, nominal the row's Value. Wet: Lavielle et al. (2024)
# effective emissivity under condensation, ~0.8 dropwise to ~0.9 filmwise.
EPS_AL_SCENARIOS: dict[str, tuple[float, float, float]] = {
    "dry": (0.04, 0.20, physics_value("Condenser (Al) emissivity (eps_Al)")),
    "wet": (0.80, 0.90, 0.85),
}

NOT_SAMPLED: dict[str, str] = {
    "Desorption enthalpy, LiCl (h_des)": "our model computes isosteric h_des(xi, T) from the LiCl isotherm; this scalar is only its NaN fallback",
    "Composite (hydrogel) density at 20% RH (rho_gel)": "never read by the model; rho_dry (sampled) is the density that sets c_s",
    "Absorber IR emissivity (eps_abs_ir)": "a coating design choice (0.05 selective to 0.95 black), not an uncertainty; held at Wilson's black paint, ABSORBER_IR_EMISSIVITY",
    "Salt weight factor (dry-mass MW scaling)": "rescales LiCl's exactly-known molar mass; bounds have no source",
    "Uptake RH / ambient RH (RH_amb)": "weather forcing; the replay days supply it",
    "Solar irradiance (Q_solar)": "weather forcing; the replay days supply it",
    "Ambient temperature (T_amb)": "weather forcing; the replay days supply it",
    "Atmospheric pressure (P0)": "weather forcing; both sites sit at sea level on a measured day",
    "Ambient convection density exponent": "multiplies (rho_amb/rho_ref)^n, which is ~1 at these sea-level sites",
    "Aluminum thermal conductivity (k_Al)": "read only by the complex-mode fin-efficiency model",
    "Air thermal conductivity (k_air)": "not read; air properties come from physics.humid_air_props",
    "Water-vapor-in-air diffusivity (D_air)": "not read; physics.d_h2o_air_m2_s (Marrero-Mason)",
    "Air kinematic viscosity (nu_air)": "not read; physics.humid_air_props",
    "Air density (rho_air)": "not read; physics.humid_air_props",
    "Air specific heat (cp_air)": "not read; physics.humid_air_props",
}

# Wilson measured daily yields (L/m2 ~ kg/m2). Atacama: Fig. 4D endpoint. Cambridge:
# "~130 mL total" over the 0.078 m2 absorber (figure3._A_C_M2).
MEASURED_YIELD_KG_M2: dict[str, float] = {"atacama": 0.62, "cambridge": 0.130 / figure3._A_C_M2}
SITES: tuple[str, ...] = ("atacama", "cambridge")
SCENARIOS: tuple[str, ...] = tuple(EPS_AL_SCENARIOS)
# Wilson's absorber is black spray paint, "estimated emissivity eps_abs = 0.95" (main
# text); the eps_abs_ir row's upper bound is the same commodity-black-paint value.
ABSORBER_IR_EMISSIVITY: float = physics_bounds("Absorber IR emissivity (eps_abs_ir)")[1]
# Desorption-phase series saved per run, in this order along timeseries.npz's axis 1.
SERIES: tuple[str, ...] = ("t_abs_c", "t_glass_c", "t_gel_c", "t_cond_c", "water_cumulative_kg_m2")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=256, help="Parameter draws (each run on every site x scenario)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, default=_OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Every empirical row must be either sampled or explained, so a row added to the
    # workbook later can't silently drop out of the analysis. The Type column is read
    # here directly because _parameters_xlsx doesn't load it.
    rows = openpyxl.load_workbook(_parameters_xlsx._XLSX_PATH, read_only=True)["Physics"].iter_rows(values_only=True)
    header = list(next(rows))
    i_name, i_type = header.index("Name"), header.index("Type")
    empirical = {r[i_name].strip() for r in rows if r[i_type] == "Empirical parameter"}
    accounted = {row for _, row, _ in SAMPLED} | set(NOT_SAMPLED) | {
        "Ambient convection coefficient (h_amb)",  # H_AMB_SCALE_BOUNDS
        "Condenser (Al) emissivity (eps_Al)",  # EPS_AL_SCENARIOS
    }
    missing = empirical - accounted
    if missing:
        raise SystemExit(f"Empirical parameters neither sampled nor in NOT_SAMPLED: {sorted(missing)}")

    # eps_al_quantile is the draw's position within whichever scenario range it runs in.
    keys = [k for k, _, _ in SAMPLED] + ["h_amb_scale", "eps_al_quantile"]
    bounds = [physics_bounds(row, mm_to_m=(k == "l_c")) for k, row, _ in SAMPLED] + [H_AMB_SCALE_BOUNDS, (0.0, 1.0)]
    log = [lg for _, _, lg in SAMPLED] + [False, False]
    u = qmc.LatinHypercube(d=len(keys), rng=args.seed).random(args.n)
    lo, hi = np.array(bounds).T
    vals = np.where(
        log,
        np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))),
        lo + u * (hi - lo),
    )
    samples = [dict(zip(keys, row)) for row in vals]
    nominal = {k: physics_value(row, mm_to_m=(k == "l_c")) for k, row, _ in SAMPLED} | {
        "h_amb_scale": 1.0,
        "eps_al_quantile": math.nan,  # nominal uses each scenario's own nominal eps_Al
    }

    runs = [(site, sc) for site in SITES for sc in SCENARIOS]
    print(f"{args.n} draws x {len(runs)} site-scenarios, {len(keys)} parameters, {args.workers} workers")
    t0 = time.time()
    tasks = [(site, sc, s) for s in [nominal] + samples for site, sc in runs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for i, res in enumerate(pool.map(run_one, tasks, chunksize=1), start=1):
            results.append(res)
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} runs  ({time.time() - t0:.0f} s)", flush=True)

    # --- timeseries.npz: "<site>_<scenario>" -> (draw, series, time), draw 0 the nominal
    # as in samples.csv; "<site>_time_s" -> seconds from desorption start. ---
    npz_path = args.out / "timeseries.npz"
    arrays: dict[str, np.ndarray] = {"series": np.array(SERIES)}
    for j, (site, sc) in enumerate(runs):
        arrays[f"{site}_{sc}"] = np.stack([r[3] for r in results[j :: len(runs)]])
        arrays[f"{site}_time_s"] = _desorption_grid_s(site)
    np.savez_compressed(npz_path, **arrays)
    results = [r[:3] for r in results]

    # --- samples.csv: one row per draw, every site-scenario's result side by side. The
    # nominal (every parameter at its workbook Value) goes first as sample "nominal". ---
    csv_path = args.out / "samples.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["sample"] + keys
            + [f"{c}_{site}_{sc}" for site, sc in runs for c in ("yield_kg_m2", "eta", "error")]
        )
        for i, s in enumerate([nominal] + samples):
            row = results[i * len(runs) : (i + 1) * len(runs)]
            w.writerow(["nominal" if i == 0 else i - 1] + [s[k] for k in keys] + [x for r in row for x in r])
    print(f"Wrote {csv_path}\n      {npz_path}\nNext: python plot_monte_carlo.py")
    return 0


def run_one(task: tuple[str, str, dict[str, float]]) -> tuple[float, float, str, np.ndarray]:
    """One steady-state daily cycle for one site, eps_Al scenario and parameter draw ->
    (yield kg/m2, eta, error, SERIES on the site's desorption grid).

    Runs in a worker process. k_gel and Rabs have no config field -- u_gel_w_m2_k reads
    them as physics module globals at call time -- so they're set on the module here,
    which is safe because each worker runs one task at a time. A failed solve is recorded
    as NaN with its message rather than aborting the other runs.
    """
    site, scenario, s = task
    grid_s = _desorption_grid_s(site)
    physics.K_GEL_W_M_K = s["k_gel"]
    physics.RABS_M2_K_W = s["rabs"]
    lo, hi, eps_al_nominal = EPS_AL_SCENARIOS[scenario]
    q = s["eps_al_quantile"]
    eps_al = eps_al_nominal if math.isnan(q) else lo + q * (hi - lo)
    base = SystemConfig.atacama_field() if site == "atacama" else SystemConfig(tilt_deg=35.0)
    profile = _site_profile(site)
    config = dataclasses.replace(
        base,
        g_conv_m_s=s["g_chamber"],
        hydrogel_density_kg_m3=s["rho_dry"],
        condenser_thickness_m=s["l_c"],
        condenser_rho_kg_m3=s["rho_al"],
        condenser_cp_j_kg_k=s["cp_al"],
    )
    # Start from the thermal params our model would build itself (isosteric h_des salt,
    # site tilt, real-emissivity radiative terms), set Wilson's black-paint absorber, and
    # override the sampled optics.
    config = dataclasses.replace(
        config,
        thermal=dataclasses.replace(
            config.thermal_params(),
            eps_abs=s["eps_abs"],
            tau_glass=s["tau_glass"],
            eps_gel=s["eps_gel"],
            eps_al=eps_al,
            eps_glass_ir=s["eps_glass_ir"],
            eps_abs_ir=ABSORBER_IR_EMISSIVITY,
        ),
    )
    profile = DailyWeatherProfile(
        absorption=dataclasses.replace(
            profile.absorption,
            h_amb_w_m2_k=tuple(h * s["h_amb_scale"] for h in profile.absorption.h_amb_w_m2_k),
        ),
        desorption=dataclasses.replace(
            profile.desorption,
            h_amb_w_m2_k=tuple(h * s["h_amb_scale"] for h in profile.desorption.h_amb_w_m2_k),
        ),
    )
    try:
        # Same warm-up as system.run_solar_simulation's replay modes (2 -> max_rounds 3).
        yield_kg, eta, _, des = run_daily_cycle(profile, config, cyclic_initial=True, cyclic_warmup_cycles=2)
    except Exception as exc:  # noqa: BLE001 -- recorded per sample, see docstring
        return math.nan, math.nan, f"{type(exc).__name__}: {exc}", np.full((len(SERIES), len(grid_s)), np.nan)
    series = np.stack([np.interp(grid_s, des.time_s, getattr(des, name)) for name in SERIES])
    return float(yield_kg), float(eta), "", series


def _site_profile(site: str) -> DailyWeatherProfile:
    """Wilson's weather for the site: Atacama Fig. 4 record (tilt 25 deg, A_r 5 come from
    SystemConfig.atacama_field) at ATACAMA_H_AMB_W_M2_K, Cambridge Fig. 3 CSVs (tilt 35
    deg, A_r 7.1 default)."""
    if site == "atacama":
        p = atacama_field_profile()
        return DailyWeatherProfile(
            absorption=dataclasses.replace(p.absorption, h_amb_w_m2_k=(ATACAMA_H_AMB_W_M2_K,) * len(p.absorption.h_amb_w_m2_k)),
            desorption=dataclasses.replace(p.desorption, h_amb_w_m2_k=(ATACAMA_H_AMB_W_M2_K,) * len(p.desorption.h_amb_w_m2_k)),
        )
    _, solar, temp = figure3._load_cambridge_weather()
    return figure3._make_profile(solar, temp, h_amb=figure3._H_AMB_MID)


def _desorption_grid_s(site: str) -> np.ndarray:
    """Seconds from desorption start at the site's weather steps, plus the end point --
    the common grid every run's series is interpolated onto so draws can be banded."""
    des = _site_profile(site).desorption
    return np.arange(len(des.temperature_c) + 1) * des.dt_s


if __name__ == "__main__":
    raise SystemExit(main())
