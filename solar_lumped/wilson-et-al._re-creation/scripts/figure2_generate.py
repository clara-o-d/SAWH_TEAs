#!/usr/bin/env python3
"""
Generate Wilson et al. (2025) Figure 2 sweep data (panels B–F).

Baseline conditions (from paper Methods):
  T_amb = 25 °C, Q_solar = 600 W/m², RH = 0.5, h_amb = 10 W/m²K,
  H₀ = 4 mm, L_g = 40 mm, ε_abs = 0.95, τ_glass = 0.9, A_r = 7.1

Error bands: h_amb = 10 ± 2.5 W/m²K (±25%) per Wilson Methods section.

Results saved to:  wilson-et-al._re-creation/outputs/figure2/figure2_data.pkl
Plot with:        python wilson-et-al._re-creation/scripts/figure2_plot.py
"""

from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).resolve()
_WILSON_DIR = _SCRIPT.parent.parent          # wilson-et-al._re-creation/
_SOLAR_ROOT = _WILSON_DIR.parent             # solar_lumped/
_SRC = _SOLAR_ROOT / "src"
for _p in (_SRC, _SOLAR_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from solar_lumped.physics import table_s3
from solar_lumped.physics.device_balances import DeviceThermalParams
from solar_lumped.simulation.device_config import DeviceConfig, register_desorption_solver_cli
from solar_lumped.simulation.ode_system import run_daily_cycle
from solar_lumped.weather.profiles import baseline_profile

_OUT_DIR = _WILSON_DIR / "outputs" / "figure2"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_DATA_PATH = _OUT_DIR / "figure2_data.pkl"

# ---------------------------------------------------------------------------
# Default baseline values (Wilson Methods)
# ---------------------------------------------------------------------------
_T_AMB_C = 25.0
_RH = 0.5
_Q_SOLAR_W_M2 = 600.0
_H_AMB = 10.0
_H_AMB_LO = 7.5
_H_AMB_HI = 12.5

_DESORPTION_SOLVER = "quasi_steady"
_H0_MM = 4.0
_LG_MM = 40.0
_A_R = 7.1
_EPS_ABS = 0.95
_TAU_GLASS = 0.9

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_thermal(
    *,
    eps_abs: float = _EPS_ABS,
    tau_glass: float = _TAU_GLASS,
    has_glass: bool = True,
    vapor_gap_m: float = _LG_MM / 1000.0,
) -> DeviceThermalParams:
    return DeviceThermalParams(
        eps_abs=eps_abs,
        tau_glass=tau_glass,
        has_glass=has_glass,
        vapor_gap_m=vapor_gap_m,
        h_des_j_per_kg=table_s3.H_DES_J_PER_KG,
    )


def make_config(
    *,
    H0_mm: float = _H0_MM,
    Lg_mm: float = _LG_MM,
    A_r: float = _A_R,
    eps_abs: float = _EPS_ABS,
    tau_glass: float = _TAU_GLASS,
    has_glass: bool = True,
) -> DeviceConfig:
    thermal = _make_thermal(
        eps_abs=eps_abs,
        tau_glass=tau_glass,
        has_glass=has_glass,
        vapor_gap_m=Lg_mm / 1000.0,
    )
    return DeviceConfig(
        hydrogel_thickness_m=H0_mm / 1000.0,
        vapor_gap_m=Lg_mm / 1000.0,
        fin_area_ratio=A_r,
        tilt_deg=30.0,
        thermal=thermal,
        desorption_solver=_DESORPTION_SOLVER,  # type: ignore[arg-type]
    )


def _run_single(config: DeviceConfig, profile) -> tuple[float, float]:
    """Return (yield_L_m2_day, thermal_efficiency_fraction).

    Single day from the fabrication initial condition (gel equilibrated at
    20% RH), matching Wilson's COMSOL approach: "all time-dependent energy
    balances ... solved ... for all times throughout a single day" with a
    12-h absorption phase followed by a 12-h desorption phase. The Cambridge
    and Atacama field tests were likewise single-day runs from a freshly
    fabricated gel, so cyclic steady state is *not* what the paper reports.
    """
    try:
        yield_kg, eta, _abs, _des = run_daily_cycle(profile, config)
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")
    return max(yield_kg, 0.0), eta


class _Job(NamedTuple):
    key: tuple
    config: DeviceConfig
    profile: object


def _worker(job: _Job) -> tuple[tuple, float, float]:
    y, eta = _run_single(job.config, job.profile)
    return job.key, y, eta


def _parallel_sweep(jobs: list[_Job]) -> dict[tuple, tuple[float, float]]:
    """Run all jobs in parallel; return {key: (yield_L_m2, eta)}."""
    results: dict[tuple, tuple[float, float]] = {}
    try:
        with ProcessPoolExecutor() as ex:
            futs = {ex.submit(_worker, j): j for j in jobs}
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    key, y, eta = fut.result()
                    results[key] = (y, eta)
                except Exception:
                    results[job.key] = (float("nan"), float("nan"))
    except Exception:
        for job in jobs:
            if job.key not in results:
                try:
                    key, y, eta = _worker(job)
                    results[key] = (y, eta)
                except Exception:
                    results[job.key] = (float("nan"), float("nan"))
    return results


def _band(results: dict, key_lo: tuple, key_mid: tuple, key_hi: tuple
          ) -> tuple[float, float, float]:
    """Return (lo, mid, hi) yield values from three h_amb variants."""
    y_lo = results.get(key_lo, (float("nan"), 0.0))[0]
    y_mid = results.get(key_mid, (float("nan"), 0.0))[0]
    y_hi = results.get(key_hi, (float("nan"), 0.0))[0]
    if np.isnan(y_mid):
        return float("nan"), float("nan"), float("nan")
    y_lo = y_lo if not np.isnan(y_lo) else y_mid
    y_hi = y_hi if not np.isnan(y_hi) else y_mid
    lo = min(y_lo, y_mid, y_hi)
    hi = max(y_lo, y_mid, y_hi)
    return lo, y_mid, hi


def sweep_B():
    tau_vals = np.linspace(0.2, 1.0, 10)
    eps_abs_vals = [0.2, 0.5, 0.8, 0.95, 0.99]
    h_amb_vals = [_H_AMB_LO, _H_AMB, _H_AMB_HI]

    jobs: list[_Job] = []
    for eps in eps_abs_vals:
        for tau in tau_vals:
            for h in h_amb_vals:
                cfg = make_config(eps_abs=eps, tau_glass=tau)
                prof = baseline_profile(h_amb_w_m2_k=h)
                jobs.append(_Job(key=(eps, tau, h), config=cfg, profile=prof))

    print(f"  Panel B: {len(jobs)} runs")
    res = _parallel_sweep(jobs)

    out = {}
    for eps in eps_abs_vals:
        lows, mids, highs = [], [], []
        for tau in tau_vals:
            lo, mid, hi = _band(
                res,
                (eps, tau, _H_AMB_LO),
                (eps, tau, _H_AMB),
                (eps, tau, _H_AMB_HI),
            )
            lows.append(lo)
            mids.append(mid)
            highs.append(hi)
        out[eps] = (tau_vals, np.array(lows), np.array(mids), np.array(highs))
    return out


def sweep_C():
    h_amb_vals = np.linspace(1.0, 10.0, 10)
    ar_vals = [1, 2, 5, 7]
    cover_states = [True, False]

    jobs: list[_Job] = []
    for ar in ar_vals:
        for has_glass in cover_states:
            for h in h_amb_vals:
                cfg = make_config(A_r=ar, has_glass=has_glass)
                prof = baseline_profile(h_amb_w_m2_k=h)
                jobs.append(_Job(key=(ar, has_glass, h), config=cfg, profile=prof))

    print(f"  Panel C: {len(jobs)} runs")
    res = _parallel_sweep(jobs)

    out = {}
    for ar in ar_vals:
        for has_glass in cover_states:
            yields = []
            for h in h_amb_vals:
                y = res.get((ar, has_glass, float(h)), (0.0, 0.0))[0]
                yields.append(y)
            out[(ar, has_glass)] = (h_amb_vals, np.array(yields))
    return out


def sweep_D():
    rh_vals = np.linspace(0.2, 0.9, 10)
    t_amb_k_vals = [280, 290, 300, 310]
    h_amb_vals = [_H_AMB_LO, _H_AMB, _H_AMB_HI]

    jobs: list[_Job] = []
    for T_k in t_amb_k_vals:
        t_c = T_k - 273.15
        for rh in rh_vals:
            for h in h_amb_vals:
                cfg = make_config()
                prof = baseline_profile(temperature_c=t_c, relative_humidity=rh, h_amb_w_m2_k=h)
                jobs.append(_Job(key=(T_k, rh, h), config=cfg, profile=prof))

    print(f"  Panel D: {len(jobs)} runs")
    res = _parallel_sweep(jobs)

    out = {}
    for T_k in t_amb_k_vals:
        lows, mids, highs = [], [], []
        for rh in rh_vals:
            lo, mid, hi = _band(
                res,
                (T_k, rh, _H_AMB_LO),
                (T_k, rh, _H_AMB),
                (T_k, rh, _H_AMB_HI),
            )
            lows.append(lo)
            mids.append(mid)
            highs.append(hi)
        out[T_k] = (rh_vals, np.array(lows), np.array(mids), np.array(highs))
    return out


def sweep_E():
    h0_vals = np.linspace(0.5, 8.0, 10)
    lg_vals = [18, 20, 40, 60]
    h_amb_vals = [_H_AMB_LO, _H_AMB, _H_AMB_HI]

    jobs: list[_Job] = []
    for lg in lg_vals:
        for h0 in h0_vals:
            for h in h_amb_vals:
                cfg = make_config(H0_mm=h0, Lg_mm=lg)
                prof = baseline_profile(h_amb_w_m2_k=h)
                jobs.append(_Job(key=(lg, h0, h), config=cfg, profile=prof))

    print(f"  Panel E: {len(jobs)} runs")
    res = _parallel_sweep(jobs)

    out = {}
    for lg in lg_vals:
        lows, mids, highs = [], [], []
        for h0 in h0_vals:
            lo, mid, hi = _band(
                res,
                (lg, h0, _H_AMB_LO),
                (lg, h0, _H_AMB),
                (lg, h0, _H_AMB_HI),
            )
            lows.append(lo)
            mids.append(mid)
            highs.append(hi)
        out[lg] = (h0_vals, np.array(lows), np.array(mids), np.array(highs))
    return out


def sweep_F():
    q_vals = np.linspace(500, 1500, 10)
    h0_vals_mm = [2, 4, 8]
    h_amb_vals = [_H_AMB_LO, _H_AMB, _H_AMB_HI]

    jobs: list[_Job] = []
    for h0 in h0_vals_mm:
        for q in q_vals:
            for h in h_amb_vals:
                cfg = make_config(H0_mm=h0)
                prof = baseline_profile(solar_w_m2=q, h_amb_w_m2_k=h)
                jobs.append(_Job(key=(h0, q, h), config=cfg, profile=prof))

    print(f"  Panel F: {len(jobs)} runs")
    res = _parallel_sweep(jobs)

    out_yield = {}
    out_eta = {}
    for h0 in h0_vals_mm:
        y_lows, y_mids, y_highs = [], [], []
        e_lows, e_mids, e_highs = [], [], []
        for q in q_vals:
            lo, mid, hi = _band(
                res,
                (h0, q, _H_AMB_LO),
                (h0, q, _H_AMB),
                (h0, q, _H_AMB_HI),
            )
            y_lows.append(lo)
            y_mids.append(mid)
            y_highs.append(hi)
            _, eta_lo = res.get((h0, q, _H_AMB_LO), (0.0, 0.0))
            _, eta_mid = res.get((h0, q, _H_AMB), (0.0, 0.0))
            _, eta_hi = res.get((h0, q, _H_AMB_HI), (0.0, 0.0))
            e_lows.append(min(eta_lo, eta_mid, eta_hi) * 100)
            e_mids.append(eta_mid * 100)
            e_highs.append(max(eta_lo, eta_mid, eta_hi) * 100)
        out_yield[h0] = (q_vals / 1000.0, np.array(y_lows), np.array(y_mids), np.array(y_highs))
        out_eta[h0] = (q_vals / 1000.0, np.array(e_lows), np.array(e_mids), np.array(e_highs))
    return out_yield, out_eta


def save_figure2_data(
    data_B,
    data_C,
    data_D,
    data_E,
    data_F_yield,
    data_F_eta,
    path: Path = _DATA_PATH,
) -> Path:
    payload = {
        "B": data_B,
        "C": data_C,
        "D": data_D,
        "E": data_E,
        "F_yield": data_F_yield,
        "F_eta": data_F_eta,
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def main() -> Path:
    import argparse
    global _DESORPTION_SOLVER

    parser = argparse.ArgumentParser(description="Wilson Figure 2 — parametric sweep data")
    register_desorption_solver_cli(parser)
    args = parser.parse_args()
    _DESORPTION_SOLVER = args.desorption_solver

    print("Wilson Figure 2 data generation — solar_lumped model")
    print("=" * 60)
    print(f"  desorption_solver = {_DESORPTION_SOLVER}")

    print("\nRunning panel B (optical properties)…")
    data_B = sweep_B()

    print("\nRunning panel C (finned condenser / glass cover)…")
    data_C = sweep_C()

    print("\nRunning panel D (T_amb × RH)…")
    data_D = sweep_D()

    print("\nRunning panel E (vapor gap × gel thickness)…")
    data_E = sweep_E()

    print("\nRunning panel F (solar flux × gel thickness)…")
    data_F_yield, data_F_eta = sweep_F()

    out_path = save_figure2_data(data_B, data_C, data_D, data_E, data_F_yield, data_F_eta)
    print(f"\nSaved data → {out_path}")
    print("Plot with:  python wilson-et-al._re-creation/scripts/figure2_plot.py")
    return out_path


if __name__ == "__main__":
    main()
