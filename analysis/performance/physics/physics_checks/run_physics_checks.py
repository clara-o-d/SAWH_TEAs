#!/usr/bin/env python3
"""Run one daily cycle and dump every physics diagnostic we have for it.

Three artefacts per run, into ``outputs/`` beside this script:

* ``diagnostics_<tag>.csv/.png`` — system and weather trajectories over the full cycle,
  including the desorption driving force (c_r, a_w, c_r - a_w) against the salt's DRH.
* ``water_inventory_<tag>.csv/.png`` — water in the gel and cumulative collected water.
* ``conservation_<tag>.png`` — the mass and enthalpy conservation check: LSODA's dense
  output for sorbent water and condenser enthalpy against what integrating the RHS
  boundary flows (m_des, dT_cond_dt) predicts. Flat drift means the integrator's state
  and its own rates agree; drift growing to a visible fraction of the signal means it
  doesn't, and the day's yield is suspect.

Usage:
    python run_physics_checks.py --weather-mode baseline
    python run_physics_checks.py --weather-mode atacama-replay --salt LiCl --cycled
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_lumped.simulation import (
    conservation_drift_series,
    detailed_series,
    plot_detailed_diagnostics,
    plot_water_inventory,
    reversal_diagnostics,
    run_daily_cycle,
    water_inventory_series,
    write_detailed_csv,
    write_water_inventory_csv,
)
from solar_lumped.system import build_system_config
from solar_lumped.weather import baseline_profile, real_day_profile, replay_profile

_OUT_DIR = Path(__file__).resolve().parent / "outputs"

_REPLAYS = ("atacama-replay", "cambridge-replay", "fig-s1-replay")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weather-mode",
        default="baseline",
        choices=("baseline", "real", *_REPLAYS),
    )
    p.add_argument("--lat", type=float, default=-23.5, help="--weather-mode real only")
    p.add_argument("--lon", type=float, default=-70.0, help="--weather-mode real only")
    p.add_argument("--day", default="2024-06-15", help="--weather-mode real only (YYYY-MM-DD)")
    p.add_argument("--cache-dir", default=None, help="Weather cache dir for real/replay modes")
    p.add_argument("--salt", default="LiCl")
    p.add_argument("--salt-loading", type=float, default=4.0)
    p.add_argument("--hydrogel-thickness-mm", type=float, default=4.0)
    p.add_argument("--vapor-gap-mm", type=float, default=40.0)
    p.add_argument("--tilt-deg", type=float, default=None, help="Default: 25 for atacama, else 30")
    p.add_argument("--fin-area-ratio", type=float, default=None, help="Default: 5 atacama, else 7.1")
    p.add_argument(
        "--cycled",
        action="store_true",
        help="Report the Aitken-converged cyclic steady state instead of day one",
    )
    args = p.parse_args(argv)

    atacama = args.weather_mode == "atacama-replay"
    config = build_system_config(
        salt=args.salt,
        salt_loading=args.salt_loading,
        hydrogel_thickness_mm=args.hydrogel_thickness_mm,
        vapor_gap_mm=args.vapor_gap_mm,
        tilt_deg=args.tilt_deg if args.tilt_deg is not None else (25.0 if atacama else 30.0),
        fin_area_ratio=(
            args.fin_area_ratio if args.fin_area_ratio is not None else (5.0 if atacama else 7.1)
        ),
    )

    if args.weather_mode == "baseline":
        profile = baseline_profile()
    elif args.weather_mode == "real":
        import dataclasses
        from datetime import date

        from solar_lumped.weather import real_site_elevation_m

        profile = real_day_profile(
            args.lat, args.lon, date.fromisoformat(args.day),
            cache_dir=args.cache_dir, poa_tilt_deg=config.tilt_deg,
        )
        # Real sites run at their own ambient pressure -- it sets every gap air property
        # and the h_amb density derate. Only the replay/baseline modes stay at sea level
        # (Antofagasta and Cambridge effectively are, and Wilson recreation must not move).
        config = dataclasses.replace(
            config,
            site_elevation_m=real_site_elevation_m(
                args.lat, args.lon, date.fromisoformat(args.day).year,
                cache_dir=args.cache_dir,
            ),
        )
    else:
        profile = replay_profile(args.weather_mode, cache_dir=args.cache_dir)

    yield_kg, eta, abs_res, des_res = run_daily_cycle(
        profile, config, cyclic_initial=args.cycled
    )
    tag = f"{args.weather_mode}_{args.salt}" + ("_cycled" if args.cycled else "")
    note = " (cyclic steady state)" if args.cycled else ""
    print(f"{tag}: yield={yield_kg:.4f} kg/m²/d  eta_thermal={eta:.4f}")

    # The desorption dc_w/dt <= 0 clamp is an approximation with no error term of its
    # own, so report how hard it worked: a large reversed fraction means the sealed
    # window extends past where the device is doing anything, and discarded_l_m2 is the
    # (loose) upper bound that mis-scheduling costs. Never subtracted from the yield.
    rev = reversal_diagnostics(des_res, config=config)
    print(
        f"{tag}: desorption clamp active {100 * rev.reversed_time_fraction:.1f}% of the "
        f"sealed window, discarding <= {1000 * rev.discarded_l_m2:.1f} g/m² "
        f"({100 * rev.discarded_l_m2 / max(yield_kg, 1e-9):.1f}% of yield)"
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    detailed = detailed_series(profile, abs_res, des_res, config)
    write_detailed_csv(_OUT_DIR / f"diagnostics_{tag}.csv", detailed)
    plot_detailed_diagnostics(
        _OUT_DIR / f"diagnostics_{tag}.png",
        detailed,
        title=f"System, weather and desorption driving force — {tag}{note}",
    )

    inventory = water_inventory_series(abs_res, des_res, config=config)
    write_water_inventory_csv(_OUT_DIR / f"water_inventory_{tag}.csv", inventory)
    plot_water_inventory(
        _OUT_DIR / f"water_inventory_{tag}.png",
        inventory,
        config=config,
        title=f"Water inventory — {tag}{note}",
    )

    # --- Conservation check. Left axis: the quantity. Right axis: the drift, on its own
    # scale because a healthy drift is orders of magnitude smaller and would be a flat
    # line at zero if shared. Percentages are of the quantity's full swing over the day.
    drift = conservation_drift_series(des_res, config=config)
    time_h = drift.time_s / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    panels = [
        (
            axes[0],
            drift.mass_l_m2,
            drift.mass_drift_l_m2,
            "Sorbent water (L/m²)",
            "Mass drift (L/m²)",
            "#4C72B0",
        ),
        (
            axes[1],
            drift.enthalpy_j_m2,
            drift.enthalpy_drift_j_m2,
            "Condenser enthalpy (J/m²)",
            "Enthalpy drift (J/m²)",
            "#C44E52",
        ),
    ]
    for ax, quantity, resid, ylabel, drift_label, color in panels:
        if quantity is None or resid is None:
            ax.text(0.5, 0.5, f"{ylabel}: not available", ha="center", transform=ax.transAxes)
            continue
        ax.plot(time_h, quantity, color=color, linewidth=2, label=ylabel)
        ax.set_ylabel(ylabel, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(True, alpha=0.3)

        ax_d = ax.twinx()
        ax_d.plot(time_h, resid, color="0.35", linewidth=1.3, linestyle="--", label=drift_label)
        ax_d.axhline(0.0, color="0.35", linewidth=0.6, alpha=0.5)
        ax_d.set_ylabel(drift_label, color="0.35")
        ax_d.tick_params(axis="y", labelcolor="0.35")

        swing = float(np.max(quantity)) - float(np.min(quantity))
        worst = float(np.max(np.abs(resid)))
        pct = 100.0 * worst / swing if swing > 0.0 else float("nan")
        ax.set_title(f"max |drift| = {worst:.3e} ({pct:.3g}% of swing)", fontsize=9)

        lines_l, labels_l = ax.get_legend_handles_labels()
        lines_r, labels_r = ax_d.get_legend_handles_labels()
        ax.legend(lines_l + lines_r, labels_l + labels_r, loc="center right", fontsize=8)

    axes[-1].set_xlabel("Desorption time (h)")
    fig.suptitle(f"Desorption conservation check — {tag}{note}")
    fig.tight_layout()
    fig.savefig(_OUT_DIR / f"conservation_{tag}.png", dpi=150)
    plt.close(fig)

    for name in (
        f"diagnostics_{tag}.csv",
        f"diagnostics_{tag}.png",
        f"water_inventory_{tag}.csv",
        f"water_inventory_{tag}.png",
        f"conservation_{tag}.png",
    ):
        print(f"Wrote {_OUT_DIR / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
