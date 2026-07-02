#!/usr/bin/env python3
"""Run passive Wilson et al. 2025 SAWH simulation and LCOW estimate."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from solar_lumped.economics.lcow import (
    lcow_cost_breakdown_from_daily_yield,
    lcow_from_daily_yield,
)
from solar_lumped.economics.params import LCOEconomicParams
from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.simulation.ode_system import run_daily_cycle
from solar_lumped.simulation.water_inventory import (
    plot_water_inventory,
    water_inventory_series,
    write_water_inventory_csv,
)
from solar_lumped.weather.profiles import (
    baseline_initial_c_w,
    baseline_profile,
    representative_mean_day_profile,
    replay_profile,
)
from solar_lumped.weather.fig_s1 import c_w_from_water_in_gel_l_m2, fig_s1_initial_c_w


def _build_config(args: argparse.Namespace) -> DeviceConfig:
    fin_ratio = args.fin_area_ratio
    if fin_ratio is None:
        fin_ratio = 5.0 if args.weather_mode == "atacama-replay" else 7.1
    tilt = args.tilt_deg
    if args.weather_mode == "atacama-replay" and tilt == 35.0:
        tilt = 25.0
    elif args.weather_mode in ("baseline", "fig-s1-replay") and tilt == 35.0:
        tilt = 30.0
    return DeviceConfig(
        salt_name=args.salt,
        salt_to_polymer_ratio=args.salt_loading,
        hydrogel_thickness_m=args.hydrogel_thickness_mm * 1e-3,
        vapor_gap_m=args.vapor_gap_mm * 1e-3,
        insulation_gap_m=args.insulation_gap_mm * 1e-3,
        tilt_deg=tilt,
        fin_area_ratio=fin_ratio,
    )


def _write_cost_breakdown_csv(
    path: Path,
    breakdown,
    *,
    subtitle: str,
    lat: float | None,
    lon: float | None,
    year: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bar_label",
                "segment",
                "lcow_usd_per_m3",
                "stack_order",
                "subtitle",
                "lat",
                "lon",
                "year",
            ]
        )
        for i, (seg, val) in enumerate(breakdown.items):
            w.writerow(
                [
                    "solar_lumped",
                    seg,
                    f"{val:.6f}",
                    i,
                    subtitle,
                    lat if lat is not None else "",
                    lon if lon is not None else "",
                    year if year is not None else "",
                ]
            )


def _uses_cycled_initial(weather_mode: str, *, initial_water_l_m2: float | None) -> bool:
    """Atacama / real weather start from post-cycle gel state unless overridden."""
    if initial_water_l_m2 is not None:
        return False
    return weather_mode in ("real", "atacama-replay", "cambridge-replay")


def main() -> None:
    p = argparse.ArgumentParser(description="Solar lumped SAWH simulation + LCOW")
    p.add_argument(
        "--weather-mode",
        choices=("real", "baseline", "atacama-replay", "cambridge-replay", "fig-s1-replay"),
        default="baseline",
    )
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--salt", type=str, default="LiCl")
    p.add_argument("--salt-loading", type=float, default=4.0)
    p.add_argument("--hydrogel-thickness-mm", type=float, default=4.0)
    p.add_argument("--vapor-gap-mm", type=float, default=40.0)
    p.add_argument("--insulation-gap-mm", type=float, default=5.0)
    p.add_argument("--tilt-deg", type=float, default=35.0)
    p.add_argument(
        "--fin-area-ratio",
        type=float,
        default=None,
        help="External fin area ratio A_r (default 5 for atacama-replay, 7 otherwise)",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Cost breakdown CSV path",
    )
    p.add_argument(
        "--water-inventory-csv",
        type=Path,
        default=None,
        help="Water-in-gel time series CSV path",
    )
    p.add_argument(
        "--water-inventory-plot",
        type=Path,
        default=None,
        help="Water-in-gel vs time PNG path",
    )
    p.add_argument(
        "--initial-water-l-m2",
        type=float,
        default=None,
        help="Initial water in gel (L/m²); overrides RH equilibrium or Fig. S1 default",
    )
    args = p.parse_args()

    if args.weather_mode == "real":
        if args.lat is None or args.lon is None:
            p.error("--weather-mode real requires --lat and --lon")
    elif args.weather_mode == "baseline":
        pass
    elif args.weather_mode == "fig-s1-replay":
        pass
    else:
        args.lat = args.lat if args.lat is not None else (-23.65 if "atacama" in args.weather_mode else 42.36)
        args.lon = args.lon if args.lon is not None else (-70.40 if "atacama" in args.weather_mode else -71.09)

    config = _build_config(args)
    if args.weather_mode == "atacama-replay":
        config = DeviceConfig.atacama_field(
            salt_name=config.salt_name,
            salt_to_polymer_ratio=config.salt_to_polymer_ratio,
            hydrogel_thickness_m=config.hydrogel_thickness_m,
            vapor_gap_m=config.vapor_gap_m,
            insulation_gap_m=config.insulation_gap_m,
        )
    elif args.weather_mode == "baseline":
        config = DeviceConfig.baseline(
            salt_name=config.salt_name,
            salt_to_polymer_ratio=config.salt_to_polymer_ratio,
            hydrogel_thickness_m=config.hydrogel_thickness_m,
            vapor_gap_m=config.vapor_gap_m,
            insulation_gap_m=config.insulation_gap_m,
        )
    elif args.weather_mode == "fig-s1-replay":
        config = DeviceConfig.comsol_table_s3(
            salt_name=config.salt_name,
            salt_to_polymer_ratio=config.salt_to_polymer_ratio,
            hydrogel_thickness_m=config.hydrogel_thickness_m,
            vapor_gap_m=config.vapor_gap_m,
            insulation_gap_m=config.insulation_gap_m,
            tilt_deg=config.tilt_deg,
            fin_area_ratio=config.fin_area_ratio,
        )
    econ = LCOEconomicParams()

    c_w_initial: float | None = None
    h_initial: float | None = None
    if args.initial_water_l_m2 is not None:
        c_w_initial = c_w_from_water_in_gel_l_m2(
            args.initial_water_l_m2, config.hydrogel_thickness_m
        )
    elif args.weather_mode == "fig-s1-replay":
        c_w_initial = fig_s1_initial_c_w(h_m=config.hydrogel_thickness_m)
    elif args.weather_mode == "baseline":
        c_w_initial = baseline_initial_c_w(h_m=config.hydrogel_thickness_m)

    use_cycled = _uses_cycled_initial(
        args.weather_mode, initial_water_l_m2=args.initial_water_l_m2
    )

    inventory_abs_res = None
    inventory_des_res = None
    inventory_note = ""

    if args.weather_mode == "real":
        profile = representative_mean_day_profile(
            args.lat,
            args.lon,
            args.year,
            cache_dir=args.cache_dir,
        )
        yield_kg, eta, inventory_abs_res, inventory_des_res = run_daily_cycle(
            profile,
            config,
            c_w_initial=c_w_initial,
            h_initial=h_initial,
            cyclic_initial=use_cycled,
        )
        result_mean = yield_kg
        n_days = 1
        eta_mean = eta
        if use_cycled:
            inventory_note = " (cycled initial state; mean diurnal weather)"
        else:
            inventory_note = f" (mean diurnal weather for {args.year})"
    elif args.weather_mode == "baseline":
        profile = baseline_profile()
        yield_kg, eta, inventory_abs_res, inventory_des_res = run_daily_cycle(
            profile, config, c_w_initial=c_w_initial, h_initial=h_initial
        )
        result_mean = yield_kg
        n_days = 1
        eta_mean = eta
    else:
        profile = replay_profile(args.weather_mode, cache_dir=args.cache_dir)
        yield_kg, eta, inventory_abs_res, inventory_des_res = run_daily_cycle(
            profile,
            config,
            c_w_initial=c_w_initial,
            h_initial=h_initial,
            cyclic_initial=use_cycled,
        )
        result_mean = yield_kg
        n_days = 1
        eta_mean = eta
        if use_cycled:
            inventory_note = " (cycled initial state after warmup days)"

    lcow = lcow_from_daily_yield(
        result_mean,
        salt_name=config.salt_name,
        salt_to_polymer_ratio=config.salt_to_polymer_ratio,
        hydrogel_thickness_m=config.hydrogel_thickness_m,
        econ=econ,
    )
    breakdown = lcow_cost_breakdown_from_daily_yield(
        result_mean,
        salt_name=config.salt_name,
        salt_to_polymer_ratio=config.salt_to_polymer_ratio,
        hydrogel_thickness_m=config.hydrogel_thickness_m,
        econ=econ,
    )

    print(f"Weather mode: {args.weather_mode}")
    if args.weather_mode == "real":
        print(f"Year aggregated to mean diurnal profile: {args.year}")
    print(f"Days simulated: {n_days}")
    print(f"Mean daily yield: {result_mean * 1000:.1f} g/m² ({result_mean:.4f} kg/m²)")
    print(f"Mean daily yield: {result_mean:.2f} L/m² (≈ kg/m² for water)")
    print(f"Mean thermal efficiency: {eta_mean * 100:.1f}%")
    print(f"LCOW: ${lcow:.4f}/m³")
    if breakdown:
        print("\nCost breakdown (USD/m³):")
        for seg, val in breakdown.items:
            print(f"  {seg:30s} {val:10.4f}")

    out = args.output_csv
    if out is None:
        tag = args.weather_mode
        if args.lat is not None:
            tag += f"_lat{args.lat:.4f}_lon{args.lon:.4f}_{args.year}"
        out = _REPO / "outputs" / "lcow" / f"cost_breakdown_{tag}.csv"
    if breakdown:
        _write_cost_breakdown_csv(
            out,
            breakdown,
            subtitle=f"LCOW=${lcow:.4f}/m³ yield={result_mean:.4f}kg/m²/d",
            lat=args.lat,
            lon=args.lon,
            year=args.year if args.weather_mode == "real" else None,
        )
        print(f"\nWrote {out}")

    if inventory_abs_res is not None and inventory_des_res is not None:
        h0_ref = config.hydrogel_thickness_m
        inventory = water_inventory_series(
            inventory_abs_res,
            inventory_des_res,
            h0_ref_m=h0_ref,
        )
        tag = args.weather_mode
        if args.lat is not None:
            tag += f"_lat{args.lat:.4f}_lon{args.lon:.4f}_{args.year}"
        inventory_csv = args.water_inventory_csv
        if inventory_csv is None:
            inventory_csv = _REPO / "outputs" / "water_inventory" / f"water_in_gel_{tag}.csv"
        inventory_plot = args.water_inventory_plot
        if inventory_plot is None:
            inventory_plot = _REPO / "outputs" / "water_inventory" / f"water_in_gel_{tag}.png"

        write_water_inventory_csv(inventory_csv, inventory)
        plot_title = f"Water in gel — {args.weather_mode}{inventory_note}"
        plot_water_inventory(inventory_plot, inventory, title=plot_title)

        if args.weather_mode in ("atacama-replay", "cambridge-replay", "baseline", "fig-s1-replay"):
            simple_csv = _REPO / "outputs" / "water_inventory" / f"water_in_gel_{args.weather_mode}.csv"
            simple_plot = _REPO / "outputs" / "water_inventory" / f"water_in_gel_{args.weather_mode}.png"
            write_water_inventory_csv(simple_csv, inventory)
            plot_water_inventory(simple_plot, inventory, title=plot_title)

        w_start = float(inventory.water_l_m2[0])
        w_peak = float(np.max(inventory.water_l_m2))
        w_end = float(inventory.water_l_m2[-1])
        print(
            f"\nWater in gel (DVS basis){inventory_note}: "
            f"start={w_start:.2f} peak={w_peak:.2f} end={w_end:.2f} L/m²"
        )
        print(f"Wrote {inventory_csv}")
        print(f"Wrote {inventory_plot}")
        if args.weather_mode in ("atacama-replay", "cambridge-replay", "baseline", "fig-s1-replay"):
            print(f"Wrote {simple_csv}")
            print(f"Wrote {simple_plot}")


if __name__ == "__main__":
    main()
