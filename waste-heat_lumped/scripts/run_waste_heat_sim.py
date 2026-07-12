#!/usr/bin/env python3
"""Run fluid-heated daily-cycle SAWH simulation and LCOW estimate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from waste_heat_lumped.economics.lcow import lcow_from_daily_yield
from waste_heat_lumped.economics.params import LCOEconomicParams
from waste_heat_lumped.physics import device_defaults as dd
from waste_heat_lumped.physics.sorbent import inventory_label
from waste_heat_lumped.simulation.device_config import DeviceConfig
from waste_heat_lumped.simulation.ode_system import run_daily_cycle
from waste_heat_lumped.simulation.water_inventory import (
    plot_water_inventory,
    water_inventory_series,
    write_water_inventory_csv,
)
from waste_heat_lumped.weather.profiles import datacenter_baseline_profile


def _build_config(args: argparse.Namespace) -> DeviceConfig:
    return DeviceConfig(
        salt_name=args.salt,
        salt_to_polymer_ratio=args.salt_loading,
        hydrogel_thickness_m=args.hydrogel_thickness_mm * 1e-3,
        g_conv_m_s=args.g_conv,
        t_f_c=args.t_f_c,
        m_dot_f_kg_s_m2=args.m_dot_f,
        ua_gel_w_k=args.ua_gel,
    )


def _build_profile(args: argparse.Namespace):
    if args.profile == "datacenter-baseline":
        return datacenter_baseline_profile(
            t_amb_c=args.t_amb_c,
            rh=args.rh,
            h_amb=args.h_amb,
        )
    raise ValueError(f"Unknown profile: {args.profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fluid-heated daily-cycle SAWH simulation")
    parser.add_argument(
        "--profile",
        default="datacenter-baseline",
        choices=["datacenter-baseline"],
    )
    parser.add_argument("--salt", default=dd.DEFAULT_SALT_NAME)
    parser.add_argument("--salt-loading", type=float, default=dd.SALT_TO_POLYMER_RATIO)
    parser.add_argument("--hydrogel-thickness-mm", type=float, default=dd.H0_M * 1e3)
    parser.add_argument("--g-conv", type=float, default=dd.G_CHAMBER_M_S)
    parser.add_argument("--t-f-c", type=float, default=dd.T_F_C, help="Loop fluid setpoint (°C)")
    parser.add_argument("--m-dot-f", type=float, default=dd.M_DOT_F_KG_S_M2, help="Loop flow (kg/s/m²)")
    parser.add_argument("--ua-gel", type=float, default=dd.UA_GEL_W_K, help="Loop→gel UA (W/K/m²)")
    parser.add_argument("--t-amb-c", type=float, default=dd.T_AMB_C)
    parser.add_argument("--rh", type=float, default=dd.RH_AMB)
    parser.add_argument("--h-amb", type=float, default=dd.H_AMB_W_M2_K)
    parser.add_argument("--cyclic-initial", action="store_true", help="Warm up to cyclic state")
    parser.add_argument("--warmup-cycles", type=int, default=2)
    parser.add_argument("--plot-water-inventory", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=_REPO / "outputs" / "water_inventory")
    args = parser.parse_args()

    config = _build_config(args)
    profile = _build_profile(args)
    yield_kg, eta, abs_res, des_res = run_daily_cycle(
        profile,
        config,
        cyclic_initial=args.cyclic_initial,
        cyclic_warmup_cycles=args.warmup_cycles,
    )

    econ = LCOEconomicParams()
    lcow = lcow_from_daily_yield(
        yield_kg,
        salt_name=config.salt_name,
        salt_to_polymer_ratio=config.salt_to_polymer_ratio,
        hydrogel_thickness_m=config.hydrogel_thickness_m,
        econ=econ,
    )

    print(f"Daily yield: {yield_kg:.4f} kg/m²")
    print(f"Thermal efficiency: {eta:.4f}")
    print(f"LCOW: {lcow:.2f} USD/m³")

    if args.plot_water_inventory:
        series = water_inventory_series(abs_res, des_res, config=config)
        tag = args.profile.replace("-", "_")
        out_dir = args.output_dir
        csv_path = out_dir / f"water_in_gel_{tag}.csv"
        png_path = out_dir / f"water_in_gel_{tag}.png"
        write_water_inventory_csv(csv_path, series)
        plot_water_inventory(
            png_path,
            series,
            config=config,
            title=f"{inventory_label(config)} — {args.profile}",
        )
        print(f"Wrote {csv_path}")
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
