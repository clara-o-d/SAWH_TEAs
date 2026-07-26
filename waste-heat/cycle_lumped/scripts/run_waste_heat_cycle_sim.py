#!/usr/bin/env python3
"""Run waste-heat two-bed SAWH simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from waste_heat_cycle_lumped.physics import rh_outside_desorber
from waste_heat_cycle_lumped.physics import (
    DEFAULT_MOF_NAME,
    DEFAULT_SALT_NAME,
    DEFAULT_SORBENT,
    G_CHAMBER_M_S,
    H0_M,
    H_AMB_W_M2_K,
    M_WH_KG_S_M2,
    P_COND_PA,
    RH_AMB,
    RH_DESORBER_SWITCH,
    SALT_TO_POLYMER_RATIO,
    TAU_HALF_S,
    T_AMB_C,
    T_WH_IN_C,
)
from waste_heat_cycle_lumped.physics import inventory_label, is_hydrogel
from waste_heat_cycle_lumped.simulation import ControllerParams, DeviceConfig
from waste_heat_cycle_lumped.simulation import (
    detailed_daily_series,
    detailed_series,
    plot_detailed_diagnostics,
    write_detailed_csv,
)
from waste_heat_cycle_lumped.simulation import run_cycle, run_daily_operation
from waste_heat_cycle_lumped.simulation import (
    plot_water_inventory,
    water_inventory_daily_series,
    water_inventory_series,
    write_water_inventory_csv,
)
from waste_heat_cycle_lumped.weather import (
    datacenter_baseline_profile,
    datacenter_diurnal_profile,
)


def _build_config(args: argparse.Namespace) -> DeviceConfig:
    controller = None
    if args.c_vac_max is not None:
        base = ControllerParams()
        controller = ControllerParams(
            m_f_base_kg_s_m2=base.m_f_base_kg_s_m2,
            m_f_min_kg_s_m2=base.m_f_min_kg_s_m2,
            m_f_max_kg_s_m2=base.m_f_max_kg_s_m2,
            c_vac_base_kg_s_pa_m2=base.c_vac_base_kg_s_pa_m2,
            c_vac_min_kg_s_pa_m2=base.c_vac_min_kg_s_pa_m2,
            c_vac_max_kg_s_pa_m2=args.c_vac_max,
            k_t_per_k=base.k_t_per_k,
            k_m_per_kg_m2=base.k_m_per_kg_m2,
            k_p_per_kg_s_m2=base.k_p_per_kg_s_m2,
        )
    return DeviceConfig(
        sorbent=args.sorbent,
        salt_name=args.salt,
        salt_to_polymer_ratio=args.salt_loading,
        hydrogel_thickness_m=args.hydrogel_thickness_mm * 1e-3,
        g_conv_m_s=args.g_conv,
        mof_name=args.mof,
        tau_half_s=args.max_half_cycle_min * 60.0,
        rh_desorber_switch=args.rh_desorber_switch,
        p_cond_pa=args.p_cond_mbar * 100.0,
        controller=controller,
    )


def _build_profile(args: argparse.Namespace, config: DeviceConfig):
    kwargs = dict(
        tau_half_s=config.tau_half_s,
        t_amb_c=args.t_amb_c,
        rh=args.rh,
        h_amb=args.h_amb,
        t_wh_in_c=args.t_wh_in_c,
        m_dot_wh_kg_s_m2=args.m_dot_wh,
    )
    if args.profile == "datacenter-diurnal":
        return datacenter_diurnal_profile(**kwargs)
    return datacenter_baseline_profile(**kwargs)


def _output_tag(args: argparse.Namespace) -> str:
    tag = args.profile.replace("-", "_")
    if args.sorbent == "mof":
        tag += f"_{args.mof}"
    if args.daily:
        tag += "_daily"
    if args.warmup_cycles > 0:
        tag += f"_warm{args.warmup_cycles}"
    if args.t_wh_in_c != T_WH_IN_C:
        tag += f"_twh{args.t_wh_in_c:.0f}"
    return tag


def _inventory_prefix(config: DeviceConfig) -> str:
    return "water_in_gel" if is_hydrogel(config) else "water_in_mof"


def _write_inventory_outputs(
    *,
    args: argparse.Namespace,
    config: DeviceConfig,
    inventory,
    note: str,
) -> None:
    prefix = _inventory_prefix(config)
    tag = _output_tag(args)
    inventory_csv = args.water_inventory_csv
    if inventory_csv is None:
        inventory_csv = _REPO / "outputs" / "water_inventory" / f"{prefix}_{tag}.csv"
    inventory_plot = args.water_inventory_plot
    if inventory_plot is None:
        inventory_plot = _REPO / "outputs" / "water_inventory" / f"{prefix}_{tag}.png"

    label = inventory_label(config)
    title = f"Water in {label} — {args.profile}{note}"
    write_water_inventory_csv(inventory_csv, inventory, config=config)
    plot_water_inventory(inventory_plot, inventory, config=config, title=title)

    simple_csv = _REPO / "outputs" / "water_inventory" / f"{prefix}_{args.profile.replace('-', '_')}.csv"
    simple_plot = _REPO / "outputs" / "water_inventory" / f"{prefix}_{args.profile.replace('-', '_')}.png"
    if simple_csv != inventory_csv or simple_plot != inventory_plot:
        write_water_inventory_csv(simple_csv, inventory, config=config)
        plot_water_inventory(simple_plot, inventory, config=config, title=title)

    w_start = float(inventory.water_l_m2[0])
    w_peak = float(np.max(inventory.water_l_m2))
    w_end = float(inventory.water_l_m2[-1])
    print(
        f"\nWater in {label}{note}: start={w_start:.2f} peak={w_peak:.2f} end={w_end:.2f} L/m²"
    )
    print(f"Wrote {inventory_csv}")
    print(f"Wrote {inventory_plot}")
    if simple_csv != inventory_csv:
        print(f"Wrote {simple_csv}")
    if simple_plot != inventory_plot:
        print(f"Wrote {simple_plot}")


def _half_cycle_note(half_a, half_b, config: DeviceConfig) -> str:
    dur_a_min = float(half_a.time_s[-1]) / 60.0
    dur_b_min = float(half_b.time_s[-1]) / 60.0
    rh_a = rh_outside_desorber(float(half_a.t_d_c[-1]), float(half_a.t_cond_c[-1]))
    rh_b = rh_outside_desorber(float(half_b.t_d_c[-1]), float(half_b.t_cond_c[-1]))
    return (
        f" (RH_out,des end={rh_a:.0%}/{rh_b:.0%}; "
        f"τ={dur_a_min:.1f}/{dur_b_min:.1f} min; "
        f"switch≤{config.rh_desorber_switch:.0%}, max={config.tau_half_s / 60.0:.0f} min)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Waste-heat two-bed SAWH simulation")
    p.add_argument("--profile", default="datacenter-baseline")
    p.add_argument("--sorbent", choices=("hydrogel", "mof"), default=DEFAULT_SORBENT)
    p.add_argument("--salt", default=DEFAULT_SALT_NAME)
    p.add_argument("--salt-loading", type=float, default=SALT_TO_POLYMER_RATIO)
    p.add_argument("--hydrogel-thickness-mm", type=float, default=H0_M * 1e3)
    p.add_argument("--g-conv", type=float, default=G_CHAMBER_M_S)
    p.add_argument("--mof", default=DEFAULT_MOF_NAME)
    p.add_argument(
        "--max-half-cycle-min",
        type=float,
        default=TAU_HALF_S / 60.0,
        help="Maximum half-cycle duration (min); ends earlier when desorber RH threshold is met",
    )
    p.add_argument(
        "--tau-half-min",
        type=float,
        dest="max_half_cycle_min",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--rh-desorber-switch",
        type=float,
        default=RH_DESORBER_SWITCH,
        help="End half-cycle when vapor-gap RH outside desorber falls to this fraction (default 0.35)",
    )
    p.add_argument("--p-cond-mbar", type=float, default=P_COND_PA / 100.0)
    p.add_argument(
        "--c-vac-max",
        type=float,
        default=None,
        help="Max vacuum conductance C_vac (kg/s/Pa/m²); default from device_defaults",
    )
    p.add_argument("--t-wh-in-c", type=float, default=T_WH_IN_C)
    p.add_argument("--m-dot-wh", type=float, default=M_WH_KG_S_M2)
    p.add_argument("--t-amb-c", type=float, default=T_AMB_C)
    p.add_argument("--rh", type=float, default=RH_AMB)
    p.add_argument("--h-amb", type=float, default=H_AMB_W_M2_K)
    p.add_argument("--daily", action="store_true", help="Simulate full day of cycles")
    p.add_argument("--n-cycles", type=int, default=None)
    p.add_argument(
        "--warmup-cycles",
        type=int,
        default=2,
        help=(
            "Aitken-accelerated warmup rounds to reach the periodic steady state "
            "before the reporting day/cycle (0 disables, starting cold)"
        ),
    )
    p.add_argument("--water-inventory-csv", type=Path, default=None)
    p.add_argument("--water-inventory-plot", type=Path, default=None)
    p.add_argument(
        "--detailed",
        action="store_true",
        help="Write CSV and plot device temperatures (contactors, HTF loop, condenser) "
        "and boundary conditions over the cycle",
    )
    p.add_argument(
        "--detailed-csv",
        type=Path,
        default=None,
        help="Detailed diagnostics CSV path (default: outputs/detailed/diagnostics_<tag>.csv)",
    )
    p.add_argument(
        "--detailed-plot",
        type=Path,
        default=None,
        help="Detailed diagnostics PNG path (default: outputs/detailed/diagnostics_<tag>.png)",
    )
    args = p.parse_args()

    config = _build_config(args)
    profile = _build_profile(args, config)
    print(f"Sorbent: {config.sorbent}")

    if args.daily:
        yield_kg, eta, results = run_daily_operation(
            profile,
            config,
            n_cycles=args.n_cycles,
            warmup_cycles=args.warmup_cycles,
        )
        print(f"Daily water yield: {yield_kg:.2f} L/m²/day")
        print(f"Thermal efficiency (vs Q_wh): {eta:.3f}")
        if args.warmup_cycles > 0:
            print(f"Warmup: {args.warmup_cycles} cycle(s) before reporting day")
        n_cyc = len(results)
        half_a = results[0].half_a
        half_b = results[0].half_b
        warmup_note = f"; {args.warmup_cycles} warmup" if args.warmup_cycles > 0 else ""
        note = _half_cycle_note(half_a, half_b, config) + f"; {n_cyc} cycles/day{warmup_note}"
        inventory = water_inventory_daily_series(results, config=config, profile=profile)
    else:
        cyc = run_cycle(profile, config, warmup_cycles=args.warmup_cycles)
        print(f"Cycle water yield: {cyc.water_collected_kg_m2:.3f} L/m²")
        ha = cyc.half_a
        imb = abs(ha.integral_ads_kg_m2 - ha.integral_des_kg_m2)
        mean_m = 0.5 * (ha.integral_ads_kg_m2 + ha.integral_des_kg_m2)
        rel = imb / mean_m if mean_m > 1e-12 else 0.0
        print(f"Half-cycle mass balance error (first half): {rel:.1%}")
        print(f"  ∫ṁ_ads = {ha.integral_ads_kg_m2:.6f} kg/m²")
        print(f"  ∫ṁ_des = {ha.integral_des_kg_m2:.6f} kg/m²")
        note = _half_cycle_note(cyc.half_a, cyc.half_b, config)
        inventory = water_inventory_series(cyc, config=config, profile=profile)

    _write_inventory_outputs(args=args, config=config, inventory=inventory, note=note)

    if args.detailed:
        tag = _output_tag(args)
        if args.daily:
            detailed = detailed_daily_series(results, config=config, profile=profile)
            warmup_note = f", {args.warmup_cycles} warmup" if args.warmup_cycles > 0 else ""
            plot_title = (
                f"Device and boundaries — {args.profile} "
                f"({len(results)} cycles{warmup_note})"
            )
        else:
            detailed = detailed_series(cyc, config=config, profile=profile)
            plot_title = f"Device and boundaries — {args.profile}"

        detailed_csv = args.detailed_csv
        if detailed_csv is None:
            detailed_csv = _REPO / "outputs" / "detailed" / f"diagnostics_{tag}.csv"
        detailed_plot = args.detailed_plot
        if detailed_plot is None:
            detailed_plot = _REPO / "outputs" / "detailed" / f"diagnostics_{tag}.png"

        write_detailed_csv(detailed_csv, detailed)
        plot_detailed_diagnostics(detailed_plot, detailed, title=plot_title)
        print(f"Wrote {detailed_csv}")
        print(f"Wrote {detailed_plot}")


if __name__ == "__main__":
    main()
