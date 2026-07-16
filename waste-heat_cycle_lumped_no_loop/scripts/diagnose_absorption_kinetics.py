#!/usr/bin/env python3
"""Compare Wilson absorption kinetics: natural flux vs integrated inventory.

Generates a diagnostic plot for the absorption half-cycle showing
water_in_gel (L/m²) alongside natural and equalized mass-transfer fluxes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from waste_heat_cycle_lumped_no_loop.physics import device_defaults as dd
from waste_heat_cycle_lumped_no_loop.physics.mass_transfer import (
    concentration_ratio_absorption,
    dH_dt,
    dc_w_dt,
    m_ads_kg_s_m2_from_state,
)
from waste_heat_cycle_lumped_no_loop.physics.salt_properties import (
    equilibrium_c_w_from_dvs_at_rh,
    FABRICATION_EQUILIBRIUM_RH,
)
from waste_heat_cycle_lumped_no_loop.physics.sorbent import mass_transfer_params
from waste_heat_cycle_lumped_no_loop.simulation.device_config import DeviceConfig
from waste_heat_cycle_lumped_no_loop.simulation.ode_system import run_cycle
from waste_heat_cycle_lumped_no_loop.simulation.water_inventory import water_inventory_series
from waste_heat_cycle_lumped_no_loop.weather.profiles import datacenter_baseline_profile

_REPO = Path(__file__).resolve().parent.parent


def _natural_adsorption_flux_at_state(
    *,
    c_w: float,
    h_m: float,
    t_c: float,
    rh: float,
    config: DeviceConfig,
) -> float:
    params = mass_transfer_params(config)
    c_r = concentration_ratio_absorption(rh)
    dc = dc_w_dt(c_w, t_gel_c=t_c, c_r=c_r, params=params, h_m=h_m, phase="absorption")
    dh = dH_dt(c_w, t_gel_c=t_c, c_r=c_r, params=params, h_m=h_m, phase="absorption")
    return m_ads_kg_s_m2_from_state(c_w, h_m, dc, dh)


def _print_state_comparison(config: DeviceConfig) -> None:
    params = mass_transfer_params(config)
    h0 = config.hydrogel_thickness_m
    c_regen = equilibrium_c_w_from_dvs_at_rh(
        FABRICATION_EQUILIBRIUM_RH, h_m=h0, h0_ref_m=h0
    )
    t_c = dd.T_AMB_C
    rh = dd.RH_AMB

    dc = dc_w_dt(
        c_regen,
        t_gel_c=t_c,
        c_r=concentration_ratio_absorption(rh),
        params=params,
        h_m=h0,
        phase="absorption",
    )
    dh = dH_dt(
        c_regen,
        t_gel_c=t_c,
        c_r=concentration_ratio_absorption(rh),
        params=params,
        h_m=h0,
        phase="absorption",
    )
    m_nat = m_ads_kg_s_m2_from_state(c_regen, h0, dc, dh)

    from waste_heat_cycle_lumped_no_loop.physics.salt_properties import WATER_MOLAR_MASS_KG_MOL

    m_dc_only = max(0.0, WATER_MOLAR_MASS_KG_MOL * dc * h0)
    m_dh_only = max(0.0, WATER_MOLAR_MASS_KG_MOL * c_regen * dh)

    print("\n--- Wilson natural absorption at dry-bed start (20% RH regen) ---")
    print(f"  T_amb = {t_c} °C, RH = {rh:.0%}, H = {h0*1e3:.1f} mm, g = {config.g_conv_m_s} m/s")
    print(f"  dc_w/dt = {dc:.3e} mol/m³/s")
    print(f"  dH/dt   = {dh:.3e} m/s")
    print(f"  m_ads (dc_w·H term)  = {m_dc_only*3600:.4f} kg/m²/h")
    print(f"  m_ads (c_w·dH term)  = {m_dh_only*3600:.4f} kg/m²/h")
    print(f"  m_ads (total)        = {m_nat*3600:.4f} kg/m²/h")
    print(f"  dc_w vs dH flux ratio = {m_dh_only / max(m_dc_only, 1e-20):.1f}x (c_w·dH dominates)")


def plot_absorption_diagnostic(
    path: Path,
    *,
    config: DeviceConfig,
    profile,
) -> None:
    cycle = run_cycle(profile, config)
    inv = water_inventory_series(cycle, config=config, profile=profile)

    abs_mask = inv.phase == "absorption"
    t_s = inv.time_s[abs_mask]
    t_h = (t_s - t_s[0]) / 3600.0
    water = inv.water_l_m2[abs_mask]
    m_nat = inv.m_ads_natural_kg_s_m2[abs_mask] * 3600.0
    m_eq = inv.m_eq_kg_s_m2[abs_mask] * 3600.0

    # Trapezoidal cumulative from natural flux (kg/m² ≈ L/m²)
    cum_nat = np.zeros(len(t_s))
    for k in range(len(t_s) - 1):
        dt = float(t_s[k + 1] - t_s[k])
        cum_nat[k + 1] = cum_nat[k] + 0.5 * (m_nat[k] + m_nat[k + 1]) * dt / 3600.0

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax0 = axes[0]
    ax0.plot(t_h, water, color="#4C72B0", linewidth=2, label="water in gel (sim)")
    ax0.set_ylabel("Water in gel (L/m²)")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Absorption half-cycle diagnostics (datacenter baseline)")

    ax1 = axes[1]
    ax1.plot(t_h, m_nat, color="#55A868", linewidth=1.5, label="m_ads natural")
    ax1.plot(t_h, m_eq, color="#C44E52", linewidth=1.5, linestyle="--", label="m_eq (min ads, des)")
    ax1.set_ylabel("Flux (kg/m²/h)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[2]
    ax2.plot(t_h, water - water[0], color="#4C72B0", linewidth=2, label="Δ water (sim)")
    ax2.plot(t_h, cum_nat, color="#55A868", linewidth=1.5, linestyle="--", label="∫ m_ads_nat dt")
    ax2.set_xlabel("Time in absorption half-cycle (h)")
    ax2.set_ylabel("Uptake (L/m²)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    swing = float(water[-1] - water[0])
    int_eq = float(np.trapezoid(inv.m_eq_kg_s_m2[abs_mask], t_s))
    int_nat = float(np.trapezoid(inv.m_ads_natural_kg_s_m2[abs_mask], t_s))
    print("\n--- Absorption half-cycle integration ---")
    print(f"  Duration = {t_h[-1]:.3f} h")
    print(f"  Δ water in gel (sim)     = {swing:.4f} L/m²")
    print(f"  ∫ m_eq dt (sim)          = {int_eq:.4f} L/m²")
    print(f"  ∫ m_ads_nat dt (natural) = {int_nat:.4f} L/m²")
    print(f"  Natural / equalized mean flux = {np.mean(m_nat)/max(np.mean(m_eq),1e-12):.2f}x")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO / "outputs" / "water_inventory" / "absorption_kinetics_diagnostic.png",
    )
    args = parser.parse_args()

    config = DeviceConfig.datacenter_baseline()
    profile = datacenter_baseline_profile(tau_half_s=config.tau_half_s)

    _print_state_comparison(config)
    plot_absorption_diagnostic(args.output, config=config, profile=profile)


if __name__ == "__main__":
    main()
