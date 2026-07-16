#!/usr/bin/env python3
"""Waterfall plot of the global energy-balance closure for half-cycle A.

Summing the four contactor/loop/condenser ODEs (governing_eq.tex) cancels every
internal exchange term (contactor A <-> loop, loop <-> contactor D, vacuum-gap +
radiative exchange between D and the condenser), leaving only: waste heat in,
three ambient-loss terms, and a net latent term. This script integrates those
terms over half-cycle A from the recorded state trajectory and plots the
cumulative bar landing next to the actual sensible-energy change (ΔT × thermal
mass, read directly off the trajectory) -- the gap between them is the
validation residual checked by test_energy_balance_closes_hydrogel/_mof in
tests/test_waste_heat_sim.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from waste_heat_cycle_lumped.physics.correlations import (
    condenser_h_conv_w_m2_k,
    hx_effectiveness_q,
    waste_heat_to_loop_q_w,
)
from waste_heat_cycle_lumped.physics.sorbent import h_ads_j_per_kg, h_des_j_per_kg
from waste_heat_cycle_lumped.simulation.control import compute_controls
from waste_heat_cycle_lumped.simulation.device_config import DeviceConfig
from waste_heat_cycle_lumped.simulation.ode_system import HalfCycleResult, run_cycle
from waste_heat_cycle_lumped.weather.profiles import datacenter_baseline_profile

_REPO = Path(__file__).resolve().parent.parent

COLOR_GAIN = "#4C72B0"
COLOR_LOSS = "#C44E52"
COLOR_TOTAL = "#3d3d3d"
COLOR_ACTUAL = "#1b1b1b"


def _env_index(t_s: float, profile, n: int) -> int:
    return min(max(int(t_s / profile.dt_s), 0), n - 1)


def energy_terms_j_m2(half: HalfCycleResult, config: DeviceConfig, profile) -> dict[str, float]:
    """Integrate each externally-visible term of the summed energy balance (J/m²)."""
    params = config.thermal_params()
    ctrl_p = config.controller_params()
    t = np.asarray(half.time_s, dtype=float)
    t_a = np.asarray(half.t_a_c, dtype=float)
    t_d = np.asarray(half.t_d_c, dtype=float)
    t_f = np.asarray(half.t_f_c, dtype=float)
    t_cond = np.asarray(half.t_cond_c, dtype=float)
    m_ads = np.asarray(half.m_ads_kg_s_m2, dtype=float)
    m_des = np.asarray(half.m_des_kg_s_m2, dtype=float)
    n_env = len(profile.temperature_c)
    h_ads = h_ads_j_per_kg(config)
    h_des = h_des_j_per_kg(config)

    q_wh = np.zeros(len(t))
    q_loss_a = np.zeros(len(t))
    q_loss_loop = np.zeros(len(t))
    q_loss_cond = np.zeros(len(t))
    q_latent = np.zeros(len(t))

    for k in range(len(t)):
        idx = _env_index(float(t[k]), profile, n_env)
        m_f = compute_controls(
            t_a_c=float(t_a[k]),
            t_d_c=float(t_d[k]),
            m_ads_kg_s_m2=float(m_ads[k]),
            m_des_kg_s_m2=float(m_des[k]),
            params=ctrl_p,
            integral_ads_kg_m2=0.0,
            integral_des_kg_m2=0.0,
        ).m_dot_f_kg_s_m2
        mdot_cp = m_f * params.fluid_cp_j_kg_k
        q_a_to_f = hx_effectiveness_q(mdot_cp, params.ua_adsorber_w_k, t_a[k] - t_f[k])
        q_f_to_d = hx_effectiveness_q(mdot_cp, params.ua_desorber_w_k, t_f[k] - t_d[k])
        q_loss_loop[k] = -params.loop_loss_fraction * (abs(q_a_to_f) + abs(q_f_to_d))
        q_wh_to_f, _ = waste_heat_to_loop_q_w(
            m_dot_wh_kg_s=profile.m_dot_wh_kg_s_m2[idx],
            cp_wh_j_kg_k=params.cp_wh_j_kg_k,
            t_wh_in_c=profile.t_wh_in_c[idx],
            t_f_c=float(t_f[k]),
            ua_wh_w_k=params.wh_hx_ua_w_k,
        )
        q_wh[k] = q_wh_to_f
        q_loss_a[k] = -(
            profile.h_amb_w_m2_k[idx] * params.contactor_area_m2 * (t_a[k] - profile.temperature_c[idx])
        )
        h_conv_cond = condenser_h_conv_w_m2_k(profile.h_amb_w_m2_k[idx], fin_area_ratio=params.fin_area_ratio)
        q_loss_cond[k] = -(h_conv_cond * (t_cond[k] - profile.temperature_c[idx]))
        q_latent[k] = m_ads[k] * h_ads - m_des[k] * h_des + m_des[k] * params.h_fg_j_per_kg

    delta_u_actual = (
        params.contactor_thermal_mass_j_m2_k * (t_a[-1] - t_a[0])
        + params.contactor_thermal_mass_j_m2_k * (t_d[-1] - t_d[0])
        + params.fluid_thermal_mass_j_m2_k * (t_f[-1] - t_f[0])
        + params.condenser_thermal_mass_j_m2_k * (t_cond[-1] - t_cond[0])
    )

    return {
        "Waste heat in": float(np.trapezoid(q_wh, t)),
        "Bed-A ambient loss": float(np.trapezoid(q_loss_a, t)),
        "HTF loop loss": float(np.trapezoid(q_loss_loop, t)),
        "Condenser ambient loss": float(np.trapezoid(q_loss_cond, t)),
        "Net latent + sorption": float(np.trapezoid(q_latent, t)),
        "_actual_delta_u": float(delta_u_actual),
    }


def plot_waterfall(ax, terms: dict[str, float], *, title: str) -> None:
    keys = [
        "Waste heat in",
        "Bed-A ambient loss",
        "HTF loop loss",
        "Condenser ambient loss",
        "Net latent + sorption",
    ]
    labels = [
        "Waste heat\nin",
        "Bed-A\nambient loss",
        "HTF loop\nloss",
        "Condenser\nambient loss",
        "Net latent\n+ sorption",
    ]
    values_kj = [terms[k] / 1000.0 for k in keys]
    actual_kj = terms["_actual_delta_u"] / 1000.0

    cum = 0.0
    x = np.arange(len(values_kj) + 1)
    for i, v in enumerate(values_kj):
        bottom = cum
        cum += v
        color = COLOR_GAIN if v >= 0 else COLOR_LOSS
        ax.bar(i, v, bottom=bottom, width=0.62, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([i - 0.31, i + 0.62 - 0.31], [cum, cum], color="0.55", linewidth=1.0, zorder=2)

    predicted_kj = cum
    ax.bar(len(values_kj), predicted_kj, width=0.62, color=COLOR_TOTAL, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(
        [len(values_kj)],
        [actual_kj],
        marker="D",
        s=70,
        color=COLOR_ACTUAL,
        zorder=5,
        label="Actual ΔU (from state trajectory)",
    )

    resid_pct = abs(predicted_kj - actual_kj) / max(abs(predicted_kj), abs(actual_kj), 1e-9) * 100.0
    ax.annotate(
        f"residual = {resid_pct:.2f}%",
        xy=(len(values_kj), actual_kj),
        xytext=(len(values_kj) - 0.05, actual_kj + 0.06 * max(abs(predicted_kj), 1.0)),
        fontsize=8,
        color="0.3",
        ha="right",
    )

    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels + ["Total"], fontsize=8)
    ax.set_ylabel("Energy (kJ/m²)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_energy_balance(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    configs = [
        ("Hydrogel (PAM-LiCl), half-cycle A", DeviceConfig.datacenter_baseline()),
        ("MOF (MIL-100(Fe)), half-cycle A", DeviceConfig.mof_baseline()),
    ]
    for ax, (title, cfg) in zip(axes, configs):
        profile = datacenter_baseline_profile(tau_half_s=cfg.tau_half_s)
        cyc = run_cycle(profile, cfg)
        terms = energy_terms_j_m2(cyc.half_a, cfg, profile)
        plot_waterfall(ax, terms, title=title)
        resid = abs(
            (sum(v for k, v in terms.items() if not k.startswith("_")) - terms["_actual_delta_u"])
        ) / max(abs(terms["_actual_delta_u"]), 1.0)
        print(f"{title}: residual = {resid:.2%}")

    diamond_handles, diamond_labels = axes[0].get_legend_handles_labels()
    handles = [
        Patch(color=COLOR_GAIN, label="Energy in"),
        Patch(color=COLOR_LOSS, label="Energy out"),
        Patch(color=COLOR_TOTAL, label="Predicted ΔU (sum of terms)"),
    ] + diamond_handles
    labels = ["Energy in", "Energy out", "Predicted ΔU (sum of terms)"] + diamond_labels
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Global energy-balance closure: waste heat in vs. losses + sensible storage", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO / "outputs" / "energy_balance" / "energy_balance_waterfall.png",
    )
    args = parser.parse_args()
    plot_energy_balance(args.output)


if __name__ == "__main__":
    main()
