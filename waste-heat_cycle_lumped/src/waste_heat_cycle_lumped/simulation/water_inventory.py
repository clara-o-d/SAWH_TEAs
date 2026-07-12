"""Water inventory time series for two-bed cycles."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from waste_heat_cycle_lumped.physics import device_defaults as dd
from waste_heat_cycle_lumped.physics.contactor_balances import ThermalEnvironment
from waste_heat_cycle_lumped.physics.mass_transfer import rh_outside_desorber
from waste_heat_cycle_lumped.physics.sorbent import (
    fluxes_for_control,
    inventory_column,
    inventory_ylabel,
    is_hydrogel,
    mass_rates,
    water_in_gel_l_m2,
    water_kg_m2_bed,
)
from waste_heat_cycle_lumped.simulation.control import compute_controls
from waste_heat_cycle_lumped.simulation.device_config import DeviceConfig
from waste_heat_cycle_lumped.simulation.ode_system import CycleResult, HalfCycleResult
from waste_heat_cycle_lumped.weather.profiles import HalfCycleProfile

MassTransferLimit = Literal["absorption", "desorption", "balanced"]
TrackedPhase = Literal["absorption", "desorption"]


@dataclass(frozen=True, slots=True)
class WaterInventorySeries:
    time_s: np.ndarray
    water_l_m2: np.ndarray
    phase: np.ndarray
    half_cycle_end_s: float
    cycle_index: np.ndarray
    half_cycle: np.ndarray
    m_ads_kg_s_m2: np.ndarray
    m_des_kg_s_m2: np.ndarray
    m_eq_kg_s_m2: np.ndarray
    m_ads_natural_kg_s_m2: np.ndarray
    m_des_natural_kg_s_m2: np.ndarray
    mass_transfer_limit: np.ndarray
    operating_flux_role: np.ndarray
    c_w_mol_m3: np.ndarray
    h_m: np.ndarray
    t_tracked_c: np.ndarray
    t_partner_c: np.ndarray
    t_f_c: np.ndarray
    t_cond_c: np.ndarray
    rh_vapor_gap: np.ndarray
    c_vac_kg_s_pa_m2: np.ndarray
    m_dot_f_kg_s_m2: np.ndarray
    collected_water_l_m2: np.ndarray


def cumulative_desorption_yield_l_m2(
    time_s: np.ndarray,
    m_des_kg_s_m2: np.ndarray,
) -> np.ndarray:
    """Trapezoidal cumulative integral of desorption flux (kg/m² ≈ L/m²)."""
    n = len(time_s)
    out = np.zeros(n, dtype=float)
    for k in range(n - 1):
        dt = float(time_s[k + 1] - time_s[k])
        out[k + 1] = out[k] + 0.5 * (m_des_kg_s_m2[k] + m_des_kg_s_m2[k + 1]) * dt
    return out


def _default_env() -> ThermalEnvironment:
    return ThermalEnvironment(
        t_amb_c=dd.T_AMB_C,
        rh_amb=dd.RH_AMB,
        h_amb_w_m2_k=dd.H_AMB_W_M2_K,
        t_wh_in_c=dd.T_WH_IN_C,
        m_dot_wh_kg_s_m2=dd.M_WH_KG_S_M2,
    )


def _env_at_time(t_s: float, profile: HalfCycleProfile | None) -> ThermalEnvironment:
    if profile is None:
        return _default_env()
    dt = profile.dt_s
    i = min(max(int(t_s / dt), 0), len(profile.temperature_c) - 1)
    return ThermalEnvironment(
        t_amb_c=profile.temperature_c[i],
        rh_amb=profile.relative_humidity[i],
        h_amb_w_m2_k=profile.h_amb_w_m2_k[i],
        t_wh_in_c=profile.t_wh_in_c[i],
        m_dot_wh_kg_s_m2=profile.m_dot_wh_kg_s_m2[i],
    )


def _mass_transfer_limit(nat_ads: float, nat_des: float) -> MassTransferLimit:
    if nat_des < 0.99 * nat_ads:
        return "desorption"
    if nat_ads < 0.99 * nat_des:
        return "absorption"
    return "balanced"


def _bed_water_l_m2(
    loading: np.ndarray,
    h_m: np.ndarray | None,
    *,
    config: DeviceConfig,
) -> np.ndarray:
    if is_hydrogel(config):
        assert h_m is not None
        return np.array(
            [
                water_in_gel_l_m2(float(q), float(h), config=config)
                for q, h in zip(loading, h_m)
            ]
        )
    return np.array([water_kg_m2_bed(float(q), config=config) for q in loading])


def _tracked_half_series(
    half: HalfCycleResult,
    *,
    config: DeviceConfig,
    profile: HalfCycleProfile | None,
    tracked_phase: TrackedPhase,
    half_label: Literal["A", "B"],
    cycle_index: int,
    t_offset_s: float,
) -> WaterInventorySeries:
    """Build detailed inventory for one physical bed through one half-cycle."""
    n = len(half.time_s)
    if tracked_phase == "absorption":
        q = half.q_a
        h = half.h_a
        t_tracked = half.t_a_c
        t_partner = half.t_d_c
    else:
        q = half.q_d
        h = half.h_d
        t_tracked = half.t_d_c
        t_partner = half.t_a_c

    ctrl_p = config.controller_params()
    water = _bed_water_l_m2(q, h, config=config)

    m_ads = np.asarray(half.m_ads_kg_s_m2, dtype=float)
    m_des = np.asarray(half.m_des_kg_s_m2, dtype=float)
    m_eq = np.minimum(m_ads, m_des)

    m_ads_nat = np.zeros(n)
    m_des_nat = np.zeros(n)
    limits: list[MassTransferLimit] = []
    c_vac = np.zeros(n)
    m_dot_f = np.zeros(n)
    rh_gap = np.zeros(n)

    for k in range(n):
        env = _env_at_time(float(half.time_s[k]), profile)
        h_a = float(half.h_a[k]) if half.h_a is not None else config.hydrogel_thickness_m
        h_d = float(half.h_d[k]) if half.h_d is not None else config.hydrogel_thickness_m
        m_ads_ctrl, m_des_ctrl = fluxes_for_control(
            loading_a=float(half.q_a[k]),
            loading_d=float(half.q_d[k]),
            h_a=h_a,
            h_d=h_d,
            t_a_c=float(half.t_a_c[k]),
            t_d_c=float(half.t_d_c[k]),
            t_cond_c=float(half.t_cond_c[k]),
            rh_amb=env.rh_amb,
            p_cond_pa=config.p_cond_pa,
            c_vac_kg_s_pa_m2=ctrl_p.c_vac_base_kg_s_pa_m2,
            config=config,
        )
        controls = compute_controls(
            t_a_c=float(half.t_a_c[k]),
            t_d_c=float(half.t_d_c[k]),
            m_ads_kg_s_m2=m_ads_ctrl,
            m_des_kg_s_m2=m_des_ctrl,
            params=ctrl_p,
            integral_ads_kg_m2=0.0,
            integral_des_kg_m2=0.0,
        )
        nat = mass_rates(
            loading_a=float(half.q_a[k]),
            loading_d=float(half.q_d[k]),
            h_a=h_a,
            h_d=h_d,
            t_a_c=float(half.t_a_c[k]),
            t_d_c=float(half.t_d_c[k]),
            t_cond_c=float(half.t_cond_c[k]),
            rh_amb=env.rh_amb,
            p_cond_pa=config.p_cond_pa,
            c_vac_kg_s_pa_m2=controls.c_vac_kg_s_pa_m2,
            config=config,
            equalize=False,
        )
        m_ads_nat[k] = nat.m_ads_kg_s_m2
        m_des_nat[k] = nat.m_des_kg_s_m2
        limits.append(_mass_transfer_limit(nat.m_ads_kg_s_m2, nat.m_des_kg_s_m2))
        c_vac[k] = controls.c_vac_kg_s_pa_m2
        m_dot_f[k] = controls.m_dot_f_kg_s_m2
        rh_gap[k] = rh_outside_desorber(float(half.t_d_c[k]), float(half.t_cond_c[k]))

    if is_hydrogel(config):
        assert h is not None
        c_w = np.asarray(q, dtype=float)
        h_arr = np.asarray(h, dtype=float)
    else:
        c_w = np.asarray(q, dtype=float)
        h_arr = np.full(n, config.hydrogel_thickness_m)

    operating_role = np.array([tracked_phase] * n, dtype=object)
    return WaterInventorySeries(
        time_s=np.asarray(half.time_s, dtype=float) + t_offset_s,
        water_l_m2=water,
        phase=np.array([tracked_phase] * n, dtype=object),
        half_cycle_end_s=0.0,
        cycle_index=np.full(n, cycle_index, dtype=int),
        half_cycle=np.array([half_label] * n, dtype=object),
        m_ads_kg_s_m2=m_ads,
        m_des_kg_s_m2=m_des,
        m_eq_kg_s_m2=m_eq,
        m_ads_natural_kg_s_m2=m_ads_nat,
        m_des_natural_kg_s_m2=m_des_nat,
        mass_transfer_limit=np.array(limits, dtype=object),
        operating_flux_role=operating_role,
        c_w_mol_m3=c_w,
        h_m=h_arr,
        t_tracked_c=np.asarray(t_tracked, dtype=float),
        t_partner_c=np.asarray(t_partner, dtype=float),
        t_f_c=np.asarray(half.t_f_c, dtype=float),
        t_cond_c=np.asarray(half.t_cond_c, dtype=float),
        rh_vapor_gap=rh_gap,
        c_vac_kg_s_pa_m2=c_vac,
        m_dot_f_kg_s_m2=m_dot_f,
        collected_water_l_m2=np.zeros(n, dtype=float),
    )


def _concat_series(chunks: list[WaterInventorySeries], *, skip_first: int) -> WaterInventorySeries:
    if not chunks:
        raise ValueError("At least one inventory chunk is required.")

    def cat(attr: str) -> np.ndarray:
        parts = [getattr(chunks[0], attr)]
        for chunk in chunks[1:]:
            arr = getattr(chunk, attr)
            parts.append(arr[skip_first:])
        return np.concatenate(parts)

    return WaterInventorySeries(
        time_s=cat("time_s"),
        water_l_m2=cat("water_l_m2"),
        phase=cat("phase"),
        half_cycle_end_s=chunks[0].half_cycle_end_s,
        cycle_index=cat("cycle_index"),
        half_cycle=cat("half_cycle"),
        m_ads_kg_s_m2=cat("m_ads_kg_s_m2"),
        m_des_kg_s_m2=cat("m_des_kg_s_m2"),
        m_eq_kg_s_m2=cat("m_eq_kg_s_m2"),
        m_ads_natural_kg_s_m2=cat("m_ads_natural_kg_s_m2"),
        m_des_natural_kg_s_m2=cat("m_des_natural_kg_s_m2"),
        mass_transfer_limit=cat("mass_transfer_limit"),
        operating_flux_role=cat("operating_flux_role"),
        c_w_mol_m3=cat("c_w_mol_m3"),
        h_m=cat("h_m"),
        t_tracked_c=cat("t_tracked_c"),
        t_partner_c=cat("t_partner_c"),
        t_f_c=cat("t_f_c"),
        t_cond_c=cat("t_cond_c"),
        rh_vapor_gap=cat("rh_vapor_gap"),
        c_vac_kg_s_pa_m2=cat("c_vac_kg_s_pa_m2"),
        m_dot_f_kg_s_m2=cat("m_dot_f_kg_s_m2"),
        collected_water_l_m2=cat("collected_water_l_m2"),
    )


def water_inventory_series(
    cycle: CycleResult,
    *,
    config: DeviceConfig,
    profile: HalfCycleProfile | None = None,
    cycle_index: int = 0,
) -> WaterInventorySeries:
    """One physical bed: absorbs in half A, desorbs in half B (same gel as solar_lumped)."""
    ha = cycle.half_a
    hb = cycle.half_b
    abs_chunk = _tracked_half_series(
        ha,
        config=config,
        profile=profile,
        tracked_phase="absorption",
        half_label="A",
        cycle_index=cycle_index,
        t_offset_s=0.0,
    )
    des_chunk = _tracked_half_series(
        hb,
        config=config,
        profile=profile,
        tracked_phase="desorption",
        half_label="B",
        cycle_index=cycle_index,
        t_offset_s=float(ha.time_s[-1]) if len(ha.time_s) else 0.0,
    )
    out = _concat_series([abs_chunk, des_chunk], skip_first=1)
    yield_abs = cumulative_desorption_yield_l_m2(ha.time_s, ha.m_des_kg_s_m2)
    yield_des = cumulative_desorption_yield_l_m2(hb.time_s, hb.m_des_kg_s_m2) + yield_abs[-1]
    collected = np.concatenate([yield_abs, yield_des[1:]])
    return WaterInventorySeries(
        time_s=out.time_s,
        water_l_m2=out.water_l_m2,
        phase=out.phase,
        half_cycle_end_s=float(ha.time_s[-1]) if len(ha.time_s) else 0.0,
        cycle_index=out.cycle_index,
        half_cycle=out.half_cycle,
        m_ads_kg_s_m2=out.m_ads_kg_s_m2,
        m_des_kg_s_m2=out.m_des_kg_s_m2,
        m_eq_kg_s_m2=out.m_eq_kg_s_m2,
        m_ads_natural_kg_s_m2=out.m_ads_natural_kg_s_m2,
        m_des_natural_kg_s_m2=out.m_des_natural_kg_s_m2,
        mass_transfer_limit=out.mass_transfer_limit,
        operating_flux_role=out.operating_flux_role,
        c_w_mol_m3=out.c_w_mol_m3,
        h_m=out.h_m,
        t_tracked_c=out.t_tracked_c,
        t_partner_c=out.t_partner_c,
        t_f_c=out.t_f_c,
        t_cond_c=out.t_cond_c,
        rh_vapor_gap=out.rh_vapor_gap,
        c_vac_kg_s_pa_m2=out.c_vac_kg_s_pa_m2,
        m_dot_f_kg_s_m2=out.m_dot_f_kg_s_m2,
        collected_water_l_m2=collected,
    )


def _append_cycle(base: WaterInventorySeries, nxt: WaterInventorySeries) -> WaterInventorySeries:
    """Append a cycle after base, skipping duplicate boundary point and shifting time."""
    t0 = float(base.time_s[-1])
    shifted = WaterInventorySeries(
        time_s=nxt.time_s[1:] + t0,
        water_l_m2=nxt.water_l_m2[1:],
        phase=nxt.phase[1:],
        half_cycle_end_s=base.half_cycle_end_s,
        cycle_index=nxt.cycle_index[1:],
        half_cycle=nxt.half_cycle[1:],
        m_ads_kg_s_m2=nxt.m_ads_kg_s_m2[1:],
        m_des_kg_s_m2=nxt.m_des_kg_s_m2[1:],
        m_eq_kg_s_m2=nxt.m_eq_kg_s_m2[1:],
        m_ads_natural_kg_s_m2=nxt.m_ads_natural_kg_s_m2[1:],
        m_des_natural_kg_s_m2=nxt.m_des_natural_kg_s_m2[1:],
        mass_transfer_limit=nxt.mass_transfer_limit[1:],
        operating_flux_role=nxt.operating_flux_role[1:],
        c_w_mol_m3=nxt.c_w_mol_m3[1:],
        h_m=nxt.h_m[1:],
        t_tracked_c=nxt.t_tracked_c[1:],
        t_partner_c=nxt.t_partner_c[1:],
        t_f_c=nxt.t_f_c[1:],
        t_cond_c=nxt.t_cond_c[1:],
        rh_vapor_gap=nxt.rh_vapor_gap[1:],
        c_vac_kg_s_pa_m2=nxt.c_vac_kg_s_pa_m2[1:],
        m_dot_f_kg_s_m2=nxt.m_dot_f_kg_s_m2[1:],
        collected_water_l_m2=nxt.collected_water_l_m2[1:] + float(base.collected_water_l_m2[-1]),
    )
    return _concat_series([base, shifted], skip_first=0)


def water_inventory_daily_series(
    cycles: list[CycleResult],
    *,
    config: DeviceConfig,
    profile: HalfCycleProfile | None = None,
) -> WaterInventorySeries:
    if not cycles:
        raise ValueError("At least one cycle is required.")
    out = water_inventory_series(
        cycles[0],
        config=config,
        profile=profile,
        cycle_index=0,
    )
    for i, cycle in enumerate(cycles[1:], start=1):
        chunk = water_inventory_series(
            cycle,
            config=config,
            profile=profile,
            cycle_index=i,
        )
        out = _append_cycle(out, chunk)
    return out


def _inventory_csv_fieldnames(config: DeviceConfig) -> list[str]:
    col = inventory_column(config)
    fields = [
        "time_s",
        "time_h",
        "cycle_index",
        "half_cycle",
        "tracked_bed_phase",
        "operating_flux_role",
        "mass_transfer_limit",
        col,
        "collected_water_l_m2",
        "c_w_mol_m3",
        "h_m",
        "m_ads_kg_s_m2",
        "m_des_kg_s_m2",
        "m_eq_kg_s_m2",
        "m_ads_natural_kg_s_m2",
        "m_des_natural_kg_s_m2",
        "t_tracked_c",
        "t_partner_c",
        "t_f_c",
        "t_cond_c",
        "rh_vapor_gap",
        "c_vac_kg_s_pa_m2",
        "m_dot_f_kg_s_m2",
    ]
    if not is_hydrogel(config):
        fields = [f for f in fields if f not in ("c_w_mol_m3", "h_m")]
    return fields


def write_water_inventory_csv(path: Path, series: WaterInventorySeries, *, config: DeviceConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    col = inventory_column(config)
    fields = _inventory_csv_fieldnames(config)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for k in range(len(series.time_s)):
            row: dict[str, object] = {
                "time_s": f"{float(series.time_s[k]):.3f}",
                "time_h": f"{float(series.time_s[k]) / 3600.0:.6f}",
                "cycle_index": int(series.cycle_index[k]),
                "half_cycle": str(series.half_cycle[k]),
                "tracked_bed_phase": str(series.phase[k]),
                "operating_flux_role": str(series.operating_flux_role[k]),
                "mass_transfer_limit": str(series.mass_transfer_limit[k]),
                col: f"{float(series.water_l_m2[k]):.6f}",
                "collected_water_l_m2": f"{float(series.collected_water_l_m2[k]):.6f}",
                "c_w_mol_m3": f"{float(series.c_w_mol_m3[k]):.3f}",
                "h_m": f"{float(series.h_m[k]):.6f}",
                "m_ads_kg_s_m2": f"{float(series.m_ads_kg_s_m2[k]):.9e}",
                "m_des_kg_s_m2": f"{float(series.m_des_kg_s_m2[k]):.9e}",
                "m_eq_kg_s_m2": f"{float(series.m_eq_kg_s_m2[k]):.9e}",
                "m_ads_natural_kg_s_m2": f"{float(series.m_ads_natural_kg_s_m2[k]):.9e}",
                "m_des_natural_kg_s_m2": f"{float(series.m_des_natural_kg_s_m2[k]):.9e}",
                "t_tracked_c": f"{float(series.t_tracked_c[k]):.4f}",
                "t_partner_c": f"{float(series.t_partner_c[k]):.4f}",
                "t_f_c": f"{float(series.t_f_c[k]):.4f}",
                "t_cond_c": f"{float(series.t_cond_c[k]):.4f}",
                "rh_vapor_gap": f"{float(series.rh_vapor_gap[k]):.6f}",
                "c_vac_kg_s_pa_m2": f"{float(series.c_vac_kg_s_pa_m2[k]):.9e}",
                "m_dot_f_kg_s_m2": f"{float(series.m_dot_f_kg_s_m2[k]):.9e}",
            }
            w.writerow([row[name] for name in fields])


def plot_water_inventory(
    path: Path,
    series: WaterInventorySeries,
    *,
    config: DeviceConfig,
    title: str | None = None,
    half_cycle_markers: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_h = series.time_s / 3600.0
    fig, (ax_inv, ax_yield) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax_inv.plot(time_h, series.water_l_m2, color="#4C72B0", linewidth=2)
    ax_yield.plot(time_h, series.collected_water_l_m2, color="#C44E52", linewidth=2)

    if half_cycle_markers:
        half_mark_h = series.half_cycle_end_s / 3600.0
        for ax in (ax_inv, ax_yield):
            ax.axvline(half_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
        cycle_period_h = 2.0 * series.half_cycle_end_s / 3600.0
        if cycle_period_h > 0.0 and time_h[-1] > cycle_period_h * 1.5:
            t_end_h = float(time_h[-1])
            t = cycle_period_h
            while t < t_end_h - 1e-9:
                for ax in (ax_inv, ax_yield):
                    ax.axvline(t, color="k", linewidth=0.5, linestyle="--", alpha=0.25)
                t += cycle_period_h

    ax_inv.set_ylabel(inventory_ylabel(config))
    ax_inv.grid(True, alpha=0.3)
    ax_yield.set_xlabel("Time (h)")
    ax_yield.set_ylabel("Collected water (L/m²)")
    ax_yield.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
