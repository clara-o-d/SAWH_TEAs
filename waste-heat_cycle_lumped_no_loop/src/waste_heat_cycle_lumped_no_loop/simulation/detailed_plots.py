"""Device temperatures and boundary conditions for two-bed waste-heat cycles."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from waste_heat_cycle_lumped_no_loop.physics.contactor_balances import ThermalEnvironment
from waste_heat_cycle_lumped_no_loop.physics import device_defaults as dd
from waste_heat_cycle_lumped_no_loop.simulation.device_config import DeviceConfig
from waste_heat_cycle_lumped_no_loop.simulation.ode_system import CycleResult, HalfCycleResult
from waste_heat_cycle_lumped_no_loop.weather.profiles import HalfCycleProfile

HalfCycleLabel = Literal["A", "B"]


@dataclass(frozen=True, slots=True)
class DetailedSeries:
    time_s: np.ndarray
    cycle_index: np.ndarray
    half_cycle: np.ndarray
    half_cycle_end_s: float
    t_a_c: np.ndarray
    t_d_c: np.ndarray
    t_cond_c: np.ndarray
    t_amb_c: np.ndarray
    relative_humidity: np.ndarray
    h_amb_w_m2_k: np.ndarray
    t_wh_in_c: np.ndarray
    m_dot_wh_kg_s_m2: np.ndarray
    n_cycles: int = 1


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


def _half_boundary_series(
    half: HalfCycleResult,
    *,
    profile: HalfCycleProfile | None,
    half_label: HalfCycleLabel,
    cycle_index: int,
    t_offset_s: float,
) -> DetailedSeries:
    n = len(half.time_s)
    t_amb = np.zeros(n)
    rh = np.zeros(n)
    h_amb = np.zeros(n)
    t_wh = np.zeros(n)
    m_dot_wh = np.zeros(n)
    for k in range(n):
        env = _env_at_time(float(half.time_s[k]), profile)
        t_amb[k] = env.t_amb_c
        rh[k] = env.rh_amb
        h_amb[k] = env.h_amb_w_m2_k
        t_wh[k] = env.t_wh_in_c
        m_dot_wh[k] = env.m_dot_wh_kg_s_m2

    return DetailedSeries(
        time_s=np.asarray(half.time_s, dtype=float) + t_offset_s,
        cycle_index=np.full(n, cycle_index, dtype=int),
        half_cycle=np.array([half_label] * n, dtype=object),
        half_cycle_end_s=0.0,
        t_a_c=np.asarray(half.t_a_c, dtype=float),
        t_d_c=np.asarray(half.t_d_c, dtype=float),
        t_cond_c=np.asarray(half.t_cond_c, dtype=float),
        t_amb_c=t_amb,
        relative_humidity=rh,
        h_amb_w_m2_k=h_amb,
        t_wh_in_c=t_wh,
        m_dot_wh_kg_s_m2=m_dot_wh,
    )


def _concat_series(chunks: list[DetailedSeries], *, skip_first: int) -> DetailedSeries:
    if not chunks:
        raise ValueError("At least one detailed chunk is required.")

    def cat(attr: str) -> np.ndarray:
        parts = [getattr(chunks[0], attr)]
        for chunk in chunks[1:]:
            arr = getattr(chunk, attr)
            parts.append(arr[skip_first:])
        return np.concatenate(parts)

    return DetailedSeries(
        time_s=cat("time_s"),
        cycle_index=cat("cycle_index"),
        half_cycle=cat("half_cycle"),
        half_cycle_end_s=chunks[0].half_cycle_end_s,
        t_a_c=cat("t_a_c"),
        t_d_c=cat("t_d_c"),
        t_cond_c=cat("t_cond_c"),
        t_amb_c=cat("t_amb_c"),
        relative_humidity=cat("relative_humidity"),
        h_amb_w_m2_k=cat("h_amb_w_m2_k"),
        t_wh_in_c=cat("t_wh_in_c"),
        m_dot_wh_kg_s_m2=cat("m_dot_wh_kg_s_m2"),
        n_cycles=chunks[0].n_cycles,
    )


def detailed_series(
    cycle: CycleResult,
    *,
    config: DeviceConfig,
    profile: HalfCycleProfile | None = None,
    cycle_index: int = 0,
) -> DetailedSeries:
    """Build temperature and boundary trajectories for one full two-bed cycle."""
    del config  # reserved for future sorbent-specific diagnostics
    ha = cycle.half_a
    hb = cycle.half_b
    chunk_a = _half_boundary_series(
        ha,
        profile=profile,
        half_label="A",
        cycle_index=cycle_index,
        t_offset_s=0.0,
    )
    chunk_b = _half_boundary_series(
        hb,
        profile=profile,
        half_label="B",
        cycle_index=cycle_index,
        t_offset_s=float(ha.time_s[-1]) if len(ha.time_s) else 0.0,
    )
    out = _concat_series([chunk_a, chunk_b], skip_first=1)
    return DetailedSeries(
        time_s=out.time_s,
        cycle_index=out.cycle_index,
        half_cycle=out.half_cycle,
        half_cycle_end_s=float(ha.time_s[-1]) if len(ha.time_s) else 0.0,
        t_a_c=out.t_a_c,
        t_d_c=out.t_d_c,
        t_cond_c=out.t_cond_c,
        t_amb_c=out.t_amb_c,
        relative_humidity=out.relative_humidity,
        h_amb_w_m2_k=out.h_amb_w_m2_k,
        t_wh_in_c=out.t_wh_in_c,
        m_dot_wh_kg_s_m2=out.m_dot_wh_kg_s_m2,
        n_cycles=1,
    )


def _append_cycle(base: DetailedSeries, nxt: DetailedSeries) -> DetailedSeries:
    t0 = float(base.time_s[-1])
    shifted = DetailedSeries(
        time_s=nxt.time_s[1:] + t0,
        cycle_index=nxt.cycle_index[1:],
        half_cycle=nxt.half_cycle[1:],
        half_cycle_end_s=base.half_cycle_end_s,
        t_a_c=nxt.t_a_c[1:],
        t_d_c=nxt.t_d_c[1:],
        t_cond_c=nxt.t_cond_c[1:],
        t_amb_c=nxt.t_amb_c[1:],
        relative_humidity=nxt.relative_humidity[1:],
        h_amb_w_m2_k=nxt.h_amb_w_m2_k[1:],
        t_wh_in_c=nxt.t_wh_in_c[1:],
        m_dot_wh_kg_s_m2=nxt.m_dot_wh_kg_s_m2[1:],
        n_cycles=base.n_cycles,
    )
    return _concat_series([base, shifted], skip_first=0)


def detailed_daily_series(
    cycles: list[CycleResult],
    *,
    config: DeviceConfig,
    profile: HalfCycleProfile | None = None,
) -> DetailedSeries:
    if not cycles:
        raise ValueError("At least one cycle is required.")
    out = detailed_series(cycles[0], config=config, profile=profile, cycle_index=0)
    for i, cycle in enumerate(cycles[1:], start=1):
        chunk = detailed_series(cycle, config=config, profile=profile, cycle_index=i)
        out = _append_cycle(out, chunk)
    return DetailedSeries(
        time_s=out.time_s,
        cycle_index=out.cycle_index,
        half_cycle=out.half_cycle,
        half_cycle_end_s=out.half_cycle_end_s,
        t_a_c=out.t_a_c,
        t_d_c=out.t_d_c,
        t_cond_c=out.t_cond_c,
        t_amb_c=out.t_amb_c,
        relative_humidity=out.relative_humidity,
        h_amb_w_m2_k=out.h_amb_w_m2_k,
        t_wh_in_c=out.t_wh_in_c,
        m_dot_wh_kg_s_m2=out.m_dot_wh_kg_s_m2,
        n_cycles=len(cycles),
    )


def write_detailed_csv(path: Path, series: DetailedSeries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time_s",
                "time_h",
                "cycle_index",
                "half_cycle",
                "t_a_c",
                "t_d_c",
                "t_cond_c",
                "t_amb_c",
                "relative_humidity",
                "h_amb_w_m2_k",
                "t_wh_in_c",
                "m_dot_wh_kg_s_m2",
            ]
        )
        for k in range(len(series.time_s)):
            w.writerow(
                [
                    f"{float(series.time_s[k]):.3f}",
                    f"{float(series.time_s[k]) / 3600.0:.6f}",
                    int(series.cycle_index[k]),
                    series.half_cycle[k],
                    f"{float(series.t_a_c[k]):.4f}",
                    f"{float(series.t_d_c[k]):.4f}",
                    f"{float(series.t_cond_c[k]):.4f}",
                    f"{float(series.t_amb_c[k]):.4f}",
                    f"{float(series.relative_humidity[k]):.6f}",
                    f"{float(series.h_amb_w_m2_k[k]):.4f}",
                    f"{float(series.t_wh_in_c[k]):.4f}",
                    f"{float(series.m_dot_wh_kg_s_m2[k]):.6f}",
                ]
            )


def plot_detailed_diagnostics(
    path: Path,
    series: DetailedSeries,
    *,
    title: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_h = series.time_s / 3600.0
    phase_mark_h = series.half_cycle_end_s / 3600.0
    show_half_mark = series.n_cycles == 1 and series.half_cycle_end_s > 0.0

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax_t, ax_wx, ax_wh = axes

    ax_t.plot(time_h, series.t_a_c, color="#8b2000", linewidth=1.8, label="Contactor A")
    ax_t.plot(time_h, series.t_d_c, color="#b06000", linewidth=1.8, label="Contactor B")
    ax_t.plot(time_h, series.t_cond_c, color="#1a5a7a", linewidth=1.8, label="Condenser")
    ax_t.plot(time_h, series.t_amb_c, color="0.45", linewidth=1.2, linestyle=":", label="Ambient")
    ax_t.plot(
        time_h,
        series.t_wh_in_c,
        color="#d95f02",
        linewidth=1.2,
        linestyle="-.",
        label="Waste-heat inlet",
    )
    if show_half_mark:
        ax_t.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_t.set_ylabel("Temperature (°C)")
    ax_t.legend(loc="upper left", fontsize=7, ncol=2)
    ax_t.grid(True, alpha=0.3)

    ax_wx.plot(time_h, series.t_amb_c, color="#d95f02", linewidth=1.6, label="T_amb")
    if show_half_mark:
        ax_wx.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_wx.set_ylabel("Temperature (°C)", color="#d95f02")
    ax_wx.tick_params(axis="y", labelcolor="#d95f02")
    ax_wx.grid(True, alpha=0.3)

    ax_rh = ax_wx.twinx()
    ax_rh.plot(
        time_h,
        series.relative_humidity * 100.0,
        color="#1b9e77",
        linewidth=1.6,
        label="RH",
    )
    ax_rh.set_ylabel("Relative humidity (%)", color="#1b9e77")
    ax_rh.tick_params(axis="y", labelcolor="#1b9e77")
    ax_rh.set_ylim(0.0, 100.0)

    lines_l, labels_l = ax_wx.get_legend_handles_labels()
    lines_r, labels_r = ax_rh.get_legend_handles_labels()
    ax_wx.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", fontsize=8)

    ax_wh.plot(
        time_h,
        series.m_dot_wh_kg_s_m2,
        color="#e6ab02",
        linewidth=1.8,
        label="m_dot_wh",
    )
    if show_half_mark:
        ax_wh.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_wh.set_ylabel("Mass flow (kg/s/m²)", color="#e6ab02")
    ax_wh.tick_params(axis="y", labelcolor="#e6ab02")
    ax_wh.grid(True, alpha=0.3)

    ax_h = ax_wh.twinx()
    ax_h.plot(time_h, series.h_amb_w_m2_k, color="#7570b3", linewidth=1.4, label="h_amb")
    ax_h.set_ylabel("h_amb (W/m²K)", color="#7570b3")
    ax_h.tick_params(axis="y", labelcolor="#7570b3")

    lines_l, labels_l = ax_wh.get_legend_handles_labels()
    lines_r, labels_r = ax_h.get_legend_handles_labels()
    ax_wh.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", fontsize=8)

    ax_wh.set_xlabel("Time (h)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
