"""Water-in-gel inventory time series from a daily absorption–desorption cycle."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from solar_lumped.simulation.ode_system import PhaseResult
from solar_lumped.weather.fig_s1 import water_in_gel_l_m2


@dataclass(frozen=True, slots=True)
class WaterInventorySeries:
    time_s: np.ndarray
    water_l_m2: np.ndarray
    phase: np.ndarray
    absorption_end_s: float
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


def water_inventory_series(
    abs_res: PhaseResult,
    des_res: PhaseResult,
    *,
    h0_ref_m: float,
) -> WaterInventorySeries:
    """Concatenate absorption and desorption phases into one water-in-gel trajectory."""
    w_abs = np.array(
        [
            water_in_gel_l_m2(float(c), float(h), h0_ref_m=h0_ref_m)
            for c, h in zip(abs_res.c_w, abs_res.H)
        ]
    )
    w_des = np.array(
        [
            water_in_gel_l_m2(float(c), float(h), h0_ref_m=h0_ref_m)
            for c, h in zip(des_res.c_w, des_res.H)
        ]
    )
    t_abs_end = float(abs_res.time_s[-1]) if len(abs_res.time_s) else 0.0
    time_s = np.concatenate([abs_res.time_s, t_abs_end + des_res.time_s[1:]])
    water_l_m2 = np.concatenate([w_abs, w_des[1:]])
    phase = np.array(
        ["absorption"] * len(w_abs) + ["desorption"] * (len(w_des) - 1),
        dtype=object,
    )
    m_des = np.concatenate([abs_res.m_des_kg_s_m2, des_res.m_des_kg_s_m2[1:]])
    collected_water_l_m2 = cumulative_desorption_yield_l_m2(time_s, m_des)
    return WaterInventorySeries(
        time_s=time_s,
        water_l_m2=water_l_m2,
        phase=phase,
        absorption_end_s=t_abs_end,
        collected_water_l_m2=collected_water_l_m2,
    )


def write_water_inventory_csv(path: Path, series: WaterInventorySeries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "time_h", "phase", "water_in_gel_l_m2", "collected_water_l_m2"])
        for t, ph, w_l, y_l in zip(
            series.time_s,
            series.phase,
            series.water_l_m2,
            series.collected_water_l_m2,
        ):
            w.writerow(
                [
                    f"{float(t):.3f}",
                    f"{float(t) / 3600.0:.6f}",
                    ph,
                    f"{float(w_l):.6f}",
                    f"{float(y_l):.6f}",
                ]
            )


def plot_water_inventory(
    path: Path,
    series: WaterInventorySeries,
    *,
    title: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_h = series.time_s / 3600.0
    phase_mark_h = series.absorption_end_s / 3600.0
    fig, (ax_inv, ax_yield) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax_inv.plot(time_h, series.water_l_m2, color="#4C72B0", linewidth=2)
    ax_inv.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_inv.set_ylabel("Water in gel (L/m²)")
    ax_inv.grid(True, alpha=0.3)

    ax_yield.plot(time_h, series.collected_water_l_m2, color="#C44E52", linewidth=2)
    ax_yield.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_yield.set_xlabel("Time (h)")
    ax_yield.set_ylabel("Collected water (L/m²)")
    ax_yield.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
