"""Device temperatures and weather time series for a fluid-heated daily SAWH cycle."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from waste_heat_lumped.simulation.device_config import DeviceConfig
from waste_heat_lumped.simulation.ode_system import PhaseResult
from waste_heat_lumped.weather.profiles import DailyWeatherProfile, PhaseProfile


@dataclass(frozen=True, slots=True)
class DetailedSeries:
    time_s: np.ndarray
    phase: np.ndarray
    absorption_end_s: float
    t_gel_c: np.ndarray
    t_cond_c: np.ndarray
    t_f_c: np.ndarray
    q_f_to_gel_w_m2: np.ndarray
    t_amb_c: np.ndarray
    relative_humidity: np.ndarray
    h_amb_w_m2_k: np.ndarray


def _profile_index(t: float, dt_s: float, n: int) -> int:
    return min(int(t / dt_s), n - 1)


def _phase_weather(
    time_s: np.ndarray,
    profile: PhaseProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(profile.temperature_c)
    dt = profile.dt_s
    t_amb: list[float] = []
    rh: list[float] = []
    h_amb: list[float] = []
    for t in time_s:
        i = _profile_index(float(t), dt, n)
        t_amb.append(profile.temperature_c[i])
        rh.append(profile.relative_humidity[i])
        h_amb.append(profile.h_amb_w_m2_k[i])
    return np.array(t_amb), np.array(rh), np.array(h_amb)


def _absorption_cond_temps(
    abs_res: PhaseResult,
    profile: PhaseProfile,
) -> np.ndarray:
    n = len(profile.temperature_c)
    dt = profile.dt_s
    t_cond: list[float] = []
    for t in abs_res.time_s:
        i = _profile_index(float(t), dt, n)
        t_cond.append(profile.temperature_c[i])
    return np.array(t_cond)


def detailed_series(
    profile: DailyWeatherProfile,
    abs_res: PhaseResult,
    des_res: PhaseResult,
    config: DeviceConfig,
) -> DetailedSeries:
    """Build full-cycle device and weather trajectories."""
    if des_res.t_cond_c is None:
        raise ValueError("Desorption result missing condenser temperature history.")

    abs_weather = _phase_weather(abs_res.time_s, profile.absorption)
    des_weather = _phase_weather(des_res.time_s, profile.desorption)

    abs_t_cond = _absorption_cond_temps(abs_res, profile.absorption)
    abs_q_f = np.zeros(len(abs_res.time_s))
    abs_t_f = np.full(len(abs_res.time_s), config.t_f_c)

    des_t_f = np.full(len(des_res.time_s), config.t_f_c)

    t_abs_end = float(abs_res.time_s[-1]) if len(abs_res.time_s) else 0.0
    time_s = np.concatenate([abs_res.time_s, t_abs_end + des_res.time_s[1:]])

    def _join(abs_arr: np.ndarray, des_arr: np.ndarray) -> np.ndarray:
        return np.concatenate([abs_arr, des_arr[1:]])

    phase = np.array(
        ["absorption"] * len(abs_res.time_s) + ["desorption"] * (len(des_res.time_s) - 1),
        dtype=object,
    )

    return DetailedSeries(
        time_s=time_s,
        phase=phase,
        absorption_end_s=t_abs_end,
        t_gel_c=_join(abs_res.t_gel_c, des_res.t_gel_c),
        t_cond_c=_join(abs_t_cond, des_res.t_cond_c),
        t_f_c=_join(abs_t_f, des_t_f),
        q_f_to_gel_w_m2=_join(abs_q_f, des_res.q_f_to_gel_w_m2),
        t_amb_c=_join(abs_weather[0], des_weather[0]),
        relative_humidity=_join(abs_weather[1], des_weather[1]),
        h_amb_w_m2_k=_join(abs_weather[2], des_weather[2]),
    )


def write_detailed_csv(path: Path, series: DetailedSeries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time_s",
                "time_h",
                "phase",
                "t_gel_c",
                "t_cond_c",
                "t_f_c",
                "q_f_to_gel_w_m2",
                "t_amb_c",
                "relative_humidity",
                "h_amb_w_m2_k",
            ]
        )
        for k in range(len(series.time_s)):
            w.writerow(
                [
                    f"{float(series.time_s[k]):.3f}",
                    f"{float(series.time_s[k]) / 3600.0:.6f}",
                    series.phase[k],
                    f"{float(series.t_gel_c[k]):.4f}",
                    f"{float(series.t_cond_c[k]):.4f}",
                    f"{float(series.t_f_c[k]):.4f}",
                    f"{float(series.q_f_to_gel_w_m2[k]):.2f}",
                    f"{float(series.t_amb_c[k]):.4f}",
                    f"{float(series.relative_humidity[k]):.6f}",
                    f"{float(series.h_amb_w_m2_k[k]):.4f}",
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
    phase_mark_h = series.absorption_end_s / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax_t, ax_wx, ax_flux = axes

    ax_t.plot(time_h, series.t_gel_c, color="#6a3d9a", linewidth=1.8, label="Gel")
    ax_t.plot(time_h, series.t_cond_c, color="#1a5a7a", linewidth=1.8, label="Condenser")
    ax_t.plot(time_h, series.t_f_c, color="#d95f02", linewidth=1.4, linestyle="--", label="Fluid setpoint")
    ax_t.plot(
        time_h,
        series.t_amb_c,
        color="0.45",
        linewidth=1.2,
        linestyle=":",
        label="Ambient",
    )
    ax_t.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_t.set_ylabel("Temperature (°C)")
    ax_t.legend(loc="upper left", fontsize=8, ncol=2)
    ax_t.grid(True, alpha=0.3)

    ax_wx.plot(time_h, series.t_amb_c, color="#d95f02", linewidth=1.6, label="T_amb")
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

    ax_flux.plot(
        time_h,
        series.q_f_to_gel_w_m2,
        color="#e6ab02",
        linewidth=1.8,
        label="Q_f→gel",
    )
    ax_flux.axvline(phase_mark_h, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_flux.set_ylabel("Heat flux (W/m²)", color="#e6ab02")
    ax_flux.tick_params(axis="y", labelcolor="#e6ab02")
    ax_flux.grid(True, alpha=0.3)

    ax_h = ax_flux.twinx()
    ax_h.plot(time_h, series.h_amb_w_m2_k, color="#7570b3", linewidth=1.4, label="h_amb")
    ax_h.set_ylabel("h_amb (W/m²K)", color="#7570b3")
    ax_h.tick_params(axis="y", labelcolor="#7570b3")

    lines_l, labels_l = ax_flux.get_legend_handles_labels()
    lines_r, labels_r = ax_h.get_legend_handles_labels()
    ax_flux.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", fontsize=8)

    ax_flux.set_xlabel("Time (h)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
