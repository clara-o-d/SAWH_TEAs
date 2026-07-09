#!/usr/bin/env python3
"""
Plot digitized Wilson et al. (2025) weather data for visual validation.

Cambridge (Fig. 3): solar flux and ambient temperature over the 10-h desorption
window — compare against the paper's incident-flux and ambient traces.

Atacama (Fig. 4B): 24-h weather from 18:00 (6 pm) — RH, solar flux, and
ambient temperature with day/night shading matching the paper panel.

Raw CSVs:   wilson-et-al._re-creation/reference/weather/
Outputs:    wilson-et-al._re-creation/outputs/weather/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).resolve()
_WILSON_DIR = _SCRIPT.parent.parent
_SOLAR_ROOT = _WILSON_DIR.parent
_SRC = _SOLAR_ROOT / "src"
for _p in (_SRC, _SOLAR_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from solar_lumped.weather.profiles import PHASE_DT_S

from solar_lumped.plotting.matlab_style import (
    figure_size_inches,
    panel_size_inches,
    plot_defaults_slides,
    print_figure,
    ref_marker_kwargs,
    scaled_fontsize,
    style_axes,
)

plot_defaults_slides()

_REF_WEATHER = _WILSON_DIR / "reference" / "weather"
_OUT_DIR = _WILSON_DIR / "outputs" / "weather"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cambridge desorption window (Wilson Fig. 3)
_CAMBRIDGE_HOURS = 10.0
_CAMBRIDGE_STEPS = int(_CAMBRIDGE_HOURS * 3600.0 / PHASE_DT_S)

# Atacama timeline origin (Wilson Fig. 4B)
_ATACAMA_ORIGIN_H = 18.0   # 6 pm
_ATACAMA_DURATION_H = 24.0
_SUNRISE_H = 13.5          # ~07:30 from 18:00 origin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t, v = data[:, 0], data[:, 1]
    order = np.argsort(t)
    t, v = t[order], v[order]
    _, uniq = np.unique(t, return_index=True)
    return t[uniq], v[uniq]


def _interp_clamped(
    t_raw: np.ndarray,
    v_raw: np.ndarray,
    t_grid: np.ndarray,
    *,
    clip_min: float | None = None,
) -> np.ndarray:
    out = np.interp(t_grid, t_raw, v_raw)
    if clip_min is not None:
        out = np.maximum(clip_min, out)
    return out


def _panel_style(ax: plt.Axes) -> None:
    style_axes(ax)


def _hours_to_clock_label(h_from_6pm: float) -> str:
    """Convert hours from 18:00 origin to '6 pm' / '9 am' style label."""
    total_h = (_ATACAMA_ORIGIN_H + h_from_6pm) % 24.0
    hour = int(total_h) % 24
    if hour == 0:
        return "12 am"
    if hour < 12:
        return f"{hour} am"
    if hour == 12:
        return "12 pm"
    return f"{hour - 12} pm"


# ---------------------------------------------------------------------------
# Cambridge
# ---------------------------------------------------------------------------

def plot_cambridge_weather() -> Path:
    solar_t, solar_v = _load_csv(_REF_WEATHER / "Cambridge_solar_W_m2.csv")
    temp_t, temp_v = _load_csv(_REF_WEATHER / "Cambridge_amb_T_C.csv")

    t_grid = np.linspace(0.0, _CAMBRIDGE_HOURS, _CAMBRIDGE_STEPS, endpoint=False)
    solar_interp = _interp_clamped(solar_t, solar_v, t_grid, clip_min=0.0)
    temp_interp = _interp_clamped(temp_t, temp_v, t_grid)

    fig, (ax_solar, ax_temp) = plt.subplots(2, 1, figsize=figure_size_inches(1, 2), sharex=True)

    # --- Solar flux ---
    ax_solar.scatter(solar_t, solar_v, label="digitized points", **ref_marker_kwargs(color="#c9a227"))
    ax_solar.plot(
        t_grid, solar_interp, color="#c9a227", alpha=0.85,
        label="interpolated (simulation grid)",
    )
    ax_solar.fill_between(t_grid, 0, solar_interp, color="#c9a227", alpha=0.12)
    ax_solar.set_ylabel("solar flux [W/m²]")
    ax_solar.set_xlim(0, _CAMBRIDGE_HOURS)
    ax_solar.set_ylim(0, 1050)
    ax_solar.legend(fontsize=8, frameon=False, loc="upper right")
    ax_solar.set_title("A  solar flux", loc="left", fontweight="bold")
    _panel_style(ax_solar)

    # --- Ambient temperature ---
    ax_temp.scatter(temp_t, temp_v, label="digitized points", **ref_marker_kwargs(color="#8b2000"))
    ax_temp.plot(
        t_grid, temp_interp, color="#8b2000", alpha=0.85,
        label="interpolated (simulation grid)",
    )
    ax_temp.set_xlabel("time from desorption start [hr]")
    ax_temp.set_ylabel("ambient temperature [°C]")
    ax_temp.set_xlim(0, _CAMBRIDGE_HOURS)
    ax_temp.set_ylim(18, 30)
    ax_temp.legend(fontsize=8, frameon=False, loc="upper right")
    ax_temp.set_title("B  ambient temperature", loc="left", fontweight="bold")
    _panel_style(ax_temp)

    fig.suptitle(
        "Wilson et al. (2025) Figure 3 — Cambridge digitized weather\n"
        f"({len(solar_t)} solar points, {len(temp_t)} temperature points; "
        f"desorption window 0–{_CAMBRIDGE_HOURS:.0f} h)",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()

    out = _OUT_DIR / "cambridge_weather.png"
    print_figure(fig, out)
    plt.close(fig)

    # Combined dual-axis panel (similar to Fig. 3C solar overlay style)
    fig2, ax_l = plt.subplots(figsize=panel_size_inches())
    ax_r = ax_l.twinx()

    ax_l.scatter(temp_t, temp_v, **ref_marker_kwargs(color="#8b2000"))
    ax_l.plot(t_grid, temp_interp, color="#8b2000", label="ambient [°C]")
    ax_l.set_ylabel("ambient temperature [°C]", color="#8b2000")
    ax_l.tick_params(axis="y", labelcolor="#8b2000")
    ax_l.set_ylim(18, 30)

    ax_r.scatter(solar_t, solar_v, **ref_marker_kwargs(color="#c9a227"))
    ax_r.fill_between(t_grid, 0, solar_interp, color="#c9a227", alpha=0.15)
    ax_r.plot(t_grid, solar_interp, color="#c9a227", label="solar [W/m²]")
    ax_r.set_ylabel("solar flux [W/m²]", color="#c9a227")
    ax_r.tick_params(axis="y", labelcolor="#c9a227")
    ax_r.set_ylim(0, 1050)

    ax_l.set_xlabel("time from desorption start [hr]")
    ax_l.set_xlim(0, _CAMBRIDGE_HOURS)
    _panel_style(ax_l)
    _panel_style(ax_r)
    fig2.suptitle(
        "Cambridge weather — dual axis (compare with Wilson Fig. 3C solar trace)",
        fontsize=9,
    )
    fig2.tight_layout()
    out2 = _OUT_DIR / "cambridge_weather_dual_axis.png"
    print_figure(fig2, out2)
    plt.close(fig2)

    print(f"  Cambridge solar:  {solar_v.min():.0f}–{solar_v.max():.0f} W/m²  ({len(solar_t)} pts)")
    print(f"  Cambridge T_amb:  {temp_v.min():.1f}–{temp_v.max():.1f} °C  ({len(temp_t)} pts)")
    print(f"Saved → {out}")
    print(f"Saved → {out2}")
    return out


# ---------------------------------------------------------------------------
# Atacama (Fig. 4B style)
# ---------------------------------------------------------------------------

def plot_atacama_weather() -> Path:
    rh_t, rh_v = _load_csv(_REF_WEATHER / "Atacama_RH.csv")
    temp_t, temp_v = _load_csv(_REF_WEATHER / "Atacama_Temp.csv")
    solar_t, solar_kw = _load_csv(_REF_WEATHER / "Atacama_solar_kW_m2.csv")

    # Fine grid for smooth interpolated curves (1-min steps over 24 h)
    n_steps = int(_ATACAMA_DURATION_H * 3600.0 / PHASE_DT_S)
    t_grid = np.linspace(0.0, _ATACAMA_DURATION_H, n_steps, endpoint=False)
    rh_interp = _interp_clamped(rh_t, rh_v, t_grid)
    temp_interp = _interp_clamped(temp_t, temp_v, t_grid)
    solar_interp = np.maximum(0.0, _interp_clamped(solar_t, solar_kw, t_grid))

    fig, ax_l = plt.subplots(figsize=panel_size_inches())
    ax_r = ax_l.twinx()

    # Day / night shading
    ax_l.axvspan(0, _SUNRISE_H, color="#d8cce8", alpha=0.35, zorder=0)
    ax_l.axvspan(_SUNRISE_H, _ATACAMA_DURATION_H, color="#fff8dc", alpha=0.45, zorder=0)

    # RH and solar on left axis (0–1, matching paper)
    ax_l.scatter(rh_t, rh_v, s=22, marker="o", facecolors="white",
                 edgecolors="#607080", linewidths=1.0, zorder=5)
    ax_l.plot(t_grid, rh_interp, color="#607080", label="RH [-]")

    ax_l.scatter(solar_t, solar_kw, s=22, marker="o", facecolors="white",
                 edgecolors="#c9a227", linewidths=1.0, zorder=5)
    ax_l.fill_between(t_grid, 0, solar_interp, color="#c9a227", alpha=0.18)
    ax_l.plot(t_grid, solar_interp, color="#c9a227", label="solar [kW/m²]")

    ax_l.set_ylabel("RH [-]  /  solar [kW/m²]")
    ax_l.set_ylim(0, 1.0)
    ax_l.set_xlim(0, _ATACAMA_DURATION_H)

    # Ambient temperature on right axis
    ax_r.scatter(temp_t, temp_v, s=22, marker="o", facecolors="white",
                 edgecolors="#8b2000", linewidths=1.0, zorder=5)
    ax_r.plot(t_grid, temp_interp, color="#8b2000", label="ambient [°C]")
    ax_r.set_ylabel("ambient temperature [°C]", color="#8b2000")
    ax_r.tick_params(axis="y", labelcolor="#8b2000")
    ax_r.set_ylim(0, 32)

    # Clock-time x labels (paper style: 6 pm, 11 pm, 4 am, 9 am, 2 pm)
    tick_hours = [0, 5, 10, 15, 20]
    ax_l.set_xticks(tick_hours)
    ax_l.set_xticklabels([_hours_to_clock_label(h) for h in tick_hours])
    ax_l.set_xlabel("time of day (cycle starts 6 pm)")

    # Moon / sun icons (simple text markers)
    trans = mtransforms.blended_transform_factory(ax_l.transData, ax_l.transAxes)
    ax_l.text(6.0, 0.97, "☾", transform=trans, ha="center", va="top", fontsize=14)
    ax_l.text(17.0, 0.97, "☼", transform=trans, ha="center", va="top", fontsize=14)

    # Vertical line at device install (~8 am = hour 14)
    install_h = 14.0
    ax_l.axvline(install_h, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_l.text(
        install_h + 0.3, 0.92, "device install\n(~8 am)",
        transform=trans, fontsize=7, color="#444444", va="top",
    )

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(
        lines_l + lines_r, labels_l + labels_r,
        fontsize=7.5, frameon=False, loc="upper left", ncol=1,
    )

    _panel_style(ax_l)
    _panel_style(ax_r)

    fig.suptitle(
        "Wilson et al. (2025) Figure 4B — Atacama digitized weather\n"
        f"({len(rh_t)} RH, {len(temp_t)} T, {len(solar_t)} solar points; "
        "open circles = digitized, solid = interpolated)",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()

    out = _OUT_DIR / "atacama_weather.png"
    print_figure(fig, out)
    plt.close(fig)

    # Individual panels for easier point-by-point checking
    fig3, axes = plt.subplots(3, 1, figsize=figure_size_inches(1, 3), sharex=True)
    series = [
        (rh_t, rh_v, rh_interp, "RH [-]", "#607080", (0, 0.55)),
        (temp_t, temp_v, temp_interp, "ambient temperature [°C]", "#8b2000", (0, 32)),
        (solar_t, solar_kw, solar_interp, "solar flux [kW/m²]", "#c9a227", (0, 0.75)),
    ]
    for ax, (tx, tv, interp, ylabel, color, ylim) in zip(axes, series):
        ax.axvspan(0, _SUNRISE_H, color="#d8cce8", alpha=0.35, zorder=0)
        ax.axvspan(_SUNRISE_H, _ATACAMA_DURATION_H, color="#fff8dc", alpha=0.45, zorder=0)
        ax.scatter(tx, tv, s=24, marker="o", facecolors="white",
                   edgecolors=color, linewidths=1.1, zorder=5, label="digitized")
        ax.plot(t_grid, interp, color=color, label="interpolated")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        _panel_style(ax)
    axes[-1].set_xlabel("hours from 6 pm")
    axes[-1].set_xlim(0, _ATACAMA_DURATION_H)
    fig3.suptitle("Atacama weather — individual series")
    fig3.tight_layout()
    out3 = _OUT_DIR / "atacama_weather_panels.png"
    print_figure(fig3, out3)
    plt.close(fig3)

    print(f"  Atacama RH:    {rh_v.min():.2f}–{rh_v.max():.2f}  ({len(rh_t)} pts)")
    print(f"  Atacama T:     {temp_v.min():.1f}–{temp_v.max():.1f} °C  ({len(temp_t)} pts)")
    print(f"  Atacama solar: {solar_kw.min():.3f}–{solar_kw.max():.3f} kW/m²  ({len(solar_t)} pts)")
    des_slice = slice(int(14 * 60), int(22 * 60))
    mean_solar_kw = float(np.mean(solar_interp[des_slice]))
    print(
        f"  Mean solar (8 am–4 pm, h=14–22): "
        f"{mean_solar_kw * 1000:.0f} W/m² ({mean_solar_kw:.3f} kW/m²; paper ≈517 W/m²)"
    )
    print(f"Saved → {out}")
    print(f"Saved → {out3}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Wilson weather validation plots")
    print("=" * 40)
    print("\nCambridge (Fig. 3 desorption window):")
    plot_cambridge_weather()
    print("\nAtacama (Fig. 4B, 24 h from 6 pm):")
    plot_atacama_weather()
    print("\nDone.")


if __name__ == "__main__":
    main()
