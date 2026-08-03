#!/usr/bin/env python3
"""
Recreation of Wilson et al. (2025) Device Figure 4 (panels C and D).

Atacama Desert field test, May 8–9, 2024:
  - 12 h absorption overnight (6 pm May 8 → 8 am May 9)
  - 8 h desorption in sun (8 am → ~4 pm May 9)
  - tilt 25° from horizontal, fin area ratio A_r = 5
  - h_amb = 1 W/m²K all day; stepped to 10 W/m²K at 2 pm (sudden desert wind)
  - Average solar flux 517 W/m² during desorption
  - Final water yield 0.62 L/m²/day, thermal efficiency 9.3%

Panel C: Predicted system temperatures (absorber, glass, condenser, ambient)
         vs time during the 8-h desorption phase.
Panel D: Predicted cumulative water output vs time; single measured endpoint
         (0.62 L/m²) from paper shown as a star marker.

Outputs saved to:  wilson-et-al._re-creation/outputs/figure4/
  - figure4.png
  - figure4_hourly_model.csv  (model estimates at each desorption hour)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Path bootstrap ---
_SCRIPT = Path(__file__).resolve()
_WILSON_DIR = _SCRIPT.parent.parent          # wilson-et-al._re-creation/
_SOLAR_ROOT = _WILSON_DIR.parent.parent.parent / "solar_lumped"
_SRC = _SOLAR_ROOT / "src"
for _p in (_SRC, _SOLAR_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from solar_lumped.physics import solve_steady_thermal
from solar_lumped.physics import SystemThermalParams
from solar_lumped.plotting import (
    figure_size_inches,
    plot_defaults_slides,
    print_figure,
    ref_marker_kwargs,
    scaled_fontsize,
    style_axes,
)
from solar_lumped.simulation import SystemConfig
from solar_lumped.simulation import PhaseResult, run_daily_cycle
from solar_lumped.simulation import cumulative_desorption_yield_l_m2
from solar_lumped.weather import (
    ATACAMA_DESORPTION_START_OFFSET_H,
    atacama_field_profile,
)

plot_defaults_slides()

_OUT_DIR = _WILSON_DIR / "outputs" / "figure4"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_REF_DIR = _WILSON_DIR / "reference" / "figure4"
_REF_LABEL = "Wilson et al. (digitized)"

# Measured final water yield from the paper (L/m²)
_MEASURED_YIELD_L_M2 = 0.62

# Note S2 / Table S3 — the paper's chamber-calibrated absorption g, used for every
# Wilson figure. The package default (0.015, parameters.xlsx) stands for our own work.
_G_CONV_M_S: float = 0.0085

# Table S3 is the COMSOL model's fin area ratio. SystemConfig.atacama_field carries the
# as-built heat sink (A_r = 5, Methods); we are recreating the model, not the hardware.
_A_R: float = 7.1

# Fig. 4D model prediction and its h_amb error bounds, digitized in mL of system total.
# The measured point on the same axis is 47.819 mL, which the paper quotes as
# 0.62 L/m² — a 0.0771 m² normalisation, i.e. A_c to three figures.
_FIG4D_CSV = "Fig4d_results_and_error.csv"
_FIG4D_MEASURED_ML = 47.81893004115226


def _fig4d_model_l_m2() -> tuple[float, float, float]:
    """(lower, central, upper) of the paper's own Fig. 4D model prediction, in L/m²."""
    ml = np.sort(np.loadtxt(_REF_DIR / _FIG4D_CSV, delimiter=",")[:, 1])
    scale = _MEASURED_YIELD_L_M2 / _FIG4D_MEASURED_ML
    return tuple(float(v) * scale for v in ml)  # type: ignore[return-value]

# Digitized Fig. 4C reference curves. The originally published panel mislabeled the
# component curves (its "glass" was flat ~38 °C, inconsistent with Eq. 3); these
# corrected re-digitizations have the glass rising with the absorber at midday.
_REF_ABSORBER_CSV = "Atacama_absorber copy.csv"
_REF_GLASS_CSV = "Atacama_glass copy.csv"
_REF_CONDENSER_CSV = "Atacama_condenser copy.csv"


def _series_r2(time_h: np.ndarray, model_y: np.ndarray, ref_csv: str) -> float | None:
    """Coefficient of determination of the model vs a digitized reference curve."""
    loaded = _load_ref_csv(ref_csv)
    if loaded is None:
        return None
    rx, ry = loaded
    mask = ~(np.isnan(rx) | np.isnan(ry))
    rx, ry = rx[mask], ry[mask]
    if rx.size < 2:
        return None
    my = np.interp(rx, time_h, model_y)
    ss_res = float(np.sum((ry - my) ** 2))
    ss_tot = float(np.sum((ry - np.mean(ry)) ** 2))
    if ss_tot == 0.0:
        return None
    return 1.0 - ss_res / ss_tot

# Fig. 4C/D x-axis: desorption window 8 am → 4 pm (0 h = install at 8 am).
_DESORPTION_START_HOUR = 8
_DESORPTION_CLOCK_TICKS_H = (0.0, 2.0, 4.0, 6.0, 8.0)
_HOURLY_MODEL_CSV = "figure4_hourly_model.csv"


def _desorption_clock_label(hours_from_8am: float) -> str:
    """Map hours from 8 am install to paper-style clock labels."""
    hour = (int(round(hours_from_8am)) + _DESORPTION_START_HOUR) % 24
    if hour == 0:
        return "12 am"
    if hour < 12:
        return f"{hour} am"
    if hour == 12:
        return "12 pm"
    return f"{hour - 12} pm"


def _hourly_desorption_times_h(duration_h: float) -> np.ndarray:
    """Integer hours from 8 am through the end of the desorption window."""
    last_hour = min(8, int(np.floor(duration_h + 1e-9)))
    return np.arange(0, last_hour + 1, dtype=float)


def _interp_at_hours(time_h: np.ndarray, values: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Linear interpolation of a model series onto hourly clock times."""
    return np.interp(hours, time_h, values)


def write_hourly_model_estimates(data: dict) -> Path:
    """Write model temperatures and water output at each desorption hour to CSV."""
    duration_h = float(data["time_h"][-1])
    hours = _hourly_desorption_times_h(duration_h)
    time_h = data["time_h"]

    t_abs = _interp_at_hours(time_h, data["t_abs"], hours)
    t_glass = _interp_at_hours(time_h, data["t_glass"], hours)
    t_cond = _interp_at_hours(time_h, data["t_cond"], hours)
    t_amb = _interp_at_hours(time_h, data["t_amb"], hours)
    cum_water = _interp_at_hours(time_h, data["cum_water_l_m2"], hours)

    out_path = _OUT_DIR / _HOURLY_MODEL_CSV
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time_h",
                "clock_time",
                "t_abs_c",
                "t_glass_c",
                "t_cond_c",
                "t_amb_c",
                "cum_water_l_m2",
            ]
        )
        for k, h in enumerate(hours):
            w.writerow(
                [
                    f"{h:.0f}",
                    _desorption_clock_label(h),
                    f"{t_abs[k]:.4f}",
                    f"{t_glass[k]:.4f}",
                    f"{t_cond[k]:.4f}",
                    f"{t_amb[k]:.4f}",
                    f"{cum_water[k]:.6f}",
                ]
            )

    print("\nHourly model estimates (desorption clock, °C unless noted):")
    print(f"  {'hour':>4}  {'clock':>6}  {'absorber':>9}  {'glass':>9}  "
          f"{'condenser':>9}  {'ambient':>9}  {'water L/m²':>10}")
    for k, h in enumerate(hours):
        print(
            f"  {h:4.0f}  {_desorption_clock_label(h):>6}  "
            f"{t_abs[k]:9.2f}  {t_glass[k]:9.2f}  "
            f"{t_cond[k]:9.2f}  {t_amb[k]:9.2f}  "
            f"{cum_water[k]:10.4f}"
        )
    print(f"\nSaved hourly estimates → {out_path}")
    return out_path


def _style_desorption_time_axis(ax: plt.Axes, *, duration_h: float) -> None:
    """Paper Fig. 4C/D: clock time from 8 am through end of desorption."""
    ax.set_xlim(0.0, duration_h)
    ax.set_xticks(_DESORPTION_CLOCK_TICKS_H)
    ax.set_xticklabels([_desorption_clock_label(t) for t in _DESORPTION_CLOCK_TICKS_H])
    ax.set_xlabel("Time", fontsize=8.5)


# --- Post-process: recover T_abs and T_glass from the stored desorption trajectory ---

def _profile_index(t: float, dt_s: float, n: int) -> int:
    return min(int(t / dt_s), n - 1)


def _post_process_desorption(
    des_res: PhaseResult,
    profile_des,
    config: SystemConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-evaluate steady thermal at each stored ODE timestep → T_abs, T_glass arrays.

    The Radau integrator stores states at t_eval points (n+1 values).  For each
    we call solve_steady_thermal with the saved (c_w, H, t_cond, weather) so we
    can reconstruct T_abs and T_glass, which are not stored in PhaseResult.
    """
    thermal = config.thermal_params()
    n_weather = len(profile_des.temperature_c)
    dt_s = profile_des.dt_s

    t_abs_list: list[float] = []
    t_glass_list: list[float] = []
    t_guess: tuple[float, float, float] | None = None

    for k in range(len(des_res.time_s)):
        t_k = float(des_res.time_s[k])
        i = _profile_index(t_k, dt_s, n_weather)

        t_cond = float(des_res.t_cond_c[k])
        t_amb = profile_des.temperature_c[i]
        q_sol = max(0.0, profile_des.solar_w_m2[i])
        h_m = max(float(des_res.H[k]), config.hydrogel_thickness_m)
        gap_eff = max(config.vapor_gap_m - h_m, 1e-6)

        # m_des from saved flux (positive, kg/m²/s)
        m_des = max(0.0, float(des_res.m_des_kg_s_m2[k]))

        state = solve_steady_thermal(
            t_cond_c=t_cond,
            t_amb_c=t_amb,
            q_solar_w_m2=q_sol,
            m_des_kg_s_m2=m_des,
            h_amb=profile_des.h_amb_w_m2_k[i],
            params=thermal,
            h_m=h_m,
            t_guess=t_guess,
            vapor_gap_m=gap_eff,
        )
        t_abs_list.append(state.t_abs_c)
        t_glass_list.append(state.t_glass_c)
        t_guess = (state.t_gel_c, state.t_abs_c, state.t_glass_c)

    return np.array(t_abs_list), np.array(t_glass_list)


# --- Simulation ---

def simulate_atacama() -> dict:
    """Run the Atacama field cycle and return all time-series needed for Fig. 4."""
    print("  Building Atacama field config…")
    config = SystemConfig.atacama_field(
        coupled_initial_temps_c=_atacama_initial_temps_c(),
        g_conv_m_s=_G_CONV_M_S,
        fin_area_ratio=_A_R,
        # Case 1: original Wilson Eqs. 3/4 blackbody/cavity radiative exchange --
        # this reproduces the paper, not the package's Case 2 default.
        thermal=SystemThermalParams(eps_abs_ir=1.0, eps_glass_ir=1.0),
    )

    print("  Loading Atacama weather profile…")
    profile = atacama_field_profile()

    print("  Integrating daily cycle (absorption 12 h + desorption 8 h)…")
    yield_kg, eta, abs_res, des_res = run_daily_cycle(profile, config)

    print(f"  → yield = {yield_kg * 1000:.1f} g/m²  "
          f"({yield_kg:.3f} L/m²),  η_thermal = {eta * 100:.1f}%")

    print("  Post-processing: recovering T_abs and T_glass…")
    if des_res.t_abs_c is not None and des_res.t_glass_c is not None:
        # Segregated solver stores surface temperatures directly.
        t_abs_arr, t_glass_arr = des_res.t_abs_c, des_res.t_glass_c
    else:
        t_abs_arr, t_glass_arr = _post_process_desorption(
            des_res, profile.desorption, config
        )

    # Time axis in hours from 8 a.m.; desorption starts ~0.15 h later (8:09 a.m.,
    # where the digitized curves begin), so shift the model clock to match.
    time_h = des_res.time_s / 3600.0 + ATACAMA_DESORPTION_START_OFFSET_H

    # Ambient temperature interpolated to ODE time grid
    n_weather = len(profile.desorption.temperature_c)
    dt_s = profile.desorption.dt_s
    t_amb_arr = np.array([
        profile.desorption.temperature_c[_profile_index(float(t), dt_s, n_weather)]
        for t in des_res.time_s
    ])

    # Cumulative water output (L/m²)
    cum_water = cumulative_desorption_yield_l_m2(des_res.time_s, des_res.m_des_kg_s_m2)

    return {
        "time_h": time_h,
        "t_abs": t_abs_arr,
        "t_glass": t_glass_arr,
        "t_cond": des_res.t_cond_c,
        "t_amb": t_amb_arr,
        "cum_water_l_m2": cum_water,
        "yield_kg": yield_kg,
        "eta": eta,
    }


# --- Plotting ---

_FIG4_STYLE = {
    "absorber":  {"color": "#c0392b", "linestyle": "-",  "label": "absorber (model)"},
    "glass":     {"color": "#e67e22", "linestyle": "--", "label": "glass (model)"},
    "condenser": {"color": "#2980b9", "linestyle": "-",  "label": "condenser (model)"},
    "ambient":   {"color": "#7f8c8d", "linestyle": ":",  "label": r"$T_\mathrm{amb}$ (measured)"},
    "water":     {"color": "#1abc9c", "linestyle": "-",  "label": "water output (model)"},
}


def _panel_style(ax: plt.Axes) -> None:
    style_axes(ax)


def _first_ref_temp(filename: str, default: float) -> float:
    """First y-value from a digitized reference CSV (Wilson cold-start point)."""
    loaded = _load_ref_csv(filename)
    if loaded is None:
        return default
    _, y = loaded
    mask = ~np.isnan(y)
    if not mask.any():
        return default
    return float(y[mask][0])


def _atacama_initial_temps_c() -> tuple[float, float, float, float]:
    """Wilson digitized Fig. 4C first points (~8:09 a.m.): gel=abs, glass, cond."""
    t_abs = _first_ref_temp(_REF_ABSORBER_CSV, 0.0)
    t_glass = _first_ref_temp(_REF_GLASS_CSV, 0.0)
    t_cond = _first_ref_temp(_REF_CONDENSER_CSV, 0.0)
    return (t_abs, t_abs, t_glass, t_cond)


def _load_ref_csv(filename: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = _REF_DIR / filename
    if not path.exists():
        return None
    data = np.loadtxt(path, delimiter=",")
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1]


def _overlay_ref(
    ax: plt.Axes,
    filename: str,
    *,
    color: str,
    label: str | None = None,
) -> bool:
    loaded = _load_ref_csv(filename)
    if loaded is None:
        return False
    x, y = loaded
    mask = ~(np.isnan(x) | np.isnan(y))
    if not mask.any():
        return False
    ax.scatter(x[mask], y[mask], label=label, **ref_marker_kwargs(color=color))
    return True


def plot_figure4(data: dict) -> Path:
    fig, (ax_C, ax_D) = plt.subplots(1, 2, figsize=figure_size_inches(2, 1))

    time_h = data["time_h"]
    duration_h = float(time_h[-1])

    # ---- Panel C: temperatures ----
    ax_C.plot(time_h, data["t_abs"],   **_FIG4_STYLE["absorber"])
    ax_C.plot(time_h, data["t_glass"], **_FIG4_STYLE["glass"])
    ax_C.plot(time_h, data["t_cond"],  **_FIG4_STYLE["condenser"])
    ax_C.plot(time_h, data["t_amb"],   **_FIG4_STYLE["ambient"])

    ref_C = False
    ref_C |= _overlay_ref(
        ax_C, _REF_ABSORBER_CSV, color=_FIG4_STYLE["absorber"]["color"],
        label=_REF_LABEL if not ref_C else None,
    )
    ref_C |= _overlay_ref(
        ax_C, _REF_GLASS_CSV, color=_FIG4_STYLE["glass"]["color"],
    )
    ref_C |= _overlay_ref(
        ax_C, _REF_CONDENSER_CSV, color=_FIG4_STYLE["condenser"]["color"],
    )

    # Per-series and pooled R² vs the (corrected) digitized reference curves.
    r2_abs = _series_r2(time_h, data["t_abs"], _REF_ABSORBER_CSV)
    r2_glass = _series_r2(time_h, data["t_glass"], _REF_GLASS_CSV)
    r2_cond = _series_r2(time_h, data["t_cond"], _REF_CONDENSER_CSV)
    r2_lines = [
        f"$R^2$ absorber = {r2_abs:.3f}" if r2_abs is not None else "",
        f"$R^2$ glass = {r2_glass:.3f}" if r2_glass is not None else "",
        f"$R^2$ condenser = {r2_cond:.3f}" if r2_cond is not None else "",
    ]
    ax_C.text(
        0.97, 0.03, "\n".join(s for s in r2_lines if s),
        transform=ax_C.transAxes, fontsize=6.5, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc",
                  alpha=0.85, linewidth=0.6),
    )

    ax_C.set_ylabel("Temperature (°C)")
    _style_desorption_time_axis(ax_C, duration_h=duration_h)
    ax_C.set_ylim(bottom=0)
    ax_C.legend(fontsize=scaled_fontsize("legend.fontsize", 0.7), loc="upper left", frameon=False)
    ax_C.set_title("C", loc="left", fontweight="bold")
    _panel_style(ax_C)

    # ---- Panel D: cumulative water output ----
    ax_D.plot(time_h, data["cum_water_l_m2"], **_FIG4_STYLE["water"])

    # The paper's own model band at the end of the run, and its measured endpoint
    band_lo, band_mid, band_hi = _fig4d_model_l_m2()
    ax_D.errorbar(
        duration_h, band_mid,
        yerr=[[band_mid - band_lo], [band_hi - band_mid]],
        fmt="s", markersize=5, color="#555555", elinewidth=1.4, capsize=5, zorder=4,
        label=f"Wilson model ({band_mid:.2f} L/m², {band_lo:.2f}–{band_hi:.2f})",
    )
    ax_D.plot(
        duration_h, _MEASURED_YIELD_L_M2,
        marker="*", markersize=11, color="#e74c3c", zorder=5,
        linestyle="none", label=f"Measured ({_MEASURED_YIELD_L_M2:.2f} L/m²)",
    )

    ax_D.set_ylabel("Cumulative water output (L/m²)")
    _style_desorption_time_axis(ax_D, duration_h=duration_h)
    ax_D.set_ylim(bottom=0)
    ax_D.legend(fontsize=scaled_fontsize("legend.fontsize", 0.7), loc="upper left", frameon=False)
    ax_D.set_title("D", loc="left", fontweight="bold")
    _panel_style(ax_D)

    eta_pct = data["eta"] * 100
    yield_val = data["yield_kg"]
    fig.suptitle(
        "Wilson et al. (2025) Figure 4 — Atacama Desert field test (May 2024)\n"
        rf"Our yield = {yield_val:.3f} L/m² (Wilson model {band_mid:.2f}, measured 0.62 L/m²),"
        rf"  $\eta_{{\mathrm{{th}}}}$ = {eta_pct:.1f}% (measured 9.3%);"
        r"  open circles = digitized paper data",
        fontsize=scaled_fontsize("axes.labelsize", 0.7), y=1.02,
    )
    fig.tight_layout()

    out_path = _OUT_DIR / "figure4.png"
    print_figure(fig, out_path)
    print(f"\nSaved → {out_path}")
    plt.close(fig)
    return out_path


# --- Main ---

def main() -> Path:
    import argparse

    parser = argparse.ArgumentParser(description="Wilson Figure 4 — Atacama field test")
    parser.parse_args()

    print("Wilson Figure 4 (Atacama field test) — solar_lumped model")
    print("=" * 60)
    data = simulate_atacama()
    write_hourly_model_estimates(data)
    print("\nComposing figure…")
    out = plot_figure4(data)
    print("Done.")
    return out


if __name__ == "__main__":
    main()
