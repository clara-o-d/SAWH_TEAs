#!/usr/bin/env python3
"""
Recreation of Wilson et al. (2025) Device Figure 3B.

Plots model-predicted temperatures (absorber, glass, condenser) over the
10-hour Cambridge desorption window using digitized solar flux and ambient
temperature from the paper figures (CSVs in the weather directory).

Uncertainty band: h_amb = 10 ± 2.5 W/m²K, matching Wilson Methods.

Optional ``--wilson-initial-temps``: start desorption at the first digitized
Fig. 3C reference point for each component (absorber / glass / condenser).

``--desorption-solver {quasi_steady,segregated,coupled_bdf}`` (default: quasi_steady).

Output saved to:  wilson-et-al._re-creation/outputs/figure3/figure3b.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (same pattern as figure2.py)
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).resolve()
_WILSON_DIR = _SCRIPT.parent.parent          # wilson-et-al._re-creation/
_SOLAR_ROOT = _WILSON_DIR.parent             # solar_lumped/
_SRC = _SOLAR_ROOT / "src"
for _p in (_SRC, _SOLAR_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from solar_lumped.physics import table_s3
from solar_lumped.simulation.coupled_dynamics import evaluate_coupled_rates
from solar_lumped.simulation.device_config import DeviceConfig, register_desorption_solver_cli
from solar_lumped.simulation.ode_system import run_daily_cycle
from solar_lumped.simulation.water_inventory import cumulative_desorption_yield_l_m2
from solar_lumped.weather.profiles import (
    DailyWeatherProfile,
    PhaseProfile,
    PHASE_DT_S,
    STEPS_PER_PHASE,
)

_OUT_DIR = _WILSON_DIR / "outputs" / "figure3"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_REF_DIR = _WILSON_DIR / "reference" / "figure3"
_WEATHER_DIR = _SRC / "solar_lumped" / "weather"
_REF_LABEL = "Wilson et al. (digitized)"

_REF_ABSORBER_CSV = "Cambridge_absorber.csv"
_REF_GLASS_CSV = "Cambridge_glass.csv"
_REF_CONDENSER_CSV = "Cambridge_condenser.csv"

# Solar absorber footprint area (Wilson Table S3). The digitized Fig. 3C water
# output is reported as device-total mL (paper: "~130 mL total", Δm_gel = 131 g);
# dividing by A_c converts it to the mL/m² basis used for the model curve.
_A_C_M2 = 0.078

# ---------------------------------------------------------------------------
# Cambridge test conditions (Wilson paper Methods / Fig. 3)
# ---------------------------------------------------------------------------
_T_ABS_C = 25.0       # absorption temperature (room adjacent to roof, ~25 °C)
_RH_ABS = 0.5         # ~50% RH absorption (paper Methods)
_RH_DES = 0.5         # RH value during desorption (sealed device; only affects CR via T_cond)
_H_AMB_MID = 10.0
_H_AMB_LO = 7.5
_H_AMB_HI = 12.5

# Desorption window: 10 h at 1-min steps
_DES_HOURS = 10.0
_DES_STEPS = int(_DES_HOURS * 3600.0 / PHASE_DT_S)   # 600


# ---------------------------------------------------------------------------
# 1. Load and interpolate Cambridge weather CSVs
# ---------------------------------------------------------------------------

def _interp_csv(
    raw: np.ndarray,
    t_grid: np.ndarray,
    *,
    clip_min: float | None = None,
) -> np.ndarray:
    """
    Linear interpolation of digitized (time, value) CSV data onto t_grid.

    Data points are sorted and deduplicated. np.interp is used so that
    values outside the data range are clamped to the nearest boundary value
    (no extrapolation oscillation from higher-order splines).
    """
    t_raw, v_raw = raw[:, 0], raw[:, 1]
    sort_idx = np.argsort(t_raw)
    t_raw, v_raw = t_raw[sort_idx], v_raw[sort_idx]
    _, uniq = np.unique(t_raw, return_index=True)
    t_raw, v_raw = t_raw[uniq], v_raw[uniq]
    out = np.interp(t_grid, t_raw, v_raw)
    if clip_min is not None:
        out = np.maximum(clip_min, out)
    return out


def _load_cambridge_weather() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_hr_grid, solar_W_m2, amb_T_C) on a 1-min grid over 0–10 h."""
    solar_raw = np.loadtxt(_WEATHER_DIR / "Cambridge_solar_W_m2.csv", delimiter=",")
    temp_raw = np.loadtxt(_WEATHER_DIR / "Cambridge_amb_T_C.csv", delimiter=",")

    t_grid = np.linspace(0.0, _DES_HOURS, _DES_STEPS, endpoint=False)

    solar_grid = _interp_csv(solar_raw, t_grid, clip_min=0.0)
    temp_grid = _interp_csv(temp_raw, t_grid)

    return t_grid, solar_grid, temp_grid


# ---------------------------------------------------------------------------
# 2. Build PhaseProfiles
# ---------------------------------------------------------------------------

def _make_absorption_profile(*, h_amb: float = _H_AMB_MID) -> PhaseProfile:
    """Standard 12-h absorption at ~25 °C, 50% RH, no solar."""
    n = STEPS_PER_PHASE
    return PhaseProfile(
        temperature_c=(_T_ABS_C,) * n,
        relative_humidity=(_RH_ABS,) * n,
        solar_w_m2=(0.0,) * n,
        h_amb_w_m2_k=(h_amb,) * n,
    )


def _make_desorption_profile(
    solar_grid: np.ndarray,
    temp_grid: np.ndarray,
    *,
    h_amb: float,
) -> PhaseProfile:
    """10-h desorption PhaseProfile driven by Cambridge CSV weather."""
    n = _DES_STEPS
    assert len(solar_grid) == n and len(temp_grid) == n
    return PhaseProfile(
        temperature_c=tuple(float(x) for x in temp_grid),
        relative_humidity=(_RH_DES,) * n,
        solar_w_m2=tuple(float(x) for x in solar_grid),
        h_amb_w_m2_k=(h_amb,) * n,
        dt_s=PHASE_DT_S,
    )


def _make_profile(
    solar_grid: np.ndarray,
    temp_grid: np.ndarray,
    *,
    h_amb: float,
) -> DailyWeatherProfile:
    return DailyWeatherProfile(
        absorption=_make_absorption_profile(h_amb=h_amb),
        desorption=_make_desorption_profile(solar_grid, temp_grid, h_amb=h_amb),
    )


# ---------------------------------------------------------------------------
# 3. Device configuration (optimised Cambridge device, Wilson Methods)
# ---------------------------------------------------------------------------

def _make_config(
    *,
    wilson_initial_temps: bool = False,
    desorption_solver: str = "quasi_steady",
) -> DeviceConfig:
    kwargs: dict[str, object] = {
        "hydrogel_thickness_m": 0.004,   # H0 = 4 mm
        "vapor_gap_m": 0.040,            # L_g = 40 mm
        "fin_area_ratio": table_s3.FIN_AREA_RATIO,  # A_r = 7.1 (Wilson Table S3)
        "tilt_deg": 35.0,                # Cambridge rooftop tilt (~35°, Methods)
        "desorption_solver": desorption_solver,
    }
    if wilson_initial_temps:
        kwargs["coupled_initial_temps_c"] = _cambridge_initial_temps_c()
    return DeviceConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4 & 5. Run ODE + reconstruct absorber and glass temperatures
# ---------------------------------------------------------------------------

def _reconstruct_all_temps(
    des_res,
    des_profile: PhaseProfile,
    config: DeviceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (t_abs_arr, t_glass_arr) at every ODE output time point by
    re-evaluating evaluate_coupled_rates with the solved (c_w, H, T_cond).
    Mirrors the internal post-processing loop in ode_system._integrate_desorption.
    """
    mass = config.mass_params()
    thermal = config.thermal_params()
    tmass = config.condenser_thermal_mass_j_m2_k()
    n = len(des_profile.temperature_c)
    dt = des_profile.dt_s

    t_abs_hist: list[float] = []
    t_glass_hist: list[float] = []
    guess: tuple[float, float, float] | None = None

    for k in range(len(des_res.time_s)):
        i = min(int(des_res.time_s[k] / dt), n - 1)
        rates = evaluate_coupled_rates(
            c_w=float(des_res.c_w[k]),
            h_m=float(des_res.H[k]),
            t_cond_c=float(des_res.t_cond_c[k]),
            t_amb_c=des_profile.temperature_c[i],
            rh=des_profile.relative_humidity[i],
            q_solar_w_m2=des_profile.solar_w_m2[i],
            h_amb=des_profile.h_amb_w_m2_k[i],
            phase="desorption",
            mass=mass,
            thermal=thermal,
            vapor_gap_m=config.vapor_gap_m,
            condenser_thermal_mass_j_m2_k=tmass,
            fin_area_ratio=config.fin_area_ratio,
            h_fg_j_per_kg=config.h_fg_j_per_kg,
            config=config,
            t_guess=guess,
        )
        t_abs_hist.append(rates.thermal.t_abs_c)
        t_glass_hist.append(rates.thermal.t_glass_c)
        guess = (rates.thermal.t_gel_c, rates.thermal.t_abs_c, rates.thermal.t_glass_c)

    return np.array(t_abs_hist), np.array(t_glass_hist)


def run_simulation(
    solar_grid: np.ndarray,
    temp_grid: np.ndarray,
    *,
    h_amb: float,
    wilson_initial_temps: bool = False,
    desorption_solver: str = "quasi_steady",
) -> dict[str, np.ndarray]:
    """
    Run one daily cycle and return a dict of temperature arrays vs time.
    Keys: 'time_hr', 't_abs', 't_glass', 't_cond', 't_gel', 't_amb'.
    """
    config = _make_config(
        wilson_initial_temps=wilson_initial_temps,
        desorption_solver=desorption_solver,
    )
    profile = _make_profile(solar_grid, temp_grid, h_amb=h_amb)

    _, _, _, des_res = run_daily_cycle(profile, config)

    if des_res.t_abs_c is not None and des_res.t_glass_c is not None:
        t_abs, t_glass = des_res.t_abs_c, des_res.t_glass_c
    else:
        t_abs, t_glass = _reconstruct_all_temps(des_res, profile.desorption, config)
    time_hr = des_res.time_s / 3600.0

    # Ambient temperature at each ODE output time point (for reference)
    n = _DES_STEPS
    dt = PHASE_DT_S
    amb_interp: list[float] = []
    for k in range(len(des_res.time_s)):
        i = min(int(des_res.time_s[k] / dt), n - 1)
        amb_interp.append(profile.desorption.temperature_c[i])

    cum_water_l_m2 = cumulative_desorption_yield_l_m2(
        des_res.time_s, des_res.m_des_kg_s_m2
    )

    return {
        "time_hr": time_hr,
        "t_abs": t_abs,
        "t_glass": t_glass,
        "t_cond": des_res.t_cond_c,
        "t_gel": des_res.t_gel_c,
        "t_amb": np.array(amb_interp),
        "cum_water_ml_m2": cum_water_l_m2 * 1000.0,
    }


# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------

# Colours matching Wilson Fig. 3B
_COL_ABS = "#8b2000"      # dark brick red — absorber
_COL_GLASS = "#b06000"    # brown/orange — glass
_COL_COND = "#007090"     # teal/cyan — condenser
_COL_AMB = "#2040b0"      # blue — ambient reference


def _fill(ax, t, lo, hi, color):
    ax.fill_between(t, lo, hi, color=color, alpha=0.18, linewidth=0)


def _first_ref_temp(filename: str, default: float) -> float:
    """First y-value from a digitized reference CSV (Wilson start-of-curve point)."""
    loaded = _load_ref_csv(filename)
    if loaded is None:
        return default
    _, y = loaded
    mask = ~np.isnan(y)
    if not mask.any():
        return default
    return float(y[mask][0])


def _cambridge_initial_temps_c() -> tuple[float, float, float, float]:
    """Wilson digitized Fig. 3C first points: gel=abs, glass, condenser."""
    t_abs = _first_ref_temp(_REF_ABSORBER_CSV, _T_ABS_C)
    t_glass = _first_ref_temp(_REF_GLASS_CSV, _T_ABS_C)
    t_cond = _first_ref_temp(_REF_CONDENSER_CSV, _T_ABS_C)
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
    y_scale: float = 1.0,
) -> bool:
    loaded = _load_ref_csv(filename)
    if loaded is None:
        return False
    x, y = loaded
    mask = ~(np.isnan(x) | np.isnan(y))
    if not mask.any():
        return False
    ax.scatter(
        x[mask],
        y[mask] * y_scale,
        s=22,
        marker="o",
        facecolors="white",
        edgecolors=color,
        linewidths=1.1,
        zorder=6,
        label=label,
    )
    return True


def _style_temp_axes(ax: plt.Axes, *, legend_fontsize: float = 7.5) -> None:
    ax.set_xlabel("time [hr]", fontsize=10)
    ax.set_ylabel("temperature [°C]", fontsize=10)
    ax.set_xlim(0, _DES_HOURS)
    ax.set_ylim(10, 80)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.legend(fontsize=legend_fontsize, frameon=False, loc="upper right")


def _plot_temp_panel(
    ax: plt.Axes,
    res_lo: dict,
    res_mid: dict,
    res_hi: dict,
    t_grid_hr: np.ndarray,
    temp_grid: np.ndarray,
    *,
    panel_title: str | None = None,
    legend_fontsize: float = 7.5,
) -> None:
    t = res_mid["time_hr"]

    def band_lo(key):
        return np.minimum(res_lo[key], np.minimum(res_mid[key], res_hi[key]))

    def band_hi(key):
        return np.maximum(res_lo[key], np.maximum(res_mid[key], res_hi[key]))

    ax.plot(t, res_mid["t_abs"], color=_COL_ABS, linewidth=1.8, label="absorber (model)")
    _fill(ax, t, band_lo("t_abs"), band_hi("t_abs"), _COL_ABS)
    ax.plot(t, res_mid["t_glass"], color=_COL_GLASS, linewidth=1.8, label="glass (model)")
    _fill(ax, t, band_lo("t_glass"), band_hi("t_glass"), _COL_GLASS)
    ax.plot(t, res_mid["t_cond"], color=_COL_COND, linewidth=1.8, label="condenser (model)")
    _fill(ax, t, band_lo("t_cond"), band_hi("t_cond"), _COL_COND)
    ax.plot(t_grid_hr, temp_grid, color=_COL_AMB, linewidth=1.0, linestyle="--",
            label="ambient (measured)")

    ref = False
    ref |= _overlay_ref(
        ax, "Cambridge_absorber.csv", color=_COL_ABS,
        label=_REF_LABEL if not ref else None,
    )
    ref |= _overlay_ref(ax, "Cambridge_glass.csv", color=_COL_GLASS)
    ref |= _overlay_ref(ax, "Cambridge_condenser.csv", color=_COL_COND)

    _style_temp_axes(ax, legend_fontsize=legend_fontsize)
    if panel_title is not None:
        ax.set_title(panel_title, loc="left", fontweight="bold", fontsize=10)


def plot_figure3b(
    res_lo: dict,
    res_mid: dict,
    res_hi: dict,
    t_grid_hr: np.ndarray,
    solar_grid: np.ndarray,
    temp_grid: np.ndarray,
) -> Path:
    fig, (ax_B, ax_C) = plt.subplots(1, 2, figsize=(12, 4.5))
    t = res_mid["time_hr"]

    _plot_temp_panel(
        ax_B, res_lo, res_mid, res_hi, t_grid_hr, temp_grid, panel_title="B",
    )

    # ---- Panel C: cumulative water output ----
    ax_C.plot(
        t, res_mid["cum_water_ml_m2"], color="#1a6b5a", linewidth=1.8,
        label="water output (model)",
    )
    # Digitized data is device-total mL; convert to mL/m² (÷ A_c) to match model.
    if not _overlay_ref(
        ax_C, "Cambridge_water_output_ml.csv", color="#1a6b5a", label=_REF_LABEL,
        y_scale=1.0 / _A_C_M2,
    ):
        print("  Warning: Cambridge_water_output_ml.csv not found")

    ax_C.set_xlabel("time [hr]", fontsize=10)
    ax_C.set_ylabel("cumulative water output [mL/m²]", fontsize=10)
    ax_C.set_xlim(0, _DES_HOURS)
    ax_C.set_ylim(bottom=0)
    ax_C.tick_params(direction="in", which="both", top=True, right=True)
    ax_C.grid(False)
    for spine in ax_C.spines.values():
        spine.set_linewidth(0.8)
    ax_C.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax_C.set_title("C", loc="left", fontweight="bold", fontsize=10)

    fig.suptitle(
        "Wilson et al. (2025) Figure 3 — Cambridge field test\n"
        r"(model lines; open circles = digitized paper data; band = $h_{amb}=10\pm2.5$ W/m²K)",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()

    out_path = _OUT_DIR / "figure3.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    fig_b, ax_b = plt.subplots(figsize=(7, 4.5))
    _plot_temp_panel(ax_b, res_lo, res_mid, res_hi, t_grid_hr, temp_grid,
                     legend_fontsize=8.5)
    out_b = _OUT_DIR / "figure3b.png"
    fig_b.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close(fig_b)
    print(f"Saved → {out_b}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Path:
    import argparse

    parser = argparse.ArgumentParser(
        description="Wilson Figure 3 — Cambridge field-test temperature model",
    )
    register_desorption_solver_cli(parser)
    parser.add_argument(
        "--wilson-initial-temps",
        action="store_true",
        help=(
            "Start desorption at the first digitized Wilson Fig. 3C temperatures "
            "(absorber, glass, condenser CSVs)."
        ),
    )
    args = parser.parse_args()

    print("Wilson Figure 3B — Cambridge temperature model")
    print("=" * 50)
    print(f"  desorption_solver = {args.desorption_solver}")
    if args.wilson_initial_temps:
        ic = _cambridge_initial_temps_c()
        print(
            f"  wilson_initial_temps: gel/abs={ic[0]:.1f} °C, "
            f"glass={ic[2]:.1f} °C, cond={ic[3]:.1f} °C"
        )

    print("\nLoading and interpolating Cambridge weather CSVs…")
    t_grid_hr, solar_grid, temp_grid = _load_cambridge_weather()

    print(
        f"  Solar: min={solar_grid.min():.0f}, max={solar_grid.max():.0f} W/m²  "
        f"  T_amb: min={temp_grid.min():.1f}, max={temp_grid.max():.1f} °C"
    )

    sim_kw = {
        "wilson_initial_temps": args.wilson_initial_temps,
        "desorption_solver": args.desorption_solver,
    }
    print("\nRunning simulations for h_amb = 7.5, 10.0, 12.5 W/m²K…")
    res_lo = run_simulation(solar_grid, temp_grid, h_amb=_H_AMB_LO, **sim_kw)
    print(f"  h_amb={_H_AMB_LO}: T_abs peak = {res_lo['t_abs'].max():.1f} °C")

    res_mid = run_simulation(solar_grid, temp_grid, h_amb=_H_AMB_MID, **sim_kw)
    print(f"  h_amb={_H_AMB_MID}: T_abs peak = {res_mid['t_abs'].max():.1f} °C")

    res_hi = run_simulation(solar_grid, temp_grid, h_amb=_H_AMB_HI, **sim_kw)
    print(f"  h_amb={_H_AMB_HI}: T_abs peak = {res_hi['t_abs'].max():.1f} °C")

    print("\nPlotting Figure 3…")
    out = plot_figure3b(res_lo, res_mid, res_hi, t_grid_hr, solar_grid, temp_grid)
    print("Done.")
    return out


if __name__ == "__main__":
    main()
