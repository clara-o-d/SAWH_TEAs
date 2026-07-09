#!/usr/bin/env python3
"""
Recreation of Díaz-Marín et al. (2024) Nature Communications Figure 5 (panels C, D, E, I).

Kinetics are simulated via ``run_solar_sim.simulate_isothermal_chamber_rh_cycle`` using
the Díaz-Marín hydrogel-only model: Eq. 5 brine thermodynamics (Conde 2004) + Eq. 8 chamber
convection (no Wilson SAWH device, vapor gap, or condenser).

Chamber boundary conditions from the paper SI:
  - g_chamber = 0.0095 m/s (Note S8)
  - T_amb = 25 °C (isothermal)
  - H₀ from Table S3 (mm, at equilibrium with RH = 20 %)
  - c_s from Methods pour inventory (12.8 mL for PAM-LiCl 2 g/g; 8 mL otherwise),
    anchored to the 4 g/g DVS dry-basis calibration; 2 g/g uses the 4 g/g reference
    H₀ (2.34 mm) for c_s per SI Note S7 constant 20 % density (Table S3 H₀ in g/H₀)
  - Uptake axis: U = (U_dry − U₂₀) / (1 + U₂₀)  (Methods)
  - RH switch times: from digitized reference curves (``reference/figure5/*.csv``;
    use ``--schedule-source esm`` for MOESM3 workbook inference)
  - Chamber integration timestep: 5000 s by default; override with ``--dt-s``

Each panel overlays three RH cycles (20 → 30/50/70 → 20 %) against digitized
model curves from the paper figure (not experimental scatter).

Output saved to:  diaz-marin-et-al._re-creation/outputs/figure5/figure5.png
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — import run_solar_sim.py from scripts/
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).resolve()
_DIAZ_DIR = _SCRIPT.parent.parent
_SOLAR_ROOT = _DIAZ_DIR.parent
_SRC = _SOLAR_ROOT / "src"
_SCRIPTS = _SOLAR_ROOT / "scripts"
for _p in (_SRC, _SCRIPTS, _SOLAR_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_solar_sim  # noqa: E402
from chamber_rh_schedule import (  # noqa: E402
    format_schedule_table,
    load_chamber_rh_schedules,
)
from solar_lumped.plotting.matlab_style import (  # noqa: E402
    figure_size_inches,
    panel_size_inches,
    plot_defaults_slides,
    print_figure,
    ref_marker_kwargs,
    scaled_fontsize,
    style_axes,
)

plot_defaults_slides()

_OUT_DIR = _DIAZ_DIR / "outputs" / "figure5"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_REF_DIR = _DIAZ_DIR / "reference" / "figure5"
_REF_LABEL = "Díaz-Marín et al. (digitized model)"

# SI Note S8 — Diaz-Marín environmental-chamber g (Wilson default is 0.0085)
_G_CONV_M_S: float = 0.0095
_TEMPERATURE_C: float = 25.0
_RH_BASELINE: float = 0.20
# Explicit Euler step for Eq. 8 chamber integration (s).
_CHAMBER_DT_S: float = 5000.0

# Table S3 — average thickness at RH = 20 % (mm)
_H0_MM: dict[str, float] = {
    "pam_licl_4": 2.34,
    "pam_licl_2": 2.16,
    "pva_licl_4": 2.15,
    "pam_licl_4_thick": 3.2,  # Fig. 5i: ~3.2 mm at RH = 20 % (50 % thicker than 2.34 mm)
}

_RH_COLORS: dict[int, str] = {
    30: "#1f77b4",
    50: "#c9a227",
    70: "#2ca02c",
}


@dataclass(frozen=True, slots=True)
class HydrogelCase:
    key: str
    panel_title: str
    salt_to_polymer_ratio: float
    h0_mm: float
    salt_name: str = "LiCl"
    t_max_min: float = 5200.0
    pour_ml: float | None = None  # None → Methods default for SL (12.8 mL at 2 g/g, else 8 mL)


_PANELS: tuple[HydrogelCase, ...] = (
    HydrogelCase("5c", "C  PAM--LiCl 4 g/g", 4.0, _H0_MM["pam_licl_4"]),
    HydrogelCase("5d", "D  PAM--LiCl 2 g/g", 2.0, _H0_MM["pam_licl_2"], t_max_min=3600.0),
    HydrogelCase("5e", "E  PVA--LiCl 4 g/g", 4.0, _H0_MM["pva_licl_4"]),
    HydrogelCase(
        "5i",
        "I  PAM--LiCl 4 g/g (~3.2 mm)",
        4.0,
        _H0_MM["pam_licl_4_thick"],
        t_max_min=7800.0,
        pour_ml=12.0,  # ~1.5× standard pour for thicker H₀ (workbook: 4 g/g 1.5H)
    ),
)


def _chamber_params(case: HydrogelCase) -> run_solar_sim.HydrogelChamberParams:
    return run_solar_sim.build_hydrogel_chamber_params(
        salt=case.salt_name,
        salt_loading=case.salt_to_polymer_ratio,
        h0_mm=case.h0_mm,
        g_conv_m_s=_G_CONV_M_S,
        pour_ml=case.pour_ml,
    )


def simulate_rh_cycle(
    case: HydrogelCase,
    rh_high: float,
    *,
    schedules: dict[tuple[str, int], object],
    dt_s: float = _CHAMBER_DT_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Delegate to ``run_solar_sim`` with ESM-inferred RH switch times."""
    params = _chamber_params(case)
    rh_pct = int(round(rh_high * 100.0))
    sched = schedules[(case.key, rh_pct)]
    return run_solar_sim.simulate_isothermal_chamber_rh_cycle(
        params,
        rh_high,
        rh_baseline=_RH_BASELINE,
        temperature_c=_TEMPERATURE_C,
        t_max_min=sched.t_end_min,
        t_high_to_20_min=sched.t_high_to_20_min,
        dt_s=dt_s,
    )


# ---------------------------------------------------------------------------
# Reference CSV helpers
# ---------------------------------------------------------------------------


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


def _style_kinetics_axes(ax: plt.Axes, *, panel_title: str, t_max: float) -> None:
    ax.set_xlabel("time [min]")
    ax.set_ylabel("uptake [g/g]")
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.05, 1.05)
    style_axes(ax)
    ax.set_title(panel_title, loc="left", fontweight="bold")


def plot_panel(
    ax: plt.Axes,
    case: HydrogelCase,
    *,
    schedules: dict[tuple[str, int], object],
    dt_s: float = _CHAMBER_DT_S,
) -> None:
    ref_shown = False
    t_max = 0.0
    for rh_pct in (30, 50, 70):
        rh = rh_pct / 100.0
        color = _RH_COLORS[rh_pct]
        t_min, uptake = simulate_rh_cycle(case, rh, schedules=schedules, dt_s=dt_s)
        t_max = max(t_max, float(t_min[-1]))
        ax.plot(t_min, uptake, color="black", zorder=4)
        csv_name = f"{case.key}_{rh_pct}.csv"
        ref_label = _REF_LABEL if not ref_shown else None
        if _overlay_ref(ax, csv_name, color=color, label=ref_label):
            ref_shown = True

    handles = [
        plt.Line2D(
            [0], [0], color=_RH_COLORS[p], linewidth=0.0, marker="o", markersize=6,
            markerfacecolor="white", markeredgecolor=_RH_COLORS[p], markeredgewidth=1.5,
        )
        for p in (30, 50, 70)
    ]
    labels = [f"20–{p}–20 % RH" for p in (30, 50, 70)]
    handles.append(plt.Line2D([0], [0], color="black"))
    labels.append("model (Eq. 5 + 8)")
    ax.legend(
        handles,
        labels,
        fontsize=scaled_fontsize("legend.fontsize", 0.7),
        frameon=False,
        loc="upper right",
    )
    _style_kinetics_axes(ax, panel_title=case.panel_title, t_max=t_max)


def plot_figure5(
    schedules: dict[tuple[str, int], object],
    *,
    dt_s: float = _CHAMBER_DT_S,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=figure_size_inches(2, 2))
    for ax, case in zip(axes.ravel(), _PANELS, strict=True):
        plot_panel(ax, case, schedules=schedules, dt_s=dt_s)

    fig.suptitle(
        "Díaz-Marín et al. (2024) Figure 5 — absorption–desorption kinetics\n"
        r"(black = hydrogel Eq. 5 + 8 via run_solar_sim, $g_{chamber}=0.0095$ m/s; "
        "circles = digitized model curves)",
        fontsize=scaled_fontsize("axes.labelsize", 0.75),
        y=1.01,
    )
    fig.tight_layout()

    out_path = _OUT_DIR / "figure5.png"
    print_figure(fig, out_path)
    plt.close(fig)
    print(f"Saved → {out_path}")

    for case in _PANELS:
        fig_s, ax_s = plt.subplots(figsize=panel_size_inches())
        plot_panel(ax_s, case, schedules=schedules, dt_s=dt_s)
        out_s = _OUT_DIR / f"figure{case.key}.png"
        print_figure(fig_s, out_s)
        plt.close(fig_s)
        print(f"Saved → {out_s}")

    return out_path


def main() -> Path:
    import argparse

    ap = argparse.ArgumentParser(description="Díaz-Marín Figure 5 recreation")
    ap.add_argument(
        "--dt-s",
        type=float,
        default=_CHAMBER_DT_S,
        help="Chamber ODE integration timestep in seconds (default: 5000)",
    )
    ap.add_argument(
        "--schedule-source",
        choices=("reference", "esm"),
        default="reference",
        help="RH step times: digitized reference CSVs (default) or ESM workbook",
    )
    args = ap.parse_args()
    if args.dt_s <= 0:
        raise SystemExit("--dt-s must be positive")

    schedules = load_chamber_rh_schedules(source=args.schedule_source)
    schedule_label = (
        "digitized reference curves"
        if args.schedule_source == "reference"
        else "MOESM3 source-data workbook"
    )
    print("Díaz-Marín Figure 5 — hydrogel Eq. 5 + 8 (run_solar_sim)")
    print("=" * 56)
    print(f"  Module: {run_solar_sim.__file__}")
    print(f"  g_chamber = {_G_CONV_M_S} m/s  |  T_amb = {_TEMPERATURE_C} °C")
    print(f"  dt = {args.dt_s:g} s ({args.dt_s / 60:.4g} min)")
    print(f"  Reference data: {_REF_DIR}")
    print(f"\nRH switch times from {schedule_label} (high → 20 %):")
    print(format_schedule_table(schedules))
    for case in _PANELS:
        print(f"  {case.panel_title}: SL={case.salt_to_polymer_ratio}, H₀={case.h0_mm:.2f} mm")
    return plot_figure5(schedules, dt_s=args.dt_s)


if __name__ == "__main__":
    main()
