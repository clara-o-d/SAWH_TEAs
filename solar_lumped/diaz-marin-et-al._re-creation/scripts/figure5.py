#!/usr/bin/env python3
"""
Recreation of Díaz-Marín et al. (2024) Nature Communications Figure 5 (panels C, D, E, I).

Kinetics are simulated via ``run_solar_sim.simulate_isothermal_chamber_rh_cycle`` using
the Díaz-Marín hydrogel-only model: Eq. 5 brine thermodynamics + Eq. 8 chamber
convection (no Wilson SAWH device, vapor gap, or condenser).

Chamber boundary conditions from the paper SI:
  - g_chamber = 0.0095 m/s (Note S8)
  - T_amb = 25 °C (isothermal)
  - H₀ from Table S3 (mm, at equilibrium with RH = 20 %)
  - Uptake axis: U = (U_dry − U₂₀) / (1 + U₂₀)  (Methods)
  - RH switch times: inferred from source-data workbook (41467_2024_53291_MOESM3_ESM.xlsx)

Each panel overlays three RH cycles (20 → 30/50/70 → 20 %) against digitized data.

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

_OUT_DIR = _DIAZ_DIR / "outputs" / "figure5"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_REF_DIR = _DIAZ_DIR / "reference" / "figure5"
_REF_LABEL = "Díaz-Marín et al. (digitized)"

# SI Note S8 — Diaz-Marín environmental-chamber g (Wilson default is 0.0085)
_G_CONV_M_S: float = 0.0095
_TEMPERATURE_C: float = 25.0
_RH_BASELINE: float = 0.20

# Table S3 — average thickness at RH = 20 % (mm)
_H0_MM: dict[str, float] = {
    "pam_licl_4": 2.34,
    "pam_licl_2": 2.16,
    "pva_licl_4": 2.15,
    "pam_licl_4_thick": 2.34 * 1.5,
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


_PANELS: tuple[HydrogelCase, ...] = (
    HydrogelCase("5c", "C  PAM--LiCl 4 g/g", 4.0, _H0_MM["pam_licl_4"]),
    HydrogelCase("5d", "D  PAM--LiCl 2 g/g", 2.0, _H0_MM["pam_licl_2"], t_max_min=3600.0),
    HydrogelCase("5e", "E  PVA--LiCl 4 g/g", 4.0, _H0_MM["pva_licl_4"]),
    HydrogelCase(
        "5i",
        "I  PAM--LiCl 4 g/g (1.5 H₀)",
        4.0,
        _H0_MM["pam_licl_4_thick"],
        t_max_min=7800.0,
    ),
)


def _chamber_params(case: HydrogelCase) -> run_solar_sim.HydrogelChamberParams:
    return run_solar_sim.build_hydrogel_chamber_params(
        salt=case.salt_name,
        salt_loading=case.salt_to_polymer_ratio,
        h0_mm=case.h0_mm,
        g_conv_m_s=_G_CONV_M_S,
    )


def simulate_rh_cycle(
    case: HydrogelCase,
    rh_high: float,
    *,
    schedules: dict[tuple[str, int], object],
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
    ax.scatter(
        x[mask],
        y[mask],
        s=18,
        marker="o",
        facecolors="white",
        edgecolors=color,
        linewidths=1.0,
        zorder=6,
        label=label,
    )
    return True


def _style_kinetics_axes(ax: plt.Axes, *, panel_title: str, t_max: float) -> None:
    ax.set_xlabel("time [min]", fontsize=10)
    ax.set_ylabel("uptake [g/g]", fontsize=10)
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_title(panel_title, loc="left", fontweight="bold", fontsize=10)


def plot_panel(
    ax: plt.Axes,
    case: HydrogelCase,
    *,
    schedules: dict[tuple[str, int], object],
) -> None:
    ref_shown = False
    t_max = 0.0
    for rh_pct in (30, 50, 70):
        rh = rh_pct / 100.0
        color = _RH_COLORS[rh_pct]
        t_min, uptake = simulate_rh_cycle(case, rh, schedules=schedules)
        t_max = max(t_max, float(t_min[-1]))
        ax.plot(t_min, uptake, color="black", linewidth=1.5, zorder=4)
        csv_name = f"{case.key}_{rh_pct}.csv"
        ref_label = _REF_LABEL if not ref_shown else None
        if _overlay_ref(ax, csv_name, color=color, label=ref_label):
            ref_shown = True

    handles = [
        plt.Line2D(
            [0], [0], color=_RH_COLORS[p], linewidth=0.0, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor=_RH_COLORS[p], markeredgewidth=1.0,
        )
        for p in (30, 50, 70)
    ]
    labels = [f"20–{p}–20 % RH" for p in (30, 50, 70)]
    handles.append(plt.Line2D([0], [0], color="black", linewidth=1.5))
    labels.append("model (Eq. 5 + 8)")
    ax.legend(handles, labels, fontsize=6.5, frameon=False, loc="upper right")
    _style_kinetics_axes(ax, panel_title=case.panel_title, t_max=t_max)


def plot_figure5(schedules: dict[tuple[str, int], object]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for ax, case in zip(axes.ravel(), _PANELS, strict=True):
        plot_panel(ax, case, schedules=schedules)

    fig.suptitle(
        "Díaz-Marín et al. (2024) Figure 5 — absorption–desorption kinetics\n"
        r"(black = hydrogel Eq. 5 + 8 via run_solar_sim, $g_{chamber}=0.0095$ m/s; "
        "circles = digitized data)",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()

    out_path = _OUT_DIR / "figure5.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    for case in _PANELS:
        fig_s, ax_s = plt.subplots(figsize=(5.5, 4.2))
        plot_panel(ax_s, case, schedules=schedules)
        out_s = _OUT_DIR / f"figure{case.key}.png"
        fig_s.savefig(out_s, dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        print(f"Saved → {out_s}")

    return out_path


def main() -> Path:
    schedules = load_chamber_rh_schedules()
    print("Díaz-Marín Figure 5 — hydrogel Eq. 5 + 8 (run_solar_sim)")
    print("=" * 56)
    print(f"  Module: {run_solar_sim.__file__}")
    print(f"  g_chamber = {_G_CONV_M_S} m/s  |  T_amb = {_TEMPERATURE_C} °C")
    print(f"  Reference data: {_REF_DIR}")
    print("\nRH switch times from source-data workbook (high → 20 %):")
    print(format_schedule_table(schedules))
    for case in _PANELS:
        print(f"  {case.panel_title}: SL={case.salt_to_polymer_ratio}, H₀={case.h0_mm:.2f} mm")
    return plot_figure5(schedules)


if __name__ == "__main__":
    main()
