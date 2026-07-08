#!/usr/bin/env python3
"""
Recreation of Díaz-Marín et al. (2024) Nature Communications Figure 3 (panels B, C, E).

Plots the thermodynamic uptake model (Eq. 5) for PAM--LiCl hydrogels using our LiCl
brine isotherm (Campbell / Conde-style activity correlation in salt_properties.py).
Digitized experimental points from the paper figure are overlaid as open markers.

Panels with reference data:
  B — salt content (0, 1, 4, 8 g LiCl / g polymer)
  C — polymer chemistry (PAM vs PVA; model curves coincide)
  E — crosslinking density (model curves coincide)

Output saved to:  diaz-marin-et-al._re-creation/outputs/figure3/figure3.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (same pattern as wilson-et-al._re-creation scripts)
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).resolve()
_DIAZ_DIR = _SCRIPT.parent.parent
_SOLAR_ROOT = _DIAZ_DIR.parent
_SRC = _SOLAR_ROOT / "src"
for _p in (_SRC, _SOLAR_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from solar_lumped.physics.salt_properties import (
    WATER_MOLAR_MASS_KG_MOL,
    get_salt,
    licl_equilibrium_brine_salt_fraction,
)

_OUT_DIR = _DIAZ_DIR / "outputs" / "figure3"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_REF_DIR = _DIAZ_DIR / "reference" / "figure3"
_REF_LABEL = "Díaz-Marín et al. (digitized)"

_TEMPERATURE_C = 25.0
_RH_GRID = np.linspace(0.0, 0.92, 200)

# ---------------------------------------------------------------------------
# Eq. 5: U = [SL / (SL + 1)] · [(x_w · i) / (1 − x_w)] · (MW_w / MW_s)
# x_w fixed by RH through our LiCl brine isotherm (a_w = RH at equilibrium).
# ---------------------------------------------------------------------------


def diaz_marin_licl_uptake_g_g(
    relative_humidity: float,
    salt_to_polymer_ratio: float,
    *,
    temperature_c: float = _TEMPERATURE_C,
) -> float:
    """Gravimetric uptake U = m_w / (m_s + m_p) [g/g] from Eq. 5."""
    rh = float(relative_humidity)
    sl = float(salt_to_polymer_ratio)
    if sl <= 0.0 or rh <= 0.0:
        return 0.0

    salt = get_salt("LiCl")
    f_b = licl_equilibrium_brine_salt_fraction(rh, temperature_c)
    if not np.isfinite(f_b) or f_b <= 0.0:
        return float("nan")

    mw_w_g_mol = WATER_MOLAR_MASS_KG_MOL * 1000.0
    mw_s_g_mol = salt.formula_weight_g_mol
    ions = salt.ions_per_formula

    mass_water = 1.0 - f_b
    mass_salt = f_b
    n_w = mass_water / mw_w_g_mol
    n_s = mass_salt / mw_s_g_mol
    x_w = n_w / (n_w + ions * n_s)
    if x_w >= 1.0:
        return float("nan")

    u_salt = (x_w * ions) / (1.0 - x_w) * (mw_w_g_mol / mw_s_g_mol)
    polymer_factor = sl / (1.0 + sl)
    return float(polymer_factor * u_salt)


def _uptake_curve(salt_to_polymer_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    uptake = np.array(
        [
            diaz_marin_licl_uptake_g_g(rh, salt_to_polymer_ratio)
            for rh in _RH_GRID
        ],
        dtype=float,
    )
    return _RH_GRID * 100.0, uptake


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
        s=24,
        marker="o",
        facecolors="white",
        edgecolors=color,
        linewidths=1.1,
        zorder=6,
        label=label,
    )
    return True


# ---------------------------------------------------------------------------
# Panel styling
# ---------------------------------------------------------------------------

# Salt-content series colours (panel B)
_COL_SL0 = "#4a4a4a"
_COL_SL1 = "#1f77b4"
_COL_SL4 = "#d62728"
_COL_SL8 = "#2ca02c"

_COL_PAM = "#d62728"
_COL_PVA = "#1f77b4"
_COL_XL = "#9467bd"


def _style_uptake_axes(ax: plt.Axes, *, panel_title: str) -> None:
    ax.set_xlabel("relative humidity [%]", fontsize=10)
    ax.set_ylabel("uptake [g/g]", fontsize=10)
    ax.set_xlim(0, 95)
    ax.set_ylim(-0.2, 10.0)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.legend(fontsize=7.0, frameon=False, loc="upper left")
    ax.set_title(panel_title, loc="left", fontweight="bold", fontsize=10)


def _plot_model_series(
    ax: plt.Axes,
    salt_to_polymer_ratio: float,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    rh_pct, uptake = _uptake_curve(salt_to_polymer_ratio)
    ax.plot(
        rh_pct,
        uptake,
        color=color,
        linewidth=1.8,
        linestyle=linestyle,
        label=f"{label} (model)",
    )


def plot_panel_b(ax: plt.Axes) -> None:
    series = [
        (0.0, "3b_0.csv", _COL_SL0, "PAM (0 g/g)"),
        (1.0, "3b_1.csv", _COL_SL1, "PAM--LiCl (1 g/g)"),
        (4.0, "3b_4.csv", _COL_SL4, "PAM--LiCl (4 g/g)"),
        (8.0, "3b_8.csv", _COL_SL8, "PAM--LiCl (8 g/g)"),
    ]
    ref_shown = False
    for sl, csv_name, color, label in series:
        if sl > 0.0:
            _plot_model_series(ax, sl, color=color, label=label)
        ref_label = _REF_LABEL if not ref_shown else None
        if _overlay_ref(ax, csv_name, color=color, label=ref_label):
            ref_shown = True
    _style_uptake_axes(ax, panel_title="B  salt content")


def plot_panel_c(ax: plt.Axes) -> None:
    """PAM and PVA at 4 g/g — Eq. 5 is independent of polymer chemistry."""
    _plot_model_series(ax, 4.0, color=_COL_PAM, label="PAM--LiCl (4 g/g)")
    _plot_model_series(
        ax,
        4.0,
        color=_COL_PVA,
        label="PVA--LiCl (4 g/g)",
        linestyle="--",
    )
    ref_shown = False
    for color in (_COL_PAM, _COL_PVA):
        ref_label = _REF_LABEL if not ref_shown else None
        if _overlay_ref(ax, "3c.csv", color=color, label=ref_label):
            ref_shown = True
    _style_uptake_axes(ax, panel_title="C  polymer chemistry")


def plot_panel_e(ax: plt.Axes) -> None:
    """Crosslinking density — Eq. 5 is independent of crosslinker content."""
    _plot_model_series(ax, 4.0, color=_COL_XL, label="PAM--LiCl (4 g/g)")
    _overlay_ref(ax, "3e.csv", color=_COL_XL, label=_REF_LABEL)
    _style_uptake_axes(ax, panel_title="E  crosslinking density")


def plot_figure3() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    plot_panel_b(axes[0])
    plot_panel_c(axes[1])
    plot_panel_e(axes[2])

    fig.suptitle(
        "Díaz-Marín et al. (2024) Figure 3 — equilibrium uptake isotherms\n"
        r"(solid = Eq. 5 with LiCl brine isotherm; open circles = digitized paper data)",
        fontsize=9,
        y=1.03,
    )
    fig.tight_layout()

    out_path = _OUT_DIR / "figure3.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    for panel_name, plot_fn in (
        ("figure3b.png", plot_panel_b),
        ("figure3c.png", plot_panel_c),
        ("figure3e.png", plot_panel_e),
    ):
        fig_s, ax_s = plt.subplots(figsize=(5.0, 4.2))
        plot_fn(ax_s)
        out_s = _OUT_DIR / panel_name
        fig_s.savefig(out_s, dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        print(f"Saved → {out_s}")

    return out_path


def main() -> Path:
    print("Díaz-Marín Figure 3 — uptake isotherm model (Eq. 5)")
    print("=" * 52)
    print(f"  LiCl isotherm: licl_equilibrium_brine_salt_fraction @ {_TEMPERATURE_C} °C")
    print(f"  Reference data: {_REF_DIR}")
    return plot_figure3()


if __name__ == "__main__":
    main()
