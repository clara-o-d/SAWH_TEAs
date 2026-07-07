#!/usr/bin/env python3
"""
Plot Wilson et al. (2025) Figure 2 from saved sweep data + digitized reference.

Expects model data at:  wilson-et-al._re-creation/outputs/figure2/figure2_data.pkl
Reference CSVs at:      wilson-et-al._re-creation/reference/figure2/
Output figure:          wilson-et-al._re-creation/outputs/figure2/figure2.png

Generate data first with:
  python wilson-et-al._re-creation/scripts/figure2_generate.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).resolve()
_WILSON_DIR = _SCRIPT.parent.parent          # wilson-et-al._re-creation/
_SOLAR_ROOT = _WILSON_DIR.parent             # solar_lumped/
_SRC = _SOLAR_ROOT / "src"
for _p in (_SRC, _SOLAR_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_OUT_DIR = _WILSON_DIR / "outputs" / "figure2"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_DATA_PATH = _OUT_DIR / "figure2_data.pkl"
_REF_DIR = _WILSON_DIR / "reference" / "figure2"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_figure2_data(path: Path = _DATA_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing sweep data: {path}\n"
            "Run:  python wilson-et-al._re-creation/scripts/figure2_generate.py"
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_TEAL_PALETTE = [
    "#1a5c5c", "#1f7a7a", "#2a9d8f", "#4db8b8", "#85d1d1"
]
_SALMON_PALETTE = [
    "#e07050", "#e09070", "#e0b090", "#e0c8a0", "#e0d8c0"
]
_MARKERS = ["+", "s", "D", "<", "*"]
_REF_LABEL = "Wilson et al. (digitized)"


def _line_colour(idx: int, n: int, dark: bool = True) -> str:
    palette = _TEAL_PALETTE if dark else _SALMON_PALETTE
    return palette[min(idx, len(palette) - 1)]


def _fig2_style(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _fill_band(ax, x, lo, hi, color, alpha=0.15):
    mask = ~(np.isnan(lo) | np.isnan(hi))
    if mask.any():
        ax.fill_between(x[mask], lo[mask], hi[mask], color=color, alpha=alpha, linewidth=0)


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
    color: str = "#111111",
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
        s=22,
        marker="o",
        facecolors="white",
        edgecolors=color,
        linewidths=1.1,
        zorder=6,
        label=label,
    )
    return True


def plot_figure2(data_B, data_C, data_D, data_E, data_F_yield, data_F_eta):
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[1, 0])
    ax_E = fig.add_subplot(gs[1, 1])
    ax_F = fig.add_subplot(gs[1, 2])

    ax_A.set_aspect("equal")
    ax_A.axis("off")
    ax_A.text(
        0.5, 0.5,
        "Device\nschematic\n(see Fig. 2A\nin paper)",
        ha="center", va="center", fontsize=8, color="gray",
        transform=ax_A.transAxes,
    )
    ax_A.set_title("A", loc="left", fontweight="bold", fontsize=9)

    # ---- Panel B ----
    eps_abs_vals = [0.2, 0.5, 0.8, 0.95, 0.99]
    n = len(eps_abs_vals)
    ref_B = False
    for i, eps in enumerate(eps_abs_vals):
        tau_vals, lows, mids, highs = data_B[eps]
        col = _line_colour(i, n, dark=(i % 2 == 1))
        mk = _MARKERS[i % len(_MARKERS)]
        ax_B.plot(tau_vals, mids, color=col, marker=mk, markersize=4,
                  linestyle="--", linewidth=1.2,
                  label=rf"$\varepsilon_{{abs}}={eps}$ (model)")
        _fill_band(ax_B, tau_vals, lows, highs, col)
        ref_B |= _overlay_ref(
            ax_B,
            f"2b_{eps}.csv",
            color=col,
            label=_REF_LABEL if not ref_B else None,
        )
    ax_B.set_xlabel(r"transmission through glass, $\tau_{glass}$", fontsize=7.5)
    ax_B.set_ylabel(r"device productivity [L/m²/day]", fontsize=7.5)
    ax_B.set_xlim(0.2, 1.0)
    ax_B.set_ylim(bottom=0)
    ax_B.legend(fontsize=6.5, loc="upper left", frameon=False)
    ax_B.set_title("B", loc="left", fontweight="bold", fontsize=9)
    _fig2_style(ax_B)

    # ---- Panel C ----
    ar_vals = [1, 2, 5, 7]
    ar_colours_glass = ["#c16a50", "#c49060", "#4a9a80", "#1a5a50"]
    ar_colours_noglass = ["#e8b0a0", "#e8c8a0", "#a0d0c0", "#70b0a8"]
    ref_C = False
    for i, ar in enumerate(ar_vals):
        h_vals, yields_g = data_C[(ar, True)]
        col_g = ar_colours_glass[i]
        ax_C.plot(h_vals, yields_g, color=col_g, marker=_MARKERS[i % len(_MARKERS)],
                  markersize=4, linestyle="--", linewidth=1.2,
                  label=rf"$A_r={ar}$ (model)")
        ref_C |= _overlay_ref(
            ax_C,
            f"2c_{ar}_glass.csv",
            color=col_g,
            label=_REF_LABEL if not ref_C else None,
        )
        h_vals, yields_ng = data_C[(ar, False)]
        col_ng = ar_colours_noglass[i]
        ax_C.plot(h_vals, yields_ng, color=col_ng, marker=_MARKERS[i % len(_MARKERS)],
                  markersize=4, linestyle=":", linewidth=1.2,
                  label=None)
        _overlay_ref(ax_C, f"2c_{ar}_no-glass.csv", color=col_ng)
    from matplotlib.lines import Line2D
    extra = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2, label="glass cover"),
        Line2D([0], [0], color="gray", linestyle=":", linewidth=1.2, label="no cover"),
    ]
    handles, labels = ax_C.get_legend_handles_labels()
    ax_C.legend(handles=handles + extra, labels=labels + ["glass cover", "no cover"],
                fontsize=6.5, loc="upper left", frameon=False, ncol=2)
    ax_C.set_xlabel(r"ambient heat transfer coefficient, $h_{amb}$ [W/m²K]", fontsize=7.5)
    ax_C.set_ylabel(r"device productivity [L/m²/day]", fontsize=7.5)
    ax_C.set_xlim(1, 10)
    ax_C.set_ylim(bottom=0)
    ax_C.set_title("C", loc="left", fontweight="bold", fontsize=9)
    _fig2_style(ax_C)

    # ---- Panel D ----
    t_amb_k_vals = [280, 290, 300, 310]
    d_colours = ["#285e6c", "#38888c", "#60b8b0", "#d08060"]
    ref_D = False
    for i, T_k in enumerate(t_amb_k_vals):
        rh_vals, lows, mids, highs = data_D[T_k]
        col = d_colours[i]
        ax_D.plot(rh_vals, mids, color=col, marker=_MARKERS[i % len(_MARKERS)],
                  markersize=4, linestyle="--", linewidth=1.2,
                  label=rf"$T_{{amb}}={T_k}$ K (model)")
        _fill_band(ax_D, rh_vals, lows, highs, col)
        ref_D |= _overlay_ref(
            ax_D,
            f"2d_{T_k}.csv",
            color=col,
            label=_REF_LABEL if not ref_D else None,
        )
    ax_D.set_xlabel(r"ambient humidity RH [ ]", fontsize=7.5)
    ax_D.set_ylabel(r"device productivity [L/m²/day]", fontsize=7.5)
    ax_D.set_xlim(0.2, 0.9)
    ax_D.set_ylim(bottom=0)
    ax_D.legend(fontsize=6.5, loc="upper left", frameon=False)
    ax_D.set_title("D", loc="left", fontweight="bold", fontsize=9)
    _fig2_style(ax_D)

    # ---- Panel E ----
    lg_vals = [18, 20, 40, 60]
    e_colours = ["#c06050", "#d09060", "#2a9070", "#1a5850"]
    ref_E = False
    for i, lg in enumerate(lg_vals):
        h0_vals, lows, mids, highs = data_E[lg]
        col = e_colours[i]
        ax_E.plot(h0_vals, mids, color=col, marker=_MARKERS[i % len(_MARKERS)],
                  markersize=4, linestyle="--", linewidth=1.2,
                  label=rf"$L_g={lg}$ mm (model)")
        _fill_band(ax_E, h0_vals, lows, highs, col)
        ref_E |= _overlay_ref(
            ax_E,
            f"2e_{lg}.csv",
            color=col,
            label=_REF_LABEL if not ref_E else None,
        )
    ax_E.set_xlabel(r"thickness of gel, $H_0$ [mm]", fontsize=7.5)
    ax_E.set_ylabel(r"device productivity [L/m²/day]", fontsize=7.5)
    ax_E.set_xlim(0, 8)
    ax_E.set_ylim(bottom=0)
    ax_E.legend(fontsize=6.5, loc="lower right", frameon=False)
    ax_E.set_title("E", loc="left", fontweight="bold", fontsize=9)
    _fig2_style(ax_E)

    # ---- Panel F ----
    h0_vals_mm = [2, 4, 8]
    f_teal = ["#1a5c5c", "#2a9d8f", "#85d1d1"]
    f_orange = ["#c86010", "#e08030", "#f0b060"]
    ax_F2 = ax_F.twinx()
    ref_F = False
    for i, h0 in enumerate(h0_vals_mm):
        q_vals_kw, y_lows, y_mids, y_highs = data_F_yield[h0]
        _, e_lows, e_mids, e_highs = data_F_eta[h0]
        col_y = f_teal[i]
        col_e = f_orange[i]
        ax_F.plot(q_vals_kw, y_mids, color=col_y, marker="o", markersize=4,
                  linestyle="--", linewidth=1.2, label=rf"$H_0={h0}$ mm (model)")
        _fill_band(ax_F, q_vals_kw, y_lows, y_highs, col_y)
        ref_F |= _overlay_ref(
            ax_F,
            f"2f_prod_{h0}.csv",
            color=col_y,
            label=_REF_LABEL if not ref_F else None,
        )
        ax_F2.plot(q_vals_kw, e_mids, color=col_e, marker="x", markersize=4,
                   linestyle="--", linewidth=1.2)
        _fill_band(ax_F2, q_vals_kw, e_lows, e_highs, col_e)
        _overlay_ref(ax_F2, f"2f_eff_{h0}.csv", color=col_e)

    ax_F.set_xlabel(r"incident solar flux, $Q_{solar}$ [kW/m²]", fontsize=7.5)
    ax_F.set_ylabel(r"device productivity [L/m²/day]", fontsize=7.5, color=f_teal[0])
    ax_F2.set_ylabel(r"thermal efficiency [%]", fontsize=7.5, color=f_orange[1])
    ax_F.set_xlim(0.5, 1.5)
    ax_F.set_ylim(bottom=0)
    ax_F2.set_ylim(bottom=0)
    ax_F.tick_params(axis="y", colors=f_teal[0])
    ax_F2.tick_params(axis="y", colors=f_orange[1])
    ax_F.legend(fontsize=6.5, loc="upper left", frameon=False)
    ax_F.set_title("F", loc="left", fontweight="bold", fontsize=9)
    _fig2_style(ax_F)
    _fig2_style(ax_F2)

    fig.suptitle(
        "Wilson et al. (2025) Figure 2 — Thermofluidic optimisation of the hydrogel SAWH device\n"
        r"(dashed = solar\_lumped model; circles = digitized paper data; bands = $h_{amb}=10\pm2.5$ W/m²K)",
        fontsize=8.5, y=1.01,
    )
    fig.tight_layout()

    out_path = _OUT_DIR / "figure2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)
    return out_path


def main(data_path: Path = _DATA_PATH) -> Path:
    payload = load_figure2_data(data_path)
    return plot_figure2(
        payload["B"],
        payload["C"],
        payload["D"],
        payload["E"],
        payload["F_yield"],
        payload["F_eta"],
    )


if __name__ == "__main__":
    main()
