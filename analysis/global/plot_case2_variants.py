"""Compare the three Case 2 sweep variants on matched (site, geometry) points.

  python analysis/global/plot_case2_variants.py

Left panel: ECDF of each variant's yield ratio against the base run. Because both
variants are physically one-sided -- a wetter c_w floor can only lose yield, an
ambient-pinned condenser can only gain it -- the curve's height where it crosses
ratio = 1.0 reads directly as the fraction of points the solver got wrong. That is
the point of plotting ECDFs rather than histograms here.

Right panel: the DRH penalty against site humidity, one point per site, which is
the mechanism -- dry sites drive the gel to its floor, humid sites never reach it.

Input is outputs/case2_variants_matched.csv (gitignored, ~4 MB; built on Sherlock,
see gpu_sweep/SHERLOCK_GPU_RUNBOOK.md). Site humidity is joined from the committed
best-per-site reduction, which is the only place it survives the matched merge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
MATCHED_CSV = _REPO / "solar_lumped" / "outputs" / "case2_variants_matched.csv"
SITES_CSV = _REPO / "analysis" / "global" / "case2_variants_best_per_site.csv"
OUTPUT = _REPO / "analysis" / "global" / "case2_variants_comparison.png"

# Categorical slots 1-2 (validated against the light chart surface). Colour follows
# the variant, not its rank, so drh stays blue in both panels.
C_DRH = "#2a78d6"
C_AMB = "#eb6834"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"

VARIANTS = (
    ("drh", C_DRH, "DRH floor", "can only lose yield"),
    ("ambient_cond", C_AMB, "Ambient condenser", "can only gain yield"),
)


def main() -> int:
    if not MATCHED_CSV.is_file():
        sys.exit(f"missing {MATCHED_CSV.relative_to(_REPO)} -- scp it from Sherlock first")
    m = pd.read_csv(MATCHED_CSV)
    ratio = pd.DataFrame({v: m[v] / m["base"] for v, *_ in VARIANTS})

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.26)

    # --- Left: paired-ratio ECDF, with the physically impossible side shaded.
    # Each variant's shaded band is the side it physically cannot land on, so anything
    # the curve puts there is solver error. Labelled, because an unexplained shade is
    # just decoration.
    ax_l.axvspan(1.0, 1.6, color=C_DRH, alpha=0.05, lw=0)
    ax_l.axvspan(0.4, 1.0, color=C_AMB, alpha=0.05, lw=0)
    ax_l.axvline(1.0, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for x, ha, colour, text in (
        (0.99, "right", C_AMB, "impossible for\nambient condenser"),
        (1.01, "left", C_DRH, "impossible for\nDRH floor"),
    ):
        ax_l.text(x, 0.985, text, ha=ha, va="top", fontsize=8.5, color=colour, alpha=0.85)

    for name, colour, label, _ in VARIANTS:
        r = np.sort(ratio[name].to_numpy())
        f = np.arange(1, len(r) + 1) / len(r)
        ax_l.plot(r, f, color=colour, lw=2.0, zorder=3, label=label)
        # Wrong-direction fraction: for drh that is P(r>1), for ambient P(r<1).
        bad = float((r > 1.0).mean()) if name == "drh" else float((r < 1.0).mean())
        y = 1.0 - bad if name == "drh" else bad
        ax_l.plot([1.0], [y], "o", ms=8, color=colour, mec=SURFACE, mew=2.0, zorder=4)
        # Each label sits on its own variant's impossible side, next to its marker.
        ax_l.annotate(
            f"{bad:.0%} wrong-signed",
            xy=(1.0, y), xytext=(8, -16) if name == "drh" else (-8, 10),
            textcoords="offset points", color=colour, fontsize=9,
            ha="left" if name == "drh" else "right",
        )

    ax_l.set_xlim(0.45, 1.35)
    ax_l.set_ylim(0, 1)
    ax_l.set_xlabel("Annual mean yield ÷ base run", color=INK_2)
    ax_l.set_ylabel("Fraction of matched points", color=INK_2)
    ax_l.set_title(
        f"Paired yield ratio, {len(m):,} matched (site × geometry) points",
        color=INK, fontsize=11, loc="left", pad=10,
    )
    leg = ax_l.legend(frameon=False, loc="upper left", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)

    # --- Right: the mechanism -- penalty tracks site humidity.
    sites = pd.read_csv(SITES_CSV).query("variant == 'base'")[["lat", "lon", "mean_rh_frac"]]
    per_site = (
        m.assign(penalty=1.0 - m["drh"] / m["base"])
        .groupby(["lat", "lon"], as_index=False)["penalty"]
        .median()
        .merge(sites, on=["lat", "lon"])
    )
    r_pearson = float(np.corrcoef(per_site["mean_rh_frac"], per_site["penalty"])[0, 1])
    ax_r.axhline(0.0, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax_r.scatter(
        per_site["mean_rh_frac"] * 100, per_site["penalty"] * 100,
        s=26, color=C_DRH, alpha=0.55, lw=0.5, edgecolor=SURFACE, zorder=3,
    )
    ax_r.set_xlabel("Site annual mean relative humidity (%)", color=INK_2)
    ax_r.set_ylabel("Median yield penalty from DRH floor (%)", color=INK_2)
    ax_r.set_title(
        f"Only dry sites reach the floor  (r = {r_pearson:+.2f}, n = {len(per_site)} sites)",
        color=INK, fontsize=11, loc="left", pad=10,
    )

    for ax in (ax_l, ax_r):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=INK_MUTED, alpha=0.18, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED)
        ax.tick_params(colors=INK_2, labelsize=9)

    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUTPUT.relative_to(_REPO)}")
    print(ratio.describe(percentiles=[0.05, 0.5, 0.95]).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
