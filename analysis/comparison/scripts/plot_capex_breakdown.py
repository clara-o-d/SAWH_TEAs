#!/usr/bin/env python3
"""CAPEX (BOM) breakdown across all four SAWH configs -- the headline cost-gap finding.

Passive's CAPEX (~$52/m^2) is 2-3 orders of magnitude below the active
configs' (~$840-9,265/m^2); this script makes that gap an explicit,
standalone, plotted/tabulated finding rather than something buried in a
combined-metric chart.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from comparison.lib.adapters import ALL_CONFIG_IDS, get_adapters  # noqa: E402

_DEFAULT_OUT_CSV = _REPO_ROOT / "comparison" / "outputs" / "capex" / "capex_breakdown.csv"
_DEFAULT_OUT_PNG = _REPO_ROOT / "comparison" / "outputs" / "capex" / "capex_breakdown_bars.png"
_DEFAULT_OUT_RATIO_CSV = _REPO_ROOT / "comparison" / "outputs" / "capex" / "capex_ratios.csv"

_ACTIVE_CONFIGS = ("single_loop", "multi_loop", "multi_noloop")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--configs", nargs="+", default=list(ALL_CONFIG_IDS), choices=list(ALL_CONFIG_IDS)
    )
    p.add_argument("--out-csv", type=Path, default=_DEFAULT_OUT_CSV)
    p.add_argument("--out-png", type=Path, default=_DEFAULT_OUT_PNG)
    p.add_argument("--out-ratio-csv", type=Path, default=_DEFAULT_OUT_RATIO_CSV)
    return p.parse_args()


def build_long_rows(config_ids: list[str]) -> list[dict]:
    adapters = get_adapters(config_ids)
    rows: list[dict] = []
    for config_id, adapter in adapters.items():
        bom = adapter.bom_line_items()
        total = sum(cost for _, cost in bom)
        for label, cost in bom:
            rows.append(
                {
                    "config_id": config_id,
                    "display_name": adapter.display_name,
                    "line_item": label,
                    "cost_usd_per_m2": cost,
                    "pct_of_total": (100.0 * cost / total) if total > 0 else float("nan"),
                    "config_total_usd_per_m2": total,
                }
            )
    return rows


def write_long_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id",
        "display_name",
        "line_item",
        "cost_usd_per_m2",
        "pct_of_total",
        "config_total_usd_per_m2",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_ratio_csv(config_ids: list[str], rows: list[dict], out_path: Path) -> None:
    totals = {}
    for r in rows:
        totals[r["config_id"]] = r["config_total_usd_per_m2"]

    passive_capex = totals.get("passive")
    active_totals = [totals[c] for c in _ACTIVE_CONFIGS if c in totals]
    cheapest_active = min(active_totals) if active_totals else float("nan")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "config_id",
                "capex_usd_per_m2",
                "ratio_to_passive",
                "ratio_to_cheapest_active",
            ],
        )
        w.writeheader()
        for cid in config_ids:
            capex = totals.get(cid, float("nan"))
            ratio_to_passive = (
                capex / passive_capex if passive_capex and passive_capex > 0 else float("nan")
            )
            ratio_to_cheapest = (
                capex / cheapest_active
                if cheapest_active and cheapest_active > 0
                else float("nan")
            )
            w.writerow(
                {
                    "config_id": cid,
                    "capex_usd_per_m2": capex,
                    "ratio_to_passive": ratio_to_passive,
                    "ratio_to_cheapest_active": ratio_to_cheapest,
                }
            )
    return totals


def plot_capex_bars(config_ids: list[str], rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: linear scale, the 3 active configs only (similar order of magnitude).
    active_ids = [c for c in config_ids if c in _ACTIVE_CONFIGS]
    _stacked_bar_panel(ax_lin, active_ids, rows, log=False)
    ax_lin.set_title("Active configs (linear scale)")
    ax_lin.set_ylabel("CAPEX (USD/m$^2$)")

    # Right panel: log scale, all configs (so passive's tiny bar stays visible).
    _stacked_bar_panel(ax_log, config_ids, rows, log=True)
    ax_log.set_title("All configs (log scale)")
    ax_log.set_ylabel("CAPEX (USD/m$^2$, log scale)")

    fig.suptitle("Device BOM / CAPEX breakdown by config", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _stacked_bar_panel(ax, config_ids: list[str], rows: list[dict], *, log: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from comparison.lib.adapters import get_adapter

    by_config: dict[str, list[tuple[str, float]]] = {c: [] for c in config_ids}
    for r in rows:
        if r["config_id"] in by_config:
            by_config[r["config_id"]].append((r["line_item"], r["cost_usd_per_m2"]))

    x = np.arange(len(config_ids))
    all_labels: list[str] = []
    for cid in config_ids:
        for label, _ in by_config[cid]:
            if label not in all_labels:
                all_labels.append(label)

    cmap = plt.get_cmap("tab20")
    label_colors = {label: cmap(i % 20) for i, label in enumerate(all_labels)}

    bottoms = np.zeros(len(config_ids))
    if log:
        # log-scale stacked bars: plot each segment as its own bar starting at
        # the running bottom, using a floor value so zero-height / near-zero
        # segments don't break the log axis.
        floor = 1e-3
        bottoms = np.full(len(config_ids), floor)
        totals = np.zeros(len(config_ids))
        for i, cid in enumerate(config_ids):
            for label, cost in by_config[cid]:
                if cost <= 0:
                    continue
                ax.bar(
                    x[i],
                    cost,
                    bottom=bottoms[i],
                    color=label_colors[label],
                    edgecolor="white",
                    linewidth=0.3,
                    width=0.6,
                )
                bottoms[i] += cost
                totals[i] += cost
        ax.set_yscale("log")
        for i, cid in enumerate(config_ids):
            ax.text(
                x[i],
                totals[i] * 1.15,
                f"${totals[i]:,.0f}/m$^2$",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_ylim(bottom=floor)
    else:
        for i, cid in enumerate(config_ids):
            for label, cost in by_config[cid]:
                ax.bar(
                    x[i],
                    cost,
                    bottom=bottoms[i],
                    color=label_colors[label],
                    edgecolor="white",
                    linewidth=0.3,
                    width=0.6,
                )
                bottoms[i] += cost
        for i, cid in enumerate(config_ids):
            ax.text(
                x[i],
                bottoms[i] * 1.02,
                f"${bottoms[i]:,.0f}/m$^2$",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([get_adapter(cid).display_name for cid in config_ids], rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.3)


def main() -> int:
    args = parse_args()
    rows = build_long_rows(args.configs)
    write_long_csv(rows, args.out_csv)
    print(f"Wrote {args.out_csv}")

    totals = write_ratio_csv(args.configs, rows, args.out_ratio_csv)
    print(f"Wrote {args.out_ratio_csv}")

    plot_capex_bars(args.configs, rows, args.out_png)
    print(f"Wrote {args.out_png}")

    print("\nCAPEX totals (USD/m^2):")
    for cid in args.configs:
        print(f"  {cid:15s} {totals.get(cid, float('nan')):>10,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
