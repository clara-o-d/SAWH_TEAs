#!/usr/bin/env python3
"""Plot year-long daily summary metrics from run_annual_simulation.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from solar_lumped.plotting import plot_defaults_slides, print_figure


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True, help="Daily summary CSV path")
    p.add_argument("--output", type=Path, default=None, help="Output PNG path")
    p.add_argument("--start-date", type=str, default=None, help="Filter start (YYYY-MM-DD)")
    p.add_argument("--end-date", type=str, default=None, help="Filter end (YYYY-MM-DD)")
    return p.parse_args(argv)


def load_summary(
    path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    if df.empty:
        raise ValueError("No rows remain after date filtering.")
    return df.sort_values("date").reset_index(drop=True)


def _plot_avg_peak(ax, dates, avg, peak, *, color: str, ylabel: str, avg_label: str, peak_label: str) -> None:
    """Shared avg-line (solid) + peak-line (dashed, faded) panel styling."""
    ax.plot(dates, avg, color=color, linewidth=1.6, label=avg_label)
    ax.plot(dates, peak, color=color, linewidth=1.2, linestyle="--", alpha=0.7, label=peak_label)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_annual_summary(df: pd.DataFrame, *, title: str | None = None) -> plt.Figure:
    plot_defaults_slides()
    dates = df["date"]

    fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
    ax_yield, ax_water, ax_temp, ax_rh, ax_solar, ax_amb = axes

    ax_yield.plot(dates, df["daily_yield_l_m2"], color="#1b9e77", linewidth=1.6)
    ax_yield.set_ylabel("Daily output\n(L/m²)")
    ax_yield.grid(True, alpha=0.3)

    ax_water.plot(
        dates,
        df["water_uptake_l_m2"],
        color="#7570b3",
        linewidth=1.6,
        label="Uptake",
    )
    ax_water.plot(
        dates,
        df["water_release_l_m2"],
        color="#d95f02",
        linewidth=1.6,
        label="Release",
    )
    ax_water.set_ylabel("Water (L/m²)")
    ax_water.legend(loc="upper right", fontsize=9)
    ax_water.grid(True, alpha=0.3)

    ax_temp.plot(dates, df["t_abs_peak_c"], linewidth=1.4, label="Absorber")
    ax_temp.plot(dates, df["t_glass_peak_c"], linewidth=1.4, label="Glass")
    ax_temp.plot(dates, df["t_cond_peak_c"], linewidth=1.4, label="Condenser")
    ax_temp.plot(dates, df["t_gel_peak_c"], linewidth=1.4, linestyle="--", label="Gel")
    ax_temp.set_ylabel("Peak temp (°C)")
    ax_temp.legend(loc="upper right", fontsize=8, ncol=2)
    ax_temp.grid(True, alpha=0.3)

    _plot_avg_peak(
        ax_rh, dates, df["rh_avg_frac"] * 100.0, df["rh_peak_frac"] * 100.0,
        color="#1b9e77", ylabel="RH (%)", avg_label="Avg RH", peak_label="Peak RH",
    )
    ax_rh.set_ylim(0.0, 100.0)

    _plot_avg_peak(
        ax_solar, dates, df["solar_avg_w_m2"], df["solar_peak_w_m2"],
        color="#e6ab02", ylabel="Solar (W/m²)", avg_label="Avg solar", peak_label="Peak solar",
    )

    _plot_avg_peak(
        ax_amb, dates, df["temp_avg_c"], df["temp_peak_c"],
        color="#d95f02", ylabel="Ambient (°C)", avg_label="Avg temp", peak_label="Peak temp",
    )

    ax_amb.xaxis.set_major_locator(mdates.MonthLocator())
    ax_amb.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_amb.set_xlabel("Month")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_summary(args.csv, start_date=args.start_date, end_date=args.end_date)
    output = args.output or (args.csv.parent / "annual_timeseries.png")

    start = df["date"].iloc[0].date()
    end = df["date"].iloc[-1].date()
    title = f"Annual simulation ({start} to {end})"
    fig = plot_annual_summary(df, title=title)
    print_figure(fig, output)
    plt.close(fig)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
