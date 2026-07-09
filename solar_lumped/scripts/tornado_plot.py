#!/usr/bin/env python3
"""One-at-a-time tornado sensitivity plot from parameter sweep CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DEFAULT_INPUT = _REPO / "parameter_sweeps" / "parameter_sweep.csv"
_DEFAULT_OUTPUT = _REPO / "tornado_plots" / "tornado_plot.png"

_EXCLUDED_PARAMS = frozenset({"humidity_high", "relative_humidity"})
_FAIL_LCO_THRESHOLD = 1e20
_BAR_COLOR = "#20A387"
_METRIC_LABEL = "LCOW (USD/m³)"

_METRIC_COLUMNS = frozenset({
    "daily_yield_kg_m2",
    "thermal_efficiency",
    "lcow_usd_per_m3",
    "capex_usd_per_m3",
    "opex_usd_per_m3",
})

_PARAM_LABELS: dict[str, str] = {
    "h_des_j_per_kg": "h_des\n(J/kg)",
    "salt_formula_weight_g_mol": "Salt MW\n(g/mol)",
    "hydrogel_lifetime_years": "Hydrogel lifetime\n(yr)",
    "hydrogel_thickness_mm": "Hydrogel thickness\n(mm)",
    "vapor_gap_mm": "Vapor gap\n(mm)",
    "humidity_high": "Uptake RH",
    "solar_irradiance_w_per_m2": "Solar GHI\n(W/m²)",
    "h_amb_w_m2_k": "h_amb\n(W/m²K)",
    "discount_rate": "Discount rate",
    "device_lifetime_years": "Device lifetime\n(yr)",
    "utilization_factor": "Utilization\nfactor",
}


def calculate_sensitivity(x1: float, y1: float, x2: float, y2: float) -> tuple[float, bool]:
    """Percent sensitivity between two (x, y) points."""
    if abs(x1) < 1e-10:
        pct_change_x = abs(x2 - x1)
    else:
        pct_change_x = (x2 - x1) / x1 * 100

    if abs(y1) < 1e-10:
        pct_change_y = y2 - y1
    else:
        pct_change_y = (y2 - y1) / y1 * 100

    if abs(pct_change_x) > 0.01:
        sensitivity = pct_change_y / pct_change_x
        if abs(sensitivity) < 1000:
            return sensitivity, True

    return 0.0, False


def _input_params(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in _METRIC_COLUMNS and c not in _EXCLUDED_PARAMS
    ]


def load_sweep(path: Path, metric: str) -> pd.DataFrame:
    """Load and clean parameter-sweep CSV."""
    df = pd.read_csv(path)
    if metric not in df.columns:
        sys.exit(f"Metric column {metric!r} not found in {path}")

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    df = df[df[metric] < _FAIL_LCO_THRESHOLD]
    return df


def oat_sensitivity(
    df: pd.DataFrame,
    target_col: str,
    input_params: list[str] | None = None,
) -> pd.DataFrame:
    """One-at-a-time sensitivity of target output to listed inputs."""
    if input_params is None:
        input_params = _input_params(df)

    print(f"Analyzing sensitivity of {target_col!r} to {len(input_params)} input parameters:")
    sensitivity_data: list[dict] = []

    for var in input_params:
        if var not in df.columns:
            print(f"  Warning: Parameter {var!r} not found in data columns, skipping")
            continue

        valid_mask = df[var].notna() & df[target_col].notna()
        valid_data = df.loc[valid_mask]
        if valid_data.empty:
            print(f"  Warning: No valid data for {var}, skipping")
            continue

        x_mean = float(valid_data[var].mean())
        y_mean = float(valid_data[target_col].mean())
        other_params = [p for p in input_params if p != var and p in df.columns]

        oat_sensitivities_increase: list[float] = []
        oat_sensitivities_decrease: list[float] = []

        for i, row1 in valid_data.iterrows():
            for j, row2 in valid_data.iterrows():
                if i >= j:
                    continue

                is_oat_pair = True
                for other_param in other_params:
                    val1, val2 = row1[other_param], row2[other_param]
                    if abs(val1 - val2) > 1e-8:
                        is_oat_pair = False
                        break

                if not is_oat_pair:
                    continue

                x1, y1 = row1[var], row1[target_col]
                x2, y2 = row2[var], row2[target_col]

                if x1 < x2:
                    x_low, y_low = x1, y1
                    x_high, y_high = x2, y2
                else:
                    x_low, y_low = x2, y2
                    x_high, y_high = x1, y1

                sens_inc, valid_inc = calculate_sensitivity(x_low, y_low, x_high, y_high)
                if valid_inc:
                    oat_sensitivities_increase.append(sens_inc)

                sens_dec, valid_dec = calculate_sensitivity(x_high, y_high, x_low, y_low)
                if valid_dec:
                    oat_sensitivities_decrease.append(sens_dec)

        avg_increase_sensitivity = (
            float(np.mean(oat_sensitivities_increase)) if oat_sensitivities_increase else 0.0
        )
        avg_decrease_sensitivity = (
            float(np.mean(oat_sensitivities_decrease)) if oat_sensitivities_decrease else 0.0
        )

        all_oat_sensitivities = oat_sensitivities_increase + oat_sensitivities_decrease
        if not all_oat_sensitivities:
            print(f"  Warning: No valid OAT pairs found for {var}")
            continue

        avg_sensitivity = float(np.mean(all_oat_sensitivities))
        abs_avg_sensitivity = float(np.mean([abs(s) for s in all_oat_sensitivities]))

        sensitivity_data.append({
            "variable": var,
            "avg_sensitivity": avg_sensitivity,
            "abs_avg_sensitivity": abs_avg_sensitivity,
            "avg_increase_sensitivity": avg_increase_sensitivity,
            "avg_decrease_sensitivity": avg_decrease_sensitivity,
            "num_increase_points": len(oat_sensitivities_increase),
            "num_decrease_points": len(oat_sensitivities_decrease),
            "num_point_sensitivities": len(all_oat_sensitivities),
            "x_min": float(valid_data[var].min()),
            "x_median": float(valid_data[var].median()),
            "x_max": float(valid_data[var].max()),
            "y_min": float(valid_data[target_col].min()),
            "y_median": float(valid_data[target_col].median()),
            "y_max": float(valid_data[target_col].max()),
            "x_mean": x_mean,
            "y_mean": y_mean,
            "valid_points": len(valid_data),
        })

        print(
            f"  {var}: increase sensitivity={avg_increase_sensitivity:.3f} "
            f"(n={len(oat_sensitivities_increase)} OAT pairs), "
            f"decrease sensitivity={avg_decrease_sensitivity:.3f} "
            f"(n={len(oat_sensitivities_decrease)} OAT pairs)"
        )

    sensitivity_df = pd.DataFrame(sensitivity_data)
    if sensitivity_df.empty:
        return sensitivity_df

    sensitivity_df["max_abs_effect"] = sensitivity_df[
        ["avg_increase_sensitivity", "avg_decrease_sensitivity"]
    ].abs().max(axis=1)
    return sensitivity_df.sort_values(by="max_abs_effect", ascending=False)


def create_tornado_plot(
    sensitivity_df: pd.DataFrame,
    target_col: str,
    title: str | None = None,
    param_name_mapping: dict[str, str] | None = None,
) -> tuple[plt.Figure | None, plt.Axes | None]:
    """Create tornado plot for sensitivity analysis results."""
    if title is None:
        title = "Parameter sensitivity"

    if sensitivity_df.empty:
        print("No valid sensitivity data to plot")
        return None, None

    plot_df = sensitivity_df.copy()
    plot_df["variable"] = plot_df["variable"].astype(str).str.lstrip("# ").str.strip()
    plot_df = plot_df.sort_values(by="max_abs_effect", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_frame_on(False)

    y_pos = np.arange(len(plot_df))

    if param_name_mapping is not None:
        plot_df["display_name"] = (
            plot_df["variable"].map(param_name_mapping).fillna(plot_df["variable"])
        )
    else:
        plot_df["display_name"] = plot_df["variable"]

    bar_width = 0.35
    increase_hatch: list[str] = []
    decrease_hatch: list[str] = []

    for _, row in plot_df.iterrows():
        if row["avg_increase_sensitivity"] >= 0:
            increase_hatch.append("")
            decrease_hatch.append("")
        else:
            increase_hatch.append("///")
            decrease_hatch.append("///")

    ax.barh(
        y_pos,
        plot_df["avg_increase_sensitivity"],
        height=bar_width,
        color=_BAR_COLOR,
        alpha=0.7,
        hatch=increase_hatch,
        label="Parameter increase effect",
    )
    ax.barh(
        y_pos,
        -plot_df["avg_decrease_sensitivity"],
        height=bar_width,
        color=_BAR_COLOR,
        alpha=0.7,
        hatch=decrease_hatch,
        label="Parameter decrease effect",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["display_name"], fontsize=16)
    ax.set_title(title, fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel(
        f"% change in {_METRIC_LABEL}\nper % change in parameter",
        fontsize=18,
    )
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(axis="x", labelsize=16)

    ax.legend(
        handles=[
            Patch(facecolor=_BAR_COLOR, alpha=0.7, label="Positive"),
            Patch(facecolor=_BAR_COLOR, alpha=0.7, hatch="///", label="Negative"),
        ],
        loc="lower right",
        fontsize=16,
    )
    plt.tight_layout()
    return fig, ax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    ap.add_argument("--table-csv", type=Path, default=None)
    ap.add_argument("--metric", default="lcow_usd_per_m3")
    ap.add_argument("--show", action="store_true", help="Display plot interactively")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"Missing {args.input}; run parameter_sweep.py first.")

    print(f"Loading data from {args.input!r}...")
    df = load_sweep(args.input, args.metric)
    print(f"Successfully loaded {len(df)} data points with {len(df.columns)} variables.")

    input_params = _input_params(df)
    for var in input_params:
        if var in df.columns:
            print(f"OK: {var!r} found in data columns.")
        else:
            print(f"WARNING: {var!r} not found in data columns.")

    total_points = len(df)
    valid_points = df[args.metric].notna().sum()
    print("\nData Quality Summary:")
    print(f"  Total data points: {total_points}")
    print(f"  Valid {args.metric} values: {valid_points}")
    print(f"  Success rate: {valid_points / total_points * 100:.1f}%")

    if valid_points < 3:
        sys.exit("Very few valid data points; cannot proceed with sensitivity analysis.")

    print(f"\nCreating tornado plot for: {args.metric}")
    sensitivity_df = oat_sensitivity(df, args.metric, input_params=input_params)

    if sensitivity_df.empty:
        sys.exit("No valid sensitivity data calculated.")

    table_csv = args.table_csv or args.output.with_suffix(".table.csv")
    sensitivity_df.to_csv(table_csv, index=False)
    print(f"Sensitivity analysis results saved to {table_csv!r}")

    fig, _ = create_tornado_plot(
        sensitivity_df,
        args.metric,
        title="Parameter sensitivity",
        param_name_mapping=_PARAM_LABELS,
    )
    if fig is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Tornado plot saved as {args.output!r}")
        if args.show:
            plt.show()
        plt.close(fig)

    print("\nSensitivity Analysis Summary:")
    print(f"  Parameters analyzed: {len(sensitivity_df)}")
    for _, row in sensitivity_df.iterrows():
        print(
            f"  {row['variable']}: {row['valid_points']} valid points, "
            f"{row['num_point_sensitivities']} OAT pairs"
        )
        print(
            f"    Parameter increase effect: {row['avg_increase_sensitivity']:.3f} "
            f"(n={row['num_increase_points']} OAT pairs)"
        )
        print(
            f"    Parameter decrease effect: {row['avg_decrease_sensitivity']:.3f} "
            f"(n={row['num_decrease_points']} OAT pairs)"
        )
        print(f"    Maximum absolute effect: {row['max_abs_effect']:.3f}")

    print("\nTrue OAT sensitivity analysis completed!")


if __name__ == "__main__":
    main()
