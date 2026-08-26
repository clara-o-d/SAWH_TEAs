#!/usr/bin/env python3
"""One location, one day: every model part's temperature and the sorbent water uptake
on a single figure, with a shared time axis and the absorption/desorption split marked.

Temperatures (absorber, glass, condenser, gel, ambient) share the left axis; water in the
hydrogel is on the right axis, so the uptake/release swing lines up with the thermal swing
that drives it.

Examples::

  python plot_single_day.py --lat 42.36 --lon -71.09 --year 2024
  python plot_single_day.py --weather-mode baseline
  python plot_single_day.py --lat -23.65 --lon -70.40 --hydrogel-thickness-mm 3 --csv out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from solar_lumped.system import (  # noqa: E402
    register_cyclic_warmup_arguments,
    register_solar_sim_arguments,
    resolve_solar_sim_arguments,
    run_solar_simulation,
)
from solar_lumped.simulation import (  # noqa: E402
    detailed_series,
    plot_detailed_diagnostics,
    water_inventory_series,
    write_detailed_csv,
)

_TEMP_SERIES = (
    ("t_abs_c", "Absorber", "#D55E00"),
    ("t_glass_c", "Glass", "#0072B2"),
    ("t_cond_c", "Condenser", "#009E73"),
    ("t_gel_c", "Gel", "#CC79A7"),
    ("t_amb_c", "Ambient", "0.45"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    register_solar_sim_arguments(ap)
    register_cyclic_warmup_arguments(ap)
    ap.set_defaults(weather_mode="real")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None, help="Also write the plotted series")
    ap.add_argument("--title", default=None)
    # The same run's full diagnostic set -- RH and T_amb on their own panel, solar, and the
    # c_r / a_w / driving-force / DRH(T_gel) stack. Already built by
    # simulation.plot_detailed_diagnostics; this just routes this run's series into it rather
    # than crowding them onto the temperature figure.
    ap.add_argument(
        "--diagnostics-output", type=Path, default=None,
        help="Also write the detailed diagnostics figure (temps+driving force, RH/T_amb, solar).",
    )
    args = ap.parse_args()

    resolve_solar_sim_arguments(args, ap)
    result = run_solar_simulation(args)
    abs_res, des_res = result.inventory_abs_res, result.inventory_des_res
    if abs_res is None or des_res is None:
        sys.exit("Simulation produced no phase results.")

    series = detailed_series(result.profile, abs_res, des_res, result.config)
    water = water_inventory_series(abs_res, des_res, config=result.config)

    if args.diagnostics_output is not None:
        plot_detailed_diagnostics(
            args.diagnostics_output, series, title=args.title,
            show_solar=False, grid=False,
        )
        write_detailed_csv(args.diagnostics_output.with_suffix(".csv"), series)
        print(f"Wrote {args.diagnostics_output}")
        print(f"Wrote {args.diagnostics_output.with_suffix('.csv')}")

    t_hr = series.time_s / 3600.0
    w_hr = water.time_s / 3600.0

    fig, ax = plt.subplots(figsize=(11, 6))
    for attr, label, color in _TEMP_SERIES:
        ax.plot(t_hr, getattr(series, attr), color=color, linewidth=1.7, label=label)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(False)

    ax_w = ax.twinx()
    ax_w.grid(False)   # twin axes draw their own grid; the parent's grid(False) misses it
    ax_w.plot(w_hr, water.water_l_m2, color="#000000", linewidth=2.4, linestyle="--",
              label="Water in gel")
    ax_w.set_ylabel("Water in gel (L/m²)")
    ax_w.set_ylim(bottom=0.0)

    # Mark where absorption hands over to desorption.
    split_hr = series.absorption_end_s / 3600.0
    ax.axvline(split_hr, color="0.3", linewidth=1.0, linestyle=":")
    ax.annotate("absorption", (split_hr, 1.0), xytext=(-6, -12), textcoords="offset points",
                xycoords=("data", "axes fraction"), ha="right", fontsize=9, color="0.35")
    ax.annotate("desorption", (split_hr, 1.0), xytext=(6, -12), textcoords="offset points",
                xycoords=("data", "axes fraction"), ha="left", fontsize=9, color="0.35")

    handles, labels = ax.get_legend_handles_labels()
    hw, lw = ax_w.get_legend_handles_labels()
    ax.legend(handles + hw, labels + lw, loc="upper left", fontsize=9, ncol=3, framealpha=0.9)
    ax.set_title(args.title or _title(result))
    fig.tight_layout()

    out = args.output or Path(f"single_day_{_tag(result)}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if args.csv:
        _write_csv(args.csv, series, water)
        print(f"Wrote {args.csv}")
    print(f"Daily yield: {result.daily_yield_kg_per_m2:.4f} kg/m²/d   "
          f"η_thermal: {result.thermal_efficiency:.3f}")
    print(f"Wrote {out}")
    return 0


def _title(result) -> str:
    parts = [f"Solar SAWH — {result.weather_mode}"]
    if result.lat is not None and result.lon is not None:
        parts.append(f"({result.lat:.2f}°, {result.lon:.2f}°)")
    if result.weather_mode == "real":
        parts.append(str(result.year))
    parts.append(f"{result.daily_yield_kg_per_m2:.2f} kg/m²/d")
    return "  ".join(parts)


def _tag(result) -> str:
    if result.lat is None or result.lon is None:
        return result.weather_mode
    return f"{result.weather_mode}_{result.lat:.2f}_{result.lon:.2f}"


def _write_csv(path: Path, series, water) -> None:
    import csv

    # Water is solved on its own time grid; resample it onto the temperature grid.
    w_on_t = np.interp(series.time_s, water.time_s, water.water_l_m2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "phase", *(a for a, _, _ in _TEMP_SERIES), "water_in_gel_l_m2",
                    "relative_humidity", "solar_w_m2"])
        for k in range(len(series.time_s)):
            w.writerow([f"{series.time_s[k] / 3600.0:.5f}", series.phase[k],
                        *(f"{getattr(series, a)[k]:.4f}" for a, _, _ in _TEMP_SERIES),
                        f"{w_on_t[k]:.6f}",
                        f"{series.relative_humidity[k]:.4f}", f"{series.solar_w_m2[k]:.2f}"])


if __name__ == "__main__":
    raise SystemExit(main())
