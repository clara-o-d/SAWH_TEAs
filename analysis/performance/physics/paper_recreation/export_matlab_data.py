#!/usr/bin/env python3
"""Export model curves as CSV for the MATLAB recreation plots in matlab/recreation/.

Usage (from analysis/paper_recreation/):
  python export_recreation_matlab_data.py [--figures diaz5 wilson3]

Díaz-Marín Figure 3 does not need export (computed in MATLAB).
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np

_PAPER_RECREATION = Path(__file__).resolve().parent
_SOLAR_ROOT = _PAPER_RECREATION.parent.parent / "solar_lumped"
_WILSON_DIR = _PAPER_RECREATION / "wilson"
_DIAZ_DIR = _PAPER_RECREATION / "diaz_marin"
# The figure modules below rely on solar_lumped being importable before they load.
for _p in (_SOLAR_ROOT / "src", _SOLAR_ROOT,
           _DIAZ_DIR / "scripts", _WILSON_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model CSVs for MATLAB recreation plots")
    parser.add_argument(
        "--figures", nargs="+", choices=list(_EXPORTERS), default=list(_EXPORTERS),
        help="Which figure datasets to export (default: all)",
    )
    args = parser.parse_args()

    print("Export recreation data for MATLAB")
    print("=" * 40)
    for key in args.figures:
        print(f"\n[{key}]")
        _EXPORTERS[key]()


def export_diaz5() -> Path:
    fig5 = _load_module("diaz_figure5", _DIAZ_DIR / "scripts" / "figure5.py")
    from chamber_rh_schedule import load_chamber_rh_schedules

    out_dir = _DIAZ_DIR / "outputs" / "matlab" / "figure5"
    schedules = load_chamber_rh_schedules()
    for case in fig5._PANELS:
        for rh_pct in (30, 50, 70):
            t_min, uptake = fig5.simulate_rh_cycle(case, rh_pct / 100.0, schedules=schedules)
            _save_csv(out_dir / f"{case.key}_{rh_pct}.csv", np.column_stack([t_min, uptake]))
            print(f"  {case.key}_{rh_pct}.csv")

    print(f"Exported Díaz-Marín Figure 5 → {out_dir}")
    return out_dir


def export_wilson2() -> Path:
    data_path = _WILSON_DIR / "outputs" / "figure2" / "figure2_data.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}\nRun: python wilson/scripts/figure2_generate.py")
    with data_path.open("rb") as fh:
        payload = pickle.load(fh)

    out_dir = _WILSON_DIR / "outputs" / "matlab" / "figure2"
    for panel, name in (("B", "2b_eps_{:.2f}"), ("D", "2d_T_{}"), ("E", "2e_Lg_{}"),
                        ("F_yield", "2f_prod_{}"), ("F_eta", "2f_eff_{}")):
        for key, (x, _lo, mid, _hi) in payload[panel].items():
            _save_csv(out_dir / f"{name.format(key)}.csv", np.column_stack([x, mid]))
    for (ar, has_glass), (x, y) in payload["C"].items():
        tag = "glass" if has_glass else "noglass"
        _save_csv(out_dir / f"2c_ar_{ar}_{tag}.csv", np.column_stack([x, y]))

    print(f"Exported Wilson Figure 2 → {out_dir}")
    return out_dir


def export_wilson3() -> Path:
    fig3 = _load_module("wilson_figure3", _WILSON_DIR / "scripts" / "figure3.py")
    out_dir = _WILSON_DIR / "outputs" / "matlab" / "figure3"

    t_grid_hr, solar_grid, temp_grid = fig3._load_cambridge_weather()
    _save_csv(
        out_dir / "weather.csv", np.column_stack([t_grid_hr, solar_grid, temp_grid]),
        header="time_hr,solar_W_m2,amb_T_C",
    )
    for label, h_amb in (("7.5", fig3._H_AMB_LO), ("10.0", fig3._H_AMB_MID), ("12.5", fig3._H_AMB_HI)):
        res = fig3.run_simulation(solar_grid, temp_grid, h_amb=h_amb)
        _save_csv(
            out_dir / f"h_amb_{label}.csv",
            np.column_stack([res[k] for k in
                             ("time_hr", "t_abs", "t_glass", "t_cond", "t_amb", "cum_water_ml_m2")]),
            header="time_hr,t_abs,t_glass,t_cond,t_amb,cum_water_ml_m2",
        )
        print(f"  h_amb_{label}.csv")

    print(f"Exported Wilson Figure 3 → {out_dir}")
    return out_dir


def export_wilson4() -> Path:
    fig4 = _load_module("wilson_figure4", _WILSON_DIR / "scripts" / "figure4.py")
    out_dir = _WILSON_DIR / "outputs" / "matlab" / "figure4"

    data = fig4.simulate_atacama()
    _save_csv(
        out_dir / "model.csv",
        np.column_stack([data[k] for k in
                         ("time_h", "t_abs", "t_glass", "t_cond", "t_amb", "cum_water_l_m2")]),
        header="time_h,t_abs,t_glass,t_cond,t_amb,cum_water_l_m2",
    )
    (out_dir / "meta.txt").write_text(f"eta={data['eta'] * 100.0:.2f}%\n", encoding="utf-8")
    print(f"Exported Wilson Figure 4 → {out_dir}")
    return out_dir


def _load_module(name: str, path: Path):
    """Import a figure script by file path (they are scripts, not importable modules)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _save_csv(path: Path, arr: np.ndarray, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, delimiter=",", fmt="%.8g", header=header, comments="")


_EXPORTERS = {
    "diaz5": export_diaz5,
    "wilson2": export_wilson2,
    "wilson3": export_wilson3,
    "wilson4": export_wilson4,
}


if __name__ == "__main__":
    main()
