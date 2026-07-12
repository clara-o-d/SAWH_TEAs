#!/usr/bin/env python3
"""
Export model curves as CSV for MATLAB recreation plotting scripts.

MATLAB scripts live in:  waste_heat_lumped/matlab/recreation/

Usage (from waste_heat_lumped/):
  python scripts/export_recreation_matlab_data.py
  python scripts/export_recreation_matlab_data.py --figures diaz5 wilson3

Díaz-Marín Figure 3 does not need export (computed in MATLAB).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
_SOLAR_ROOT = _SCRIPT.parent.parent
_SRC = _SOLAR_ROOT / "src"
_SCRIPTS = _SOLAR_ROOT / "scripts"
_WILSON_DIR = _SOLAR_ROOT / "wilson-et-al._re-creation"
_DIAZ_DIR = _SOLAR_ROOT / "diaz-marin-et-al._re-creation"
for _p in (_SRC, _SCRIPTS, _SOLAR_ROOT, _DIAZ_DIR / "scripts", _WILSON_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _save_xy(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([x, y]), delimiter=",", fmt="%.8g")


def export_diaz5() -> Path:
    import importlib.util

    fig5_path = _DIAZ_DIR / "scripts" / "figure5.py"
    spec = importlib.util.spec_from_file_location("diaz_figure5", fig5_path)
    fig5 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = fig5
    spec.loader.exec_module(fig5)

    from chamber_rh_schedule import load_chamber_rh_schedules

    out_dir = _DIAZ_DIR / "outputs" / "matlab" / "figure5"
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules = load_chamber_rh_schedules()

    for case in fig5._PANELS:
        for rh_pct in (30, 50, 70):
            rh = rh_pct / 100.0
            t_min, uptake = fig5.simulate_rh_cycle(case, rh, schedules=schedules)
            _save_xy(out_dir / f"{case.key}_{rh_pct}.csv", t_min, uptake)
            print(f"  {case.key}_{rh_pct}.csv")

    print(f"Exported Díaz-Marín Figure 5 → {out_dir}")
    return out_dir


def export_wilson3() -> Path:
    import importlib.util

    fig3_path = _WILSON_DIR / "scripts" / "figure3.py"
    spec = importlib.util.spec_from_file_location("wilson_figure3", fig3_path)
    fig3 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = fig3
    spec.loader.exec_module(fig3)

    out_dir = _WILSON_DIR / "outputs" / "matlab" / "figure3"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_grid_hr, solar_grid, temp_grid = fig3._load_cambridge_weather()
    weather = np.column_stack([t_grid_hr, solar_grid, temp_grid])
    np.savetxt(
        out_dir / "weather.csv",
        weather,
        delimiter=",",
        fmt="%.8g",
        header="time_hr,solar_W_m2,amb_T_C",
        comments="",
    )

    for label, h_amb in (
        ("7.5", fig3._H_AMB_LO),
        ("10.0", fig3._H_AMB_MID),
        ("12.5", fig3._H_AMB_HI),
    ):
        res = fig3.run_simulation(solar_grid, temp_grid, h_amb=h_amb)
        arr = np.column_stack(
            [
                res["time_hr"],
                res["t_abs"],
                res["t_glass"],
                res["t_cond"],
                res["t_amb"],
                res["cum_water_ml_m2"],
            ]
        )
        path = out_dir / f"h_amb_{label}.csv"
        np.savetxt(
            path,
            arr,
            delimiter=",",
            fmt="%.8g",
            header="time_hr,t_abs,t_glass,t_cond,t_amb,cum_water_ml_m2",
            comments="",
        )
        print(f"  {path.name}")

    print(f"Exported Wilson Figure 3 → {out_dir}")
    return out_dir


def export_wilson4() -> Path:
    import importlib.util

    fig4_path = _WILSON_DIR / "scripts" / "figure4.py"
    spec = importlib.util.spec_from_file_location("wilson_figure4", fig4_path)
    fig4 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = fig4
    spec.loader.exec_module(fig4)

    out_dir = _WILSON_DIR / "outputs" / "matlab" / "figure4"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = fig4.simulate_atacama()
    arr = np.column_stack(
        [
            data["time_h"],
            data["t_abs"],
            data["t_glass"],
            data["t_cond"],
            data["t_amb"],
            data["cum_water_l_m2"],
        ]
    )
    np.savetxt(
        out_dir / "model.csv",
        arr,
        delimiter=",",
        fmt="%.8g",
        header="time_h,t_abs,t_glass,t_cond,t_amb,cum_water_l_m2",
        comments="",
    )
    eta_pct = data["eta"] * 100.0
    (out_dir / "meta.txt").write_text(f"eta={eta_pct:.2f}%\n", encoding="utf-8")
    print(f"Exported Wilson Figure 4 → {out_dir}")
    return out_dir


def export_wilson2() -> Path:
    data_path = _WILSON_DIR / "outputs" / "figure2" / "figure2_data.pkl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}\n"
            "Run: python wilson-et-al._re-creation/scripts/figure2_generate.py"
        )

    with data_path.open("rb") as fh:
        payload = pickle.load(fh)

    out_dir = _WILSON_DIR / "outputs" / "matlab" / "figure2"
    out_dir.mkdir(parents=True, exist_ok=True)

    for eps, (x, _lo, mid, _hi) in payload["B"].items():
        _save_xy(out_dir / f"2b_eps_{eps:.2f}.csv", x, mid)

    for (ar, has_glass), (x, y) in payload["C"].items():
        tag = "glass" if has_glass else "noglass"
        _save_xy(out_dir / f"2c_ar_{ar}_{tag}.csv", x, y)

    for t_k, (x, _lo, mid, _hi) in payload["D"].items():
        _save_xy(out_dir / f"2d_T_{t_k}.csv", x, mid)

    for lg, (x, _lo, mid, _hi) in payload["E"].items():
        _save_xy(out_dir / f"2e_Lg_{lg}.csv", x, mid)

    for h0, (x, _lo, mid, _hi) in payload["F_yield"].items():
        _save_xy(out_dir / f"2f_prod_{h0}.csv", x, mid)
    for h0, (x, _lo, mid, _hi) in payload["F_eta"].items():
        _save_xy(out_dir / f"2f_eff_{h0}.csv", x, mid)

    print(f"Exported Wilson Figure 2 → {out_dir}")
    return out_dir


_EXPORTERS = {
    "diaz5": export_diaz5,
    "wilson2": export_wilson2,
    "wilson3": export_wilson3,
    "wilson4": export_wilson4,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model CSVs for MATLAB recreation plots")
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=list(_EXPORTERS),
        default=list(_EXPORTERS),
        help="Which figure datasets to export (default: all)",
    )
    args = parser.parse_args()

    print("Export recreation data for MATLAB")
    print("=" * 40)
    for key in args.figures:
        print(f"\n[{key}]")
        _EXPORTERS[key]()


if __name__ == "__main__":
    main()
