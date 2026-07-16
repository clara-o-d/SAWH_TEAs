#!/usr/bin/env python3
"""Full-factorial parameter sweep for the fluid-heated daily-cycle SAWH LCOW/NPV."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_SRC = _REPO / "src"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_waste_heat_sim import (  # noqa: E402
    WasteHeatSimResult,
    register_cyclic_warmup_arguments,
    register_waste_heat_sim_arguments,
    run_waste_heat_simulation,
)
from waste_heat_lumped.economics.lcow import LcowCostBreakdown  # noqa: E402
from waste_heat_lumped.economics.npv import npv_from_daily_yield  # noqa: E402
from waste_heat_lumped.economics.params import LCOEconomicParams  # noqa: E402
from waste_heat_lumped.physics import device_defaults as dd  # noqa: E402
from waste_heat_lumped.physics.salt_properties import get_salt  # noqa: E402

DEFAULT_WATER_PRICE_USD_PER_M3: float = 5.0

DEFAULT_SWEEP_KEYS: tuple[str, ...] = (
    "hydrogel_thickness_mm",
    "salt_weight_factor",
    "hydrogel_lifetime_years",
)


@dataclass(frozen=True, slots=True)
class SweepParam:
    key: str
    label: str
    lo: float
    hi: float
    baseline: float
    is_int: bool = False


def _sweep_grid(sp: SweepParam, n: int) -> list[float]:
    if n < 1:
        raise ValueError("n_levels must be >= 1")
    if n == 1:
        return [sp.baseline]
    vals = [sp.lo + (sp.hi - sp.lo) * i / (n - 1) for i in range(n)]
    if sp.is_int:
        return [float(int(round(v))) for v in vals]
    return vals


def _capex_opex_usd_per_m3(breakdown: LcowCostBreakdown | None) -> tuple[float, float]:
    if breakdown is None:
        return float("nan"), float("nan")
    capex = 0.0
    opex = 0.0
    for label, val in breakdown.items:
        if label.startswith("CAPEX:"):
            capex += val
        else:
            opex += val
    return capex, opex


def _metrics_from_result(
    result: WasteHeatSimResult,
    *,
    water_price_usd_per_m3: float = DEFAULT_WATER_PRICE_USD_PER_M3,
) -> dict[str, float]:
    capex, opex = _capex_opex_usd_per_m3(result.breakdown)

    npv_result = npv_from_daily_yield(
        result.daily_yield_kg_per_m2,
        water_price_usd_per_m3,
        salt_name=result.config.salt_name,
        salt_to_polymer_ratio=result.config.salt_to_polymer_ratio,
        hydrogel_thickness_m=result.config.hydrogel_thickness_m,
        econ=result.econ,
    )
    if npv_result is None:
        npv_usd_per_m2 = float("nan")
        payback_years_simple = float("nan")
        payback_years_discounted = float("nan")
    else:
        npv_usd_per_m2 = npv_result.npv_usd_per_m2
        payback_years_simple = npv_result.payback_years_simple
        payback_years_discounted = npv_result.payback_years_discounted

    return {
        "daily_yield_kg_m2": result.daily_yield_kg_per_m2,
        "thermal_efficiency": result.thermal_efficiency,
        "lcow_usd_per_m3": result.lcow_usd_per_m3,
        "capex_usd_per_m3": capex,
        "opex_usd_per_m3": opex,
        "npv_usd_per_m2": npv_usd_per_m2,
        "payback_years_simple": payback_years_simple,
        "payback_years_discounted": payback_years_discounted,
    }


def _apply_combo(
    combo: dict[str, float],
    base_args: argparse.Namespace,
    base_econ: LCOEconomicParams,
) -> tuple[WasteHeatSimResult, float]:
    """Run one sweep point; returns (result, water_price_usd_per_m3 for that point)."""
    args = copy.copy(base_args)
    config_overrides: dict[str, object] = {}
    profile_kwargs: dict[str, float] = {}
    econ_overrides: dict[str, object] = {}
    water_price_usd_per_m3 = DEFAULT_WATER_PRICE_USD_PER_M3

    for key, value in combo.items():
        if key == "hydrogel_thickness_mm":
            config_overrides["hydrogel_thickness_m"] = value * 1e-3
        elif key == "vapor_gap_mm":
            config_overrides["vapor_gap_m"] = value * 1e-3
        elif key == "salt_weight_factor":
            config_overrides["salt_weight_factor"] = value
        elif key == "t_f_c":
            config_overrides["t_f_c"] = value
        elif key == "m_dot_f_kg_s_m2":
            config_overrides["m_dot_f_kg_s_m2"] = value
        elif key == "ua_gel_w_k":
            config_overrides["ua_gel_w_k"] = value
        elif key == "t_amb_c":
            profile_kwargs["t_amb_c"] = value
        elif key == "relative_humidity":
            profile_kwargs["rh"] = value
        elif key == "h_amb_w_m2_k":
            profile_kwargs["h_amb"] = value
        elif key == "discount_rate":
            econ_overrides["discount_rate"] = value
        elif key == "device_lifetime_years":
            econ_overrides["device_lifetime_years"] = int(value)
        elif key == "hydrogel_lifetime_years":
            econ_overrides["hydrogel_lifetime_years"] = value
        elif key == "utilization_factor":
            econ_overrides["utilization_factor"] = value
        elif key == "water_price_usd_per_m3":
            water_price_usd_per_m3 = value
        else:
            raise ValueError(f"Unknown sweep parameter: {key}")

    econ = LCOEconomicParams(**econ_overrides) if econ_overrides else base_econ

    result = run_waste_heat_simulation(
        args,
        econ=econ,
        config_overrides=config_overrides or None,
        profile_kwargs=profile_kwargs or None,
    )
    return result, water_price_usd_per_m3


def make_sweep_params(
    base_args: argparse.Namespace,
    base_econ: LCOEconomicParams,
) -> list[SweepParam]:
    salt = get_salt(base_args.salt)
    return [
        # Material
        SweepParam(
            "hydrogel_thickness_mm",
            "Hydrogel thickness (mm)",
            1.0,
            10.0,
            base_args.hydrogel_thickness_mm,
        ),
        SweepParam(
            "salt_weight_factor",
            "Salt weight factor",
            35.0 / salt.formula_weight_g_mol,
            50.0 / salt.formula_weight_g_mol,
            1.0,
        ),
        SweepParam(
            "hydrogel_lifetime_years",
            "Hydrogel lifetime (yr)",
            0.5,
            2.0,
            base_econ.hydrogel_lifetime_years,
        ),
        # Heat transfer
        SweepParam("t_f_c", "Loop fluid setpoint (°C)", 40.0, 75.0, base_args.t_f_c),
        SweepParam(
            "m_dot_f_kg_s_m2", "Loop flow (kg/s/m²)", 0.05, 0.30, base_args.m_dot_f
        ),
        SweepParam("ua_gel_w_k", "Loop→gel UA (W/K/m²)", 400.0, 2000.0, base_args.ua_gel),
        SweepParam("vapor_gap_mm", "Vapor gap (mm)", 7.0, 60.0, dd.VAPOR_GAP_M * 1e3),
        # External
        SweepParam("t_amb_c", "Ambient temperature (°C)", 25.0, 40.0, base_args.t_amb_c),
        SweepParam("relative_humidity", "Ambient RH", 0.15, 0.80, base_args.rh),
        SweepParam("h_amb_w_m2_k", "h_amb (W/m²K)", 5.0, 25.0, base_args.h_amb),
        # Financial
        SweepParam("discount_rate", "Discount rate", 0.04, 0.12, base_econ.discount_rate),
        SweepParam(
            "device_lifetime_years",
            "Device lifetime (yr)",
            10,
            30,
            base_econ.device_lifetime_years,
            is_int=True,
        ),
        SweepParam(
            "utilization_factor",
            "Utilization factor",
            0.7,
            1.0,
            base_econ.utilization_factor,
        ),
        SweepParam(
            "water_price_usd_per_m3",
            "Water price (USD/m³)",
            1.0,
            25.0,
            DEFAULT_WATER_PRICE_USD_PER_M3,
        ),
    ]


def _factorial_combos(
    params: list[SweepParam],
    n_levels: int,
) -> list[dict[str, float]]:
    grids = {sp.key: _sweep_grid(sp, n_levels) for sp in params}
    keys = [sp.key for sp in params]
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grids[k] for k in keys))
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full-factorial parameter sweep (uses run_waste_heat_simulation)",
    )
    register_waste_heat_sim_arguments(ap)
    register_cyclic_warmup_arguments(ap)
    ap.add_argument(
        "--n-levels",
        "--n-points",
        dest="n_levels",
        type=int,
        default=5,
        help="Grid points per parameter (total runs = n_levels ** n_params)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO / "outputs" / "parameter_sweeps" / "parameter_sweep.csv",
    )
    ap.add_argument(
        "--params",
        nargs="*",
        default=None,
        help=(
            "Parameter keys to sweep (default: hydrogel thickness, salt weight factor, "
            f"hydrogel lifetime). Available: {', '.join(DEFAULT_SWEEP_KEYS)} and others "
            "from make_sweep_params."
        ),
    )
    args = ap.parse_args()

    if args.warmup_cycles < 0:
        ap.error("--warmup-cycles must be >= 0")

    base_args = copy.copy(args)
    base_econ = LCOEconomicParams()

    all_params = make_sweep_params(base_args, base_econ)
    sweep_keys = tuple(args.params) if args.params else DEFAULT_SWEEP_KEYS
    known = {p.key for p in all_params}
    unknown = [k for k in sweep_keys if k not in known]
    if unknown:
        ap.error(f"Unknown sweep parameter(s): {', '.join(unknown)}")
    params = [p for p in all_params if p.key in sweep_keys]

    combos = _factorial_combos(params, args.n_levels)
    n_runs = len(combos)
    print(f"Full factorial: {len(params)} parameters x {args.n_levels} levels = {n_runs} runs")

    param_keys = [p.key for p in params]
    rows: list[dict] = []
    for i, combo in enumerate(combos, start=1):
        result, water_price = _apply_combo(combo, base_args, base_econ)
        rows.append(
            {**combo, **_metrics_from_result(result, water_price_usd_per_m3=water_price)}
        )
        if i % max(1, n_runs // 10) == 0 or i == n_runs:
            print(f"  {i}/{n_runs} complete", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        *param_keys,
        "daily_yield_kg_m2",
        "thermal_efficiency",
        "lcow_usd_per_m3",
        "capex_usd_per_m3",
        "opex_usd_per_m3",
        "npv_usd_per_m2",
        "payback_years_simple",
        "payback_years_discounted",
    ]
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
