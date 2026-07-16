#!/usr/bin/env python3
"""Full-factorial parameter sweep for solar lumped SAWH LCOW."""

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

from run_solar_sim import (  # noqa: E402
    SolarSimResult,
    _lcow_kwargs,
    register_cyclic_warmup_arguments,
    register_solar_sim_arguments,
    resolve_solar_sim_arguments,
    run_solar_simulation,
)
from solar_lumped.economics.lcow import LcowCostBreakdown  # noqa: E402
from solar_lumped.economics.npv import npv_from_daily_yield  # noqa: E402
from solar_lumped.economics.params import LCOEconomicParams  # noqa: E402
from solar_lumped.physics import table_s3  # noqa: E402
from solar_lumped.physics.salt_properties import get_salt  # noqa: E402

DEFAULT_SWEEP_KEYS: tuple[str, ...] = (
    "h_des_j_per_kg",
    "salt_weight_factor",
    "hydrogel_lifetime_years",
)

# Baseline water price used for NPV/payback metrics when
# "water_price_usd_per_m3" isn't part of the current sweep subset; matches
# the baseline in the SweepParam registered by make_sweep_params().
_BASELINE_WATER_PRICE_USD_PER_M3: float = 5.0


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
    result: SolarSimResult,
    water_price_usd_per_m3: float = _BASELINE_WATER_PRICE_USD_PER_M3,
) -> dict[str, float]:
    capex, opex = _capex_opex_usd_per_m3(result.breakdown)
    lcow_kw = _lcow_kwargs(result.config)
    npv_result = npv_from_daily_yield(
        result.daily_yield_kg_per_m2,
        water_price_usd_per_m3,
        salt_name=result.config.salt_name,
        salt_to_polymer_ratio=result.config.salt_to_polymer_ratio,
        hydrogel_thickness_m=result.config.hydrogel_thickness_m,
        econ=result.econ,
        cycles_per_day=1.0,
        **lcow_kw,
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
) -> SolarSimResult:
    args = copy.copy(base_args)
    baseline_profile_kwargs: dict[str, float] = {}
    econ = base_econ
    h_des_j_per_kg: float | None = None
    salt_formula_weight_g_mol: float | None = None
    salt_weight_factor: float | None = None

    for key, value in combo.items():
        if key == "hydrogel_thickness_mm":
            args.hydrogel_thickness_mm = value
        elif key == "vapor_gap_mm":
            args.vapor_gap_mm = value
        elif key == "humidity_high":
            baseline_profile_kwargs["relative_humidity"] = value
        elif key == "solar_irradiance_w_per_m2":
            baseline_profile_kwargs["solar_w_m2"] = value
        elif key == "h_amb_w_m2_k":
            baseline_profile_kwargs["h_amb_w_m2_k"] = value
        elif key == "discount_rate":
            econ = LCOEconomicParams(discount_rate=value)
        elif key == "device_lifetime_years":
            econ = LCOEconomicParams(device_lifetime_years=int(value))
        elif key == "hydrogel_lifetime_years":
            econ = LCOEconomicParams(hydrogel_lifetime_years=value)
        elif key == "utilization_factor":
            econ = LCOEconomicParams(utilization_factor=value)
        elif key == "h_des_j_per_kg":
            h_des_j_per_kg = value
        elif key == "salt_formula_weight_g_mol":
            salt_formula_weight_g_mol = value
        elif key == "salt_weight_factor":
            salt_weight_factor = value
        elif key == "water_price_usd_per_m3":
            # Pure economics input consumed directly in main()'s per-combo
            # loop (via _metrics_from_result); no physics/profile dispatch.
            pass
        elif key == "salt_to_polymer_ratio":
            args.salt_loading = value
        elif key == "insulation_gap_mm":
            args.insulation_gap_mm = value
        elif key == "tilt_deg":
            args.tilt_deg = value
        elif key == "fin_area_ratio":
            args.fin_area_ratio = value
        elif key == "temperature_c":
            baseline_profile_kwargs["temperature_c"] = value
        elif key == "total_investment_factor":
            econ = LCOEconomicParams(total_investment_factor=value)
        elif key == "maintenance_cost_fraction":
            econ = LCOEconomicParams(maintenance_cost_fraction=value)
        elif key == "electricity_price_usd_per_kwh":
            econ = LCOEconomicParams(electricity_price_usd_per_kwh=value)
        elif key == "c_acrylamide_usd_per_kg":
            econ = LCOEconomicParams(c_acrylamide_usd_per_kg=value)
        elif key == "c_additives_usd_per_kg_composite":
            econ = LCOEconomicParams(c_additives_usd_per_kg_composite=value)
        else:
            raise ValueError(f"Unknown sweep parameter: {key}")

    if baseline_profile_kwargs and args.weather_mode != "baseline":
        raise ValueError(
            "Weather-profile sweeps require --weather-mode baseline "
            f"(got {args.weather_mode!r})"
        )

    return run_solar_simulation(
        args,
        econ=econ,
        baseline_profile_kwargs=baseline_profile_kwargs or None,
        h_des_j_per_kg=h_des_j_per_kg,
        salt_formula_weight_g_mol=salt_formula_weight_g_mol,
        salt_weight_factor=salt_weight_factor,
    )


def make_sweep_params(
    base_args: argparse.Namespace,
    base_econ: LCOEconomicParams,
) -> list[SweepParam]:
    salt = get_salt(base_args.salt)
    return [
        SweepParam(
            "h_des_j_per_kg",
            "h_des (J/kg)",
            1.8e6,
            3.2e6,
            table_s3.H_DES_J_PER_KG,
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
        SweepParam(
            "hydrogel_thickness_mm",
            "Hydrogel thickness (mm)",
            1.0,
            10.0,
            base_args.hydrogel_thickness_mm,
        ),
        SweepParam("vapor_gap_mm", "Vapor gap (mm)", 7.0, 60.0, base_args.vapor_gap_mm),
        SweepParam("humidity_high", "Uptake RH", 0.15, 0.80, 0.5),
        SweepParam("solar_irradiance_w_per_m2", "Solar GHI (W/m²)", 400.0, 800.0, 600.0),
        SweepParam("h_amb_w_m2_k", "h_amb (W/m²K)", 1.0, 12.5, 10.0),
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
            _BASELINE_WATER_PRICE_USD_PER_M3,
        ),
        # --- Material ---
        SweepParam("salt_to_polymer_ratio", "Salt:polymer ratio (S/L)", 1.0, 8.0, base_args.salt_loading),
        SweepParam(
            "c_acrylamide_usd_per_kg",
            "Acrylamide price (USD/kg)",
            0.5 * base_econ.c_acrylamide_usd_per_kg,
            1.5 * base_econ.c_acrylamide_usd_per_kg,
            base_econ.c_acrylamide_usd_per_kg,
        ),
        SweepParam(
            "c_additives_usd_per_kg_composite",
            "Additives price (USD/kg composite)",
            0.5 * base_econ.c_additives_usd_per_kg_composite,
            1.5 * base_econ.c_additives_usd_per_kg_composite,
            base_econ.c_additives_usd_per_kg_composite,
        ),
        # --- Heat transfer / geometry ---
        SweepParam("insulation_gap_mm", "Insulation gap (mm)", 1.0, 20.0, base_args.insulation_gap_mm),
        SweepParam("fin_area_ratio", "Condenser fin area ratio", 3.0, 12.0, 7.0),
        SweepParam("tilt_deg", "Tilt angle (deg)", 0.0, 60.0, base_args.tilt_deg),
        # --- External / climate ---
        SweepParam("temperature_c", "Ambient temperature (°C)", 10.0, 40.0, 25.0),
        # --- Financial ---
        SweepParam(
            "total_investment_factor",
            "Total investment factor",
            0.7,
            1.5,
            base_econ.total_investment_factor,
        ),
        SweepParam(
            "maintenance_cost_fraction",
            "Maintenance cost (frac CAPEX/yr)",
            0.02,
            0.10,
            base_econ.maintenance_cost_fraction,
        ),
        SweepParam(
            "electricity_price_usd_per_kwh",
            "Electricity price (USD/kWh)",
            0.05,
            0.30,
            base_econ.electricity_price_usd_per_kwh,
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
        description="Full-factorial parameter sweep (uses run_solar_simulation)",
    )
    register_solar_sim_arguments(ap)
    register_cyclic_warmup_arguments(ap)
    ap.set_defaults(weather_mode="baseline")
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
            "Parameter keys to sweep (default: h_des, salt MW, hydrogel lifetime). "
            f"Available: {', '.join(DEFAULT_SWEEP_KEYS)} and others from make_sweep_params."
        ),
    )
    args = ap.parse_args()

    resolve_solar_sim_arguments(args, ap)
    if args.no_cyclic and args.cyclic:
        ap.error("Cannot use both --cyclic and --no-cyclic")
    if args.warmup_cycles < 0:
        ap.error("--warmup-cycles must be >= 0")
    if args.no_cyclic and args.warmup_cycles != 2:
        print("Note: --warmup-cycles ignored when --no-cyclic is set.", flush=True)

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
    use_cyclic = args.cyclic or (
        not args.no_cyclic
        and args.weather_mode in ("real", "atacama-replay", "cambridge-replay")
        and args.initial_water_l_m2 is None
    )
    n_ode_days = 1 if not use_cyclic else args.warmup_cycles + 1
    print(
        f"Full factorial: {len(params)} parameters x {args.n_levels} levels "
        f"= {n_runs} runs"
    )
    cyclic_desc = "off"
    if use_cyclic:
        cyclic_desc = f"on ({args.warmup_cycles} warmup + 1 report)"
    print(f"  cyclic={cyclic_desc}  ODE days/run={n_ode_days}", flush=True)

    param_keys = [p.key for p in params]
    rows: list[dict] = []
    for i, combo in enumerate(combos, start=1):
        result = _apply_combo(combo, base_args, base_econ)
        water_price = combo.get("water_price_usd_per_m3", _BASELINE_WATER_PRICE_USD_PER_M3)
        rows.append({**combo, **_metrics_from_result(result, water_price)})
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
