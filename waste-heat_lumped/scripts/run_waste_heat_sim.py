#!/usr/bin/env python3
"""Run fluid-heated daily-cycle SAWH simulation and LCOW estimate."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from waste_heat_lumped.economics.lcow import (
    LcowCostBreakdown,
    lcow_cost_breakdown_from_daily_yield,
    lcow_from_daily_yield,
)
from waste_heat_lumped.economics.params import LCOEconomicParams
from waste_heat_lumped.physics import device_defaults as dd
from waste_heat_lumped.physics.sorbent import inventory_label
from waste_heat_lumped.simulation.device_config import DeviceConfig
from waste_heat_lumped.simulation.detailed_plots import (
    detailed_series,
    plot_detailed_diagnostics,
    write_detailed_csv,
)
from waste_heat_lumped.simulation.ode_system import PhaseResult, run_daily_cycle
from waste_heat_lumped.simulation.water_inventory import (
    plot_water_inventory,
    water_inventory_series,
    write_water_inventory_csv,
)
from waste_heat_lumped.weather.profiles import (
    DEFAULT_DESORPTION_START_HOUR,
    DailyWeatherProfile,
    datacenter_baseline_profile,
    monthly_mean_day_profiles,
    representative_mean_day_profile,
)


def register_waste_heat_sim_arguments(p: argparse.ArgumentParser) -> None:
    """CLI arguments shared by ``run_waste_heat_sim.py`` and parameter-sweep scripts."""
    p.add_argument(
        "--profile",
        default="datacenter-baseline",
        choices=["datacenter-baseline", "real"],
    )
    p.add_argument("--lat", type=float, default=None, help="Required for --profile real")
    p.add_argument("--lon", type=float, default=None, help="Required for --profile real")
    p.add_argument("--year", type=int, default=2024, help="Weather year for --profile real")
    p.add_argument(
        "--cache-dir",
        type=str,
        default=str(_REPO / ".weather_cache"),
        help="Open-Meteo response cache dir for --profile real",
    )
    p.add_argument(
        "--desorption-start-hour",
        type=int,
        default=DEFAULT_DESORPTION_START_HOUR,
        help="Clock hour (0-23) the loop fluid starts desorption for --profile real "
        "(default 8: desorption 08:00-20:00, absorption overnight)",
    )
    p.add_argument(
        "--resolution",
        choices=["annual", "monthly"],
        default="annual",
        help="--profile real only: one annual mean-day profile (default) or "
        "one mean-day profile per calendar month, day-weighted",
    )
    p.add_argument("--salt", default=dd.DEFAULT_SALT_NAME)
    p.add_argument("--salt-loading", type=float, default=dd.SALT_TO_POLYMER_RATIO)
    p.add_argument("--hydrogel-thickness-mm", type=float, default=dd.H0_M * 1e3)
    p.add_argument("--g-conv", type=float, default=dd.G_CHAMBER_M_S)
    p.add_argument("--t-f-c", type=float, default=dd.T_F_C, help="Loop fluid setpoint (°C)")
    p.add_argument(
        "--m-dot-f", type=float, default=dd.M_DOT_F_KG_S_M2, help="Loop flow (kg/s/m²)"
    )
    p.add_argument("--ua-gel", type=float, default=dd.UA_GEL_W_K, help="Loop→gel UA (W/K/m²)")
    p.add_argument("--t-amb-c", type=float, default=dd.T_AMB_C)
    p.add_argument("--rh", type=float, default=dd.RH_AMB)
    p.add_argument("--h-amb", type=float, default=dd.H_AMB_W_M2_K)


def resolve_waste_heat_sim_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Validate lat/lon for --profile real (mutates nothing; raises via parser.error)."""
    if args.profile == "real" and (args.lat is None or args.lon is None):
        parser.error("--profile real requires --lat and --lon")
    if args.resolution == "monthly" and (
        getattr(args, "detailed", False) or getattr(args, "plot_water_inventory", False)
    ):
        parser.error(
            "--resolution monthly reports a day-weighted annual mean only; "
            "--detailed/--plot-water-inventory need a single day (--resolution annual)"
        )


def register_cyclic_warmup_arguments(p: argparse.ArgumentParser) -> None:
    """CLI flags for optional warmup cycles before the reporting day."""
    p.add_argument("--cyclic-initial", action="store_true", help="Warm up to cyclic state")
    p.add_argument("--warmup-cycles", type=int, default=2)


def _build_config(args: argparse.Namespace) -> DeviceConfig:
    return DeviceConfig(
        salt_name=args.salt,
        salt_to_polymer_ratio=args.salt_loading,
        hydrogel_thickness_m=args.hydrogel_thickness_mm * 1e-3,
        g_conv_m_s=args.g_conv,
        t_f_c=args.t_f_c,
        m_dot_f_kg_s_m2=args.m_dot_f,
        ua_gel_w_k=args.ua_gel,
    )


def _build_profile(
    args: argparse.Namespace,
    profile_kwargs: dict[str, float] | None = None,
) -> DailyWeatherProfile:
    if args.profile == "real":
        if args.lat is None or args.lon is None:
            raise ValueError("--profile real requires --lat and --lon")
        return representative_mean_day_profile(
            args.lat,
            args.lon,
            args.year,
            cache_dir=args.cache_dir,
            desorption_start_hour=args.desorption_start_hour,
        )
    kwargs: dict[str, float] = {
        "t_amb_c": args.t_amb_c,
        "rh": args.rh,
        "h_amb": args.h_amb,
    }
    if profile_kwargs:
        kwargs.update(profile_kwargs)
    if args.profile == "datacenter-baseline":
        return datacenter_baseline_profile(**kwargs)
    raise ValueError(f"Unknown profile: {args.profile}")


def _lcow_kwargs(config: DeviceConfig) -> dict[str, str]:
    return {"sorbent": config.sorbent}


@dataclass(frozen=True, slots=True)
class WasteHeatSimResult:
    config: DeviceConfig
    econ: LCOEconomicParams
    profile: DailyWeatherProfile
    daily_yield_kg_per_m2: float
    thermal_efficiency: float
    lcow_usd_per_m3: float
    breakdown: LcowCostBreakdown | None
    abs_res: PhaseResult
    des_res: PhaseResult
    monthly_breakdown: list[tuple[int, float, float, int]] | None = None


def _day_weighted_yield_eta(
    profiles: list[tuple[int, DailyWeatherProfile, int]],
    config: DeviceConfig,
    *,
    cyclic_initial: bool,
    warmup_cycles: int,
) -> tuple[float, float, PhaseResult, PhaseResult, DailyWeatherProfile, list[tuple[int, float, float, int]]]:
    """Day-weighted mean yield/eta across monthly profiles (solar_lumped combo_yield_kg_m2 pattern)."""
    per_month: list[tuple[int, float, float, int]] = []
    weights: list[int] = []
    yields: list[float] = []
    etas: list[float] = []
    last_abs_res: PhaseResult | None = None
    last_des_res: PhaseResult | None = None
    last_profile: DailyWeatherProfile | None = None
    for month, profile, n_days in profiles:
        yield_kg, eta, abs_res, des_res = run_daily_cycle(
            profile,
            config,
            cyclic_initial=cyclic_initial,
            cyclic_warmup_cycles=warmup_cycles,
        )
        yields.append(yield_kg)
        etas.append(eta)
        weights.append(n_days)
        per_month.append((month, yield_kg, eta, n_days))
        last_abs_res, last_des_res, last_profile = abs_res, des_res, profile
    total_w = sum(weights)
    mean_yield = sum(y * w for y, w in zip(yields, weights)) / total_w
    mean_eta = sum(e * w for e, w in zip(etas, weights)) / total_w
    assert last_abs_res is not None and last_des_res is not None and last_profile is not None
    return mean_yield, mean_eta, last_abs_res, last_des_res, last_profile, per_month


def run_waste_heat_simulation(
    args: argparse.Namespace,
    *,
    econ: LCOEconomicParams | None = None,
    config_overrides: dict[str, object] | None = None,
    profile_kwargs: dict[str, float] | None = None,
) -> WasteHeatSimResult:
    """Run one SAWH daily cycle and LCOW (same logic as ``main()``)."""
    config = _build_config(args)
    if config_overrides:
        config = replace(config, **config_overrides)
    econ = econ or LCOEconomicParams()

    monthly_breakdown: list[tuple[int, float, float, int]] | None = None
    if args.profile == "real" and getattr(args, "resolution", "annual") == "monthly":
        if args.lat is None or args.lon is None:
            raise ValueError("--profile real requires --lat and --lon")
        profiles = monthly_mean_day_profiles(
            args.lat,
            args.lon,
            args.year,
            cache_dir=args.cache_dir,
            desorption_start_hour=args.desorption_start_hour,
        )
        yield_kg, eta, abs_res, des_res, profile, monthly_breakdown = _day_weighted_yield_eta(
            profiles,
            config,
            cyclic_initial=getattr(args, "cyclic_initial", False),
            warmup_cycles=getattr(args, "warmup_cycles", 2),
        )
    else:
        profile = _build_profile(args, profile_kwargs)
        yield_kg, eta, abs_res, des_res = run_daily_cycle(
            profile,
            config,
            cyclic_initial=getattr(args, "cyclic_initial", False),
            cyclic_warmup_cycles=getattr(args, "warmup_cycles", 2),
        )

    lcow_kw = _lcow_kwargs(config)
    lcow = lcow_from_daily_yield(
        yield_kg,
        salt_name=config.salt_name,
        salt_to_polymer_ratio=config.salt_to_polymer_ratio,
        hydrogel_thickness_m=config.hydrogel_thickness_m,
        econ=econ,
        **lcow_kw,
    )
    breakdown = lcow_cost_breakdown_from_daily_yield(
        yield_kg,
        salt_name=config.salt_name,
        salt_to_polymer_ratio=config.salt_to_polymer_ratio,
        hydrogel_thickness_m=config.hydrogel_thickness_m,
        econ=econ,
        **lcow_kw,
    )

    return WasteHeatSimResult(
        config=config,
        econ=econ,
        profile=profile,
        daily_yield_kg_per_m2=float(yield_kg),
        thermal_efficiency=float(eta),
        lcow_usd_per_m3=float(lcow),
        breakdown=breakdown,
        abs_res=abs_res,
        des_res=des_res,
        monthly_breakdown=monthly_breakdown,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fluid-heated daily-cycle SAWH simulation")
    register_waste_heat_sim_arguments(parser)
    register_cyclic_warmup_arguments(parser)
    parser.add_argument("--plot-water-inventory", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=_REPO / "outputs" / "water_inventory")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Write CSV and plot device temperatures (gel, condenser, fluid) "
        "and weather variables over the full daily cycle",
    )
    parser.add_argument(
        "--detailed-csv",
        type=Path,
        default=None,
        help="Detailed diagnostics CSV path (default: outputs/detailed/diagnostics_<tag>.csv)",
    )
    parser.add_argument(
        "--detailed-plot",
        type=Path,
        default=None,
        help="Detailed diagnostics PNG path (default: outputs/detailed/diagnostics_<tag>.png)",
    )
    args = parser.parse_args()
    resolve_waste_heat_sim_arguments(args, parser)

    result = run_waste_heat_simulation(args)
    config = result.config
    profile = result.profile
    yield_kg = result.daily_yield_kg_per_m2
    eta = result.thermal_efficiency
    lcow = result.lcow_usd_per_m3
    abs_res = result.abs_res
    des_res = result.des_res

    if result.monthly_breakdown is not None:
        print(f"{'Month':>5}  {'Yield (kg/m²)':>14}  {'Eta':>6}  {'Days':>5}")
        for month, m_yield, m_eta, n_days in result.monthly_breakdown:
            print(f"{month:>5}  {m_yield:>14.4f}  {m_eta:>6.4f}  {n_days:>5}")
        print(f"Day-weighted mean yield: {yield_kg:.4f} kg/m²")
        print(f"Day-weighted mean thermal efficiency: {eta:.4f}")
    else:
        print(f"Daily yield: {yield_kg:.4f} kg/m²")
        print(f"Thermal efficiency: {eta:.4f}")
    print(f"LCOW: {lcow:.2f} USD/m³")

    tag = args.profile.replace("-", "_")
    if args.resolution == "monthly":
        tag += "_monthly"
    if args.profile == "real":
        tag += f"_lat{args.lat:.4f}_lon{args.lon:.4f}_{args.year}"

    if args.plot_water_inventory:
        series = water_inventory_series(abs_res, des_res, config=config)
        out_dir = args.output_dir
        csv_path = out_dir / f"water_in_gel_{tag}.csv"
        png_path = out_dir / f"water_in_gel_{tag}.png"
        write_water_inventory_csv(csv_path, series)
        plot_water_inventory(
            png_path,
            series,
            config=config,
            title=f"{inventory_label(config)} — {args.profile}",
        )
        print(f"Wrote {csv_path}")
        print(f"Wrote {png_path}")

    if args.detailed:
        detailed = detailed_series(profile, abs_res, des_res, config)
        detailed_csv = args.detailed_csv
        if detailed_csv is None:
            detailed_csv = _REPO / "outputs" / "detailed" / f"diagnostics_{tag}.csv"
        detailed_plot = args.detailed_plot
        if detailed_plot is None:
            detailed_plot = _REPO / "outputs" / "detailed" / f"diagnostics_{tag}.png"

        write_detailed_csv(detailed_csv, detailed)
        plot_title = f"Device and weather — {args.profile}"
        plot_detailed_diagnostics(detailed_plot, detailed, title=plot_title)
        print(f"Wrote {detailed_csv}")
        print(f"Wrote {detailed_plot}")


if __name__ == "__main__":
    main()
