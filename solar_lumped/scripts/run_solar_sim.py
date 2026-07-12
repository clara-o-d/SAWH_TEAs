#!/usr/bin/env python3
"""Run passive Wilson et al. 2025 SAWH simulation and LCOW estimate."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from solar_lumped.economics.lcow import (
    LcowCostBreakdown,
    lcow_cost_breakdown_from_daily_yield,
    lcow_from_daily_yield,
)
from solar_lumped.economics.params import LCOEconomicParams
from solar_lumped.simulation.device_config import DeviceConfig
from solar_lumped.simulation.ode_system import run_daily_cycle
from solar_lumped.simulation.detailed_plots import (
    detailed_series,
    plot_detailed_diagnostics,
    write_detailed_csv,
)
from solar_lumped.simulation.water_inventory import (
    plot_water_inventory,
    water_inventory_series,
    write_water_inventory_csv,
)
from solar_lumped.weather.profiles import (
    baseline_initial_c_w,
    baseline_profile,
    representative_mean_day_profile,
    replay_profile,
)
from solar_lumped.physics import table_s3
from solar_lumped.physics.adsorbent import DEFAULT_MOF_NAME
from solar_lumped.physics.salt_properties import (
    DRY_COMPOSITE_DENSITY_KG_M3,
    GAS_CONSTANT_J_MOL_K,
    WATER_MOLAR_MASS_KG_MOL,
    chamber_c_s_from_synthesis,
    chamber_c_s_with_constant_density,
    equilibrium_c_w_at_rh,
    get_salt,
    salt_molarity_from_composite,
    saturation_vapor_pressure_pa,
    water_activity_from_c_w,
)
from solar_lumped.physics.mass_transfer import C_W_MAX_MOL_M3, C_W_MIN_MOL_M3
from solar_lumped.physics.sorbent import SorbentKind, inventory_label, inventory_prefix
from solar_lumped.weather.fig_s1 import c_w_from_water_in_gel_l_m2, fig_s1_initial_c_w


def _lcow_kwargs(config: DeviceConfig) -> dict:
    if config.sorbent == "mof":
        props = config.mof()
        return {
            "sorbent": "mof",
            "mof_mass_kg_m2": props.m_ads_kg_m2,
            "mof_price_usd_per_kg": props.price_usd_per_kg,
        }
    return {"sorbent": "hydrogel"}


def _initial_loading_from_water_l_m2(
    water_l_m2: float,
    config: DeviceConfig,
) -> float:
    if config.sorbent == "mof":
        props = config.mof()
        if props.m_ads_kg_m2 <= 0.0:
            raise ValueError("MOF m_ads_kg_m2 must be positive")
        return water_l_m2 / props.m_ads_kg_m2
    return c_w_from_water_in_gel_l_m2(water_l_m2, config.hydrogel_thickness_m)


def _config_overrides(config: DeviceConfig) -> dict:
    out = {
        "sorbent": config.sorbent,
        "mof_name": config.mof_name,
        "salt_name": config.salt_name,
        "salt_to_polymer_ratio": config.salt_to_polymer_ratio,
        "hydrogel_thickness_m": config.hydrogel_thickness_m,
        "vapor_gap_m": config.vapor_gap_m,
        "insulation_gap_m": config.insulation_gap_m,
    }
    if config.salt_formula_weight_g_mol is not None:
        out["salt_formula_weight_g_mol"] = config.salt_formula_weight_g_mol
    if config.salt_weight_factor != 1.0:
        out["salt_weight_factor"] = config.salt_weight_factor
    if config.thermal is not None:
        out["thermal"] = config.thermal
    return out


def _apply_physics_overrides(
    config: DeviceConfig,
    *,
    h_des_j_per_kg: float | None = None,
    salt_formula_weight_g_mol: float | None = None,
    salt_weight_factor: float | None = None,
) -> DeviceConfig:
    updates: dict[str, object] = {}
    if salt_formula_weight_g_mol is not None:
        updates["salt_formula_weight_g_mol"] = salt_formula_weight_g_mol
    if salt_weight_factor is not None:
        updates["salt_weight_factor"] = salt_weight_factor
    if h_des_j_per_kg is not None:
        updates["thermal"] = replace(
            config.thermal_params(),
            h_des_j_per_kg=h_des_j_per_kg,
        )
    if not updates:
        return config
    return replace(config, **updates)


def build_device_config(
    *,
    sorbent: SorbentKind = "hydrogel",
    mof: str = DEFAULT_MOF_NAME,
    salt: str = "LiCl",
    salt_loading: float = 4.0,
    hydrogel_thickness_mm: float = 4.0,
    vapor_gap_mm: float = 40.0,
    insulation_gap_mm: float = 5.0,
    tilt_deg: float = 30.0,
    fin_area_ratio: float = 7.1,
    g_conv_m_s: float | None = None,
) -> DeviceConfig:
    """Construct a ``DeviceConfig`` (shared by CLI and chamber-kinetics callers).

    Sorption/brine state uses ``DeviceConfig``'s default dry-basis composite density
    (DVS isotherm); gel thermal mass in the ODE uses ``table_s3.RHO_COMPOSITE_KG_M3``.
    """
    get_salt(salt)
    kwargs: dict[str, object] = {
        "sorbent": sorbent,
        "mof_name": mof,
        "salt_name": salt,
        "salt_to_polymer_ratio": salt_loading,
        "hydrogel_thickness_m": hydrogel_thickness_mm * 1e-3,
        "vapor_gap_m": vapor_gap_mm * 1e-3,
        "insulation_gap_m": insulation_gap_mm * 1e-3,
        "tilt_deg": tilt_deg,
        "fin_area_ratio": fin_area_ratio,
    }
    if g_conv_m_s is not None:
        kwargs["g_conv_m_s"] = g_conv_m_s
    return DeviceConfig(**kwargs)  # type: ignore[arg-type]


def _build_config(args: argparse.Namespace) -> DeviceConfig:
    fin_ratio = args.fin_area_ratio
    if fin_ratio is None:
        fin_ratio = 5.0 if args.weather_mode == "atacama-replay" else 7.1
    tilt = args.tilt_deg
    if args.weather_mode == "atacama-replay" and tilt == 35.0:
        tilt = 25.0
    elif args.weather_mode in ("baseline", "fig-s1-replay") and tilt == 35.0:
        tilt = 30.0
    return build_device_config(
        sorbent=args.sorbent,
        mof=args.mof,
        salt=args.salt,
        salt_loading=args.salt_loading,
        hydrogel_thickness_mm=args.hydrogel_thickness_mm,
        vapor_gap_mm=args.vapor_gap_mm,
        insulation_gap_mm=args.insulation_gap_mm,
        tilt_deg=tilt,
        fin_area_ratio=fin_ratio,
    )


@dataclass(frozen=True, slots=True)
class HydrogelChamberParams:
    """Open-chamber hydrogel kinetics (Díaz-Marín Eqs. 5 + 8); no SAWH device."""

    salt_name: str
    salt_to_polymer_ratio: float
    h0_m: float
    g_conv_m_s: float
    c_s_mol_m3: float
    ions_per_formula: int
    formula_weight_g_mol: float


def build_hydrogel_chamber_params(
    *,
    salt: str = "LiCl",
    salt_loading: float = 4.0,
    h0_mm: float,
    g_conv_m_s: float,
    dry_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3,
    use_synthesis_c_s: bool = True,
    pour_ml: float | None = None,
) -> HydrogelChamberParams:
    """Hydrogel-only chamber parameters (Eq. 5 thermodynamics + Eq. 8 kinetics)."""
    salt_rec = get_salt(salt)
    h0_m = h0_mm * 1e-3
    if salt == "LiCl" and use_synthesis_c_s:
        c_s = chamber_c_s_with_constant_density(
            salt_loading,
            h0_mm,
            formula_weight_g_mol=salt_rec.formula_weight_g_mol,
            pour_ml=pour_ml,
        )
    else:
        c_s = salt_molarity_from_composite(
            salt_loading,
            dry_density_kg_m3,
            salt_rec.formula_weight_g_mol,
        )
    return HydrogelChamberParams(
        salt_name=salt,
        salt_to_polymer_ratio=salt_loading,
        h0_m=h0_m,
        g_conv_m_s=g_conv_m_s,
        c_s_mol_m3=c_s,
        ions_per_formula=salt_rec.ions_per_formula,
        formula_weight_g_mol=salt_rec.formula_weight_g_mol,
    )


def chamber_equilibrium_c_w(
    params: HydrogelChamberParams,
    rh: float,
    *,
    temperature_c: float = 25.0,
) -> float:
    """Invert Eq. 5 brine isotherm to ``c_w`` at fixed ``H₀``."""
    return equilibrium_c_w_at_rh(
        rh,
        c_s=params.c_s_mol_m3,
        ions_per_formula=params.ions_per_formula,
        temperature_c=temperature_c,
        salt_name=params.salt_name,
        formula_weight_g_mol=params.formula_weight_g_mol,
        salt_to_polymer_ratio=params.salt_to_polymer_ratio,
        h0_ref_m=params.h0_m,
    )


def _dry_uptake_g_g(c_w: float, params: HydrogelChamberParams) -> float:
    """Eq. 5 gravimetric uptake U = m_w / (m_s + m_p) at fixed ``H₀``."""
    h = params.h0_m
    mass_water = max(0.0, c_w) * h * WATER_MOLAR_MASS_KG_MOL
    mass_salt = params.c_s_mol_m3 * h * params.formula_weight_g_mol / 1000.0
    mass_polymer = mass_salt / max(params.salt_to_polymer_ratio, 1e-9)
    dry_mass = mass_salt + mass_polymer
    if dry_mass <= 0.0:
        return 0.0
    return mass_water / dry_mass


def chamber_relative_uptake(
    c_w: float,
    params: HydrogelChamberParams,
    *,
    u_baseline: float,
) -> float:
    """Díaz-Marín Fig. 5 uptake relative to mass at the baseline RH."""
    return (_dry_uptake_g_g(c_w, params) - u_baseline) / (1.0 + u_baseline)


def chamber_dc_w_dt(
    c_w: float,
    rh: float,
    params: HydrogelChamberParams,
    *,
    temperature_c: float = 25.0,
) -> float:
    """Díaz-Marín Eq. 8 / SI Eq. S31 with brine ``a_w`` from Eq. 5 only."""
    aw = water_activity_from_c_w(
        c_w,
        c_s=params.c_s_mol_m3,
        ions_per_formula=params.ions_per_formula,
        temperature_c=temperature_c,
        salt_name=params.salt_name,
        formula_weight_g_mol=params.formula_weight_g_mol,
        salt_to_polymer_ratio=params.salt_to_polymer_ratio,
        h0_ref_m=params.h0_m,
    )
    if not np.isfinite(aw):
        return 0.0
    t_k = max(temperature_c + 273.15, 200.0)
    p_sat = saturation_vapor_pressure_pa(temperature_c)
    driving = rh - aw
    rate = (
        params.g_conv_m_s
        / params.h0_m
        * (p_sat / (GAS_CONSTANT_J_MOL_K * t_k))
        * driving
    )
    if not np.isfinite(rate):
        return 0.0
    if c_w >= C_W_MAX_MOL_M3 and rate > 0.0:
        return 0.0
    if c_w <= C_W_MIN_MOL_M3 and rate < 0.0:
        return 0.0
    return float(rate)


def simulate_isothermal_chamber_rh_cycle(
    params: HydrogelChamberParams,
    rh_high: float,
    *,
    rh_baseline: float = 0.20,
    temperature_c: float = 25.0,
    t_max_min: float = 5200.0,
    t_high_to_20_min: float | None = None,
    uptake_rate_eps: float = 2.0e-5,
    eq_hold_s: float = 600.0,
    dt_s: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Isothermal open-chamber 20 % → ``rh_high`` → 20 % RH cycle.

    Hydrogel-only Díaz-Marín model: Eq. 8 convection-limited ``dc_w/dt`` with
    brine water activity from Eq. 5; fixed ``H₀`` (no vapor gap / condenser).
    Returns ``(time_min, uptake_g_g)`` in the paper's reporting convention.

    When ``t_high_to_20_min`` is set, RH follows the experimental schedule from
    the source-data workbook: ``rh_high`` from ``t = 0`` until that time, then
    ``rh_baseline`` until ``t_max_min``. Otherwise RH advances when uptake
    equilibrates (legacy behaviour).
    """
    c_w = chamber_equilibrium_c_w(params, rh_baseline, temperature_c=temperature_c)
    u_baseline = _dry_uptake_g_g(c_w, params)

    def uptake_now(cw: float) -> float:
        return chamber_relative_uptake(cw, params, u_baseline=u_baseline)

    def uptake_target(rh: float) -> float:
        cw_eq = chamber_equilibrium_c_w(params, rh, temperature_c=temperature_c)
        return uptake_now(cw_eq)

    use_fixed_schedule = t_high_to_20_min is not None
    rh_schedule = (rh_baseline, rh_high, rh_baseline)
    phase = 0
    eq_hold = 0.0
    t_s = 0.0
    t_max_s = t_max_min * 60.0
    u_prev = uptake_now(c_w)

    times: list[float] = [0.0]
    uptakes: list[float] = [u_prev]

    while t_s < t_max_s:
        t_min = t_s / 60.0
        if use_fixed_schedule:
            rh = rh_high if t_min < t_high_to_20_min else rh_baseline
        else:
            rh = rh_schedule[phase]

        dc_w = chamber_dc_w_dt(c_w, rh, params, temperature_c=temperature_c)
        c_w = float(np.clip(c_w + dc_w * dt_s, C_W_MIN_MOL_M3, C_W_MAX_MOL_M3))

        t_s += dt_s
        u_now = uptake_now(c_w)
        times.append(t_s / 60.0)
        uptakes.append(u_now)

        if not use_fixed_schedule:
            du_dt = abs(u_now - u_prev) / dt_s
            u_prev = u_now
            u_target = uptake_target(rh)
            u_span = max(abs(u_target), 1e-6)
            near_target = abs(u_now - u_target) < 0.02 * u_span
            if du_dt < uptake_rate_eps and near_target:
                eq_hold += dt_s
            else:
                eq_hold = 0.0

            if eq_hold >= eq_hold_s and phase < len(rh_schedule) - 1:
                phase += 1
                eq_hold = 0.0

    return np.array(times), np.array(uptakes)


def _write_cost_breakdown_csv(
    path: Path,
    breakdown,
    *,
    subtitle: str,
    lat: float | None,
    lon: float | None,
    year: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bar_label",
                "segment",
                "lcow_usd_per_m3",
                "stack_order",
                "subtitle",
                "lat",
                "lon",
                "year",
            ]
        )
        for i, (seg, val) in enumerate(breakdown.items):
            w.writerow(
                [
                    "solar_lumped",
                    seg,
                    f"{val:.6f}",
                    i,
                    subtitle,
                    lat if lat is not None else "",
                    lon if lon is not None else "",
                    year if year is not None else "",
                ]
            )


def _uses_cycled_initial(weather_mode: str, *, initial_water_l_m2: float | None) -> bool:
    """Atacama / real weather start from post-cycle gel state unless overridden."""
    if initial_water_l_m2 is not None:
        return False
    return weather_mode in ("real", "atacama-replay", "cambridge-replay")


def register_cyclic_warmup_arguments(p: argparse.ArgumentParser) -> None:
    """CLI flags for optional warmup cycles before the reporting day."""
    p.add_argument(
        "--cyclic",
        action="store_true",
        help="Run warmup daily cycles before the reporting day (required for baseline/fig-s1).",
    )
    p.add_argument(
        "--no-cyclic",
        action="store_true",
        help="Skip warmup cycles; single-day ODE only (~3× faster, less accurate IC).",
    )
    p.add_argument(
        "--warmup-cycles",
        type=int,
        default=2,
        metavar="N",
        help="Warmup daily cycles before the reporting day when cyclic (default: 2).",
    )


def _cyclic_settings(
    args: argparse.Namespace,
    *,
    cyclic_initial: bool | None = None,
    cyclic_warmup_cycles: int | None = None,
) -> tuple[bool, int]:
    """Resolve warmup flags from explicit kwargs or ``args`` (if registered)."""
    n_warmup = 2 if cyclic_warmup_cycles is None else cyclic_warmup_cycles
    if cyclic_initial is not None:
        return cyclic_initial, n_warmup
    if hasattr(args, "no_cyclic"):
        n_warmup = getattr(args, "warmup_cycles", n_warmup)
        if args.no_cyclic:
            return False, n_warmup
        if getattr(args, "cyclic", False):
            return True, n_warmup
        return (
            _uses_cycled_initial(
                args.weather_mode,
                initial_water_l_m2=args.initial_water_l_m2,
            ),
            n_warmup,
        )
    return (
        _uses_cycled_initial(args.weather_mode, initial_water_l_m2=args.initial_water_l_m2),
        n_warmup,
    )


def register_solar_sim_arguments(p: argparse.ArgumentParser) -> None:
    """CLI arguments shared by ``run_solar_sim.py`` and site LCOW breakdown scripts."""
    p.add_argument(
        "--weather-mode",
        choices=("real", "baseline", "atacama-replay", "cambridge-replay", "fig-s1-replay"),
        default="baseline",
    )
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument(
        "--sorbent",
        choices=("hydrogel", "mof"),
        default="hydrogel",
        help="Sorbent material: PAM-salt hydrogel (default) or MOF (tabulated MIL-100(Fe) isotherm)",
    )
    p.add_argument(
        "--mof",
        default=DEFAULT_MOF_NAME,
        help="MOF catalog name when --sorbent mof",
    )
    p.add_argument("--salt", type=str, default="LiCl")
    p.add_argument("--salt-loading", type=float, default=4.0)
    p.add_argument("--hydrogel-thickness-mm", type=float, default=4.0)
    p.add_argument("--vapor-gap-mm", type=float, default=40.0)
    p.add_argument("--insulation-gap-mm", type=float, default=5.0)
    p.add_argument("--tilt-deg", type=float, default=35.0)
    p.add_argument(
        "--fin-area-ratio",
        type=float,
        default=None,
        help="External fin area ratio A_r (default 5 for atacama-replay, 7 otherwise)",
    )
    p.add_argument(
        "--initial-water-l-m2",
        type=float,
        default=None,
        help="Initial water in gel (L/m²); overrides RH equilibrium or Fig. S1 default",
    )


def resolve_solar_sim_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Apply weather-mode defaults and validate lat/lon (mutates ``args`` in place)."""
    if args.weather_mode == "real":
        if args.lat is None or args.lon is None:
            parser.error("--weather-mode real requires --lat and --lon")
    elif args.weather_mode not in ("baseline", "fig-s1-replay"):
        args.lat = args.lat if args.lat is not None else (
            -23.65 if "atacama" in args.weather_mode else 42.36
        )
        args.lon = args.lon if args.lon is not None else (
            -70.40 if "atacama" in args.weather_mode else -71.09
        )


def default_solar_sim_args() -> argparse.Namespace:
    """CLI defaults for ``run_solar_simulation`` (scripting / parameter sweeps)."""
    p = argparse.ArgumentParser()
    register_solar_sim_arguments(p)
    return p.parse_args([])


def output_tag(args: argparse.Namespace, config: DeviceConfig) -> str:
    tag = args.weather_mode
    if config.sorbent == "mof":
        tag += f"_{config.mof_name}"
    if args.lat is not None:
        tag += f"_lat{args.lat:.4f}_lon{args.lon:.4f}_{args.year}"
    return tag


@dataclass(frozen=True, slots=True)
class SolarSimResult:
    weather_mode: str
    config: DeviceConfig
    econ: LCOEconomicParams
    profile: object
    daily_yield_kg_per_m2: float
    thermal_efficiency: float
    lcow_usd_per_m3: float
    breakdown: LcowCostBreakdown | None
    lat: float | None
    lon: float | None
    year: int
    inventory_note: str
    inventory_abs_res: object | None
    inventory_des_res: object | None


def run_solar_simulation(
    args: argparse.Namespace,
    *,
    econ: LCOEconomicParams | None = None,
    baseline_profile_kwargs: dict[str, float] | None = None,
    h_des_j_per_kg: float | None = None,
    salt_formula_weight_g_mol: float | None = None,
    salt_weight_factor: float | None = None,
    cyclic_initial: bool | None = None,
    cyclic_warmup_cycles: int | None = None,
) -> SolarSimResult:
    """Run one SAWH daily cycle and LCOW (same logic as ``main()``)."""
    config = _build_config(args)
    overrides = _config_overrides(config)
    if args.weather_mode == "atacama-replay":
        config = DeviceConfig.atacama_field(**overrides)
    elif args.weather_mode == "baseline":
        config = DeviceConfig.baseline(**overrides)
    elif args.weather_mode == "fig-s1-replay":
        config = DeviceConfig.comsol_table_s3(
            **overrides,
            tilt_deg=config.tilt_deg,
            fin_area_ratio=config.fin_area_ratio,
        )
    config = _apply_physics_overrides(
        config,
        h_des_j_per_kg=h_des_j_per_kg,
        salt_formula_weight_g_mol=salt_formula_weight_g_mol,
        salt_weight_factor=salt_weight_factor,
    )
    econ = econ or LCOEconomicParams()

    c_w_initial: float | None = None
    h_initial: float | None = None
    if args.initial_water_l_m2 is not None:
        c_w_initial = _initial_loading_from_water_l_m2(args.initial_water_l_m2, config)
    elif config.sorbent == "hydrogel" and args.weather_mode == "fig-s1-replay":
        c_w_initial = fig_s1_initial_c_w(h_m=config.hydrogel_thickness_m)
    elif config.sorbent == "hydrogel" and args.weather_mode == "baseline":
        c_w_initial = baseline_initial_c_w(h_m=config.hydrogel_thickness_m)

    use_cycled, n_warmup = _cyclic_settings(
        args,
        cyclic_initial=cyclic_initial,
        cyclic_warmup_cycles=cyclic_warmup_cycles,
    )

    inventory_abs_res = None
    inventory_des_res = None
    inventory_note = ""
    if use_cycled:
        inventory_note = f" ({n_warmup} warmup day(s) + 1 reporting day)"

    if args.weather_mode == "real":
        profile = representative_mean_day_profile(
            args.lat,
            args.lon,
            args.year,
            cache_dir=args.cache_dir,
        )
        yield_kg, eta, inventory_abs_res, inventory_des_res = run_daily_cycle(
            profile,
            config,
            c_w_initial=c_w_initial,
            h_initial=h_initial,
            cyclic_initial=use_cycled,
            cyclic_warmup_cycles=n_warmup,
        )
        if use_cycled:
            inventory_note = (
                f" (cycled initial state; mean diurnal weather; "
                f"{n_warmup} warmup + 1 report)"
            )
        else:
            inventory_note = f" (mean diurnal weather for {args.year})"
    elif args.weather_mode == "baseline":
        profile = baseline_profile(**(baseline_profile_kwargs or {}))
        yield_kg, eta, inventory_abs_res, inventory_des_res = run_daily_cycle(
            profile,
            config,
            c_w_initial=c_w_initial,
            h_initial=h_initial,
            cyclic_initial=use_cycled,
            cyclic_warmup_cycles=n_warmup,
        )
    else:
        profile = replay_profile(args.weather_mode, cache_dir=args.cache_dir)
        yield_kg, eta, inventory_abs_res, inventory_des_res = run_daily_cycle(
            profile,
            config,
            c_w_initial=c_w_initial,
            h_initial=h_initial,
            cyclic_initial=use_cycled,
            cyclic_warmup_cycles=n_warmup,
        )
        if use_cycled:
            inventory_note = (
                f" (cycled initial state after {n_warmup} warmup day(s))"
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

    return SolarSimResult(
        weather_mode=args.weather_mode,
        config=config,
        econ=econ,
        profile=profile,
        daily_yield_kg_per_m2=float(yield_kg),
        thermal_efficiency=float(eta),
        lcow_usd_per_m3=float(lcow),
        breakdown=breakdown,
        lat=args.lat,
        lon=args.lon,
        year=int(args.year),
        inventory_note=inventory_note,
        inventory_abs_res=inventory_abs_res,
        inventory_des_res=inventory_des_res,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Solar lumped SAWH simulation + LCOW")
    register_solar_sim_arguments(p)
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Cost breakdown CSV path",
    )
    p.add_argument(
        "--water-inventory-csv",
        type=Path,
        default=None,
        help="Water-in-gel time series CSV path",
    )
    p.add_argument(
        "--water-inventory-plot",
        type=Path,
        default=None,
        help="Water-in-gel vs time PNG path",
    )
    p.add_argument(
        "--detailed",
        action="store_true",
        help="Write CSV and plot device temperatures (absorber, glass, condenser, gel) "
        "and weather variables over the full daily cycle",
    )
    p.add_argument(
        "--detailed-csv",
        type=Path,
        default=None,
        help="Detailed diagnostics CSV path (default: outputs/detailed/diagnostics_<tag>.csv)",
    )
    p.add_argument(
        "--detailed-plot",
        type=Path,
        default=None,
        help="Detailed diagnostics PNG path (default: outputs/detailed/diagnostics_<tag>.png)",
    )
    args = p.parse_args()

    resolve_solar_sim_arguments(args, p)
    result = run_solar_simulation(args)
    config = result.config
    profile = result.profile
    result_mean = result.daily_yield_kg_per_m2
    eta_mean = result.thermal_efficiency
    lcow = result.lcow_usd_per_m3
    breakdown = result.breakdown
    inventory_abs_res = result.inventory_abs_res
    inventory_des_res = result.inventory_des_res
    inventory_note = result.inventory_note
    n_days = 1

    print(f"Weather mode: {args.weather_mode}")
    print(f"Sorbent: {config.sorbent}" + (f" ({config.mof_name})" if config.sorbent == "mof" else f" ({config.salt_name})"))
    if args.weather_mode == "real":
        print(f"Year aggregated to mean diurnal profile: {args.year}")
    print(f"Days simulated: {n_days}")
    print(f"Mean daily yield: {result_mean * 1000:.1f} g/m² ({result_mean:.4f} kg/m²)")
    print(f"Mean daily yield: {result_mean:.2f} L/m² (≈ kg/m² for water)")
    print(f"Mean thermal efficiency: {eta_mean * 100:.1f}%")
    print(f"LCOW: ${lcow:.4f}/m³")
    if breakdown:
        print("\nCost breakdown (USD/m³):")
        for seg, val in breakdown.items:
            print(f"  {seg:30s} {val:10.4f}")

    out = args.output_csv
    if out is None:
        out = _REPO / "outputs" / "lcow" / f"cost_breakdown_{output_tag(args, config)}.csv"
    if breakdown:
        _write_cost_breakdown_csv(
            out,
            breakdown,
            subtitle=f"LCOW=${lcow:.4f}/m³ yield={result_mean:.4f}kg/m²/d",
            lat=args.lat,
            lon=args.lon,
            year=args.year if args.weather_mode == "real" else None,
        )
        print(f"\nWrote {out}")

    if inventory_abs_res is not None and inventory_des_res is not None:
        inventory = water_inventory_series(
            inventory_abs_res,
            inventory_des_res,
            config=config,
        )
        inv_prefix = inventory_prefix(config)
        tag = output_tag(args, config)
        inventory_csv = args.water_inventory_csv
        if inventory_csv is None:
            inventory_csv = _REPO / "outputs" / "water_inventory" / f"{inv_prefix}_{tag}.csv"
        inventory_plot = args.water_inventory_plot
        if inventory_plot is None:
            inventory_plot = _REPO / "outputs" / "water_inventory" / f"{inv_prefix}_{tag}.png"

        write_water_inventory_csv(inventory_csv, inventory)
        plot_title = f"Water in {inventory_label(config)} — {args.weather_mode}{inventory_note}"
        plot_water_inventory(inventory_plot, inventory, config=config, title=plot_title)

        if args.weather_mode in ("atacama-replay", "cambridge-replay", "baseline", "fig-s1-replay"):
            simple_csv = _REPO / "outputs" / "water_inventory" / f"{inv_prefix}_{args.weather_mode}.csv"
            simple_plot = _REPO / "outputs" / "water_inventory" / f"{inv_prefix}_{args.weather_mode}.png"
            write_water_inventory_csv(simple_csv, inventory)
            plot_water_inventory(simple_plot, inventory, config=config, title=plot_title)

        w_start = float(inventory.water_l_m2[0])
        w_peak = float(np.max(inventory.water_l_m2))
        w_end = float(inventory.water_l_m2[-1])
        print(
            f"\nWater in {inventory_label(config)}{inventory_note}: "
            f"start={w_start:.2f} peak={w_peak:.2f} end={w_end:.2f} L/m²"
        )
        print(f"Wrote {inventory_csv}")
        print(f"Wrote {inventory_plot}")
        if args.weather_mode in ("atacama-replay", "cambridge-replay", "baseline", "fig-s1-replay"):
            print(f"Wrote {simple_csv}")
            print(f"Wrote {simple_plot}")

    if args.detailed:
        if inventory_abs_res is None or inventory_des_res is None:
            print("\n--detailed skipped: simulation did not return phase results.", flush=True)
        else:
            tag = output_tag(args, config)
            detailed = detailed_series(profile, inventory_abs_res, inventory_des_res, config)
            detailed_csv = args.detailed_csv
            if detailed_csv is None:
                detailed_csv = _REPO / "outputs" / "detailed" / f"diagnostics_{tag}.csv"
            detailed_plot = args.detailed_plot
            if detailed_plot is None:
                detailed_plot = _REPO / "outputs" / "detailed" / f"diagnostics_{tag}.png"

            write_detailed_csv(detailed_csv, detailed)
            plot_title = f"Device and weather — {args.weather_mode}{inventory_note}"
            plot_detailed_diagnostics(detailed_plot, detailed, title=plot_title)
            print(f"\nWrote {detailed_csv}")
            print(f"Wrote {detailed_plot}")


if __name__ == "__main__":
    main()
