#!/usr/bin/env python3
"""Full-factorial device-parameter sweep at one grid site, over one representative mean day
per calendar month (day-weighted). Monthly is the default because this feeds a map that must
be accurate everywhere: a single annual mean day is 12x cheaper and within ~0.2% at desert
sites, but ~14-21% off at high-seasonal-variance ones (see compare_annual_vs_mean_day.py).

Built for cluster job arrays -- one invocation is one site, chosen by --site-index (e.g.
$SLURM_ARRAY_TASK_ID) out of the --step land grid, or --lat/--lon for a local run. Each runs
the full grid (default 5x5x5 = 125 combos over hydrogel thickness, fin area ratio, vapor
gap), appending one row per combo to --output-csv and skipping existing rows under --resume,
so a preempted task can be resubmitted. eps_abs/tau_glass are fixed per case, not swept.
Weather is fetched once per site and reused across all combos."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from solar_lumped.physics import DeviceThermalParams, TILT_DEG  # noqa: E402
from solar_lumped.simulation import DeviceConfig  # noqa: E402
from solar_lumped.simulation import find_cyclic_state, run_daily_cycle  # noqa: E402
from solar_lumped.weather import fetch_year_weather  # noqa: E402
from solar_lumped.weather import representative_mean_day_df  # noqa: E402
from solar_lumped.weather import grid_land_points  # noqa: E402
from solar_lumped.weather import DailyWeatherProfile, profile_from_day_df  # noqa: E402

# Baselines: table_s3.H0_M=4mm, L_G_M=40mm, EPS_ABS=0.95, TAU_GLASS=0.9, FIN_AREA_RATIO=7.1.
DEFAULT_HYDROGEL_THICKNESS_MM: tuple[float, ...] = (1.0, 3.25, 5.5, 7.75, 10.0)
DEFAULT_FIN_AREA_RATIO: tuple[float, ...] = (3.0, 5.275, 7.55, 9.825, 12.0)
DEFAULT_VAPOR_GAP_MM: tuple[float, ...] = (20.0, 30.0, 40.0, 50.0, 60.0)
# eps_abs and tau_glass are fixed constants per case (not swept) -- see --eps-abs/--tau-glass.
DEFAULT_EPS_ABS: float = 0.95
DEFAULT_TAU_GLASS: float = 0.90


@dataclass(frozen=True, slots=True)
class Combo:
    hydrogel_thickness_mm: float
    fin_area_ratio: float
    vapor_gap_mm: float


def combo_grid(
    *,
    hydrogel_thickness_mm: list[float],
    fin_area_ratio: list[float],
    vapor_gap_mm: list[float],
) -> list[Combo]:
    return [
        Combo(*vals)
        for vals in itertools.product(hydrogel_thickness_mm, fin_area_ratio, vapor_gap_mm)
    ]


def build_device_config(
    combo: Combo,
    *,
    salt: str,
    salt_loading: float,
    insulation_gap_mm: float,
    tilt_deg: float,
    eps_abs: float,
    tau_glass: float,
    eps_abs_ir: float | None = None,
    eps_glass_ir: float | None = None,
) -> DeviceConfig:
    thermal = DeviceThermalParams(
        insulation_gap_m=insulation_gap_mm * 1e-3,
        vapor_gap_m=combo.vapor_gap_mm * 1e-3,
        eps_abs=eps_abs,
        tau_glass=tau_glass,
        tilt_deg=tilt_deg,
        eps_abs_ir=eps_abs_ir,
        eps_glass_ir=eps_glass_ir,
    )
    return DeviceConfig(
        salt_name=salt,
        salt_to_polymer_ratio=salt_loading,
        hydrogel_thickness_m=combo.hydrogel_thickness_mm * 1e-3,
        vapor_gap_m=combo.vapor_gap_mm * 1e-3,
        insulation_gap_m=insulation_gap_mm * 1e-3,
        fin_area_ratio=combo.fin_area_ratio,
        tilt_deg=tilt_deg,
        thermal=thermal,
    )


def monthly_mean_profiles(df, *, poa_tilt_deg: float | None = None) -> list[tuple[int, DailyWeatherProfile, int]]:
    """One representative mean-day profile per calendar month present in *df*.

    Returns (month, profile, n_days_in_month) so callers can day-weight the average.
    """
    import pandas as pd

    out: list[tuple[int, DailyWeatherProfile, int]] = []
    for m in sorted(set(df.index.month)):
        month_df = df[df.index.month == m]
        if month_df.empty:
            continue
        ref_day = month_df.index[len(month_df) // 2].date()
        mean_day_df = representative_mean_day_df(month_df, reference_day=ref_day)
        profile = profile_from_day_df(mean_day_df, poa_tilt_deg=poa_tilt_deg)
        n_days = len(pd.unique(month_df.index.date))
        out.append((m, profile, n_days))
    return out


def single_mean_profile(df, *, poa_tilt_deg: float | None = None) -> list[tuple[int, DailyWeatherProfile, int]]:
    """One representative mean-day profile for the whole year. Validated against the full
    365-day simulation across 3 climates: within ~0.2% at low-variance (desert, LCOW-
    competitive) sites, ~14-21% off at high-variance ones. Use --resolution monthly to
    re-check specific sites instead of paying 12x across the whole grid."""
    import pandas as pd

    ref_day = df.index[len(df) // 2].date()
    mean_day_df = representative_mean_day_df(df, reference_day=ref_day)
    profile = profile_from_day_df(mean_day_df, poa_tilt_deg=poa_tilt_deg)
    n_days = len(pd.unique(df.index.date))
    return [(0, profile, n_days)]


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def combo_yield_kg_m2(
    profiles: list[tuple[int, DailyWeatherProfile, int]],
    config: DeviceConfig,
    *,
    warmup_method: str,
    fixed_warmup_cycles: int,
) -> tuple[float, float]:
    """Day-weighted mean daily yield (kg/m^2) and mean eta_thermal across profiles."""
    yields: list[float] = []
    etas: list[float] = []
    weights: list[int] = []
    for _month, profile, n_days in profiles:
        if warmup_method == "aitken":
            cw, h = find_cyclic_state(profile, config)
        else:
            cw, h = None, None
            for _ in range(max(1, fixed_warmup_cycles)):
                _, _, _, des_res = run_daily_cycle(profile, config, c_w_initial=cw, h_initial=h)
                cw, h = float(des_res.c_w[-1]), float(des_res.H[-1])
        yield_kg, eta, _, _ = run_daily_cycle(profile, config, c_w_initial=cw, h_initial=h)
        yields.append(yield_kg)
        etas.append(eta)
        weights.append(n_days)
    return _weighted_mean(yields, weights), _weighted_mean(etas, weights)


def mean_weather_stats(
    profiles: list[tuple[int, DailyWeatherProfile, int]],
) -> tuple[float, float, float]:
    """Day-weighted mean RH (fraction), ambient temperature (C), and daylight solar
    irradiance (W/m²) across profiles -- a site property, computed once and reused across
    combos. Solar averages the desorption phase only, since absorption is by definition the
    low-solar half of the day."""
    rh_means: list[float] = []
    t_means: list[float] = []
    solar_means: list[float] = []
    weights: list[int] = []
    for _period, profile, n_days in profiles:
        rh = list(profile.absorption.relative_humidity) + list(profile.desorption.relative_humidity)
        t = list(profile.absorption.temperature_c) + list(profile.desorption.temperature_c)
        solar = profile.desorption.solar_w_m2
        rh_means.append(sum(rh) / len(rh))
        t_means.append(sum(t) / len(t))
        solar_means.append(sum(solar) / len(solar))
        weights.append(n_days)
    return (
        _weighted_mean(rh_means, weights),
        _weighted_mean(t_means, weights),
        _weighted_mean(solar_means, weights),
    )


_CSV_COLUMNS: tuple[str, ...] = (
    "lat",
    "lon",
    "mean_rh_frac",
    "mean_t_amb_c",
    "mean_solar_w_m2",
    "salt",
    "hydrogel_thickness_mm",
    "eps_abs",
    "tau_glass",
    "eps_abs_ir",
    "eps_glass_ir",
    "fin_area_ratio",
    "vapor_gap_mm",
    "warmup_method",
    "resolution",
    "mean_yield_kg_m2",
    "mean_eta_thermal",
    "n_periods",
)


def _existing_combo_keys(path: Path, lat: float, lon: float) -> set[tuple]:
    if not path.is_file():
        return set()
    import pandas as pd

    df = pd.read_csv(path)
    df = df[(df["lat"] == lat) & (df["lon"] == lon)]
    keys = set()
    for _, row in df.iterrows():
        keys.add(
            (
                round(float(row["hydrogel_thickness_mm"]), 6),
                round(float(row["fin_area_ratio"]), 6),
                round(float(row["vapor_gap_mm"]), 6),
            )
        )
    return keys


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    site = p.add_mutually_exclusive_group(required=True)
    site.add_argument("--lat-lon", type=float, nargs=2, metavar=("LAT", "LON"))
    site.add_argument("--site-index", type=int, help="Index into the --step land grid (e.g. $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--step", type=float, default=3.0, help="Grid spacing in degrees, used with --site-index")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cache-dir", type=str, default=str(_REPO / ".weather_cache"))
    p.add_argument("--salt", type=str, default="LiCl")
    p.add_argument("--salt-loading", type=float, default=4.0)
    p.add_argument("--insulation-gap-mm", type=float, default=5.0)
    p.add_argument("--tilt-deg", type=float, default=TILT_DEG)
    p.add_argument(
        "--poa",
        action="store_true",
        help=(
            "Drive the absorber with plane-of-array irradiance at --tilt-deg instead of "
            "raw GHI (Erbs + Liu-Jordan isotropic). Off by default: Wilson's Eq. 4 uses "
            "GHI, so tilt otherwise only affects Hollands gap convection. With --poa, "
            "tilt trades solar gain against gap convection at once."
        ),
    )
    p.add_argument("--hydrogel-thickness-mm", type=float, nargs="+", default=list(DEFAULT_HYDROGEL_THICKNESS_MM))
    p.add_argument("--fin-area-ratio", type=float, nargs="+", default=list(DEFAULT_FIN_AREA_RATIO))
    p.add_argument("--vapor-gap-mm", type=float, nargs="+", default=list(DEFAULT_VAPOR_GAP_MM))
    p.add_argument(
        "--eps-abs", type=float, default=DEFAULT_EPS_ABS,
        help="Absorber solar absorptivity (fixed constant, not swept). Case 1/2 baseline: 0.95; "
        "Case 3 'optical material limits': 1.0.",
    )
    p.add_argument(
        "--tau-glass", type=float, default=DEFAULT_TAU_GLASS,
        help="Glass solar transmittance (fixed constant, not swept). Case 1/2 baseline: 0.90; "
        "Case 3: 1.0.",
    )
    p.add_argument(
        "--eps-abs-ir", type=float, default=None,
        help="Absorber IR emissivity for the modified Eqs. 3/4 radiative exchange (fixed constant, "
        "not swept). Default None reproduces the original blackbody/cavity approximation "
        "(eps_ag=eps_ga=1.0) exactly -- set together with --eps-glass-ir to activate the modified "
        "physics (Case 2: 0.05; Case 3 'optical material limits': 0.0).",
    )
    p.add_argument(
        "--eps-glass-ir", type=float, default=None,
        help="Glass IR emissivity for the modified Eqs. 3/4 radiative exchange (fixed constant, not "
        "swept). See --eps-abs-ir. (Case 2: 0.95; Case 3: 0.0.)",
    )
    p.add_argument(
        "--warmup-method",
        choices=("aitken", "fixed-cycle"),
        default="aitken",
        help="aitken: converge to the true steady periodic state (find_cyclic_state, "
        "~3-6 rounds). fixed-cycle: cheaper but can be badly off at strongly "
        "seasonal sites (see compare_annual_vs_mean_day.py findings).",
    )
    p.add_argument(
        "--resolution",
        choices=("single", "monthly"),
        default="monthly",
        help="monthly (default): 12 Aitken-converged mean days per combo, day-weighted. "
        "Needed for a map that must be accurate everywhere, not just at "
        "LCOW-competitive sites -- single mean-day is ~14-21%% off at "
        "high-seasonal-variance sites (see compare_annual_vs_mean_day.py findings). "
        "single: 12x cheaper, only accurate at low-seasonal-variance sites.",
    )
    p.add_argument("--fixed-warmup-cycles", type=int, default=1, help="Used when --warmup-method=fixed-cycle")
    p.add_argument(
        "--combo-offset",
        type=int,
        default=0,
        help="Start index into the full combo grid (for splitting one site's combos "
        "across multiple job-array tasks -- a site's full combo grid can take "
        "hours to tens of hours serially, too long for one SLURM task).",
    )
    p.add_argument(
        "--combo-limit",
        type=int,
        default=None,
        help="Number of combos to run starting at --combo-offset (default: all remaining).",
    )
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Skip combos already present for this site in --output-csv")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.lat_lon is not None:
        lat, lon = args.lat_lon
    else:
        points = grid_land_points(args.step)
        if not (0 <= args.site_index < len(points)):
            print(f"--site-index must be in [0, {len(points) - 1}] for --step {args.step}", file=sys.stderr)
            return 1
        lat, lon = points[args.site_index]

    all_combos = combo_grid(
        hydrogel_thickness_mm=args.hydrogel_thickness_mm,
        fin_area_ratio=args.fin_area_ratio,
        vapor_gap_mm=args.vapor_gap_mm,
    )
    end = None if args.combo_limit is None else args.combo_offset + args.combo_limit
    combos = all_combos[args.combo_offset : end]
    if not (0 <= args.combo_offset < max(len(all_combos), 1)):
        print(f"--combo-offset must be in [0, {len(all_combos) - 1}]", file=sys.stderr)
        return 1
    print(
        f"Site ({lat:+.4f}, {lon:+.4f}): {len(combos)} combo(s) "
        f"(offset {args.combo_offset} of {len(all_combos)} total)",
        flush=True,
    )

    print(f"Fetching {args.year} weather [cache={args.cache_dir}]…", flush=True)
    df = fetch_year_weather(lat, lon, args.year, cache_dir=args.cache_dir)

    poa_tilt_deg = args.tilt_deg if args.poa else None
    builder = single_mean_profile if args.resolution == "single" else monthly_mean_profiles
    profiles = builder(df, poa_tilt_deg=poa_tilt_deg)
    print(
        f"Built {len(profiles)} {args.resolution} mean-day profile(s)"
        + (f" [POA @ {args.tilt_deg:g} deg tilt]" if args.poa else " [GHI]"),
        flush=True,
    )

    mean_rh, mean_t_amb, mean_solar = mean_weather_stats(profiles)
    print(
        f"  Site mean RH={mean_rh:.3f}  T_amb={mean_t_amb:.1f}C  Q_solar={mean_solar:.0f}W/m²",
        flush=True,
    )

    done = _existing_combo_keys(args.output_csv, lat, lon) if args.resume else set()
    if done:
        print(f"Resume: {len(done)} combo(s) already done for this site.", flush=True)

    t0 = time.perf_counter()
    n_done = 0
    for i, combo in enumerate(combos, start=1):
        key = (
            round(combo.hydrogel_thickness_mm, 6),
            round(combo.fin_area_ratio, 6),
            round(combo.vapor_gap_mm, 6),
        )
        if key in done:
            continue
        config = build_device_config(
            combo,
            salt=args.salt,
            salt_loading=args.salt_loading,
            insulation_gap_mm=args.insulation_gap_mm,
            tilt_deg=args.tilt_deg,
            eps_abs=args.eps_abs,
            tau_glass=args.tau_glass,
            eps_abs_ir=args.eps_abs_ir,
            eps_glass_ir=args.eps_glass_ir,
        )
        mean_yield, mean_eta = combo_yield_kg_m2(
            profiles,
            config,
            warmup_method=args.warmup_method,
            fixed_warmup_cycles=args.fixed_warmup_cycles,
        )
        _append_row(
            args.output_csv,
            {
                "lat": lat,
                "lon": lon,
                "mean_rh_frac": f"{mean_rh:.6f}",
                "mean_t_amb_c": f"{mean_t_amb:.4f}",
                "mean_solar_w_m2": f"{mean_solar:.2f}",
                "salt": args.salt,
                "hydrogel_thickness_mm": combo.hydrogel_thickness_mm,
                "eps_abs": args.eps_abs,
                "tau_glass": args.tau_glass,
                "eps_abs_ir": args.eps_abs_ir if args.eps_abs_ir is not None else "",
                "eps_glass_ir": args.eps_glass_ir if args.eps_glass_ir is not None else "",
                "fin_area_ratio": combo.fin_area_ratio,
                "vapor_gap_mm": combo.vapor_gap_mm,
                "warmup_method": args.warmup_method,
                "resolution": args.resolution,
                "mean_yield_kg_m2": f"{mean_yield:.6f}",
                "mean_eta_thermal": f"{mean_eta:.6f}",
                "n_periods": len(profiles),
            },
        )
        n_done += 1
        print(
            f"  [{i}/{len(combos)}] h={combo.hydrogel_thickness_mm:.2f}mm eps_abs={args.eps_abs:.2f} "
            f"tau_glass={args.tau_glass:.2f} fin={combo.fin_area_ratio:.2f} "
            f"gap={combo.vapor_gap_mm:.1f}mm -> yield={mean_yield:.6f} kg/m² "
            f"({time.perf_counter() - t0:.1f}s elapsed)",
            flush=True,
        )

    print(f"Done: {n_done} combo(s) run, {len(done)} skipped, {time.perf_counter() - t0:.1f}s total.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
