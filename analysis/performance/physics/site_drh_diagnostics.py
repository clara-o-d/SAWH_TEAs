#!/usr/bin/env python3
"""One-day CPU diagnostics at five real sites, for LiCl and LiBr, with the DRH question
made quantitative.

Two questions, one script, because they are the same question from two ends.

**How often does desorption run below the deliquescence RH?** Desorption is driven by
c_r = (p_sat(T_cond)/p_sat(T_gel))(T_gel/T_cond), and the brine's activity a_w cannot fall
below the salt's DRH at the gel temperature -- past that the salt precipitates and a
saturated solution's activity stops falling. So whenever c_r < DRH(T_gel), the gel is
still drying (a_w > c_r, so Eq. 5 still points outward) but it is doing so with the brine
already saturated. Removing that water is a dehydration reaction this model does not
represent. This script reports what fraction of the desorption phase -- by duration and
by yield -- sits in that regime, and by how much c_r undercuts DRH.

**What does it cost to forbid it?** ``SystemConfig.c_w_floor_mode`` is exactly that
choice: "hydrate" lets c_r dip below DRH and stops only at the crystal-hydrate water
n*c_s, while "drh" stops at the loading whose equilibrium activity IS the DRH. The yield
matrix runs every scenario x salt x floor-mode combination so the first question's answer
can be read against the second's price.

DRH is temperature-dependent here, not the 25 C catalog value: solubility rises with
temperature, so DRH FALLS as the gel heats (LiCl 0.150 at 0 C -> 0.096 at 80 C). A
desorbing gel therefore pins lower than a room-temperature number would suggest, which
matters precisely because desorption is the hot half of the cycle.

Everything runs at the parameters.xlsx baseline design and the default optical scenario
(scenario 2, "reasonable improvements": eps_abs 0.95, tau_glass 0.90, eps_abs_ir 0.05,
eps_glass_ir 0.95) unless the scenario axis says otherwise.

    python3 analysis/performance/physics/site_drh_diagnostics.py
"""

from __future__ import annotations

import dataclasses
import sys
import traceback
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SOLAR = _REPO / "solar_lumped"
sys.path.insert(0, str(_SOLAR / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from solar_lumped import site_sweep as ss  # noqa: E402
from solar_lumped.physics import (  # noqa: E402
    concentration_ratio_desorption,
    deliquescence_rh,
    drh_floor_c_w,
    water_activity_from_c_w,
)
from solar_lumped.simulation import (  # noqa: E402
    detailed_series,
    plot_detailed_diagnostics,
    plot_water_inventory,
    run_daily_cycle,
    water_inventory_series,
    write_detailed_csv,
    write_water_inventory_csv,
)
from solar_lumped.weather import real_day_profile  # noqa: E402

# Named sites rather than the land grid: these are the ones with a reason to be here --
# two temperate (one Mediterranean-dry, one the Cambridge test's own climate), the
# hyper-arid Atacama the field test used, and two humid tropics that stress the opposite
# end of the isotherm.
SITES: tuple[tuple[str, float, float], ...] = (
    ("palo_alto_us", 37.4419, -122.1430),
    ("cambridge_us", 42.3736, -71.1097),
    ("atacama_cl", -23.6500, -70.4000),
    ("san_jose_cr", 9.9281, -84.0907),
    ("singapore_sg", 1.3521, 103.8198),
)
SALTS: tuple[str, ...] = ("LiCl", "LiBr")
# The repo's own default diagnostic day (system.py --weather-mode real). June puts the
# three northern sites near their solar peak and Atacama in winter; that asymmetry is
# real and is called out in the report rather than averaged away.
DAY = date(2024, 6, 15)
CACHE_DIR = str(_REPO / ".weather_cache")
OUT_DIR = _SOLAR / "outputs" / "site_diagnostics"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = {}
    for name, lat, lon in SITES:
        profiles[name] = real_day_profile(
            lat, lon, DAY, cache_dir=CACHE_DIR, poa_tilt_deg=ss.TILT_DEG
        )
        print(f"weather loaded: {name} ({lat:+.4f}, {lon:+.4f})", flush=True)

    drh_rows = []
    for name, _lat, _lon in SITES:
        for salt in SALTS:
            drh_rows.append(_detailed_diagnostic(name, profiles[name], salt))
    drh = pd.DataFrame(drh_rows)
    drh.to_csv(OUT_DIR / "drh_diagnostics.csv", index=False)

    yields = _yield_matrix(profiles)
    yields.to_csv(OUT_DIR / "yield_matrix.csv", index=False)

    _report(drh, yields)
    print(f"\nArtifacts in {OUT_DIR}")
    return 0


def _config(salt: str, scenario_name: str, floor_mode: str):
    """Baseline design, one scenario, one salt, one c_w floor rule."""
    sc = ss.SCENARIOS[scenario_name]
    cfg = ss.build_system_config(
        ss.BASELINE_COMBO,
        salt=salt,
        salt_loading=ss.BASELINE_SALT_LOADING,
        insulation_gap_mm=ss.BASELINE_INSULATION_GAP_MM,
        tilt_deg=ss.TILT_DEG,
        eps_abs=sc.eps_abs,
        tau_glass=sc.tau_glass,
        eps_abs_ir=sc.eps_abs_ir,
        eps_glass_ir=sc.eps_glass_ir,
    )
    return dataclasses.replace(
        cfg,
        instant_equilibrium=sc.instant_equilibrium,
        condenser_tracks_ambient=sc.condenser_ambient,
        c_w_floor_mode=floor_mode,
    )


def _detailed_diagnostic(site: str, profile, salt: str) -> dict:
    """Full one-day diagnostic (the --detailed artifacts) plus the c_r-vs-DRH accounting.

    Runs the default scenario at the "hydrate" floor -- i.e. c_r IS allowed to dip below
    DRH -- because that is the configuration whose trajectory the question is about. The
    price of forbidding it is the yield matrix's job.
    """
    config = _config(salt, "improved", "hydrate")
    water, eta, abs_res, des_res = run_daily_cycle(profile, config)

    tag = f"{site}_{salt}"
    detailed = detailed_series(profile, abs_res, des_res, config)
    write_detailed_csv(OUT_DIR / f"detailed_{tag}.csv", detailed)
    plot_detailed_diagnostics(
        OUT_DIR / f"detailed_{tag}.png",
        detailed,
        title=f"System and weather — {site}, {salt}, {DAY.isoformat()} (scenario 2)",
    )
    inventory = water_inventory_series(abs_res, des_res, config=config)
    write_water_inventory_csv(OUT_DIR / f"inventory_{tag}.csv", inventory)
    plot_water_inventory(
        OUT_DIR / f"inventory_{tag}.png",
        inventory,
        config=config,
        title=f"Water in gel — {site}, {salt}, {DAY.isoformat()}",
    )

    # --- c_r against the temperature-dependent DRH, over the desorption phase ---
    t_gel = np.asarray(des_res.t_gel_c, dtype=float)
    t_cond = np.asarray(des_res.t_cond_c, dtype=float)
    c_r = np.array([concentration_ratio_desorption(g, c) for g, c in zip(t_gel, t_cond)])
    drh = np.array([deliquescence_rh(salt, g) for g in t_gel])
    below = c_r < drh
    deficit = np.where(below, drh - c_r, 0.0)

    # Interval-weighted, not sample-counted: the reporting grid is uniform here, but a
    # fraction that silently assumes that is a fraction that breaks when the grid changes.
    t = np.asarray(des_res.time_s, dtype=float)
    dt = np.diff(t)
    mid_below = below[:-1] & below[1:]
    duration_frac = float(dt[mid_below].sum() / dt.sum()) if dt.sum() > 0 else float("nan")

    # Yield accrued while below DRH, from the integrated W state rather than a
    # re-trapezoid of m_des, so it agrees with the reported daily yield exactly.
    cum = np.asarray(des_res.water_cumulative_kg_m2, dtype=float)
    d_water = np.diff(cum)
    total = float(d_water.sum())
    yield_frac = float(d_water[mid_below].sum() / total) if total > 0 else float("nan")

    # c_r < DRH means the equilibrium the gel is being driven TOWARD lies past brine
    # saturation, so desorption will not self-terminate there -- it runs to whichever
    # floor the model imposes. Whether the brine is saturated at a given instant is the
    # separate, sharper question of whether a_w has actually reached its DRH floor, so
    # both are reported. The yield matrix's hydrate-vs-drh ratio is the price of the gap.
    mass = config.mass_params()
    a_w = np.array([
        water_activity_from_c_w(
            float(c), c_s=mass.c_s_mol_m3, ions_per_formula=mass.ions_per_formula,
            temperature_c=float(g), salt_name=mass.salt_name,
            formula_weight_g_mol=mass.formula_weight_g_mol, salt_loading=mass.salt_loading,
            h_m=float(h), h0_ref_m=mass.h0_ref_m, salt_weight_factor=mass.salt_weight_factor,
        )
        for c, g, h in zip(des_res.c_w, t_gel, des_res.H)
    ])
    pinned = a_w <= drh * (1.0 + 1e-6)
    mid_pinned = pinned[:-1] & pinned[1:]
    pinned_duration = float(dt[mid_pinned].sum() / dt.sum()) if dt.sum() > 0 else float("nan")
    pinned_yield = float(d_water[mid_pinned].sum() / total) if total > 0 else float("nan")

    return {
        "site": site,
        "salt": salt,
        "yield_kg_m2": float(water),
        "eta_thermal": float(eta),
        "desorption_h": float((t[-1] - t[0]) / 3600.0),
        "pct_duration_c_r_below_drh": 100.0 * duration_frac,
        "pct_yield_c_r_below_drh": 100.0 * yield_frac,
        "mean_deficit_when_below": float(deficit[below].mean()) if below.any() else 0.0,
        "max_deficit": float(deficit.max()),
        "mean_c_r": float(c_r.mean()),
        "mean_drh_at_t_gel": float(drh.mean()),
        "drh_at_25c": deliquescence_rh(salt, 25.0),
        "mean_t_gel_c": float(t_gel.mean()),
        "max_t_gel_c": float(t_gel.max()),
        "pct_duration_a_w_pinned_at_drh": 100.0 * pinned_duration,
        "pct_yield_a_w_pinned_at_drh": 100.0 * pinned_yield,
        "c_w_end": float(des_res.c_w[-1]),
        "hydrate_floor_c_w": float(mass.c_w_min_mol_m3),
        # Same call SystemConfig.mass_params() makes for c_w_floor_mode="drh", so this is
        # the floor the yield matrix's "drh" column actually ran against.
        "drh_floor_c_w": drh_floor_c_w(
            c_s_mol_m3=mass.c_s_mol_m3, salt_name=mass.salt_name,
            formula_weight_g_mol=mass.formula_weight_g_mol,
        ),
    }


def _yield_matrix(profiles: dict) -> pd.DataFrame:
    """Every site x salt x scenario x floor-mode yield.

    "hydrate" lets c_r dip below DRH (desorption continues to the crystal-hydrate water);
    "drh" pins the floor at the loading whose equilibrium activity is the DRH. The
    difference is the yield attributable to drying a brine that has already saturated.

    Failures are recorded per cell, not allowed to abort the matrix -- but they are
    counted and printed, because a blank cell that reads as "no data" when it means "the
    solver gave up" is how a bad number gets into a figure.
    """
    rows = []
    failures = 0
    total = len(SITES) * len(SALTS) * len(ss.SCENARIOS) * 2
    done = 0
    for site, _lat, _lon in SITES:
        for salt in SALTS:
            for scenario in ss.SCENARIOS:
                for floor_mode in ("hydrate", "drh"):
                    done += 1
                    try:
                        water, eta, _abs_res, _des_res = run_daily_cycle(
                            profiles[site], _config(salt, scenario, floor_mode)
                        )
                        rows.append({
                            "site": site, "salt": salt, "scenario": scenario,
                            "c_w_floor_mode": floor_mode,
                            "yield_kg_m2": float(water), "eta_thermal": float(eta),
                            "error": "",
                        })
                    except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
                        failures += 1
                        rows.append({
                            "site": site, "salt": salt, "scenario": scenario,
                            "c_w_floor_mode": floor_mode,
                            "yield_kg_m2": float("nan"), "eta_thermal": float("nan"),
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        print(f"  FAILED {site}/{salt}/{scenario}/{floor_mode}: "
                              f"{type(exc).__name__}: {exc}", flush=True)
                        traceback.print_exc(limit=2)
                    if done % 10 == 0:
                        print(f"  yield matrix {done}/{total}", flush=True)
    print(f"yield matrix complete: {total - failures}/{total} succeeded, {failures} failed")
    return pd.DataFrame(rows)


def _report(drh: pd.DataFrame, yields: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("\n=== c_r vs temperature-dependent DRH, desorption phase (scenario 2, hydrate floor) ===")
    print(drh[[
        "site", "salt", "yield_kg_m2", "pct_duration_c_r_below_drh",
        "pct_yield_c_r_below_drh", "pct_duration_a_w_pinned_at_drh",
        "pct_yield_a_w_pinned_at_drh", "mean_deficit_when_below", "max_deficit",
        "mean_c_r", "mean_drh_at_t_gel", "drh_at_25c", "max_t_gel_c",
        "c_w_end", "hydrate_floor_c_w", "drh_floor_c_w",
    ]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Yield by salt (scenario 2, hydrate floor) ===")
    base = yields[(yields.scenario == "improved") & (yields.c_w_floor_mode == "hydrate")]
    print(base.pivot(index="site", columns="salt", values="yield_kg_m2").to_string(
        float_format=lambda v: f"{v:.4f}"))

    print("\n=== Cost of pinning at DRH: yield_drh / yield_hydrate, by scenario ===")
    piv = yields.pivot_table(
        index=["salt", "scenario"], columns="c_w_floor_mode", values="yield_kg_m2",
        aggfunc="mean",
    )
    piv["ratio_drh_over_hydrate"] = piv["drh"] / piv["hydrate"]
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== Same ratio, by site (mean over scenarios) ===")
    piv2 = yields.pivot_table(
        index=["site", "salt"], columns="c_w_floor_mode", values="yield_kg_m2", aggfunc="mean",
    )
    piv2["ratio_drh_over_hydrate"] = piv2["drh"] / piv2["hydrate"]
    print(piv2.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    raise SystemExit(main())
