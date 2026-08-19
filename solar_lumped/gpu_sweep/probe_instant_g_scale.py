#!/usr/bin/env python3
"""Is _INSTANT_EQUILIBRIUM_G_SCALE (parameters.xlsx: 1e6) bigger than it needs to be?

The instant-equilibrium limit multiplies g by that factor so the residual driving force
sits far below the ODE tolerance -- the workbook itself calls the value "numerical, not
physical". But it also makes the c_w relaxation stiff, and explicit Tsit5 pays for that
in tiny steps: measured on the serc A100, the instant groups cost ~1.0 s per
instance-day against ~0.023 s for finite g, a 45x penalty that puts the full grid at
~430 GPU-hours.

If the yield is already converged at a smaller factor, that penalty is mostly
avoidable. This runs the same sites and days at several factors and reports both the
yield and the cost, so the choice is a measurement:

    python3 gpu_sweep/probe_instant_g_scale.py            # 3 sites, 20 days, 4 factors

Both backends read the constant from one place, but jax_physics binds it at import, so
each factor is patched into both modules -- and the step function is rebuilt per factor,
which is what forces a retrace rather than reusing the previous factor's compiled code.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

import pandas as pd  # noqa: E402

import solar_lumped.physics as physics  # noqa: E402
import jax_physics  # noqa: E402
import run_gpu_sweep as driver  # noqa: E402
from solar_lumped.weather import WeatherClient  # noqa: E402

SCALES = (1e4, 1e5, 1e6)  # 1e5 is now the workbook value; 1e6 is the old reference
SITES, DAYS, SCENARIO = 20, 20, "improved_instant_g"


def main() -> int:
    out_dir = _HERE.parent / "outputs" / "gpu_scenario_sweep" / "probe"
    args = driver.parse_args([
        "--num-sites", str(SITES), "--scenarios", SCENARIO,
        "--max-days", str(DAYS), "--max-rounds", "1", "--progress-every", "0",
        "--output-csv", str(out_dir / "unused.csv"),
    ])
    client = WeatherClient(cache_dir=args.cache_dir)
    loaded = [
        w for lat, lon in driver._site_list(args)
        if (w := driver._load_site(lat, lon, args, client)) is not None
    ]
    n_days = min(DAYS, min(len(w.days) for w in loaded))
    instances = [(w, SCENARIO) for w in loaded]
    print(f"{len(instances)} instance(s) x {n_days} day(s) per factor", flush=True)

    rows = []
    for scale in SCALES:
        # Both bindings, or the JAX path silently keeps the workbook value.
        physics._INSTANT_EQUILIBRIUM_G_SCALE = scale
        jax_physics._INSTANT_EQUILIBRIUM_G_SCALE = scale
        csv = out_dir / f"g_scale_{scale:.0e}.csv"
        csv.unlink(missing_ok=True)
        args.output_csv = csv
        t0 = time.perf_counter()
        driver.run_group(instances, n_days, args, instant_equilibrium=True, condenser_ambient=False)
        elapsed = time.perf_counter() - t0
        d = pd.read_csv(csv).sort_values(["lat", "lon"])
        rows.append({"scale": scale, "s_per_day": elapsed / n_days,
                     "yields": d.mean_yield_kg_m2.to_numpy()})
        print(f"  g x {scale:.0e}: {elapsed / n_days:6.2f} s/day  "
              f"yields {[f'{y:.4f}' for y in rows[-1]['yields']]}", flush=True)

    ref = rows[-1]
    print(f"\nvs the current {ref['scale']:.0e}:")
    for r in rows:
        drift = max(abs(r["yields"] / ref["yields"] - 1.0)) * 100
        print(f"  g x {r['scale']:.0e}: worst yield drift {drift:7.4f}%   "
              f"cost {r['s_per_day'] / ref['s_per_day']:.3f}x")
    print("\nPick the smallest factor whose drift is below the noise you care about "
          "(the ODE runs at rtol=1e-4, so <0.01% is not distinguishable from it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
