# Session summary: solar_lumped global sweep + Sherlock migration

Reference doc for everything done in this session. See also
[`sherlock_param_sweep.tex`](sherlock_param_sweep.tex) for the formal written plan —
**note**: its "Job structure" section describes an earlier (single large `sbatch --array`)
submission design that turned out not to work on this cluster. The real, working mechanism
is the `topup_sweep.sh` retry-loop described below; the `.tex` doc was not updated after
that pivot.

## Where things stand

A 1405-site × 135-combo device-parameter sweep is running on Sherlock via a self-submitting
background script (`topup_sweep.sh`), started under `nohup`+`disown` on the login node
(`sh03-ln02`), writing to `outputs/grid_sweep/site_*_chunk_*.csv`. It survives SSH disconnects
and stray `Ctrl+C` (those only affect whatever's in your terminal foreground, e.g. a `tail -f`).
Expected total runtime: **hours to a few days**, gated by a per-user Slurm submission cap (see
below), not by compute capacity.

To check on it later:
```bash
ssh claraod@login.sherlock.stanford.edu
cd /home/groups/cdiazm/SAWH_TEAs/solar_lumped
tail -f topup_sweep2.log            # live progress
squeue -u $USER -r | wc -l           # current true outstanding task count
ls outputs/grid_sweep/*.csv | wc -l  # completed chunk files (target: 63225)
```

Once it's done (or "done enough"), merge everything into one dataset:
```bash
n=$(ls outputs/grid_sweep/site_*_chunk_*.csv 2>/dev/null | wc -l)
echo "found $n of 63225 expected chunk files"
{ head -1 "$(ls outputs/grid_sweep/site_*_chunk_*.csv | head -1)"
  for f in outputs/grid_sweep/site_*_chunk_*.csv; do tail -n +2 "$f"; done
} > outputs/grid_sweep_full.csv
```
Max possible rows: 189,675 (1405 sites × 135 combos). This is **yield and thermal efficiency
only — no LCOW/economics**; that would be a separate downstream calculation from
`mean_yield_kg_m2`, the same way `lcow_full_global_map.py` derives LCOW from yield.

## What the sweep actually computes

**Goal**: build a map that needs to be accurate *everywhere*, not just find winning sites —
this is the reasoning behind several choices below that favor accuracy over raw speed.

- **Grid**: 3° lat/lon, land points only (Natural Earth 110m polygons), lat ∈ [-54, 72],
  **minus** Canada and Russia above 60°N (excluded by country polygon, not a blunt latitude
  line — Alaska/Greenland/Iceland/Scandinavia's far north are unaffected). **1405 sites total.**
- **Swept parameters** (4 axes, 135 combinations):

  | Parameter | Values | Baseline |
  |---|---|---|
  | Hydrogel thickness (mm) | 1.0, 3.25, 5.5, 7.75, 10.0 | 4.0 |
  | Absorber emissivity (`eps_abs`) | 0.85, 0.90, 0.95 | 0.95 |
  | Glass transmittance (`tau_glass`) | 0.80, 0.85, 0.90 | 0.90 |
  | Condenser fin area ratio | 3.0, 7.1, 12.0 | 7.1 |

  Vapor gap was considered as a 5th axis but fixed at 40mm instead (not swept) to cut the
  combo count by 3x (405 → 135).
- **Fixed**: salt = LiCl only (no fallback to other salts), salt:polymer ratio 4.0, insulation
  gap 5mm, tilt 35°, weather year 2024, `h_amb` (ambient convection coefficient) **fixed at
  10 W/m²K** (was wind-derived from Open-Meteo's `wind_speed_10m`, changed to a flat constant
  per explicit request).
- **Time resolution: monthly** (12 Aitken-converged representative mean days, day-weighted),
  not single-mean-day, not weekly, not full-annual. See validation table below for why.
- **Output columns**: `lat, lon, mean_rh_frac, mean_t_amb_c, mean_solar_w_m2, salt,
  hydrogel_thickness_mm, eps_abs, tau_glass, fin_area_ratio, vapor_gap_mm, warmup_method,
  resolution, mean_yield_kg_m2, mean_eta_thermal, n_periods`. The `mean_*` weather columns are
  site properties (identical across all 135 rows for a site) computed once per site at ~zero
  extra cost. `mean_solar_w_m2` is averaged over the desorption (daylight) phase only — blending
  in the absorption (night) phase would dilute it with ~12h of near-zero values.

## Key finding: why monthly, not single-mean-day

Validated by running the real day-by-day sequential year (365/366 days, gel state carried
forward) against single-mean-day and monthly-mean-day shortcuts, at three test climates:

| Site | Annual (kg/m²) | 1 mean-day | 12 months | 52 weeks |
|---|---|---|---|---|
| Patagonian coast (-50, -75), high seasonal variance | 0.3809 | 0.3010 (-21.0%) | 0.3287 (-13.7%) | 0.3321 (-12.8%) |
| Atacama desert (-23.6, -70.4), low seasonal variance | 1.9052 | 1.9018 (-0.2%) | 1.9011 (-0.2%) | 1.9007 (-0.2%) |
| Cambridge, MA (42.36, -71.09), moderate seasonal variance | 0.9032 | 0.7393 (-18.1%) | 0.8606 (-4.7%) | not run |

Single mean-day is off by double digits exactly at the high-seasonal-variance sites where a
map most needs to be trustworthy. Monthly closes most of that gap for 12x the compute (the
right price for a map product); weekly buys almost nothing further for another 4.3x on top.
Full annual is ~33x monthly's cost — reserve it for spot-checking a specific site later via
`compare_annual_vs_mean_day.py`, not the whole grid.

## Numerics: `find_cyclic_state` and the 2-cycle bifurcation

`ode_system.py`'s `find_cyclic_state` finds the steady periodic post-desorption state via
restarted vector Aitken Δ² extrapolation (typically 3-6 rounds) instead of brute-force
fixed-cycle warmup (which could need 100+ cycles at strongly seasonal sites, or silently return
an under-converged state if capped too low). Some (profile, config) pairs have no single fixed
point — the one-cycle map bifurcates into a stable period-2 orbit (an alternating
wetter-day/drier-day pattern). This is detected as two consecutive rounds where the relative
step fails to shrink by at least half, handled by returning the average of the two alternating
states (logged with a one-line warning) instead of an arbitrary snapshot. The full sweep's logs
give a free count of how often the grid actually hits this — worth checking after the fact.

## Sherlock environment notes

- Account `claraod`, group `cdiazm` (Diaz-Marin group), repo at
  `/home/groups/cdiazm/SAWH_TEAs/solar_lumped`. Unix groups: `cdiazm sh_o-serc sh_s-dss sh_users`.
- `uv pip install cartopy pyproj ...` (no flag) tries to **build pyproj from source** and fails
  (`proj executable not found`) — the resolved versions' wheels target a newer glibc than this
  cluster's nodes ship. Fix: `uv pip install --only-binary :all: numpy scipy pandas
  requests-cache retry-requests shapely cartopy` — resolves to slightly older, wheel-compatible
  versions (cartopy 0.24.1, pyproj 3.7.1) with no compilation needed.
- Build venvs / run anything heavier than trivial on `sh_dev` (interactive compute node
  session), not the login node.
- **Sherlock runs this workload ~2.66x slower per combo than a 2023 M3 Pro MacBook** (measured:
  1011.1s/859.9s on Sherlock vs 386.9s/318.2s locally for identical combos). Always calibrate
  cost estimates against the actual target hardware before sizing a real job — a laptop
  benchmark alone understated the real cost by exactly this factor.
- **No GPU path**: this code is scalar SciPy (`solve_ivp`, Radau/BDF) — single-threaded CPU
  work. A100s are irrelevant to this workload.
- Weather cache: originally planned to build locally and `rsync` to Sherlock, but home-network
  upload bandwidth (~1.3 MB/s measured) made a 21GB transfer impractically slow (and it dropped
  mid-transfer once). Switched to running `prefetch_weather.py` directly on Sherlock instead —
  no reason to build the cache locally and ship it when Sherlock's own network link to
  Open-Meteo has nothing to do with a laptop's home upload speed.

## The Slurm submission saga (read this before resubmitting anything)

Sizing the array job took three wrong turns before landing on a working design:

1. **`--array=0-63224%2000` rejected outright**: `sbatch: error: Invalid job array specification`.
   Bisected the real ceiling empirically (100/500/1000 succeeded, 5000/10000 failed) — looked
   at first like a dynamic cluster-wide job-count limit.
2. **Real cause, found via `sacctmgr show qos format=name,MaxSubmitJobsPerUser`**: the `owners`
   QOS caps outstanding submitted jobs (pending+running, **each array task counted
   individually**) at **3000 per user** — completely unrelated to the `MaxArraySize=1,000,000`
   global config (checked, not the bottleneck) or `dev` QOS's limit of 2 (that's for `sh_dev`
   interactive sessions specifically, a different partition entirely).

   Full QOS table for reference:
   | QOS | MaxSubmitJobsPerUser |
   |---|---|
   | normal | 2000 |
   | dev | 2 |
   | long | 20 |
   | bigmem | 10 |
   | gpu | 100 |
   | owner / owners | **3000** |
   | high_p | 100 |
   | service | 4 |
   | system | 99999 |

3. **Fix, attempt 1 (broken)**: batch submissions into chunks of 1000 tasks, with a script that
   checks `squeue -u $USER -h | wc -l` before submitting each new batch and waits if too high.
   Failed silently — Slurm **collapses pending array ranges into one display line**
   (e.g. `34516315_[31-999%32]`), so `wc -l` massively undercounts. Added `-r`/`--array` to
   un-collapse — **still failed** (root cause of that second failure not fully pinned down,
   possibly a propagation-delay race between `sbatch` returning and `squeue` reflecting it).

4. **Fix, attempt 2 (working, current design)**: stopped trying to *predict* headroom via a
   separate query entirely. Instead, `topup_sweep.sh` just **tries** each batch's `sbatch` call
   for real and only advances once Slurm's own output confirms `"Submitted batch job ..."` — on
   failure, it retries the *same* batch every 120s indefinitely. This can't race through
   incorrectly because it checks the actual authoritative outcome, not a query that can be stale.
   Verified via a mocked-`sbatch` integration test (fails N times then succeeds) before deploying.

### Final job structure

- 135 combos/site split into **45 chunks of 3** (not the originally-planned 9 — the 2.66x
  Sherlock slowdown made 9-combo chunks risk multi-hour single tasks, bad for `owners`
  preemption exposure). `--combo-offset`/`--combo-limit` (added to `grid_param_sweep.py`) slice
  a site's combo grid.
- 1405 sites × 45 chunks = **63,225 total array tasks**, submitted as **64 separate `sbatch`
  calls of ≤1000 tasks each** (not one giant array) by `topup_sweep.sh`, because of the QOS cap.
- Each `sbatch` batch: `--time=05:00:00`, `--partition=owners`, `%32` internal throttle,
  `--cpus-per-task=1`, `--mem-per-cpu=4G`.
- Output: one CSV per (site, chunk) — **not** one shared file per site, because multiple chunks
  of the same site can run concurrently on different nodes and concurrent header-writes to one
  file would race. `--resume` on each task skips combos already written to its own chunk's file,
  so a preempted/retried task loses at most the one in-progress combo.
- Recalibrated cost: **~53,000–185,000 core-hours** total (up from an early
  Mac-only estimate of ~20,000–70,000, before the 2.66x Sherlock-slowdown finding).

### `topup_sweep.sh` (the actual working script)

```bash
#!/bin/bash
CHUNK_TOTAL=63225
BATCH_SIZE=1000
THROTTLE=32

for (( batch_start=0; batch_start<CHUNK_TOTAL; batch_start+=BATCH_SIZE )); do
  job_name="sawh-sweep-b${batch_start}"

  if squeue -u "$USER" -h -o "%j" | grep -qx "$job_name"; then
    echo "$(date): $job_name already queued, skipping"
    continue
  fi

  batch_end=$(( batch_start + BATCH_SIZE - 1 ))
  if (( batch_end >= CHUNK_TOTAL )); then batch_end=$(( CHUNK_TOTAL - 1 )); fi
  arr_max=$(( batch_end - batch_start ))

  while true; do
    output=$(sbatch <<EOF2
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --array=0-${arr_max}%${THROTTLE}
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=owners
GLOBAL_INDEX=\$(( ${batch_start} + SLURM_ARRAY_TASK_ID ))
N_CHUNKS_PER_SITE=45
CHUNK_SIZE=3
SITE_INDEX=\$(( GLOBAL_INDEX / N_CHUNKS_PER_SITE ))
CHUNK_INDEX=\$(( GLOBAL_INDEX % N_CHUNKS_PER_SITE ))
COMBO_OFFSET=\$(( CHUNK_INDEX * CHUNK_SIZE ))
source .venv/bin/activate
export PYTHONUNBUFFERED=True
python scripts/grid_param_sweep.py --site-index \$SITE_INDEX --step 3 --combo-offset \$COMBO_OFFSET --combo-limit \$CHUNK_SIZE --output-csv outputs/grid_sweep/site_\${SITE_INDEX}_chunk_\${CHUNK_INDEX}.csv --resume
EOF2
    2>&1)
    if echo "$output" | grep -q "^Submitted batch job"; then
      echo "$(date): submitted $job_name -- $output"
      break
    else
      echo "$(date): $job_name submission failed, retrying in 120s -- $output"
      sleep 120
    fi
  done
done

echo "$(date): all 64 batches submitted"
```
Launched via `nohup bash topup_sweep.sh > topup_sweep2.log 2>&1 & disown` on the login node —
survives SSH disconnects and stray `Ctrl+C` (nohup+disown detaches it from the controlling
terminal and the shell's job table entirely). To stop it deliberately:
```bash
pkill -u $USER -f topup_sweep.sh
squeue -u $USER -h -o "%A %j" | awk '$2 ~ /^sawh-sweep-b/ {print $1}' | xargs -r scancel
```

## Files changed this session (all pushed to `origin/main`)

| File | What changed |
|---|---|
| `scripts/lcow_full_global_map.py` | `ProcessPoolExecutor`-based multiprocessing (`--workers`); LiCl-only default salt with feasibility fallback; `grid_land_points` extracted out (see below) |
| `scripts/lcow_random_global_map.py` | LiCl-only default salt; `stop_at_first_feasible` option |
| `src/solar_lumped/weather/client.py` | Cache TTL bug fix: `expire_after` was 6 hours (found 14,632/14,634 cached responses already expired), changed to `NEVER_EXPIRE` since historical weather for a past year never changes |
| `src/solar_lumped/simulation/ode_system.py` | `find_cyclic_state` (Aitken-accelerated convergence) + 2-cycle bifurcation detection/averaging fallback |
| `scripts/compare_annual_vs_mean_day.py` | A/B/C/D comparison tool (full-annual vs single/monthly/weekly mean-day), used to produce the validation table above |
| `src/solar_lumped/weather/land_grid.py` | **New.** `grid_land_points`/`_prepared_land_union`/`_prepared_country_union` extracted out of `lcow_full_global_map.py` so headless callers (job-array workers, prefetch) don't need matplotlib just to build a grid. Includes the Canada/Russia >60°N country-polygon exclusion. |
| `scripts/grid_param_sweep.py` | **New.** The main sweep script: per-site full-factorial combo grid, `--resolution {single,monthly}`, `--combo-offset`/`--combo-limit` chunking, weather-stat columns |
| `scripts/prefetch_weather.py` | **New.** Single-process weather prefetch (~6x faster than a subprocess-per-site loop) with exponential-cooldown retry on Open-Meteo 429s |
| `src/solar_lumped/weather/profiles.py` | `h_amb` fixed at 10 W/m²K (was wind-derived) |
| `docs/sherlock_param_sweep.tex` | Written plan doc — **stale** re: final submission mechanics (see above) |

## Open items / things to revisit

- The `.tex` doc's job-structure section doesn't reflect the final `topup_sweep.sh` retry-based
  design — update it if the doc needs to stay authoritative.
- Never determined `serc`'s own QOS submit limit (not in the QOS table pulled — it likely maps
  to one of the listed QOSes rather than having its own). Could be worth routing some load there
  too if `owners` throughput becomes limiting, but unconfirmed.
- The root cause of the *second* headroom-check failure (even with `squeue -r`) was never fully
  nailed down — worked around by abandoning the predict-then-submit approach rather than fully
  diagnosed. If a similar counting issue crops up elsewhere, don't trust `squeue`-based
  pre-checks; prefer "try it and check the real result."
- Vapor gap and full-annual resolution were both deliberately dropped from the production sweep
  for cost reasons — both are easy to add back for a targeted re-check on specific sites
  (`grid_param_sweep.py --resolution single` swapped for a direct
  `compare_annual_vs_mean_day.py` call; vapor gap as a 5th swept axis) without resubmitting the
  whole grid.
