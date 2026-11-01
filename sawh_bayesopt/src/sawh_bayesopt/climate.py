"""Climate featurization and location sampling.

A day is its hourly [T, RH, GHI] cycle (72 numbers) projected onto a PCA fitted over every
cached day on Earth, plus scalars the decomposition can under-weight. Only T, RH and GHI:
those are the three channels the physics consumes. Wind is fetched but h_amb is fixed at
10 W/m2K (solar_lumped.weather._resample_phase), and nothing radiates to a sky temperature,
so wind or T_sky features would be variance a surrogate can only overfit.

A cell (location) is summarized by the annual mean and std of its day features plus its
seasonal RH amplitude; sampling, nearest-neighbour and out-of-distribution questions are
all asked in that standardized descriptor space.

The cell set is every 2024 response already in the Open-Meteo requests-cache sqlite --
~14.7k sites, read without network. Frames are keyed by their cache key so a selected cell
can be re-read later without re-scanning the 22 GB file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Column order of a day's hourly block, and of the (n_days, 72) matrix built from it.
CHANNELS: tuple[str, ...] = ("temperature_2m", "relative_humidity_2m", "shortwave_radiation")
HOURS = 24
YEAR = 2024
N_DAYS = 366  # 2024 is a leap year; missing days stay NaN / invalid
# Retained principal components. The previous campaign's validate-features run simulated
# real years against years rebuilt from k components (featval_summary.csv): at k = 20 the
# median annual-yield error was 0.5-3% in every regime and the 90th percentile fell below
# 13% everywhere, where k = 12 still left coastal-humid and high-altitude p90 above 40%.
N_COMPONENTS = 20

# Physically motivated per-day scalars appended to the PCs. The last two are geometry, not
# weather: tilt only matters through the sun's path, which the GHI shape alone cannot tell
# a surrogate (same GHI at 5 deg vs 45 deg latitude transposes very differently).
SCALARS: tuple[str, ...] = (
    "rh_at_tmin_pct",
    "ghi_daily_kwh_m2",
    "dewpoint_depression_at_peak_sun_c",
    "t_diurnal_amplitude_c",
    "elevation_km",
    "abs_latitude_deg",
    "noon_zenith_deg",
)

# Koppen-Geiger classes for every cached cell (Rubel et al. 2017, doi:10.1127/metz/2016/0816,
# via kgcpy's 100 arc-second raster; built by analysis/two_stage/koppen_labels.py). Koppen
# needs monthly precipitation, which the weather cache does not have, so classes are looked
# up rather than computed.
KOPPEN_CSV = Path(__file__).resolve().parents[2] / "data" / "koppen.csv"
DESERT_CLASSES: tuple[str, ...] = ("BWh", "BWk")


def select_cells(descriptors: np.ndarray, koppen_group: np.ndarray, koppen_class: np.ndarray, *, n_space_filling: int,
                 n_desert: int, split_fracs: tuple[float, float, float], seed: int) -> tuple[pd.DataFrame, dict]:
    """The training locations: a space-filling subset of every cell, then extra desert cells.

    Both stages are greedy subset selection (Kochenderfer & Wheeler, Algorithms for
    Optimization, 2nd ed., algorithm 16.8) on standardized descriptors: add, one at a time,
    the cell that most reduces the largest distance from any cell to its nearest selected
    cell (eq. 16.4). The desert stage runs the same greedy over the desert cells only
    (DESERT_CLASSES), seeded with the deserts the first stage already took, so the extra
    cells fill the gaps in desert climate space rather than piling onto its densest part.

    The split is by cell, stratified by Koppen group: a cell's rows are thousands of
    correlated days of one climate, and a row split would leak each climate across it.
    Returns the selection table and both stages' coverage curves (d_max after each pick).
    """
    rng = np.random.default_rng(seed)
    z = (descriptors - descriptors.mean(axis=0)) / (descriptors.std(axis=0) + 1e-12)
    chosen, curve_all = greedy_maximin_subset(z, n_space_filling, rng=rng)
    desert = np.flatnonzero(np.isin(koppen_class, DESERT_CLASSES))
    seed_local = np.flatnonzero(np.isin(desert, chosen))
    picked_local, curve_desert = greedy_maximin_subset(z[desert], len(seed_local) + n_desert, rng=rng,
                                                       initial=seed_local)
    extra = desert[picked_local[len(seed_local):]]

    sel = pd.DataFrame({"cell": np.r_[chosen, extra],
                        "stage": ["space_filling"] * len(chosen) + ["desert"] * len(extra)})
    sel["koppen"] = koppen_class[sel["cell"]]
    sel["koppen_group"] = koppen_group[sel["cell"]]
    sel["split"] = "train"
    f_train, f_val, _f_test = split_fracs
    for _group, members in sel.groupby("koppen_group"):
        order = rng.permutation(members.index.to_numpy())
        n_train = int(round(f_train * len(order)))
        n_val = int(round(f_val * len(order)))
        sel.loc[order[n_train:n_train + n_val], "split"] = "val"
        sel.loc[order[n_train + n_val:], "split"] = "test"
    curves = {"space_filling_d_max": curve_all, "desert_d_max": curve_desert[len(seed_local):]}
    return sel, curves


def greedy_maximin_subset(X: np.ndarray, m: int, *, rng: np.random.Generator, initial: np.ndarray | None = None,
                          col_chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Kochenderfer & Wheeler algorithm 16.8 with d = d_max (eq. 16.4), exactly: S starts
    from ``initial`` (or one random point), and each step adds the x not in S that minimizes
    max_i min(nearest_i, |X_i - x|). Keeping nearest_i (distance from X_i to S) makes a step
    one pass over the distance matrix instead of a recomputation of d_max per candidate.

    Returns the m selected row indices in pick order and d_max(X, S) after each pick.
    O(n^2) memory for the distance matrix (0.9 GB at the full 14.7k cells, float32).
    """
    from scipy.spatial.distance import cdist

    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    D = cdist(X, X).astype(np.float32)
    S = list(np.asarray(initial, int)) if initial is not None and len(initial) else [int(rng.integers(n))]
    nearest = D[:, S].min(axis=1)
    curve = []
    for k in range(m):
        if k < len(S):
            curve.append(float(nearest.max()) if k == len(S) - 1 else np.nan)
            continue
        best_val, best_x = np.inf, -1
        for a in range(0, n, col_chunk):
            # d_max if column x joined S, for every x in this chunk of columns.
            vals = np.minimum(nearest[:, None], D[:, a:a + col_chunk]).max(axis=0)
            vals[np.isin(np.arange(a, min(a + col_chunk, n)), S)] = np.inf
            j = int(np.argmin(vals))
            if vals[j] < best_val:
                best_val, best_x = float(vals[j]), a + j
        S.append(best_x)
        nearest = np.minimum(nearest, D[:, best_x])
        curve.append(float(nearest.max()))
    return np.array(S[:m]), np.array(curve[:m])


def koppen_for(lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    """Koppen class and group for each (lat, lon), matched to KOPPEN_CSV's cells. Raises if
    any cell has no entry within 1e-3 degrees: a silent nearest match across the globe would
    label a cell with someone else's climate."""
    from scipy.spatial import cKDTree

    table = pd.read_csv(KOPPEN_CSV)
    dist, idx = cKDTree(table[["lat", "lon"]].to_numpy()).query(np.column_stack([lat, lon]))
    if (dist > 1e-3).any():
        raise ValueError(f"{int((dist > 1e-3).sum())} cell(s) have no Koppen entry in {KOPPEN_CSV}; "
                         "regenerate it with analysis/two_stage/koppen_labels.py")
    return table.iloc[idx][["koppen", "koppen_group"]].reset_index(drop=True)


def featurize_cache(cache_dir: str | Path, out_path: str | Path, *, pca_fit_days: int = 500_000,
                    seed: int = 0) -> dict:
    """Scan the requests-cache once and write every 2024 cell's per-day features.

    Writes ``out_path`` (npz): lat, lon, elevation_m, cache_key, valid (n, 366),
    pc_scores (n, 366, 72) -- ALL components, sliced to N_COMPONENTS on load --
    scalars (n, 366, len(SCALARS)), month (366,), and the per-cell summary stats the
    descriptors use. The PCA bundle goes next to it as ``pca.npz``.

    ponytail: one serial pass over ~15k JSON blobs (~22 GB); parallelize the parse with a
    process pool if this becomes a bottleneck -- it runs once per cache.
    """
    # Only the day matrix (float32) and three site scalars survive the scan: holding every
    # frame would be ~30 GB at 15k cells.
    rows = []
    for key, df in iter_cached_frames(cache_dir):
        matrix, dates = daily_weather_matrix(df)
        if len(dates) < 300:  # a partial year cannot describe a climate
            continue
        day_idx = np.array([d.timetuple().tm_yday - 1 for d in dates])
        site = tuple(float(df[c].iloc[0]) for c in ("latitude", "longitude", "elevation_m"))
        rows.append((key, site, matrix.astype(np.float32), day_idx))
        if len(rows) % 500 == 0:
            print(f"  featurized {len(rows)} cells", flush=True)
    if not rows:
        raise ValueError(f"no usable {YEAR} responses in {cache_dir}")

    # PCA fit on an equal random share of days from every cell.
    rng = np.random.default_rng(seed)
    per_cell = max(1, pca_fit_days // len(rows))
    pca = fit_climate_pca(np.concatenate([
        m[rng.choice(len(m), size=min(per_cell, len(m)), replace=False)] for _k, _s, m, _d in rows
    ]).astype(float))

    n = len(rows)
    scores = np.full((n, N_DAYS, 3 * HOURS), np.nan, dtype=np.float32)
    scalars = np.full((n, N_DAYS, len(SCALARS)), np.nan, dtype=np.float32)
    valid = np.zeros((n, N_DAYS), dtype=bool)
    meta = {k: np.zeros(n) for k in ("lat", "lon", "elevation_m", "mean_rh_pct",
                                     "rh_seasonal_amplitude_pct", "t_diurnal_amplitude_c")}
    keys = []
    for i, (key, (lat, lon, elev), matrix, day_idx) in enumerate(rows):
        matrix = matrix.astype(float)
        scores[i, day_idx] = project(pca, matrix)
        scalars[i, day_idx] = day_scalars(matrix, day_idx, lat=lat, elevation_m=elev)
        valid[i, day_idx] = True
        keys.append(key)
        rh = matrix[:, HOURS:2 * HOURS]
        t = matrix[:, :HOURS]
        month = pd.DatetimeIndex(pd.Timestamp(f"{YEAR}-01-01") + pd.to_timedelta(day_idx, "D")).month
        monthly_rh = pd.Series(rh.mean(axis=1)).groupby(month).mean()
        meta["lat"][i], meta["lon"][i], meta["elevation_m"][i] = lat, lon, elev
        meta["mean_rh_pct"][i] = rh.mean()
        meta["rh_seasonal_amplitude_pct"][i] = monthly_rh.max() - monthly_rh.min()
        meta["t_diurnal_amplitude_c"][i] = (t.max(axis=1) - t.min(axis=1)).mean()

    month = pd.date_range(f"{YEAR}-01-01", periods=N_DAYS, freq="D").month.to_numpy()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, cache_key=np.array(keys), valid=valid, pc_scores=scores, scalars=scalars,
             month=month, **meta)
    np.savez(out_path.with_name("pca.npz"), **pca)
    return {"n_cells": n}


def iter_cached_frames(cache_dir: str | Path):
    """(cache key, weather frame) for every cached 2024 response, parsed exactly as
    WeatherClient would have returned it. 15-minute data is preferred when a response
    carries it (historical-forecast API); archive responses are hourly."""
    import requests_cache

    session = requests_cache.CachedSession(cache_name=str(Path(cache_dir) / "openmeteo_cache"),
                                           backend="sqlite")
    seen: dict[tuple[float, float], str] = {}
    for key in list(session.cache.responses.keys()):
        df = frame_from_cache(session, key)
        if df is None:
            continue
        site = (round(float(df["latitude"].iloc[0]), 4), round(float(df["longitude"].iloc[0]), 4))
        if site in seen:  # one frame per site; the cache can hold both APIs' answers
            continue
        seen[site] = key
        yield key, df


def frame_from_cache(session_or_dir, key: str) -> pd.DataFrame | None:
    """One cached response as a weather frame, or None if it is not a usable 2024 year.
    Accepts an open CachedSession (bulk scans) or the cache directory (one-off reads)."""
    from solar_lumped.weather import WeatherClient

    if not hasattr(session_or_dir, "cache"):
        import requests_cache

        session_or_dir = requests_cache.CachedSession(
            cache_name=str(Path(session_or_dir) / "openmeteo_cache"), backend="sqlite")
    response = session_or_dir.cache.responses.get(key)
    if response is None:
        return None
    try:
        data = json.loads(response.content)
    except (ValueError, UnicodeDecodeError):
        return None
    series_key = "minutely_15" if data.get("minutely_15") else "hourly"
    block = data.get(series_key) or {}
    if not all(c in block for c in CHANNELS) or not block.get("time"):
        return None
    if not str(block["time"][0]).startswith(str(YEAR)):
        return None
    df = WeatherClient._series_to_dataframe(
        data, series_key, float(data["latitude"]), float(data["longitude"]))
    df[list(CHANNELS)] = df[list(CHANNELS)].astype(float).interpolate(limit_direction="both")
    return df


def daily_weather_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    """(n_days, 72) hourly [T | RH | GHI] per complete calendar day, and those days' dates.

    Local clock time, the frame's own fixed-offset index. ponytail: not solar time, so a
    site at the edge of its time zone has its cycle shifted up to ~1-2 h against the pooled
    PCA basis, which costs a component or two; realign on lon/utc_offset if that matters.
    """
    hourly = df[list(CHANNELS)].resample("1h").mean()
    hourly = hourly[hourly.index.year == YEAR]
    hourly["date"] = hourly.index.date
    hourly["hour"] = hourly.index.hour
    blocks = [hourly.pivot(index="date", columns="hour", values=c).reindex(columns=range(HOURS))
              for c in CHANNELS]
    wide = pd.concat(blocks, axis=1).dropna()  # drops any day missing an hour
    return wide.to_numpy(dtype=float), list(wide.index)


def fit_climate_pca(matrix: np.ndarray) -> dict:
    """PCA over standardized day matrices. Standardized per *channel* (one mean/std for all
    24 hours of T, etc.), not per column, so the basis keeps the diurnal shape physical and
    a reconstruction is plain arithmetic. Full rank: the retained count is a later slice."""
    from sklearn.decomposition import PCA

    chan_mean = np.array([matrix[:, i * HOURS:(i + 1) * HOURS].mean() for i in range(3)])
    chan_std = np.array([matrix[:, i * HOURS:(i + 1) * HOURS].std() for i in range(3)])
    pca = PCA().fit(_standardize(matrix, chan_mean, chan_std))
    return {"chan_mean": chan_mean, "chan_std": chan_std, "mean": pca.mean_,
            "components": pca.components_, "explained_variance_ratio": pca.explained_variance_ratio_}


def _standardize(matrix, chan_mean, chan_std):
    return (matrix - np.repeat(chan_mean, HOURS)) / np.repeat(chan_std, HOURS)


def project(pca: dict, matrix: np.ndarray) -> np.ndarray:
    """All PC scores of each day row."""
    z = _standardize(matrix, pca["chan_mean"], pca["chan_std"]) - pca["mean"]
    return z @ pca["components"].T


def day_scalars(matrix: np.ndarray, day_idx: np.ndarray, *, lat: float, elevation_m: float) -> np.ndarray:
    """SCALARS for each day row, in SCALARS order."""
    t, rh, ghi = matrix[:, :HOURS], matrix[:, HOURS:2 * HOURS], matrix[:, 2 * HOURS:]
    rows = np.arange(len(matrix))
    peak = ghi.argmax(axis=1)
    t_peak, rh_peak = t[rows, peak], np.clip(rh[rows, peak], 1.0, 100.0)
    # Magnus dewpoint
    gamma = np.log(rh_peak / 100.0) + 17.62 * t_peak / (243.12 + t_peak)
    dewpoint = 243.12 * gamma / (17.62 - gamma)
    declination = 23.44 * np.sin(2.0 * np.pi * (284 + day_idx + 1) / 365.0)
    return np.column_stack([
        rh[rows, t.argmin(axis=1)],
        ghi.sum(axis=1) / 1000.0,
        t_peak - dewpoint,
        t.max(axis=1) - t.min(axis=1),
        np.full(len(matrix), elevation_m / 1000.0),
        np.full(len(matrix), abs(lat)),
        np.abs(lat - declination),
    ])


def load_features(path: str | Path) -> dict:
    """features.npz with the PC block sliced to N_COMPONENTS and joined to the scalars, the
    missing days filled, and each cell's Koppen class attached:

      day_features (n_cells, 366, N_COMPONENTS + len(SCALARS)), every day filled -- a
      missing day takes the nearest earlier valid day's features, then the nearest later
      one, so the inner loop can walk all 366 calendar days for every cell;
      descriptors (n_cells, 2F + 1), from the valid days only;
      koppen, koppen_group (n_cells,).
    """
    z = dict(np.load(path, allow_pickle=False))
    raw = np.concatenate([z["pc_scores"][..., :N_COMPONENTS], z["scalars"]], axis=-1)
    masked = np.where(z["valid"][..., None], raw, np.nan)
    z["descriptors"] = np.column_stack([np.nanmean(masked, axis=1), np.nanstd(masked, axis=1),
                                        z["rh_seasonal_amplitude_pct"]])
    n, d, f = masked.shape
    flat = pd.DataFrame(masked.transpose(1, 0, 2).reshape(d, n * f)).ffill().bfill()
    z["day_features"] = flat.to_numpy(dtype=np.float32).reshape(d, n, f).transpose(1, 0, 2)
    kg = koppen_for(z["lat"], z["lon"])
    z["koppen"], z["koppen_group"] = kg["koppen"].to_numpy(), kg["koppen_group"].to_numpy()
    return z


def mahalanobis_ood(train_desc: np.ndarray, desc: np.ndarray, *, quantile: float = 0.99
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """(squared distance, is_ood, threshold) for every row of ``desc`` against the
    training cells' descriptor distribution. The threshold is the training cells' own
    ``quantile`` -- in-sample, so it errs toward flagging more cells, not fewer."""
    from sklearn.covariance import EmpiricalCovariance

    mu, sd = train_desc.mean(axis=0), train_desc.std(axis=0) + 1e-12
    cov = EmpiricalCovariance().fit((train_desc - mu) / sd)
    d2 = cov.mahalanobis((desc - mu) / sd)
    threshold = float(np.quantile(cov.mahalanobis((train_desc - mu) / sd), quantile))
    return d2, d2 > threshold, threshold
