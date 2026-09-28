"""Climate featurization for the two-stage pipeline: each day of each cell becomes a compact
descriptor of its diurnal cycle, and cells are clustered, stratified and OOD-flagged in
that feature space.

A day is its hourly [T, RH, GHI] cycle (72 numbers) projected onto a PCA fitted over every
cached day on Earth, plus scalars the decomposition can under-weight. Only T, RH and GHI:
those are the three channels the physics consumes. Wind is fetched but h_amb is fixed at
10 W/m2K (solar_lumped.weather._resample_phase), and nothing radiates to a sky
temperature, so wind or T_sky features would be variance the surrogate can only overfit.

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

# Physically motivated per-day scalars appended to the PCs. The last two are geometry, not
# weather: tilt only matters through the sun's path, which the GHI shape alone cannot tell
# the surrogate (same GHI at 5 deg vs 45 deg latitude transposes very differently).
SCALARS: tuple[str, ...] = (
    "rh_at_tmin_pct",
    "ghi_daily_kwh_m2",
    "dewpoint_depression_at_peak_sun_c",
    "t_diurnal_amplitude_c",
    "elevation_km",
    "abs_latitude_deg",
    "noon_zenith_deg",
)

# Rule-labelled regimes, rare by area but decisive for the headline result, so selection
# oversamples them. First match wins, in this order.
REGIMES: tuple[str, ...] = ("high_altitude", "hyper_arid", "monsoonal", "coastal_humid", "other")


def featurize_cache(cache_dir: str | Path, out_path: str | Path, *, pca_fit_days: int = 500_000,
                    seed: int = 0) -> dict:
    """Scan the requests-cache once and write every 2024 cell's per-day features.

    Writes ``out_path`` (npz): lat, lon, elevation_m, cache_key, valid (n, 366),
    pc_scores (n, 366, 72) -- ALL components, so the retained count is chosen later
    (validate-features) by slicing -- scalars (n, 366, len(SCALARS)), month (366,), and
    the per-cell regime label and summary stats selection uses. The PCA bundle goes next
    to it as ``pca.npz``.

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

    regime = label_regimes(meta)
    month = pd.date_range(f"{YEAR}-01-01", periods=N_DAYS, freq="D").month.to_numpy()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, cache_key=np.array(keys), valid=valid, pc_scores=scores, scalars=scalars,
             month=month, regime=regime, **meta)
    np.savez(out_path.with_name("pca.npz"), **pca)
    return {"n_cells": n, "regime_counts": dict(zip(*np.unique(regime, return_counts=True)))}


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
    PCA basis, which costs a component or two; realign on lon/utc_offset if
    validate-features shows the retained count creeping up for that reason.
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


def reconstruct(pca: dict, scores: np.ndarray, n_components: int) -> np.ndarray:
    """Day matrices back from their leading ``n_components`` scores."""
    z = scores[:, :n_components] @ pca["components"][:n_components] + pca["mean"]
    return z * np.repeat(pca["chan_std"], HOURS) + np.repeat(pca["chan_mean"], HOURS)


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


def label_regimes(meta: dict) -> np.ndarray:
    """REGIMES label per cell from its annual summary stats. Thresholds are deliberately
    blunt -- they only steer oversampling and break out error reports, they do not enter
    the physics."""
    labels = np.full(len(meta["lat"]), "other", dtype=object)
    rules = {
        "high_altitude": meta["elevation_m"] > 2500.0,
        "hyper_arid": meta["mean_rh_pct"] < 25.0,
        # A >55-point swing in monthly-mean RH inside the (sub)tropics: ~5% of cached
        # cells. At 35 points with no latitude limit it caught 28%, mostly mid-latitude
        # continental seasonality -- no longer a tail.
        "monsoonal": (meta["rh_seasonal_amplitude_pct"] > 55.0) & (np.abs(meta["lat"]) < 35.0),
        "coastal_humid": (meta["mean_rh_pct"] > 75.0) & (meta["t_diurnal_amplitude_c"] < 6.0),
    }
    for name in reversed(REGIMES[:-1]):  # reversed so the first-listed rule wins
        labels[rules[name]] = name
    return labels.astype(str)


def load_features(path: str | Path, n_components: int) -> dict:
    """features.npz with the PC block sliced to ``n_components`` and joined to the scalars:
    ``day_features`` is (n_cells, 366, n_components + len(SCALARS))."""
    z = dict(np.load(path, allow_pickle=False))
    z["day_features"] = np.concatenate([z["pc_scores"][..., :n_components], z["scalars"]], axis=-1)
    z["n_components"] = n_components
    return z


def reconstruct_day_frame(df: pd.DataFrame, pca: dict, n_components: int) -> pd.DataFrame:
    """The frame with T/RH/GHI replaced by their ``n_components`` PCA reconstruction, at
    the frame's own time step. Site columns (lat/lon/utc_offset/elevation) are kept so the
    POA and physics paths run on it unchanged -- this is what validate-features simulates
    to check that the retained features carry the yield-relevant weather."""
    matrix, dates = daily_weather_matrix(df)
    rebuilt = reconstruct(pca, project(pca, matrix), n_components)
    # Hourly means are centred on the half hour; interpolate back onto the native index.
    centres = pd.DatetimeIndex([pd.Timestamp(d) + pd.Timedelta(hours=h + 0.5)
                                for d in dates for h in range(HOURS)])
    keep = pd.Index(df.index.date).isin(dates)
    out = df[keep].copy()
    # Wall clock on both sides: the frame index is tz-aware wherever the zone has no DST
    # (tz_localize succeeded), and asi8 of a tz-aware index is UTC, not local. as_unit
    # because pandas 3 indexes carry s/us/ns resolution and asi8 is in that unit.
    wall = out.index.tz_localize(None) if out.index.tz is not None else out.index
    x_new = wall.as_unit("ns").asi8.astype(float)
    x_src = centres.as_unit("ns").asi8.astype(float)
    for i, col in enumerate(CHANNELS):
        out[col] = np.interp(x_new, x_src, rebuilt[:, i * HOURS:(i + 1) * HOURS].reshape(-1))
    out["relative_humidity_2m"] = out["relative_humidity_2m"].clip(0.0, 100.0)
    # Sun below the horizon means no GHI -- geometry, not a feature. Without this the
    # hourly interpolation smears light into dawn/dusk and truncation ringing puts a few
    # W/m2 into the night; the physics splits absorption/desorption on GHI >= 5 W/m2, so
    # the pilot's rebuilt years ran 1-2.5 h longer lit days and yields 5-40% low at every
    # k. POA at zero tilt is exactly GHI by day and zero at night.
    from solar_lumped.weather import plane_of_array_w_m2

    out["shortwave_radiation"] = plane_of_array_w_m2(
        out["shortwave_radiation"].clip(lower=0.0).to_numpy(), out.index,
        latitude_deg=float(out["latitude"].iloc[0]), longitude_deg=float(out["longitude"].iloc[0]),
        utc_offset_h=float(out["utc_offset_s"].iloc[0]) / 3600.0, tilt_deg=0.0)
    return out


def cell_descriptors(day_features: np.ndarray, valid: np.ndarray, meta: dict) -> np.ndarray:
    """(n_cells, 2F + 1): annual mean and std of every day feature, plus the seasonal RH
    amplitude (which the annual mean/std of PCs can blur)."""
    masked = np.where(valid[..., None], day_features, np.nan)
    return np.column_stack([np.nanmean(masked, axis=1), np.nanstd(masked, axis=1),
                            meta["rh_seasonal_amplitude_pct"]])


def select_representatives(descriptors: np.ndarray, regime: np.ndarray, *, n_clusters: int,
                           regime_quota: dict[str, int], test_frac: float = 0.15,
                           seed: int = 0) -> pd.DataFrame:
    """Representative cells for the physics campaign, stratified over the real climate
    manifold rather than a uniform climate box.

    KMeans on standardized descriptors, keeping the member nearest each centroid; then each
    tail regime is topped up to its quota with random extra members. Returns cell index,
    regime and a regime-stratified train/test split -- held out by *cell*, which is the
    only split that does not leak climate across it.
    """
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    z = (descriptors - descriptors.mean(axis=0)) / (descriptors.std(axis=0) + 1e-12)
    km = KMeans(n_clusters=min(n_clusters, len(z)), n_init=4, random_state=seed).fit(z)
    dist = np.linalg.norm(z - km.cluster_centers_[km.labels_], axis=1)
    chosen = {int(np.flatnonzero(km.labels_ == c)[np.argmin(dist[km.labels_ == c])])
              for c in range(km.n_clusters)}
    for name, quota in regime_quota.items():
        have = sum(regime[i] == name for i in chosen)
        pool = [i for i in np.flatnonzero(regime == name) if i not in chosen]
        extra = rng.choice(pool, size=min(max(quota - have, 0), len(pool)), replace=False)
        chosen.update(int(i) for i in extra)

    sel = pd.DataFrame({"cell": sorted(chosen)})
    sel["regime"] = regime[sel["cell"]]
    sel["split"] = "train"
    for _name, group in sel.groupby("regime"):
        if len(group) < 2:
            continue
        n_test = max(1, int(round(test_frac * len(group))))
        sel.loc[rng.choice(group.index, size=n_test, replace=False), "split"] = "test"
    return sel


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
