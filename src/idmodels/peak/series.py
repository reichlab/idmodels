"""
Season-aligned series, state features, peak targets and the "season replay" training set used by the direct peak
models.

Every (source, location, season) series is stored as one row of an array with one column per season week
(column j holds season week j + 1). Features describe the season as observed through season week t and are
scale-free (log ratios and week counts), so that rows from NHSN, ILINet and FluSurvNet can be pooled.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# offset added before taking logs, in each source's units (NHSN and FluSurvNet: rate per 100k; ILINet: weighted
# ILI x proportion positive). Roughly 1/1000 of a typical national peak for each source.
LOG_EPS = {"nhsn": 0.01, "flusurvnet": 0.01, "ilinet": 0.001}

N_SEASON_WEEKS = 53

# features used by the models; KCDE_FEATURES must never be missing for rows that have a current observation
ALL_FEATURES = [
    "season_week",
    "rel_max",
    "wks_since_max",
    "g1",
    "g2",
    "g3",
    "rm3",
    "cum_rel",
    "hist_rel",
    "nat_rel_max",
    "nat_wks_since_max",
    "nat_g3",
    "src_code",
]
SOURCE_CODES = {"nhsn": 0, "ilinet": 1, "flusurvnet": 2}


@dataclass
class SeasonArrays:
    """keys has one row per series (source, agg_level, location, season); y[i, j] is the value in season week j+1."""

    keys: pd.DataFrame
    y: np.ndarray

    def subset(self, mask: np.ndarray) -> "SeasonArrays":
        return SeasonArrays(keys=self.keys.loc[mask].reset_index(drop=True), y=self.y[mask])

    @property
    def eps(self) -> np.ndarray:
        return self.keys["source"].map(LOG_EPS).fillna(0.01).to_numpy()


def build_season_arrays(df: pd.DataFrame, max_gap: int = 2) -> SeasonArrays:
    """
    Pivot long-format iddata output (columns source, agg_level, location, season, season_week, inc) to SeasonArrays.
    Interior gaps of at most `max_gap` weeks are filled by linear interpolation; leading and trailing missing values
    are left missing.
    """
    df = df.dropna(subset=["season_week"])
    wide = df.pivot_table(
        index=["source", "agg_level", "location", "season"],
        columns="season_week",
        values="inc",
        aggfunc="last",
        dropna=False,
    )
    wide = wide.reindex(columns=np.arange(1, N_SEASON_WEEKS + 1, dtype=float))
    wide = wide.loc[wide.notna().any(axis=1)]
    filled = wide.interpolate(axis=1, limit_area="inside")
    # only fill gaps of at most max_gap consecutive missing weeks
    missing = wide.isna().to_numpy()
    run_len = np.zeros_like(missing, dtype=int)
    for j in range(missing.shape[1]):  # length of the missing run each cell belongs to
        run_len[:, j] = np.where(missing[:, j], (run_len[:, j - 1] if j > 0 else 0) + 1, 0)
    for j in range(missing.shape[1] - 2, -1, -1):
        run_len[:, j] = np.where(missing[:, j] & missing[:, j + 1], run_len[:, j + 1], run_len[:, j])
    wide = wide.where(~missing | (run_len > max_gap), filled)
    keys = wide.index.to_frame(index=False)
    return SeasonArrays(keys=keys, y=wide.to_numpy(dtype=float))


def national_index(keys: pd.DataFrame) -> np.ndarray:
    """For each series, the row index of the national series with the same source and season (-1 if none)."""
    nat = keys.reset_index().query("agg_level == 'national'")
    lookup = dict(zip(zip(nat["source"], nat["season"]), nat["index"]))
    return np.array([lookup.get((s, ssn), -1) for s, ssn in zip(keys["source"], keys["season"])], dtype=int)


def _ffill(y: np.ndarray) -> np.ndarray:
    """Forward fill missing values along axis 1."""
    idx = np.where(np.isnan(y), 0, np.arange(y.shape[1]))
    np.maximum.accumulate(idx, axis=1, out=idx)
    return y[np.arange(y.shape[0])[:, None], idx]


def season_peaks(y: np.ndarray, window_start: int, window_end: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Peak size and peak week (first week attaining the maximum) within season weeks [window_start, window_end].
    Rows with no in-window observations get NaN.
    """
    w = y[:, window_start - 1 : window_end]
    has = ~np.all(np.isnan(w), axis=1)
    peak = np.full(y.shape[0], np.nan)
    peak_week = np.full(y.shape[0], np.nan)
    peak[has] = np.nanmax(w[has], axis=1)
    peak_week[has] = window_start + np.nanargmax(w[has], axis=1)
    return peak, peak_week


def running_max(y: np.ndarray, t: int, window_start: int) -> tuple[np.ndarray, np.ndarray]:
    """
    The running maximum M_t over in-window weeks observed through season week t, and the week attaining it (first
    such week). Before the window opens (t < window_start), or if no in-window week is observed, M_t is the most
    recent observed value and its week is t.
    """
    yf = _ffill(y[:, :t])
    latest = yf[:, t - 1]
    m = latest.copy()
    m_week = np.full(y.shape[0], float(t))
    if t >= window_start:
        w = y[:, window_start - 1 : t]
        has = ~np.all(np.isnan(w), axis=1)
        m[has] = np.nanmax(w[has], axis=1)
        m_week[has] = window_start + np.nanargmax(w[has], axis=1)
    return m, m_week


def state_features(
    y: np.ndarray,
    t: int,
    eps: np.ndarray,
    window_start: int,
    nat_idx: np.ndarray | None = None,
    hist_peak: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Features describing each season as observed through season week t (the most recent observed week). Only
    y[:, :t] is used, so there is no look-ahead. Also returns the helper columns `lm` (log of M_t + eps) and
    `max_week` needed to map relative predictions back to the season.
    """
    n = y.shape[0]
    yf = _ffill(y[:, :t])
    eps = np.asarray(eps, dtype=float)

    def lag_log(lag):
        if t - 1 - lag < 0:
            return np.full(n, np.nan)
        return np.log(yf[:, t - 1 - lag] + eps)

    lx = lag_log(0)
    m, m_week = running_max(y, t, window_start)
    lm = np.log(m + eps)
    recent = yf[:, max(t - 3, 0) : t]
    with warnings.catch_warnings():  # all-missing rows (no data yet this season) give NaN, as intended
        warnings.simplefilter("ignore", category=RuntimeWarning)
        recent_mean = np.nanmean(recent, axis=1)
    cum = np.nansum(y[:, 4:t], axis=1) if t > 4 else np.nansum(y[:, :t], axis=1)
    feats = pd.DataFrame(
        {
            "season_week": np.full(n, float(t)),
            "rel_max": lx - lm,
            "wks_since_max": t - m_week,
            "g1": lx - lag_log(1),
            "g2": lag_log(1) - lag_log(2),
            "g3": lx - lag_log(3),
            "rm3": np.log(recent_mean + eps) - lm,
            "cum_rel": np.log(cum + eps) - lm,
            "hist_rel": lm - hist_peak if hist_peak is not None else np.full(n, np.nan),
            "lm": lm,
            "max_week": m_week,
            "observed": ~np.isnan(yf[:, t - 1]),
        }
    )
    if nat_idx is not None:
        has_nat = nat_idx >= 0
        for col, src in [("nat_rel_max", "rel_max"), ("nat_wks_since_max", "wks_since_max"), ("nat_g3", "g3")]:
            vals = np.full(n, np.nan)
            vals[has_nat] = feats[src].to_numpy()[nat_idx[has_nat]]
            feats[col] = vals
    else:
        feats["nat_rel_max"] = np.nan
        feats["nat_wks_since_max"] = np.nan
        feats["nat_g3"] = np.nan
    return feats


def historical_log_peaks(arrays: SeasonArrays, window_start: int, window_end: int, min_window_obs: int) -> pd.DataFrame:
    """log(peak + eps) for every complete series; columns source, location, season, log_peak, peak_week."""
    peak, peak_week = season_peaks(arrays.y, window_start, window_end)
    n_obs = np.sum(~np.isnan(arrays.y[:, window_start - 1 : window_end]), axis=1)
    out = arrays.keys[["source", "location", "season"]].copy()
    out["log_peak"] = np.log(peak + arrays.eps)
    out["peak_week"] = peak_week
    return out.loc[n_obs >= min_window_obs].reset_index(drop=True)


def prior_mean_log_peak(keys: pd.DataFrame, hist: pd.DataFrame) -> np.ndarray:
    """For each series, the mean log peak over strictly earlier seasons of the same source and location."""
    out = np.full(len(keys), np.nan)
    grouped = {k: g for k, g in hist.groupby(["source", "location"])}
    for i, (src, loc, season) in enumerate(zip(keys["source"], keys["location"], keys["season"])):
        g = grouped.get((src, loc))
        if g is not None:
            prev = g.loc[g["season"] < season, "log_peak"]
            if len(prev) > 0:
                out[i] = prev.mean()
    return out


def build_replay_rows(
    arrays: SeasonArrays, window_start: int, window_end: int, replay_start: int, min_window_obs: int
) -> pd.DataFrame:
    """
    Season replay: for each complete historical series and each season week t in [replay_start, window_end] with an
    observation, the features computed from weeks 1..t, together with the targets
        z = log(peak + eps) - log(M_t + eps)      (log growth still to come in the running maximum)
        k = peak_week - t                         (weeks until the peak; k <= 0 means already peaked)
    computed from the completed season.
    """
    n_obs = np.sum(~np.isnan(arrays.y[:, window_start - 1 : window_end]), axis=1)
    arrays = arrays.subset(n_obs >= min_window_obs)
    peak, peak_week = season_peaks(arrays.y, window_start, window_end)
    hist = historical_log_peaks(arrays, window_start, window_end, min_window_obs)
    hist_peak = prior_mean_log_peak(arrays.keys, hist)
    nat_idx = national_index(arrays.keys)
    eps = arrays.eps

    frames = []
    for t in range(replay_start, window_end + 1):
        feats = state_features(arrays.y, t, eps, window_start, nat_idx=nat_idx, hist_peak=hist_peak)
        feats = pd.concat([arrays.keys, feats], axis=1)
        feats["z"] = np.log(peak + eps) - feats["lm"]
        feats["k"] = peak_week - t
        # only rows whose current week is actually observed (not carried forward)
        feats = feats.loc[~np.isnan(arrays.y[:, t - 1])]
        frames.append(feats)
    rows = pd.concat(frames, ignore_index=True)
    rows["src_code"] = rows["source"].map(SOURCE_CODES).astype(int)
    rows["peak_week"] = rows["season_week"] + rows["k"]
    return rows.drop(columns=["observed"])


def peak_week_climatology(hist: pd.DataFrame, window_start: int, window_end: int, smoothing_sd: float) -> np.ndarray:
    """
    Kernel-smoothed distribution of the peak week over season weeks window_start..window_end, with every season
    receiving equal total weight (so that sources with many locations do not dominate). Returns an array indexed by
    season week - window_start.
    """
    weeks = np.arange(window_start, window_end + 1)
    w = 1.0 / hist.groupby("season")["season"].transform("size").to_numpy()
    dens = np.exp(-0.5 * ((weeks[None, :] - hist["peak_week"].to_numpy()[:, None]) / smoothing_sd) ** 2)
    dens = dens / dens.sum(axis=1, keepdims=True)
    clim = (w[:, None] * dens).sum(axis=0)
    return clim / clim.sum()
