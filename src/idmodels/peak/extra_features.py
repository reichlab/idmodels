"""
Optional candidate features for the direct peak models, beyond series.state_features (ported from the exploratory
analysis in the sandbox, analysis/peak-models/relative_size_predictors.py, with identical definitions and names).

All are computed from data through the current season week t only (plus a calendar that is known in advance, fixed
state centroids, and influenza type/subtype counts through week t). Groups:
    trend:     local Taylor fits of the log series over trailing windows, rolling means and lags, all relative to the
               log running maximum
    recession: decline since the running maximum (rec_rate, rec_consist) and records (n_rec4, rec_frac)
    holiday:   Christmas / New Year weeks (week-ending Saturday Dec 22 - Jan 7)
    latlon:    state centroid latitude and longitude (state series only)
    bshare:    influenza B share of positives and A/B growth (own geography)
    h3:        A(H3) share of subtyped A positives (HHS region, and national)
"""

import datetime
import warnings

import numpy as np
import pandas as pd

STATE_CENTROIDS = {
    "01": (32.59, -86.75),
    "02": (64.0, -150.0),
    "04": (34.22, -111.62),
    "05": (34.73, -92.30),
    "06": (36.53, -119.77),
    "08": (38.68, -105.51),
    "09": (41.59, -72.36),
    "10": (38.68, -74.98),
    "11": (38.90, -77.03),
    "12": (27.87, -81.69),
    "13": (32.33, -83.37),
    "15": (20.8, -156.3),
    "16": (43.56, -113.93),
    "17": (40.05, -89.38),
    "18": (40.05, -86.08),
    "19": (41.94, -93.37),
    "20": (38.42, -98.12),
    "21": (37.39, -84.77),
    "22": (30.62, -92.27),
    "23": (45.62, -68.98),
    "24": (39.28, -76.65),
    "25": (42.36, -71.58),
    "26": (43.14, -84.69),
    "27": (46.39, -94.60),
    "28": (32.68, -89.81),
    "29": (38.33, -92.51),
    "30": (46.82, -109.32),
    "31": (41.34, -99.59),
    "32": (39.11, -116.85),
    "33": (43.39, -71.39),
    "34": (39.96, -74.23),
    "35": (34.48, -105.94),
    "36": (43.14, -75.14),
    "37": (35.42, -78.47),
    "38": (47.25, -100.10),
    "39": (40.22, -82.60),
    "40": (35.51, -97.12),
    "41": (43.91, -120.07),
    "42": (40.91, -77.45),
    "44": (41.59, -71.12),
    "45": (33.62, -80.51),
    "46": (44.34, -99.72),
    "47": (35.68, -86.46),
    "48": (31.39, -98.79),
    "49": (39.11, -111.33),
    "50": (44.25, -72.55),
    "51": (37.56, -78.20),
    "53": (47.42, -119.75),
    "54": (38.42, -80.67),
    "55": (44.59, -89.99),
    "56": (43.05, -107.26),
}
HHS_REGION = {
    f: r
    for r, fs in {
        1: ["09", "23", "25", "33", "44", "50"],
        2: ["34", "36", "72", "78"],
        3: ["10", "11", "24", "42", "51", "54"],
        4: ["01", "12", "13", "21", "28", "37", "45", "47"],
        5: ["17", "18", "26", "27", "39", "55"],
        6: ["05", "22", "35", "40", "48"],
        7: ["19", "20", "29", "31"],
        8: ["08", "30", "38", "46", "49", "56"],
        9: ["04", "06", "15", "32"],
        10: ["02", "16", "41", "53"],
    }.items()
    for f in fs
}

FEATURE_GROUPS = {
    "trend": [
        "tay1_w4_lvl",
        "tay1_w4_slope",
        "tay1_w6_lvl",
        "tay1_w6_slope",
        "tay2_w6_lvl",
        "tay2_w6_slope",
        "tay2_w6_curv",
        "tay2_w8_lvl",
        "tay2_w8_slope",
        "tay2_w8_curv",
        "rm2",
        "rm4",
        "lag1_rel",
        "lag2_rel",
        "lag3_rel",
        "lag4_rel",
    ],
    "recession": ["rec_rate", "rec_consist", "n_rec4", "rec_frac"],
    "holiday": [
        "hol_now",
        "max_in_holiday",
        "weeks_since_holiday_end",
        "holiday_excess_max",
        "rel_max_nohol",
        "wks_since_max_nohol",
    ],
    "latlon": ["lat", "lon"],
    "bshare": [
        "b_share_3wk",
        "b_share_cum",
        "b_share_trend",
        "b_rising",
        "a_rising",
        "b_minus_a_growth",
        "b_frac_of_peak",
    ],
    "h3": ["h3_share_cum", "h3_share_3wk", "h3_share_nat"],
}
# minimum positives (A + B, or subtyped A) in a window for a share to be computed
MIN_POS = 20
N_WEEKS = 53


def _ffill(y: np.ndarray) -> np.ndarray:
    idx = np.where(np.isnan(y), 0, np.arange(y.shape[1]))
    np.maximum.accumulate(idx, axis=1, out=idx)
    return y[np.arange(y.shape[0])[:, None], idx]


def _taylor_pinv(w: int, deg: int) -> np.ndarray:
    """Least-squares map from a trailing window of w log values (lags -w+1..0) to (level, slope, curvature) at lag 0."""
    lags = np.arange(-w + 1, 1, dtype=float)
    X = np.column_stack([np.ones(w)] + [lags**d / np.prod(np.arange(1, d + 1)) for d in range(1, deg + 1)])
    return np.linalg.pinv(X)


_TAYLOR = {
    (1, 4): _taylor_pinv(4, 1),
    (1, 6): _taylor_pinv(6, 1),
    (2, 6): _taylor_pinv(6, 2),
    (2, 8): _taylor_pinv(8, 2),
}


def holiday_weeks(season: str) -> list[int]:
    """Season weeks of `season` whose Saturday week-ending date falls in Dec 22 - Jan 7."""
    from idmodels.peak.base import season_week_to_date

    y0 = int(season[:4])
    lo, hi = datetime.date(y0, 12, 22), datetime.date(y0 + 1, 1, 7)
    return [w for w in range(15, 30) if lo <= season_week_to_date(season, w) <= hi]


def holiday_matrix(seasons) -> np.ndarray:
    """H[i, j] is True when season week j + 1 of series i (with season seasons[i]) is a holiday week."""
    seasons = list(seasons)
    cal = {s: holiday_weeks(s) for s in set(seasons)}
    H = np.zeros((len(seasons), N_WEEKS), dtype=bool)
    for i, s in enumerate(seasons):
        H[i, np.array(cal[s]) - 1] = True
    return H


def centroids(locations, agg_levels) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude of state series (NaN for national and HHS-region series)."""
    ll = [
        STATE_CENTROIDS.get(loc, (np.nan, np.nan)) if agg == "state" else (np.nan, np.nan)
        for loc, agg in zip(locations, agg_levels)
    ]
    return np.array([a for a, _ in ll], dtype=float), np.array([b for _, b in ll], dtype=float)


def type_arrays(locations, agg_levels, seasons, strain: dict) -> dict:
    """
    Type/subtype count arrays (n, 53) for each series: A and B of its own geography (state FIPS; 'Region k' for HHS
    regions; 'US'), H1 and H3 of its HHS region (national for national series), and national H1, H3 (H1n, H3n).
    `strain` maps (geography, season) to an array (4, 53) of weekly A, B, A(H1), A(H3) positives by season week.
    """
    n = len(locations)
    out = {k: np.full((n, N_WEEKS), np.nan) for k in ["A", "B", "H1", "H3", "H1n", "H3n"]}
    for i, (loc, agg, ssn) in enumerate(zip(locations, agg_levels, seasons)):
        reg = (
            "US"
            if loc == "US"
            else (loc if agg == "hhs region" else (f"Region {HHS_REGION[loc]}" if loc in HHS_REGION else None))
        )
        if (loc, ssn) in strain:
            out["A"][i], out["B"][i] = strain[(loc, ssn)][0], strain[(loc, ssn)][1]
        if reg is not None and (reg, ssn) in strain:
            out["H1"][i], out["H3"][i] = strain[(reg, ssn)][2], strain[(reg, ssn)][3]
        if ("US", ssn) in strain:
            out["H1n"][i], out["H3n"][i] = strain[("US", ssn)][2], strain[("US", ssn)][3]
    return out


def extra_features(
    y: np.ndarray,
    t: int,
    eps: np.ndarray,
    window_start: int,
    lm: np.ndarray,
    max_week: np.ndarray,
    holiday: np.ndarray | None = None,
    latlon: tuple[np.ndarray, np.ndarray] | None = None,
    types: dict | None = None,
) -> pd.DataFrame:
    """
    Candidate features at season week t (see module docstring). y (n, 53) is the season so far (only y[:, :t] is
    used); lm and max_week are the log running maximum and its week from series.state_features. Groups whose inputs
    are not supplied (holiday, latlon, types) are NaN.
    """
    n = y.shape[0]
    W0 = window_start
    eps = np.asarray(eps, dtype=float)
    yt = y[:, :t]
    yf = _ffill(yt)
    L = np.log(yf + eps[:, None])
    lx = L[:, t - 1]
    wsm = t - max_week
    f: dict[str, np.ndarray] = {}

    # trend: Taylor fits, rolling means, lags
    for (deg, win), P in _TAYLOR.items():
        B = L[:, t - win : t] @ P.T if t >= win else np.full((n, deg + 1), np.nan)
        f[f"tay{deg}_w{win}_lvl"] = B[:, 0] - lm
        f[f"tay{deg}_w{win}_slope"] = B[:, 1]
        if deg == 2:
            f[f"tay{deg}_w{win}_curv"] = B[:, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for k in (2, 4):
            f[f"rm{k}"] = np.log(np.nanmean(yf[:, max(t - k, 0) : t], axis=1) + eps) - lm
    for lag in range(1, 5):
        f[f"lag{lag}_rel"] = (L[:, t - 1 - lag] - lm) if t - 1 - lag >= 0 else np.full(n, np.nan)

    # recession since the running max, and records
    D = lm - lx
    f["rec_rate"] = D / np.maximum(wsm, 1)
    neg = np.zeros((n, t + 1))
    if t >= 2:
        neg[:, 2 : t + 1] = np.cumsum((L[:, 1:t] - L[:, : t - 1]) < 0, axis=1)
    mwi = max_week.astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["rec_consist"] = np.where(wsm > 0, (neg[:, t] - neg[np.arange(n), mwi]) / wsm, np.nan)
    if t >= W0:
        w = yt[:, W0 - 1 : t]
        prev = np.fmax.accumulate(np.concatenate([np.full((n, 1), np.nan), w[:, :-1]], axis=1), axis=1)
        recs = ~np.isnan(w) & (np.isnan(prev) | (w > prev))
        f["n_rec4"] = recs[:, -4:].sum(axis=1).astype(float)
        f["rec_frac"] = recs.sum(axis=1) / (t - W0 + 1)
    else:
        f["n_rec4"] = np.full(n, np.nan)
        f["rec_frac"] = np.full(n, np.nan)

    # holiday weeks
    for c in FEATURE_GROUPS["holiday"]:
        f[c] = np.full(n, np.nan)
    if holiday is not None:
        H = holiday
        last_hol = H.shape[1] - np.argmax(H[:, ::-1], axis=1)
        first_hol = np.argmax(H, axis=1) + 1
        f["hol_now"] = H[:, t - 1].astype(float)
        f["max_in_holiday"] = (H[np.arange(n), mwi - 1] & (t >= W0)).astype(float)
        f["weeks_since_holiday_end"] = np.where(
            t < first_hol, -1.0, np.where(t <= last_hol, 0.0, np.minimum(t - last_hol, 10))
        )
        if t >= W0:
            w = yt[:, W0 - 1 : t].copy()
            w[H[:, W0 - 1 : t]] = np.nan
            has = ~np.all(np.isnan(w), axis=1)
            m_nh, wk_nh = np.full(n, np.nan), np.full(n, np.nan)
            m_nh[has] = np.nanmax(w[has], axis=1)
            wk_nh[has] = W0 + np.nanargmax(w[has], axis=1)
            lm_nh = np.log(m_nh + eps)
        else:
            lm_nh, wk_nh = lm, max_week.astype(float)
        f["holiday_excess_max"] = lm - lm_nh
        f["rel_max_nohol"] = lx - lm_nh
        f["wks_since_max_nohol"] = t - wk_nh

    # location
    f["lat"], f["lon"] = latlon if latlon is not None else (np.full(n, np.nan), np.full(n, np.nan))

    # influenza type / subtype
    for c in FEATURE_GROUPS["bshare"] + FEATURE_GROUPS["h3"]:
        f[c] = np.full(n, np.nan)
    if types is not None:
        T = {k: v[:, :t] for k, v in types.items()}  # nothing after week t

        def wsum(x, lo, hi):  # sum over season weeks lo..hi (1-based, inclusive), NaN if all missing
            lo = max(lo, 1)
            if hi < lo:
                return np.full(n, np.nan)
            v = x[:, lo - 1 : hi]
            return np.where(np.all(np.isnan(v), axis=1), np.nan, np.nansum(v, axis=1))

        def share(num, den):
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(den >= MIN_POS, num / den, np.nan)

        A, B = T["A"], T["B"]
        A3, B3 = wsum(A, t - 2, t), wsum(B, t - 2, t)
        A3p, B3p = wsum(A, t - 5, t - 3), wsum(B, t - 5, t - 3)
        f["b_share_3wk"] = share(B3, A3 + B3)
        f["b_share_cum"] = share(wsum(B, 5, t), wsum(A, 5, t) + wsum(B, 5, t))
        f["b_share_trend"] = f["b_share_3wk"] - share(B3p, A3p + B3p)
        f["b_rising"] = np.log((B3 + 1) / (B3p + 1))
        f["a_rising"] = np.log((A3 + 1) / (A3p + 1))
        f["b_minus_a_growth"] = f["b_rising"] - f["a_rising"]
        roll = np.column_stack([wsum(B, s - 2, s) for s in range(3, t + 1)]) if t >= 3 else np.full((n, 1), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bmax = np.nanmax(roll, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            f["b_frac_of_peak"] = np.where(bmax > 0, B3 / bmax, np.nan)
        H1c, H3c = wsum(T["H1"], 5, t), wsum(T["H3"], 5, t)
        f["h3_share_cum"] = share(H3c, H1c + H3c)
        f["h3_share_3wk"] = share(wsum(T["H3"], t - 2, t), wsum(T["H1"], t - 2, t) + wsum(T["H3"], t - 2, t))
        H1n, H3n = wsum(T["H1n"], 5, t), wsum(T["H3n"], 5, t)
        f["h3_share_nat"] = share(H3n, H1n + H3n)
    return pd.DataFrame(f)
