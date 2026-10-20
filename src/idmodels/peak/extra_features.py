"""
Optional feature groups for the direct peak models, in addition to the core features of series.state_features. A
model uses a group only if its configuration lists it (see PeakGBQRModelConfig.size_feature_groups).

All are computed from data through the current season week t only, plus a calendar that is known in advance. Groups:
    holiday:   Christmas / New Year weeks (week-ending Saturday Dec 22 - Jan 7)
"""

import datetime

import numpy as np
import pandas as pd

from idmodels.peak.dates import season_week_to_date

FEATURE_GROUPS = {
    "holiday": [
        "hol_now",
        "max_in_holiday",
        "weeks_since_holiday_end",
        "holiday_excess_max",
        "rel_max_nohol",
        "wks_since_max_nohol",
    ],
}
N_WEEKS = 53


def ffill(y: np.ndarray) -> np.ndarray:
    """Forward fill missing values along axis 1."""
    idx = np.where(np.isnan(y), 0, np.arange(y.shape[1]))
    np.maximum.accumulate(idx, axis=1, out=idx)
    return y[np.arange(y.shape[0])[:, None], idx]


def holiday_weeks(season: str) -> list[int]:
    """Season weeks of `season` whose Saturday week-ending date falls in Dec 22 - Jan 7."""
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


def extra_features(
    y: np.ndarray,
    t: int,
    eps: np.ndarray,
    window_start: int,
    lm: np.ndarray,
    max_week: np.ndarray,
    holiday: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Optional features at season week t (see module docstring). y (n, 53) is the season so far (only y[:, :t] is
    used); lm and max_week are the log running maximum and its week from series.state_features. The holiday group is
    NaN if `holiday` (holiday_matrix) is not supplied.
    """
    n = y.shape[0]
    W0 = window_start
    eps = np.asarray(eps, dtype=float)
    yt = y[:, :t]
    lx = np.log(ffill(yt)[:, t - 1] + eps)
    mwi = max_week.astype(int)
    f: dict[str, np.ndarray] = {}

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

    return pd.DataFrame(f)
