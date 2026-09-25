"""
Model for revisions ("backfill") to recently reported NHSN admission counts.

For a data vintage released on date a, let L_a be the most recent week in that vintage. The revision at lag j is
    rho_j = log((final count for week L_a - j + 1) / (count reported in vintage a for week L_a - j + 1)),
where "final" is the most recent vintage available. A revision vector (rho_0, ..., rho_{J-1}) is recorded for every
(location, vintage) pair whose final values have had at least J weeks to mature. New revision draws are produced by
resampling whole vectors (a block bootstrap), which keeps the dependence between revisions at different lags within a
single release. Vectors are grouped into strata by the size of the reported lag-0 count, because small counts are
revised proportionally more.
"""

import datetime
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NHSN_PATTERNS = [
    "infectious-disease-data/data-raw/influenza-hhs/hhs-????-??-??.csv",
    "infectious-disease-data/data-raw/influenza-nhsn/nhsn-????-??-??.csv",
]
_VINTAGE_CACHE: dict[datetime.date, pd.DataFrame | None] = {}


def list_nhsn_vintage_dates(as_of: datetime.date) -> list[datetime.date]:
    """Dates of all NHSN (and pre-2024-11-15 HHS) snapshots released on or before as_of."""
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    dates = []
    for pattern in _NHSN_PATTERNS:
        dates += [datetime.date.fromisoformat(f[-14:-4]) for f in fs.glob(pattern)]
    return sorted(d for d in set(dates) if d <= as_of)


def _load_one_vintage(as_of: datetime.date) -> pd.DataFrame | None:
    from iddata.sources.nhsn import NHSNDataSource

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dat = NHSNDataSource(rates=False).load(as_of=as_of)
    except (KeyError, ValueError) as err:
        # a few snapshots use a different column schema; skip them
        logger.warning(f"skipping NHSN vintage {as_of}: {err!r}")
        return None
    dat = dat.dropna(subset=["location"])[["location", "season", "season_week", "wk_end_date", "inc"]].copy()
    dat["as_of"] = pd.Timestamp(as_of)
    return dat


def load_nhsn_vintages(as_of: datetime.date, max_workers: int = 8) -> pd.DataFrame:
    """
    All NHSN vintages (admission counts) released on or before as_of, stacked with an `as_of` column. Vintages are
    cached in memory for the life of the process, so repeated calls (e.g. in hindcasts) only download new files.
    """
    dates = list_nhsn_vintage_dates(as_of)
    missing = [d for d in dates if d not in _VINTAGE_CACHE]
    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for d, dat in zip(missing, ex.map(_load_one_vintage, missing)):
                _VINTAGE_CACHE[d] = dat
    frames = [_VINTAGE_CACHE[d] for d in dates if _VINTAGE_CACHE[d] is not None]
    if not frames:
        raise FileNotFoundError(f"no NHSN vintages available on or before {as_of}")
    return pd.concat(frames, ignore_index=True)


class RevisionModel:
    """Block-bootstrap model for the log revision ratios of the most recent `max_lag` weeks."""

    def __init__(
        self,
        max_lag: int = 10,
        strata_bounds: tuple[float, ...] = (50.0, 500.0),
        min_vectors: int = 200,
        min_stratum_vectors: int = 30,
    ):
        self.max_lag = max_lag
        self.strata_bounds = np.asarray(strata_bounds, dtype=float)
        self.min_vectors = min_vectors
        self.min_stratum_vectors = min_stratum_vectors
        self.vectors: np.ndarray = np.zeros((0, max_lag))
        self.strata: np.ndarray = np.zeros(0, dtype=int)

    def _stratum(self, counts: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.strata_bounds, np.asarray(counts, dtype=float), side="right")

    def fit(self, vintages: pd.DataFrame) -> "RevisionModel":
        """vintages: output of load_nhsn_vintages (columns location, wk_end_date, inc, as_of)."""
        vintages = vintages.dropna(subset=["inc"])
        final_as_of = vintages["as_of"].max()
        final = vintages.loc[vintages["as_of"] == final_as_of, ["location", "wk_end_date", "inc"]]
        final = final.rename(columns={"inc": "final"})
        eligible = vintages.loc[vintages["as_of"] <= final_as_of - pd.Timedelta(weeks=self.max_lag)]
        if len(eligible) == 0:
            return self

        latest = eligible.groupby(["as_of", "location"])["wk_end_date"].transform("max")
        eligible = eligible.assign(lag=((latest - eligible["wk_end_date"]).dt.days // 7).astype(int))
        eligible = eligible.loc[eligible["lag"] < self.max_lag].merge(final, on=["location", "wk_end_date"])
        eligible["rho"] = np.log((eligible["final"] + 1.0) / (eligible["inc"] + 1.0))

        wide_rho = eligible.pivot_table(index=["as_of", "location"], columns="lag", values="rho")
        wide_inc = eligible.pivot_table(index=["as_of", "location"], columns="lag", values="inc")
        wide_rho = wide_rho.reindex(columns=range(self.max_lag)).dropna()
        self.vectors = wide_rho.to_numpy()
        self.strata = self._stratum(wide_inc.loc[wide_rho.index, 0].to_numpy())
        return self

    def sample(self, lag0_counts: np.ndarray, num_draws: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw revision vectors for series whose most recent reported counts are lag0_counts.
        Returns an array of shape (len(lag0_counts), num_draws, max_lag) of log revision ratios.
        """
        lag0_counts = np.asarray(lag0_counts, dtype=float)
        out = np.empty((len(lag0_counts), num_draws, self.max_lag))
        if len(self.vectors) < self.min_vectors:
            # too little vintage history (e.g. early in the 2023/24 season): fall back to a small, symmetric,
            # lag-decaying perturbation so past weeks still receive some probability
            sd = 0.1 / (1.0 + np.arange(self.max_lag))
            return rng.normal(size=out.shape) * sd
        strata = self._stratum(lag0_counts)
        for i, s in enumerate(strata):
            pool = np.flatnonzero(self.strata == s)
            if len(pool) < self.min_stratum_vectors:
                pool = np.arange(len(self.vectors))
            out[i] = self.vectors[rng.choice(pool, size=num_draws, replace=True)]
        return out


def apply_revisions(counts: np.ndarray, last_week: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Apply revision draws to season-aligned count arrays.

    counts: (n_series, n_weeks) reported counts; last_week: (n_series,) index (0-based column) of each series' most
    recent reported week; rho: (n_series, num_draws, max_lag) log revision ratios.
    Returns (n_series, num_draws, n_weeks) revised counts, (c + 1) * exp(rho) - 1 truncated at 0.
    """
    n, num_draws, max_lag = rho.shape
    revised = np.repeat(counts[:, None, :], num_draws, axis=1).astype(float)
    for j in range(max_lag):
        cols = last_week - j
        ok = cols >= 0
        rows = np.flatnonzero(ok)
        c = counts[rows, cols[ok]]
        revised[rows, :, cols[ok]] = np.maximum((c[:, None] + 1.0) * np.exp(rho[rows, :, j]) - 1.0, 0.0)
    return revised
