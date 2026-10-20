"""Nowcast by scaling up for the hospitals that have not reported yet."""

import datetime
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from idmodels.config import ScaleByCompletenessConfig
from idmodels.nowcast.base import Nowcaster
from idmodels.nowcast.delphi import nhsn_hosp_reporting_archive


def read_archive(path: Path | None, default: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """
    An as-of archive (one row per published value, with columns location, wk_end_date, report_date and a value
    column): a csv or parquet file at `path`, or `default()` (a Delphi archive) if `path` is None.
    """
    if path is None:
        archive = default()
    elif Path(path).suffix == ".parquet":
        archive = pd.read_parquet(path)
    else:
        archive = pd.read_csv(path, dtype={"location": str})
    return archive.assign(
        wk_end_date=pd.to_datetime(archive["wk_end_date"]), report_date=pd.to_datetime(archive["report_date"])
    )


def latest_as_of(archive: pd.DataFrame, as_of: datetime.date) -> pd.DataFrame:
    """For each location and week in an as-of archive, the row with the latest value published on or before `as_of`."""
    known = archive.loc[archive["report_date"] <= pd.Timestamp(as_of)]
    latest = known.sort_values("report_date").groupby(["location", "wk_end_date"], as_index=False).last()
    return latest.drop(columns="report_date")


def values_as_of(archive: pd.DataFrame, keys: pd.DataFrame, column: str) -> np.ndarray:
    """
    For each row of `keys` (columns location, wk_end_date, release), the latest value of `column` in an as-of archive
    published on or before `release`; NaN if none was.
    """
    left = keys[["location", "wk_end_date", "release"]].reset_index(drop=True).assign(_row=lambda d: d.index)
    merged = pd.merge_asof(
        left.sort_values("release"),
        archive[["location", "wk_end_date", "report_date", column]].sort_values("report_date"),
        left_on="release",
        right_on="report_date",
        by=["location", "wk_end_date"],
        direction="backward",
    )
    return merged.sort_values("_row")[column].to_numpy(dtype=float)


def load_reporting(path: Path | None, as_of: datetime.date) -> pd.DataFrame:
    """
    Number of hospitals reporting each week, as known on `as_of`: from a csv or parquet as-of archive at `path` (with
    columns location, wk_end_date, report_date, hosp_reporting), or the Delphi Epidata archive of NHSN counts if `path`
    is None. Returns columns location, wk_end_date, hosp_reporting: for each location and week, the latest value
    published on or before `as_of`.
    """
    return latest_as_of(read_archive(path, nhsn_hosp_reporting_archive), as_of)[
        ["location", "wk_end_date", "hosp_reporting"]
    ]


class ScaleByCompletenessNowcaster(Nowcaster):
    """
    Divide each of the most recent weeks' values by its completeness: hospitals reporting that week divided by the
    usual number (the median number reporting the weeks config.typical_lags weeks before the location's newest week),
    both as known at the release the data come from (see release_date(); from config.reporting_path, or the Delphi
    archive). Completeness is clipped to [config.min_completeness, 1]. A week whose completeness cannot be computed
    (no reporting data for the location or week, for example when that week's release came later than usual) is left
    unchanged.
    """

    def __init__(self, config: ScaleByCompletenessConfig):
        # Narrow self.config from NowcastConfig to ScaleByCompletenessConfig so that
        # type checkers resolve ScaleByCompletenessConfig-specific attributes in this class.
        super().__init__(config)
        self.config: ScaleByCompletenessConfig = config

    def release_date(self, as_of: datetime.date) -> datetime.date:
        """The date of the release the data come from: the latest config.release_weekday on or before `as_of`."""
        weekday = self.config.release_weekday
        if weekday is None:
            return as_of
        return as_of - datetime.timedelta(days=(as_of.weekday() - weekday) % 7)

    def completeness(self, df: pd.DataFrame, as_of: datetime.date) -> pd.Series:
        """Completeness (unclipped) of each of the most recent weeks of config.source in `df`; NaN elsewhere."""
        cfg = self.config
        reporting = load_reporting(cfg.reporting_path, self.release_date(as_of)).set_index(["location", "wk_end_date"])[
            "hosp_reporting"
        ]
        rows = df.loc[self.recent_weeks(df), ["agg_level", "location", "wk_end_date"]]
        weeks = pd.to_datetime(rows["wk_end_date"])
        newest = weeks.groupby([rows["agg_level"], rows["location"]]).transform("max")

        def reported(week: pd.Series) -> pd.Series:
            idx = pd.MultiIndex.from_arrays([rows["location"], week])
            return pd.Series(reporting.reindex(idx).to_numpy(dtype=float), index=rows.index)

        typical = pd.concat([reported(newest - pd.Timedelta(weeks=lag)) for lag in cfg.typical_lags], axis=1).median(
            axis=1
        )
        return (reported(weeks) / typical.where(typical > 0)).reindex(df.index)

    def _adjust(self, df: pd.DataFrame, as_of: datetime.date) -> pd.DataFrame:
        completeness = self.completeness(df, as_of).clip(lower=self.config.min_completeness, upper=1.0)
        scale = completeness.notna()
        df.loc[scale, "inc"] = df.loc[scale, "inc"] / completeness[scale]
        return df
