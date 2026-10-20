"""Nowcast by scaling up for the hospitals that have not reported yet."""

import datetime
from pathlib import Path

import pandas as pd

from idmodels.config import ScaleByCompletenessConfig
from idmodels.nowcast.base import Nowcaster
from idmodels.nowcast.delphi import nhsn_hosp_reporting_archive


def load_reporting(path: Path | None, as_of: datetime.date) -> pd.DataFrame:
    """
    Number of hospitals reporting each week, as known on `as_of`, from an as-of archive (one row per published value,
    with columns location, wk_end_date, report_date, hosp_reporting): a csv or parquet file at `path`, or the Delphi
    Epidata archive of NHSN counts if `path` is None. Returns columns location, wk_end_date, hosp_reporting: for each
    location and week, the latest value published on or before `as_of`.
    """
    if path is None:
        archive = nhsn_hosp_reporting_archive()
    elif Path(path).suffix == ".parquet":
        archive = pd.read_parquet(path)
    else:
        archive = pd.read_csv(path, dtype={"location": str})
    archive = archive.assign(
        wk_end_date=pd.to_datetime(archive["wk_end_date"]), report_date=pd.to_datetime(archive["report_date"])
    )
    known = archive.loc[archive["report_date"] <= pd.Timestamp(as_of)]
    latest = known.sort_values("report_date").groupby(["location", "wk_end_date"], as_index=False).last()
    return latest[["location", "wk_end_date", "hosp_reporting"]]


class ScaleByCompletenessNowcaster(Nowcaster):
    """
    Divide each of the most recent weeks' values by its completeness: hospitals reporting that week divided by the
    usual number (the median number reporting the weeks config.typical_lags weeks before the location's newest week),
    both as known on the reference date (from config.reporting_path, or the Delphi archive). Completeness is clipped
    to [config.min_completeness, 1]. A week whose completeness cannot be computed (no reporting data for the location
    or week) is left unchanged.
    """

    def __init__(self, config: ScaleByCompletenessConfig):
        # Narrow self.config from NowcastConfig to ScaleByCompletenessConfig so that
        # type checkers resolve ScaleByCompletenessConfig-specific attributes in this class.
        super().__init__(config)
        self.config: ScaleByCompletenessConfig = config

    def completeness(self, df: pd.DataFrame, as_of: datetime.date) -> pd.Series:
        """Completeness (unclipped) of each of the most recent weeks of config.source in `df`; NaN elsewhere."""
        cfg = self.config
        reporting = load_reporting(cfg.reporting_path, as_of).set_index(["location", "wk_end_date"])["hosp_reporting"]
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
