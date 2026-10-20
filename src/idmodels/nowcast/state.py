"""Nowcast by scaling up for the hospitals that have not reported yet, corrected by a factor for each location."""

import datetime

import numpy as np
import pandas as pd

from idmodels.config import ScaleByStateConfig
from idmodels.nowcast.completeness import (
    ScaleByCompletenessNowcaster,
    latest_as_of,
    read_archive,
    values_as_of,
)
from idmodels.nowcast.delphi import nhsn_admissions_archive, nhsn_hosp_reporting_archive


def season_start(as_of: datetime.date, month: int) -> pd.Timestamp:
    """The first day of the season that `as_of` falls in, for seasons starting on the first day of `month`."""
    year = as_of.year if as_of.month >= month else as_of.year - 1
    return pd.Timestamp(year, month, 1)


class ScaleByStateNowcaster(ScaleByCompletenessNowcaster):
    """
    Scale each of the most recent weeks' values by completeness, as ScaleByCompletenessNowcaster does, then correct
    it by its location's factor (see ScaleByStateConfig). The correction is applied on the count scale: with
    count = inc * pop / 100000 (iddata's NHSN values are rates per 100,000), the adjusted count is exp(log(count / completeness + 1) + effect) - 1. A week whose
    completeness cannot be computed is left unchanged; a location without settled weeks this season gets no
    correction.

    Scaling over-corrects in locations where the missing hospitals have few admissions or never report, and
    under-corrects where hospitals that did report revise their counts upward; the factors capture both. They do not
    carry over from one season to the next, so they are learned from the current season only.
    """

    def __init__(self, config: ScaleByStateConfig):
        # Narrow self.config from ScaleByCompletenessConfig to ScaleByStateConfig so that
        # type checkers resolve ScaleByStateConfig-specific attributes in this class.
        super().__init__(config)
        self.config: ScaleByStateConfig = config

    def training_weeks(self, as_of: datetime.date) -> pd.DataFrame:
        """
        This season's settled weeks as of the release the data come from (release_date(as_of)), one row per location
        and week, with columns location, wk_end_date, release (the earlier release in which the week was the newest),
        first (its value then), completeness (then, unclipped) and settled (its value in the current release).
        """
        cfg = self.config
        admissions = read_archive(cfg.admissions_path, nhsn_admissions_archive)
        reporting = read_archive(cfg.reporting_path, nhsn_hosp_reporting_archive)
        release = self.release_date(as_of)
        current = latest_as_of(admissions, release).rename(columns={"admissions": "settled"})
        newest = current.groupby("location")["wk_end_date"].transform("max")
        weeks = current.loc[
            (current["wk_end_date"] >= season_start(as_of, cfg.season_start_month))
            & (current["wk_end_date"] <= newest - pd.Timedelta(weeks=cfg.settle_weeks))
        ].copy()
        weeks["release"] = pd.Timestamp(release) - (newest[weeks.index] - weeks["wk_end_date"])
        weeks["first"] = values_as_of(admissions, weeks, "admissions")
        reported = values_as_of(reporting, weeks, "hosp_reporting")
        typical = pd.DataFrame(
            {
                lag: values_as_of(
                    reporting,
                    weeks.assign(wk_end_date=weeks["wk_end_date"] - pd.Timedelta(weeks=lag)),
                    "hosp_reporting",
                )
                for lag in cfg.typical_lags
            }
        ).median(axis=1)
        weeks["completeness"] = reported / typical.where(typical > 0).to_numpy()
        return weeks[["location", "wk_end_date", "release", "first", "completeness", "settled"]].reset_index(drop=True)

    def state_effects(self, as_of: datetime.date) -> pd.Series:
        """Each location's effect (log scale; see ScaleByStateConfig), for locations with settled weeks this season."""
        cfg = self.config
        weeks = self.training_weeks(as_of).dropna(subset=["first", "completeness", "settled"])
        scaled = weeks["first"] / weeks["completeness"].clip(lower=cfg.min_completeness, upper=1.0)
        log_err = np.log1p(weeks["settled"]) - np.log1p(scaled)
        by_location = log_err.groupby(weeks["location"])
        return by_location.sum() / (by_location.count() + cfg.shrink)

    def _adjust(self, df: pd.DataFrame, as_of: datetime.date) -> pd.DataFrame:
        completeness = self.completeness(df, as_of).clip(lower=self.config.min_completeness, upper=1.0)
        rows = completeness.notna()
        if not rows.any():
            return df
        effect = df.loc[rows, "location"].map(self.state_effects(as_of)).fillna(0.0)
        scaled = df.loc[rows, "inc"] / completeness[rows]
        # iddata gives NHSN values as rates per 100,000; without a population, the factor is applied to the rate
        to_count = df.loc[rows, "pop"] / 100000
        adjusted = np.maximum(np.expm1(np.log1p(scaled * to_count) + effect), 0) / to_count
        df.loc[rows, "inc"] = adjusted.where(to_count.notna(), scaled * np.exp(effect))
        return df
