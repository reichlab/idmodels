"""
Nowcasting: adjusting the most recent values of the input data before a model is fit.

The most recent weeks of a surveillance series are often incomplete when first published (for NHSN admissions, some
hospitals have not reported yet) and are revised later. A Nowcaster replaces those values by estimates of their
eventual (settled) values, using only data available on the reference date. It works on the long-format table that
iddata's DiseaseDataLoader returns (one row per source, location and week, values in `inc`), so models use it without
any changes: IDModel.run() applies it to the loaded data when the model configuration has a `nowcast`.
"""

import datetime
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from idmodels.config import NowcastConfig


class Nowcaster(ABC):
    """
    Abstract base for nowcasts. Subclasses implement _adjust(); apply() runs it and checks that it changed only what a
    nowcast may change.
    """

    def __init__(self, config: NowcastConfig):
        self.config = config

    def apply(self, df: pd.DataFrame, as_of: datetime.date) -> pd.DataFrame:
        """
        Return a copy of `df` (long-format iddata output) in which the values (`inc`) of the most recent weeks of
        config.source are replaced by nowcasts, using data available on `as_of`. All other rows and columns are
        unchanged.
        """
        adjusted = self._adjust(df.copy(), as_of)
        self._check(df, adjusted)
        return adjusted

    @abstractmethod
    def _adjust(self, df: pd.DataFrame, as_of: datetime.date) -> pd.DataFrame:
        """Return `df` with adjusted `inc` values; may modify `df`, which is a copy."""
        ...

    def recent_weeks(self, df: pd.DataFrame) -> pd.Series:
        """
        True for the rows of config.source that are among the config.num_weeks most recent weeks with a value, for
        each (agg_level, location).
        """
        has_value = (df["source"] == self.config.source.value) & df["inc"].notna()
        rows = df.loc[has_value]
        weeks = pd.to_datetime(rows["wk_end_date"])
        newest = weeks.groupby([rows["agg_level"], rows["location"]]).transform("max")
        lag = ((newest - weeks).dt.days // 7).reindex(df.index)
        return has_value & (lag < self.config.num_weeks)

    def _check(self, original: pd.DataFrame, adjusted: pd.DataFrame) -> None:
        """Raise if the nowcast changed anything other than `inc` of the most recent weeks of config.source."""
        name = type(self).__name__
        if not adjusted.index.equals(original.index) or list(adjusted.columns) != list(original.columns):
            raise ValueError(f"{name} changed the rows or columns of the data")
        others = original.columns.drop("inc")
        if not adjusted[others].equals(original[others]):
            raise ValueError(f"{name} changed columns other than inc")
        changed = ~np.isclose(adjusted["inc"], original["inc"], equal_nan=True)
        if (changed & ~self.recent_weeks(original)).any():
            raise ValueError(f"{name} changed values other than the most recent weeks of {self.config.source.value}")
        if (adjusted["inc"].isna() & original["inc"].notna()).any():
            raise ValueError(f"{name} produced missing values")
