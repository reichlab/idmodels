from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from idmodels.constants import IN_SEASON_WEEK_MAX, IN_SEASON_WEEK_MIN, POWER_TRANSFORM_OFFSET


class Transform(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the forward transformation, writing result columns into df."""
        ...


    @abstractmethod
    def invert(self, values: np.ndarray, context: pd.DataFrame) -> np.ndarray:
        """Apply the inverse transformation. context may contain factor columns."""
        ...


class FourthRootTransform(Transform):
    """f(x) = (x + additive_shift + offset)^0.25"""


    def __init__(self, additive_shift: float = 0.0, offset: float = POWER_TRANSFORM_OFFSET):
        self.additive_shift = additive_shift
        self.offset = offset


    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df["inc_trans"] = (df["inc"] + self.additive_shift + self.offset) ** 0.25
        return df


    def invert(self, values: np.ndarray, context: pd.DataFrame) -> np.ndarray:
        return np.maximum(values, 0.0) ** 4 - self.offset - self.additive_shift


class IdentityTransform(Transform):
    """f(x) = x + additive_shift + offset (no power transform)."""


    def __init__(self, additive_shift: float = 0.0, offset: float = POWER_TRANSFORM_OFFSET):
        self.additive_shift = additive_shift
        self.offset = offset


    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df["inc_trans"] = df["inc"] + self.additive_shift + self.offset
        return df


    def invert(self, values: np.ndarray, context: pd.DataFrame) -> np.ndarray:
        return np.maximum(values, 0.0) - self.offset - self.additive_shift


class CenterScaleTransform(Transform):
    """
    Scales by in-season 95th-percentile then centers by in-season mean, per (source, location). Writes factor columns
    into df.

    Output column: inc_trans_cs
    Factor columns: inc_trans_scale_factor, inc_trans_center_factor
    """


    def __init__(self, in_season_week_min: int = IN_SEASON_WEEK_MIN, in_season_week_max: int = IN_SEASON_WEEK_MAX):
        self.in_season_week_min = in_season_week_min
        self.in_season_week_max = in_season_week_max


    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df["inc_trans_scale_factor"] = (
            df.assign(
                inc_trans_in_season=lambda x: np.where(
                    (x["season_week"] < self.in_season_week_min)
                    | (x["season_week"] > self.in_season_week_max),
                    np.nan,
                    x["inc_trans"],
                )
            )
            .groupby(["source", "location"])["inc_trans_in_season"]
            .transform(lambda x: x.quantile(0.95))
        )
        df["inc_trans_cs"] = df["inc_trans"] / (df["inc_trans_scale_factor"] + 0.01)
        df["inc_trans_center_factor"] = (
            df.assign(
                inc_trans_cs_in_season=lambda x: np.where(
                    (x["season_week"] < self.in_season_week_min)
                    | (x["season_week"] > self.in_season_week_max),
                    np.nan,
                    x["inc_trans_cs"],
                )
            )
            .groupby(["source", "location"])["inc_trans_cs_in_season"]
            .transform("mean")
        )
        df["inc_trans_cs"] = df["inc_trans_cs"] - df["inc_trans_center_factor"]
        return df


    def invert(self, values: np.ndarray, context: pd.DataFrame) -> np.ndarray:
        return (values + context["inc_trans_center_factor"].values) * (
                context["inc_trans_scale_factor"].values + 0.01)


class ComposedTransform(Transform):
    """Chains multiple transforms; apply() in order, invert() in reverse."""


    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms


    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for t in self.transforms:
            df = t.apply(df)
        return df


    def invert(self, values: np.ndarray, context: pd.DataFrame) -> np.ndarray:
        for t in reversed(self.transforms):
            values = t.invert(values, context)
        return values
