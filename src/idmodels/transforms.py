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
    """f(x) = (x + offset)^0.25"""


    def __init__(self, offset: float = POWER_TRANSFORM_OFFSET):
        self.offset = offset


    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df["inc_trans"] = (df["inc"] + self.offset) ** 0.25
        return df


    def invert(self, values: np.ndarray, context: pd.DataFrame) -> np.ndarray:
        return np.maximum(values, 0.0) ** 4 - self.offset


class IdentityTransform(Transform):
    """f(x) = x + offset (no power transform)"""


    def __init__(self, offset: float = POWER_TRANSFORM_OFFSET):
        self.offset = offset


    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df["inc_trans"] = df["inc"] + self.offset
        return df


    def invert(self, values: np.ndarray, context: pd.DataFrame) -> np.ndarray:
        return np.maximum(values, 0.0) - self.offset


class SourceScaleTransform(Transform):
    """
    Applies (inc + floor) * scale per source before the power transform; inverts after.
    params: dict mapping source name string to (floor, scale).
    Sources absent from params pass through unchanged.
    """


    def __init__(self, params: dict[str, tuple[float, float]]):
        self.params = params


    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # look up per-row floor and scale by source name; sources not in params get identity defaults
        floors = df["source"].map({k: v[0] for k, v in self.params.items()}).fillna(0.0)  # default floor: 0
        scales = df["source"].map({k: v[1] for k, v in self.params.items()}).fillna(1.0)  # default scale: 1
        df["inc"] = (df["inc"] + floors) * scales
        return df


    def invert(self, values: np.ndarray, context: pd.DataFrame) -> np.ndarray:
        # same lookup on context["source"] to recover the per-row params used in apply()
        floors = context["source"].map({k: v[0] for k, v in self.params.items()}).fillna(0.0).values
        scales = context["source"].map({k: v[1] for k, v in self.params.items()}).fillna(1.0).values
        return values / scales - floors  # inverse of (x + floor) * scale


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
