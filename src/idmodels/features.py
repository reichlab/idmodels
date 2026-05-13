import fnmatch
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from iddata.utils import get_holidays
from timeseriesutils import featurize

from idmodels.spatial_utils import get_directional_neighbors, get_location_centroids, validate_wave_directions


class Feature(ABC):
    """A single feature-engineering step applied to a DataFrame."""


    @abstractmethod
    def apply(
            self,
            df: pd.DataFrame,
            feat_names: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Augment df with new feature columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input data sorted by ["source", "location", "wk_end_date"].
        feat_names : list[str]
            Running list of active feature column names.

        Returns
        -------
        tuple of (augmented df, updated feat_names)
        """
        ...


class OneHotEncodingFeature(Feature):
    """One-hot encode categorical columns."""


    def __init__(self, columns: list[str]):
        self.columns = columns


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        for c in self.columns:
            ohe = pd.get_dummies(df[c], prefix=c)
            df = pd.concat([df, ohe], axis=1)
            feat_names = feat_names + list(ohe.columns)
        return df, feat_names


class HolidayFeature(Feature):
    """Adds delta_xmas (signed distance in weeks from Christmas) to df and feat_names."""


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        df = df.merge(
            get_holidays()
            .query("holiday == 'Christmas Day'")
            .drop(columns=["holiday", "date"])
            .rename(columns={"season_week": "xmas_week"}),
            how="left",
            on="season",
        ).assign(delta_xmas=lambda x: x["season_week"] - x["xmas_week"])
        feat_names = feat_names + ["delta_xmas"]
        return df, feat_names


class TaylorFeature(Feature):
    """Windowed Taylor polynomial coefficients via timeseriesutils."""


    def __init__(
            self,
            column: str,
            degree: int,
            window_sizes: list[int],
            window_align: str = "trailing",
            fill_edges: bool = False,
    ):
        self.column = column
        self.degree = degree
        self.window_sizes = window_sizes
        self.window_align = window_align
        self.fill_edges = fill_edges


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        df, new_feat_names = featurize.featurize_data(
            df,
            group_columns=["source", "location"],
            features=[
                {
                    "fun": "windowed_taylor_coefs",
                    "args": {
                        "columns": self.column,
                        "taylor_degree": self.degree,
                        "window_align": self.window_align,
                        "window_size": self.window_sizes,
                        "fill_edges": self.fill_edges,
                    },
                }
            ],
        )
        feat_names = feat_names + new_feat_names
        return df, feat_names


class RollingMeanFeature(Feature):
    """Rolling mean over specified window sizes."""


    def __init__(
            self,
            column: str,
            window_sizes: list[int],
            group_columns: list[str] | None = None,
    ):
        self.column = column
        self.window_sizes = window_sizes
        self.group_columns = group_columns if group_columns is not None else ["location"]


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        df, new_feat_names = featurize.featurize_data(
            df,
            group_columns=["source", "location"],
            features=[
                {
                    "fun": "rollmean",
                    "args": {
                        "columns": self.column,
                        "group_columns": self.group_columns,
                        "window_size": self.window_sizes,
                    },
                }
            ],
        )
        feat_names = feat_names + new_feat_names
        return df, feat_names


class LagFeature(Feature):
    """
    Create lagged versions of specified columns.

    When columns=None, FeaturePipeline resolves this to all columns accumulated
    since the previous LagFeature step.
    """


    def __init__(self, columns: list[str] | None, lags: list[int]):
        self.columns = columns
        self.lags = lags


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        if self.columns is None:
            raise ValueError("LagFeature.columns must be resolved by FeaturePipeline before calling apply().")
        df, new_feat_names = featurize.featurize_data(
            df,
            group_columns=["source", "location"],
            features=[
                {
                    "fun": "lag",
                    "args": {
                        "columns": self.columns,
                        "lags": self.lags,
                    },
                }
            ],
        )
        feat_names = feat_names + new_feat_names
        return df, feat_names


class HorizonTargetFeature(Feature):
    """
    Expand df to max_horizon rows per original row; add inc_trans_cs_target,
    horizon, and delta_target columns. Only horizon is added to feat_names.
    """


    def __init__(self, column: str, max_horizon: int):
        self.column = column
        self.max_horizon = max_horizon


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        df, new_feat_names = featurize.featurize_data(
            df,
            group_columns=["source", "location"],
            features=[
                {
                    "fun": "horizon_targets",
                    "args": {
                        "columns": self.column,
                        "horizons": list(range(1, self.max_horizon + 1)),
                    },
                }
            ],
        )
        df["delta_target"] = df[self.column + "_target"] - df[self.column]
        feat_names = feat_names + [f for f in new_feat_names if f == "horizon"]
        return df, feat_names


class LevelFeatureFilter(Feature):
    """
    Remove absolute-level features from feat_names (does not drop df columns).
    Used when incl_level_feats=False.
    """


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        level_feats = (
                ["inc_trans_cs", "inc_trans_cs_lag1", "inc_trans_cs_lag2"]
                + fnmatch.filter(feat_names, "*taylor_d?_c0*")
                + fnmatch.filter(feat_names, "*inc_trans_cs_rollmean*")
        )
        feat_names = [f for f in feat_names if f not in level_feats]
        return df, feat_names


class DirectionalWaveFeature(Feature):
    """
    Spatial wave propagation features based on directional neighbor averages.

    For each direction in `directions`, computes an inverse-distance-weighted average
    of `inc_trans_cs` from neighbors in that direction within `max_distance_km`. Also
    supports an omnidirectional aggregate feature and temporal lags.

    Parameters
    ----------
    directions : list[str]
        Cardinal directions to compute wave features for (subset of {"N", "S", "E", "W"}).
    temporal_lags : list[int]
        Number of weeks to lag each base directional feature.
    max_distance_km : float
        Maximum neighbor distance (km) to include in weighted averages.
    include_velocity : bool
        If True, adds a velocity feature (current minus lag-1) for each base feature.
    include_aggregate : bool
        If True, adds an omnidirectional weighted-average feature (``inc_trans_cs_wave_avg``).
    """


    def __init__(
            self,
            directions: list[str],
            temporal_lags: list[int],
            max_distance_km: float,
            include_velocity: bool = False,
            include_aggregate: bool = True,
    ):
        self.directions = directions
        self.temporal_lags = temporal_lags
        self.max_distance_km = max_distance_km
        self.include_velocity = include_velocity
        self.include_aggregate = include_aggregate


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        df, wave_feat_names = self._compute(df)
        feat_names = feat_names + wave_feat_names
        return df, feat_names


    def _compute(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Compute directional wave features and return augmented df plus feature names."""
        validate_wave_directions(self.directions)

        agg_levels = df["agg_level"].unique()
        if len(agg_levels) > 1:
            raise ValueError(f"Multiple aggregation levels found: {agg_levels}. "
                             "Directional wave features currently support only one agg_level at a time.")

        agg_level = agg_levels[0]

        location_coords = get_location_centroids(agg_level=agg_level)

        locations_in_df = set(df["location"].unique())
        locations_with_coords = set(location_coords.keys())
        missing = locations_in_df - locations_with_coords
        if missing:
            raise ValueError(f"Missing coordinates for locations: {missing}. Cannot compute directional wave features.")

        # Precompute directional neighbors
        neighbor_cache: dict = {}
        for loc in locations_in_df:
            neighbor_cache[loc] = {}
            for direction in self.directions:
                neighbor_cache[loc][direction] = get_directional_neighbors(
                    origin_loc=loc,
                    origin_coord=location_coords[loc],
                    all_coords=location_coords,
                    direction=direction,
                    max_distance_km=self.max_distance_km,
                )

        # Precompute all-direction neighbors for aggregate feature
        all_neighbor_cache: dict = {}
        if self.include_aggregate:
            from idmodels.spatial_utils import haversine_distance
            for loc in locations_in_df:
                neighbors = [(other_loc, haversine_distance(location_coords[loc], coord))
                             for other_loc, coord in location_coords.items()
                             if other_loc != loc]
                neighbors = [(loc, dist) for loc, dist in neighbors if dist <= self.max_distance_km]
                neighbors.sort(key=lambda x: x[1])
                all_neighbor_cache[loc] = neighbors

        df_sorted = df.sort_values(["location", "wk_end_date"]).reset_index(drop=True)
        wave_features: dict = {}


        def _weighted_avg(neighbors, date):
            ws, wt = 0.0, 0.0
            for nloc, dist in neighbors:
                val = df_sorted.loc[
                    (df_sorted["location"] == nloc) & (df_sorted["wk_end_date"] == date),
                    "inc_trans_cs",
                ]
                if len(val) > 0 and not pd.isna(val.iloc[0]):
                    w = 1.0 / dist if dist > 0 else 1.0
                    ws += w * val.iloc[0]
                    wt += w
            return ws / wt if wt > 0 else np.nan


        # Base directional features
        for direction in self.directions:
            feat_name = f"inc_trans_cs_wave_{direction}"
            wave_features[feat_name] = [_weighted_avg(neighbor_cache[row["location"]][direction], row["wk_end_date"])
                                        for _, row in df_sorted.iterrows()]

        # Aggregate feature
        if self.include_aggregate:
            wave_features["inc_trans_cs_wave_avg"] = [
                _weighted_avg(all_neighbor_cache[row["location"]], row["wk_end_date"])
                for _, row in df_sorted.iterrows()]

        for feat_name, vals in wave_features.items():
            df_sorted[feat_name] = vals

        base_feat_names = list(wave_features.keys())

        # Temporal lags
        for feat_name in base_feat_names:
            for lag in self.temporal_lags:
                lagged = f"{feat_name}_lag{lag}"
                df_sorted[lagged] = df_sorted.groupby("location")[feat_name].shift(lag)

        # Velocity features
        if self.include_velocity:
            for feat_name in base_feat_names:
                lag1 = f"{feat_name}_lag1"
                if lag1 not in df_sorted.columns:
                    df_sorted[lag1] = df_sorted.groupby("location")[feat_name].shift(1)
                df_sorted[f"{feat_name}_velocity"] = df_sorted[feat_name] - df_sorted[lag1]

        df_sorted = df_sorted.sort_index()

        wave_feat_names = list(base_feat_names)
        wave_feat_names += [f"{fn}_lag{lag}" for fn in base_feat_names for lag in self.temporal_lags]
        if self.include_velocity:
            wave_feat_names += [f"{fn}_velocity" for fn in base_feat_names]

        return df_sorted, wave_feat_names


class FeaturePipeline:
    """
    Applies a sequence of Feature steps to a DataFrame.

    LagFeature(columns=None) resolves to all columns accumulated since the last
    LagFeature step. Accumulator resets after each LagFeature step.
    initial_feat_names columns are NOT included in the accumulator.
    """


    def __init__(self, features: list[Feature], initial_feat_names: list[str] | None = None):
        self.features = features
        self.initial_feat_names = initial_feat_names or []


    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        feat_names = list(self.initial_feat_names)
        accumulated_new: list[str] = []

        for feature in self.features:
            feat_names_before = list(feat_names)

            if isinstance(feature, LagFeature) and feature.columns is None:
                feature = LagFeature(columns=list(accumulated_new), lags=feature.lags)

            df, feat_names = feature.apply(df, feat_names)
            new_this_step = [f for f in feat_names if f not in feat_names_before]

            if isinstance(feature, LagFeature):
                accumulated_new = []
            else:
                accumulated_new.extend(new_this_step)

        return df, feat_names
