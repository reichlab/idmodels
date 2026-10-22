import fnmatch
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from iddata.utils import get_holidays
from timeseriesutils import featurize

from idmodels.spatial_utils import (
    NATIONAL_LOCATION,
    get_directional_neighbors,
    get_location_centroids,
    get_neighbors_within_distance,
    validate_wave_directions,
)


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
            group_columns: list[str] | None = None,
    ):
        self.column = column
        self.degree = degree
        self.window_sizes = window_sizes
        self.window_align = window_align
        self.fill_edges = fill_edges
        self.group_columns = group_columns if group_columns is not None else ["source", "location"]


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        df, new_feat_names = featurize.featurize_data(
            df,
            group_columns=self.group_columns,
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
        self.group_columns = group_columns if group_columns is not None else ["source", "location"]


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        df, new_feat_names = featurize.featurize_data(
            df,
            group_columns=self.group_columns,
            features=[
                {
                    "fun": "rollmean",
                    "args": {
                        "columns": self.column,
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


    def __init__(self, columns: list[str] | None, lags: list[int], group_columns: list[str] | None = None):
        self.columns = columns
        self.lags = lags
        self.group_columns = group_columns if group_columns is not None else ["source", "location"]


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        if self.columns is None:
            raise ValueError("LagFeature.columns must be resolved by FeaturePipeline before calling apply().")
        df, new_feat_names = featurize.featurize_data(
            df,
            group_columns=self.group_columns,
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


    def __init__(self, column: str, max_horizon: int, group_columns: list[str] | None = None):
        self.column = column
        self.max_horizon = max_horizon
        self.group_columns = group_columns if group_columns is not None else ["source", "location"]


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        df, new_feat_names = featurize.featurize_data(
            df,
            group_columns=self.group_columns,
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
        # delta_target and feat_names filtering are bundled here to enforce the FeaturePipeline
        # contract and prevent callers from accidentally omitting delta_target. An argument could
        # be made for splitting them out: delta_target is a training target, not a feature, and
        # its construction inside a class named HorizonTargetFeature conflates two concerns.
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

    Neighbor averages are computed separately within each series defined by
    `group_columns` (minus "location"), so e.g. NHSN values are never averaged into
    NSSP rows.

    Known limitation: only state-level data is supported. Features are computed for
    rows with ``agg_level == "state"``; all other rows (e.g. national, HSA) get NaN,
    and the national ("US") centroid is never used as a neighbor. Data with no state
    rows raises ValueError.

    Parameters
    ----------
    directions : list[str]
        Cardinal directions to compute wave features for (subset of {"N", "S", "E", "W"}).
    temporal_lags : list[int]
        List of weeks to lag each base directional feature.
    max_distance_km : float
        Maximum neighbor distance (km) to include in weighted averages.
    include_velocity : bool
        If True, adds a velocity feature (current minus lag-1) for each base feature.
    include_aggregate : bool
        If True, adds an omnidirectional weighted-average feature (``inc_trans_cs_wave_avg``).
    group_columns : list[str] | None
        Columns identifying a single time series; must include "location". Defaults to
        ["source", "location"].
    """


    def __init__(
            self,
            directions: list[str],
            temporal_lags: list[int],
            max_distance_km: float,
            include_velocity: bool = False,
            include_aggregate: bool = True,
            group_columns: list[str] | None = None,
    ):
        self.directions = directions
        self.temporal_lags = temporal_lags
        self.max_distance_km = max_distance_km
        self.include_velocity = include_velocity
        self.include_aggregate = include_aggregate
        self.group_columns = group_columns if group_columns is not None else ["source", "location"]
        if "location" not in self.group_columns:
            raise ValueError(f"group_columns must include 'location', got {self.group_columns}.")


    def apply(self, df: pd.DataFrame, feat_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
        df, wave_feat_names = self._compute(df)
        feat_names = feat_names + wave_feat_names
        return df, feat_names


    def _compute(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Compute directional wave features and return augmented df plus feature names."""
        validate_wave_directions(self.directions)

        # Only state-level rows are featurized; the national centroid is not a valid neighbor.
        location_coords = get_location_centroids(agg_level="state")
        location_coords.pop(NATIONAL_LOCATION, None)

        # Work on a RangeIndex copy so shifts/assignments align; the caller's index is restored at the end
        orig_index = df.index
        df = df.reset_index(drop=True)
        df_state = df.loc[df["agg_level"] == "state"]
        if df_state.empty:
            raise ValueError(f"Directional wave features require state-level rows; found only agg_level values "
                             f"{sorted(df['agg_level'].unique())}.")

        locations = sorted(df_state["location"].unique())
        missing = set(locations) - set(location_coords.keys())
        if missing:
            raise ValueError(f"Missing coordinates for locations: {missing}. Cannot compute directional wave features.")

        series_cols = [c for c in self.group_columns if c != "location"]
        key_cols = series_cols + ["location", "wk_end_date"]
        dupes = df_state.duplicated(subset=key_cols)
        if dupes.any():
            raise ValueError(f"Directional wave features require one row per {key_cols}; found {dupes.sum()} "
                             f"duplicate rows. Check group_columns.")

        # Inverse-distance weight matrix per base feature: weights[feat][i, j] is the weight location j
        # gets in location i's neighbor average (0 if j is not a neighbor of i).
        loc_idx = {loc: i for i, loc in enumerate(locations)}

        def _weight_matrix(neighbors_of) -> np.ndarray:
            w = np.zeros((len(locations), len(locations)))
            for loc in locations:
                for nloc, dist in neighbors_of(loc):
                    if nloc in loc_idx:
                        w[loc_idx[loc], loc_idx[nloc]] = 1.0 / dist if dist > 0 else 1.0
            return w

        weights: dict[str, np.ndarray] = {}
        for direction in self.directions:
            weights[f"inc_trans_cs_wave_{direction}"] = _weight_matrix(
                lambda loc, direction=direction: get_directional_neighbors(
                    origin_loc=loc,
                    origin_coord=location_coords[loc],
                    all_coords=location_coords,
                    direction=direction,
                    max_distance_km=self.max_distance_km,
                )
            )
        if self.include_aggregate:
            weights["inc_trans_cs_wave_avg"] = _weight_matrix(
                lambda loc: get_neighbors_within_distance(
                    origin_loc=loc,
                    origin_coord=location_coords[loc],
                    all_coords=location_coords,
                    max_distance_km=self.max_distance_km,
                )
            )
        base_feat_names = list(weights.keys())

        # Weighted neighbor averages, computed separately within each series (e.g. source)
        series_groups = df_state.groupby(series_cols, sort=False, dropna=False) if series_cols else [((), df_state)]
        long_frames = []
        for key, grp in series_groups:
            pivot = grp.pivot(index="wk_end_date", columns="location", values="inc_trans_cs").reindex(columns=locations)
            values = pivot.to_numpy(dtype=float)
            valid = ~np.isnan(values)
            filled = np.where(valid, values, 0.0)

            long = pd.DataFrame({
                "wk_end_date": np.repeat(pivot.index.to_numpy(), len(locations)),
                "location": np.tile(np.array(locations, dtype=object), len(pivot.index)),
            })
            for feat_name, w in weights.items():
                weighted_sum = filled @ w.T
                weight_total = valid.astype(float) @ w.T
                with np.errstate(invalid="ignore", divide="ignore"):
                    result = weighted_sum / weight_total
                result[weight_total == 0] = np.nan
                long[feat_name] = result.ravel()

            key = key if isinstance(key, tuple) else (key,)
            for col, val in zip(series_cols, key):
                long[col] = val
            long_frames.append(long)

        if long_frames:
            wave_long = pd.concat(long_frames, ignore_index=True)
        else:
            wave_long = pd.DataFrame(columns=key_cols + base_feat_names)
        wave_long["agg_level"] = "state"

        merged = df[["agg_level"] + key_cols].merge(wave_long, on=["agg_level"] + key_cols, how="left")
        for feat_name in base_feat_names:
            df[feat_name] = merged[feat_name].to_numpy(dtype=float)

        # Temporal lags and velocity, shifted within each series in date order
        lag_groups = (
            df.sort_values(self.group_columns + ["wk_end_date"], kind="stable")
            .groupby(["agg_level"] + self.group_columns, sort=False, dropna=False)
        )
        for feat_name in base_feat_names:
            for lag in self.temporal_lags:
                df[f"{feat_name}_lag{lag}"] = lag_groups[feat_name].shift(lag)

        if self.include_velocity:
            for feat_name in base_feat_names:
                lag1 = f"{feat_name}_lag1"
                if lag1 not in df.columns:
                    warnings.warn(f"include_velocity=True: adding extra column '{lag1}' (not in temporal_lags) to "
                                  f"compute velocity.")
                    df[lag1] = lag_groups[feat_name].shift(1)
                df[f"{feat_name}_velocity"] = df[feat_name] - df[lag1]

        df.index = orig_index

        # Collect all feature names
        wave_feat_names = list(base_feat_names)
        wave_feat_names += [f"{fn}_lag{lag}" for fn in base_feat_names for lag in self.temporal_lags]
        if self.include_velocity:
            wave_feat_names += [f"{fn}_velocity" for fn in base_feat_names]

        return df, wave_feat_names


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
                feature = LagFeature(columns=list(accumulated_new), lags=feature.lags,
                                     group_columns=feature.group_columns)

            df, feat_names = feature.apply(df, feat_names)
            new_this_step = [f for f in feat_names if f not in feat_names_before]

            if isinstance(feature, LagFeature):
                accumulated_new = []
            else:
                accumulated_new.extend(new_this_step)

        return df, feat_names
