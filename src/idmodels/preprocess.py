import fnmatch

import numpy as np
import pandas as pd
from iddata.utils import get_holidays
from timeseriesutils import featurize

from idmodels.spatial_utils import (
    get_location_centroids,
    get_directional_neighbors,
    validate_wave_directions
)


def create_features_and_targets(df, incl_level_feats, max_horizon, curr_feat_names = []):
    '''
    Create features and targets for prediction
    
    Parameters
    ----------
    df: pandas dataframe
      data frame with data to "featurize"
    incl_level_feats: boolean
      include features that are a measure of local level of the signal?
    max_horizon: int
      maximum forecast horizon
    curr_feat_names: list of strings
      list of names of columns in `df` containing existing features
    
    Returns
    -------
    tuple with:
    - the input data frame, augmented with additional columns with feature and
      target values
    - a list of all feature names, columns in the data frame
    '''
    
    # current features; will be updated
    feat_names = curr_feat_names
    
    # one-hot encodings of data source, agg_level, and location
    for c in ["source", "agg_level", "location"]:
        ohe = pd.get_dummies(df[c], prefix=c)
        df = pd.concat([df, ohe], axis=1)
        feat_names = feat_names + list(ohe.columns)
    
    # season week relative to christmas
    df = df.merge(
            get_holidays() \
                .query("holiday == 'Christmas Day'") \
                .drop(columns=["holiday", "date"]) \
                .rename(columns={"season_week": "xmas_week"}),
            how="left",
            on="season") \
        .assign(delta_xmas = lambda x: x["season_week"] - x["xmas_week"])
    
    feat_names = feat_names + ["delta_xmas"]
    
    # features summarizing data within each combination of source and location
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=["source", "location"],
        features = [
            {
                "fun": "windowed_taylor_coefs",
                "args": {
                    "columns": "inc_trans_cs",
                    "taylor_degree": 2,
                    "window_align": "trailing",
                    "window_size": [4, 6],
                    "fill_edges": False
                }
            },
            {
                "fun": "windowed_taylor_coefs",
                "args": {
                    "columns": "inc_trans_cs",
                    "taylor_degree": 1,
                    "window_align": "trailing",
                    "window_size": [3, 5],
                    "fill_edges": False
                }
            },
            {
                "fun": "rollmean",
                "args": {
                    "columns": "inc_trans_cs",
                    "group_columns": ["location"],
                    "window_size": [2, 4]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=["source", "location"],
        features = [
            {
                "fun": "lag",
                "args": {
                    "columns": ["inc_trans_cs"] + new_feat_names,
                    "lags": [1, 2]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    # add forecast targets
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=["source", "location"],
        features = [
            {
                "fun": "horizon_targets",
                "args": {
                    "columns": "inc_trans_cs",
                    "horizons": [(i + 1) for i in range(max_horizon)]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    # we will model the differences between the prediction target and the most
    # recent observed value
    df["delta_target"] = df["inc_trans_cs_target"] - df["inc_trans_cs"]
    
    # if requested, drop features that involve absolute level
    if not incl_level_feats:
        feat_names = _drop_level_feats(feat_names)
    
    return df, feat_names


def _drop_level_feats(feat_names):
    level_feats = ["inc_trans_cs", "inc_trans_cs_lag1", "inc_trans_cs_lag2"] + \
                  fnmatch.filter(feat_names, "*taylor_d?_c0*") + \
                  fnmatch.filter(feat_names, "*inc_trans_cs_rollmean*")
    feat_names = [f for f in feat_names if f not in level_feats]
    return feat_names


def create_directional_wave_features(df, wave_config=None):
    """
    Create spatial directional wave features.

    For each location and time point, computes distance-weighted averages
    of neighboring locations' incidence in specified directions (e.g., N, S, E, W).
    Also computes lagged versions and optionally velocity (rate of change) features.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame with columns: location, wk_end_date, inc_trans_cs, agg_level
    wave_config : dict, optional
        Configuration dictionary with keys:
        - 'enabled': bool (default: False) - whether to generate features
        - 'directions': list of str (default: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
          Subset of: N, NE, E, SE, S, SW, W, NW
        - 'temporal_lags': list of int (default: [1, 2]) - temporal lags to include
          lag1 means t-1, lag2 means t-2, etc.
        - 'max_distance_km': float (default: 1000) - max distance for neighbors
        - 'include_velocity': bool (default: False) - include rate-of-change features
        - 'include_aggregate': bool (default: True) - include overall weighted average

    Returns
    -------
    df : pandas.DataFrame
        Input dataframe augmented with wave features
    wave_feat_names : list of str
        List of new feature names added to df

    Notes
    -----
    - Lag semantics: lag1 uses time t-1, lag2 uses time t-2, etc.
    - Velocity features compute: wave(t) - wave(t-1)
    - Distance weighting uses inverse distance: weight = 1 / distance
    - Features are computed per location and time point
    """
    # Return early if not enabled
    if wave_config is None or not wave_config.get('enabled', False):
        return df, []

    # Extract configuration with defaults
    directions = wave_config.get('directions', ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    temporal_lags = wave_config.get('temporal_lags', [1, 2])
    max_distance_km = wave_config.get('max_distance_km', 1000)
    include_velocity = wave_config.get('include_velocity', False)
    include_aggregate = wave_config.get('include_aggregate', True)

    # Validate directions
    validate_wave_directions(directions)

    # Get aggregation level(s) from dataframe
    agg_levels = df['agg_level'].unique()
    if len(agg_levels) > 1:
        raise ValueError(
            f"Multiple aggregation levels found: {agg_levels}. "
            f"Directional wave features currently support only one agg_level at a time."
        )
    agg_level = agg_levels[0]

    # Get location centroids for this aggregation level
    try:
        location_coords = get_location_centroids(agg_level=agg_level)
    except ValueError as e:
        raise ValueError(
            f"Cannot create directional wave features: {str(e)}"
        )

    # Filter to locations present in both data and coordinate lookup
    locations_in_df = set(df['location'].unique())
    locations_with_coords = set(location_coords.keys())
    locations_to_use = locations_in_df.intersection(locations_with_coords)

    if len(locations_to_use) < len(locations_in_df):
        missing = locations_in_df - locations_with_coords
        raise ValueError(
            f"Missing coordinates for locations: {missing}. "
            f"Cannot compute directional wave features."
        )

    # Precompute directional neighbors for each location
    neighbor_cache = {}
    for loc in locations_to_use:
        neighbor_cache[loc] = {}
        for direction in directions:
            neighbors = get_directional_neighbors(
                origin_loc=loc,
                origin_coord=location_coords[loc],
                all_coords=location_coords,
                direction=direction,
                max_distance_km=max_distance_km
            )
            neighbor_cache[loc][direction] = neighbors

    # Also compute all neighbors (for aggregate feature)
    if include_aggregate:
        all_neighbor_cache = {}
        for loc in locations_to_use:
            # Get all neighbors regardless of direction
            neighbors = []
            for other_loc, other_coord in location_coords.items():
                if other_loc == loc:
                    continue
                from idmodels.spatial_utils import haversine_distance
                distance = haversine_distance(location_coords[loc], other_coord)
                if distance <= max_distance_km:
                    neighbors.append((other_loc, distance))
            neighbors.sort(key=lambda x: x[1])
            all_neighbor_cache[loc] = neighbors

    # Create features for each direction
    wave_features = {}

    # Sort dataframe by location and date for efficient processing
    df_sorted = df.sort_values(['location', 'wk_end_date']).reset_index(drop=True)

    # Compute base directional features (at time t)
    for direction in directions:
        feat_name = f'inc_trans_cs_wave_{direction}'
        feat_values = []

        for idx, row in df_sorted.iterrows():
            loc = row['location']
            date = row['wk_end_date']

            # Get neighbors in this direction
            neighbors = neighbor_cache[loc][direction]

            if len(neighbors) == 0:
                # No neighbors in this direction
                feat_values.append(np.nan)
                continue

            # Compute distance-weighted average
            weighted_sum = 0.0
            weight_sum = 0.0

            for neighbor_loc, distance in neighbors:
                # Get neighbor's inc_trans_cs at same time point
                neighbor_value = df_sorted[
                    (df_sorted['location'] == neighbor_loc) &
                    (df_sorted['wk_end_date'] == date)
                ]['inc_trans_cs']

                if len(neighbor_value) > 0 and not pd.isna(neighbor_value.iloc[0]):
                    # Inverse distance weighting
                    weight = 1.0 / distance if distance > 0 else 1.0
                    weighted_sum += weight * neighbor_value.iloc[0]
                    weight_sum += weight

            if weight_sum > 0:
                feat_values.append(weighted_sum / weight_sum)
            else:
                feat_values.append(np.nan)

        wave_features[feat_name] = feat_values

    # Compute aggregate feature (overall weighted average)
    if include_aggregate:
        feat_name = 'inc_trans_cs_wave_avg'
        feat_values = []

        for idx, row in df_sorted.iterrows():
            loc = row['location']
            date = row['wk_end_date']

            neighbors = all_neighbor_cache[loc]

            if len(neighbors) == 0:
                feat_values.append(np.nan)
                continue

            # Compute distance-weighted average
            weighted_sum = 0.0
            weight_sum = 0.0

            for neighbor_loc, distance in neighbors:
                neighbor_value = df_sorted[
                    (df_sorted['location'] == neighbor_loc) &
                    (df_sorted['wk_end_date'] == date)
                ]['inc_trans_cs']

                if len(neighbor_value) > 0 and not pd.isna(neighbor_value.iloc[0]):
                    weight = 1.0 / distance if distance > 0 else 1.0
                    weighted_sum += weight * neighbor_value.iloc[0]
                    weight_sum += weight

            if weight_sum > 0:
                feat_values.append(weighted_sum / weight_sum)
            else:
                feat_values.append(np.nan)

        wave_features[feat_name] = feat_values

    # Add base features to dataframe
    for feat_name, feat_values in wave_features.items():
        df_sorted[feat_name] = feat_values

    # Create lagged features
    lagged_features = {}
    base_feat_names = list(wave_features.keys())

    for feat_name in base_feat_names:
        for lag in temporal_lags:
            lagged_feat_name = f'{feat_name}_lag{lag}'
            # Use groupby to create lags within each location
            df_sorted[lagged_feat_name] = df_sorted.groupby('location')[feat_name].shift(lag)
            lagged_features[lagged_feat_name] = None  # Just track the name

    # Create velocity features (rate of change)
    if include_velocity:
        velocity_features = {}
        for feat_name in base_feat_names:
            # Velocity = current - lag1
            lag1_name = f'{feat_name}_lag1'
            if lag1_name in df_sorted.columns or 1 in temporal_lags:
                velocity_feat_name = f'{feat_name}_velocity'
                if lag1_name not in df_sorted.columns:
                    # Need to create lag1 if it doesn't exist
                    df_sorted[lag1_name] = df_sorted.groupby('location')[feat_name].shift(1)
                df_sorted[velocity_feat_name] = df_sorted[feat_name] - df_sorted[lag1_name]
                velocity_features[velocity_feat_name] = None

    # Restore original index order
    df_sorted = df_sorted.sort_index()

    # Collect all feature names
    wave_feat_names = list(wave_features.keys())
    wave_feat_names += list(lagged_features.keys())
    if include_velocity:
        wave_feat_names += list(velocity_features.keys())

    return df_sorted, wave_feat_names

