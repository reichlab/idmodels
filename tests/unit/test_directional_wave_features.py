"""Unit tests for directional wave feature generation."""

import numpy as np
import pandas as pd
import pytest
from idmodels.features import DirectionalWaveFeature


def create_test_dataframe():
    """Create a simple test dataframe with synthetic data."""
    dates = pd.date_range("2024-01-01", periods=5, freq="W")
    locations = ["01", "06", "36"]  # Alabama, California, New York

    data = []
    for loc in locations:
        for date in dates:
            data.append({"location": loc,
                         "wk_end_date": date,
                         "inc_trans_cs": np.random.randn(),
                         "agg_level": "state",
                         "source": "nhsn"})

    return pd.DataFrame(data)


def test_create_directional_wave_features_basic():
    """Test basic directional wave feature generation."""
    df = create_test_dataframe()

    df_result, feat_names = DirectionalWaveFeature(directions=["N", "S"],
                                                   temporal_lags=[1],
                                                   max_distance_km=5000,
                                                   include_velocity=False,
                                                   include_aggregate=False,
                                                   ).apply(df, [])

    expected_feats = ["inc_trans_cs_wave_N",
                      "inc_trans_cs_wave_S",
                      "inc_trans_cs_wave_N_lag1",
                      "inc_trans_cs_wave_S_lag1"]

    assert set(feat_names) == set(expected_feats)

    for feat in feat_names:
        assert feat in df_result.columns


def test_create_directional_wave_features_all_directions():
    """Test with all 8 directions."""
    df = create_test_dataframe()

    _, feat_names = DirectionalWaveFeature(directions=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                                           temporal_lags=[],
                                           max_distance_km=5000,
                                           include_velocity=False,
                                           include_aggregate=False,
                                           ).apply(df, [])

    assert len(feat_names) == 8

    for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
        assert f"inc_trans_cs_wave_{direction}" in feat_names


def test_create_directional_wave_features_with_aggregate():
    """Test with aggregate feature enabled."""
    df = create_test_dataframe()

    _, feat_names = DirectionalWaveFeature(directions=["N", "S"],
                                           temporal_lags=[],
                                           max_distance_km=5000,
                                           include_velocity=False,
                                           include_aggregate=True,
                                           ).apply(df, [])

    assert "inc_trans_cs_wave_N" in feat_names
    assert "inc_trans_cs_wave_S" in feat_names
    assert "inc_trans_cs_wave_avg" in feat_names


def test_create_directional_wave_features_with_lags():
    """Test with multiple temporal lags."""
    df = create_test_dataframe()

    _, feat_names = DirectionalWaveFeature(directions=["N"],
                                           temporal_lags=[1, 2],
                                           max_distance_km=5000,
                                           include_velocity=False,
                                           include_aggregate=False,
                                           ).apply(df, [])

    expected_feats = ["inc_trans_cs_wave_N",
                      "inc_trans_cs_wave_N_lag1",
                      "inc_trans_cs_wave_N_lag2"]

    assert set(feat_names) == set(expected_feats)


def test_create_directional_wave_features_with_velocity():
    """Test with velocity features enabled."""
    df = create_test_dataframe()

    df_result, feat_names = DirectionalWaveFeature(directions=["N"],
                                                   temporal_lags=[1],
                                                   max_distance_km=5000,
                                                   include_velocity=True,
                                                   include_aggregate=False,
                                                   ).apply(df, [])

    assert "inc_trans_cs_wave_N" in feat_names
    assert "inc_trans_cs_wave_N_lag1" in feat_names
    assert "inc_trans_cs_wave_N_velocity" in feat_names

    for loc in df_result["location"].unique():
        loc_data = df_result[df_result["location"] == loc].reset_index(drop=True)
        for i in range(1, len(loc_data)):
            base_val = loc_data.loc[i, "inc_trans_cs_wave_N"]
            lag1_val = loc_data.loc[i, "inc_trans_cs_wave_N_lag1"]
            velocity_val = loc_data.loc[i, "inc_trans_cs_wave_N_velocity"]

            if not pd.isna(base_val) and not pd.isna(lag1_val):
                expected_velocity = base_val - lag1_val
                assert abs(velocity_val - expected_velocity) < 1e-6


def test_create_directional_wave_features_lag_semantics():
    """Test that lag1 refers to t-1, lag2 to t-2."""
    df = create_test_dataframe()

    df_result, _ = DirectionalWaveFeature(directions=["N"],
                                          temporal_lags=[1, 2],
                                          max_distance_km=5000,
                                          include_velocity=False,
                                          include_aggregate=False,
                                          ).apply(df, [])

    loc_data = df_result[df_result["location"] == "01"].sort_values("wk_end_date").reset_index(drop=True)

    for i in range(1, len(loc_data)):
        base_prev = loc_data.loc[i - 1, "inc_trans_cs_wave_N"]
        lag1_curr = loc_data.loc[i, "inc_trans_cs_wave_N_lag1"]

        if not pd.isna(base_prev) and not pd.isna(lag1_curr):
            assert abs(base_prev - lag1_curr) < 1e-6


def test_create_directional_wave_features_preserves_index():
    """Test that original dataframe index is preserved."""
    df = create_test_dataframe()
    original_index = df.index.tolist()

    df_result, _ = DirectionalWaveFeature(directions=["N"],
                                          temporal_lags=[],
                                          max_distance_km=5000,
                                          include_velocity=False,
                                          include_aggregate=False,
                                          ).apply(df, [])

    assert df_result.index.tolist() == original_index


def test_create_directional_wave_features_invalid_direction():
    """Test that invalid directions raise ValueError."""
    df = create_test_dataframe()

    with pytest.raises(ValueError, match="Invalid direction"):
        DirectionalWaveFeature(directions=["N", "INVALID"],
                               temporal_lags=[],
                               max_distance_km=5000,
                               include_velocity=False,
                               include_aggregate=False,
                               ).apply(df, [])


def test_create_directional_wave_features_multiple_agg_levels():
    """Test that multiple agg_levels raise ValueError."""
    df = create_test_dataframe()

    df.loc[len(df)] = {"location": "01001",
                       "wk_end_date": pd.Timestamp("2024-01-01"),
                       "inc_trans_cs": 0.5,
                       "agg_level": "county",
                       "source": "nhsn"}

    with pytest.raises(ValueError, match="Multiple aggregation levels"):
        DirectionalWaveFeature(directions=["N"],
                               temporal_lags=[],
                               max_distance_km=5000,
                               include_velocity=False,
                               include_aggregate=False,
                               ).apply(df, [])


def test_create_directional_wave_features_missing_coordinates():
    """Test that missing coordinates raise ValueError."""
    df = create_test_dataframe()

    df.loc[len(df)] = {"location": "FAKE99",
                       "wk_end_date": pd.Timestamp("2024-01-01"),
                       "inc_trans_cs": 0.5,
                       "agg_level": "state",
                       "source": "nhsn"}

    with pytest.raises(ValueError, match="Missing coordinates"):
        DirectionalWaveFeature(directions=["N"],
                               temporal_lags=[],
                               max_distance_km=5000,
                               include_velocity=False,
                               include_aggregate=False,
                               ).apply(df, [])



def test_create_directional_wave_features_feature_count():
    """Test that the correct number of features is generated."""
    df = create_test_dataframe()

    _, feat_names = DirectionalWaveFeature(directions=["N", "S", "E", "W"],
                                           temporal_lags=[1, 2],
                                           max_distance_km=5000,
                                           include_velocity=True,
                                           include_aggregate=True,
                                           ).apply(df, [])

    # 4 base directional + 1 aggregate + (4+1)*2 lags + 5 velocity = 20
    assert len(feat_names) == 20


def test_all_nan_inc_trans_cs_gives_nan_wave_features():
    """When all inc_trans_cs values are NaN, _weighted_avg should return NaN for every row."""
    df = create_test_dataframe()
    df["inc_trans_cs"] = np.nan

    df_result, feat_names = DirectionalWaveFeature(
        directions=["N"],
        temporal_lags=[],
        max_distance_km=5000,
        include_velocity=False,
        include_aggregate=False,
    ).apply(df, [])

    assert "inc_trans_cs_wave_N" in feat_names
    assert df_result["inc_trans_cs_wave_N"].isna().all()


def test_velocity_without_lag1_in_temporal_lags():
    """velocity is computed correctly when temporal_lags doesn't include 1 (lag1 created internally)."""
    df = create_test_dataframe()

    df_result, feat_names = DirectionalWaveFeature(
        directions=["N"],
        temporal_lags=[2],  # lag1 deliberately excluded
        max_distance_km=5000,
        include_velocity=True,
        include_aggregate=False,
    ).apply(df, [])

    assert "inc_trans_cs_wave_N_velocity" in feat_names
    # lag1 was created internally for velocity but should NOT be exposed in feat_names
    assert "inc_trans_cs_wave_N_lag1" not in feat_names
    # lag1 column exists in df (needed for velocity calculation)
    assert "inc_trans_cs_wave_N_lag1" in df_result.columns

    # Velocity semantics: base - lag1 wherever both are non-NaN
    for loc in df_result["location"].unique():
        loc_data = df_result[df_result["location"] == loc].sort_values("wk_end_date").reset_index(drop=True)
        for i in range(len(loc_data)):
            base = loc_data.loc[i, "inc_trans_cs_wave_N"]
            lag1 = loc_data.loc[i, "inc_trans_cs_wave_N_lag1"]
            vel = loc_data.loc[i, "inc_trans_cs_wave_N_velocity"]
            if not (pd.isna(base) or pd.isna(lag1) or pd.isna(vel)):
                assert abs(vel - (base - lag1)) < 1e-10


def test_create_directional_wave_features_no_neighbors():
    """Test behavior when locations have no neighbors in a direction."""
    df = pd.DataFrame([
        {
            "location": "01",
            "wk_end_date": pd.Timestamp("2024-01-01"),
            "inc_trans_cs": 0.5,
            "agg_level": "state",
            "source": "nhsn"
        }
    ])

    df_result, feat_names = DirectionalWaveFeature(directions=["N"],
                                                   temporal_lags=[],
                                                   max_distance_km=10,
                                                   include_velocity=False,
                                                   include_aggregate=False,
                                                   ).apply(df, [])

    assert "inc_trans_cs_wave_N" in feat_names
    assert pd.isna(df_result.loc[0, "inc_trans_cs_wave_N"])
