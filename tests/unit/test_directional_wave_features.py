"""Unit tests for directional wave feature generation."""

import numpy as np
import pandas as pd
import pytest

from idmodels.preprocess import create_directional_wave_features


def create_test_dataframe():
    """Create a simple test dataframe with synthetic data."""
    # Create 3 locations over 5 time points
    dates = pd.date_range("2024-01-01", periods=5, freq="W")
    locations = ["01", "06", "36"]  # Alabama, California, New York

    data = []
    for loc in locations:
        for date in dates:
            data.append({
                "location": loc,
                "wk_end_date": date,
                "inc_trans_cs": np.random.randn(),
                "agg_level": "state",
                "source": "nhsn"
            })

    return pd.DataFrame(data)


def test_create_directional_wave_features_disabled():
    """Test that function returns empty list when disabled."""
    df = create_test_dataframe()
    original_cols = set(df.columns)

    # Test with None config
    df_result, feat_names = create_directional_wave_features(df, wave_config=None)
    assert feat_names == []
    assert set(df_result.columns) == original_cols

    # Test with enabled=False
    wave_config = {"enabled": False}
    df_result, feat_names = create_directional_wave_features(df, wave_config)
    assert feat_names == []
    assert set(df_result.columns) == original_cols


def test_create_directional_wave_features_basic():
    """Test basic directional wave feature generation."""
    df = create_test_dataframe()

    wave_config = {
        "enabled": True,
        "directions": ["N", "S"],
        "temporal_lags": [1],
        "max_distance_km": 5000,
        "include_velocity": False,
        "include_aggregate": False
    }

    df_result, feat_names = create_directional_wave_features(df, wave_config)

    # Should have base features + lag1 for each direction
    expected_feats = [
        "inc_trans_cs_wave_N",
        "inc_trans_cs_wave_S",
        "inc_trans_cs_wave_N_lag1",
        "inc_trans_cs_wave_S_lag1"
    ]

    assert set(feat_names) == set(expected_feats)

    # Check that features were added to dataframe
    for feat in feat_names:
        assert feat in df_result.columns


def test_create_directional_wave_features_all_directions():
    """Test with all 8 directions."""
    df = create_test_dataframe()

    wave_config = {
        "enabled": True,
        "directions": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        "temporal_lags": [],  # No lags for simplicity
        "max_distance_km": 5000,
        "include_velocity": False,
        "include_aggregate": False
    }

    df_result, feat_names = create_directional_wave_features(df, wave_config)

    # Should have 8 base features (one per direction)
    assert len(feat_names) == 8

    expected_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    for direction in expected_directions:
        assert f"inc_trans_cs_wave_{direction}" in feat_names


def test_create_directional_wave_features_with_aggregate():
    """Test with aggregate feature enabled."""
    df = create_test_dataframe()

    wave_config = {
        "enabled": True,
        "directions": ["N", "S"],
        "temporal_lags": [],
        "max_distance_km": 5000,
        "include_velocity": False,
        "include_aggregate": True
    }

    df_result, feat_names = create_directional_wave_features(df, wave_config)

    # Should have N, S, and avg
    assert "inc_trans_cs_wave_N" in feat_names
    assert "inc_trans_cs_wave_S" in feat_names
    assert "inc_trans_cs_wave_avg" in feat_names


def test_create_directional_wave_features_with_lags():
    """Test with multiple temporal lags."""
    df = create_test_dataframe()

    wave_config = {
        "enabled": True,
        "directions": ["N"],
        "temporal_lags": [1, 2],
        "max_distance_km": 5000,
        "include_velocity": False,
        "include_aggregate": False
    }

    df_result, feat_names = create_directional_wave_features(df, wave_config)

    # Should have base + lag1 + lag2
    expected_feats = [
        "inc_trans_cs_wave_N",
        "inc_trans_cs_wave_N_lag1",
        "inc_trans_cs_wave_N_lag2"
    ]

    assert set(feat_names) == set(expected_feats)


def test_create_directional_wave_features_with_velocity():
    """Test with velocity features enabled."""
    df = create_test_dataframe()

    wave_config = {
        "enabled": True,
        "directions": ["N"],
        "temporal_lags": [1],
        "max_distance_km": 5000,
        "include_velocity": True,
        "include_aggregate": False
    }

    df_result, feat_names = create_directional_wave_features(df, wave_config)

    # Should have base + lag1 + velocity
    assert "inc_trans_cs_wave_N" in feat_names
    assert "inc_trans_cs_wave_N_lag1" in feat_names
    assert "inc_trans_cs_wave_N_velocity" in feat_names

    # Check that velocity is computed correctly (current - lag1)
    # For locations with enough history
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

    wave_config = {
        "enabled": True,
        "directions": ["N"],
        "temporal_lags": [1, 2],
        "max_distance_km": 5000,
        "include_velocity": False,
        "include_aggregate": False
    }

    df_result, feat_names = create_directional_wave_features(df, wave_config)

    # Check lag semantics for one location
    loc_data = df_result[df_result["location"] == "01"].sort_values("wk_end_date").reset_index(drop=True)

    # At index i, lag1 should equal base value at index i-1
    for i in range(1, len(loc_data)):
        base_prev = loc_data.loc[i-1, "inc_trans_cs_wave_N"]
        lag1_curr = loc_data.loc[i, "inc_trans_cs_wave_N_lag1"]

        # If both exist and are not NaN, they should match
        if not pd.isna(base_prev) and not pd.isna(lag1_curr):
            assert abs(base_prev - lag1_curr) < 1e-6


def test_create_directional_wave_features_preserves_index():
    """Test that original dataframe index is preserved."""
    df = create_test_dataframe()
    original_index = df.index.tolist()

    wave_config = {
        "enabled": True,
        "directions": ["N"],
        "temporal_lags": [],
        "max_distance_km": 5000,
        "include_velocity": False,
        "include_aggregate": False
    }

    df_result, _ = create_directional_wave_features(df, wave_config)

    # Index should be preserved
    assert df_result.index.tolist() == original_index


def test_create_directional_wave_features_invalid_direction():
    """Test that invalid directions raise ValueError."""
    df = create_test_dataframe()

    wave_config = {
        "enabled": True,
        "directions": ["N", "INVALID"],
        "temporal_lags": [],
        "max_distance_km": 5000,
        "include_velocity": False,
        "include_aggregate": False
    }

    with pytest.raises(ValueError, match="Invalid direction"):
        create_directional_wave_features(df, wave_config)


def test_create_directional_wave_features_multiple_agg_levels():
    """Test that multiple agg_levels raise ValueError."""
    df = create_test_dataframe()

    # Add a row with different agg_level
    df.loc[len(df)] = {
        "location": "01001",
        "wk_end_date": pd.Timestamp("2024-01-01"),
        "inc_trans_cs": 0.5,
        "agg_level": "county",
        "source": "nhsn"
    }

    wave_config = {
        "enabled": True,
        "directions": ["N"],
        "temporal_lags": [],
        "max_distance_km": 5000,
        "include_velocity": False,
        "include_aggregate": False
    }

    with pytest.raises(ValueError, match="Multiple aggregation levels"):
        create_directional_wave_features(df, wave_config)


def test_create_directional_wave_features_missing_coordinates():
    """Test that missing coordinates raise ValueError."""
    df = create_test_dataframe()

    # Add a location without coordinates
    df.loc[len(df)] = {
        "location": "FAKE99",
        "wk_end_date": pd.Timestamp("2024-01-01"),
        "inc_trans_cs": 0.5,
        "agg_level": "state",
        "source": "nhsn"
    }

    wave_config = {
        "enabled": True,
        "directions": ["N"],
        "temporal_lags": [],
        "max_distance_km": 5000,
        "include_velocity": False,
        "include_aggregate": False
    }

    with pytest.raises(ValueError, match="Missing coordinates"):
        create_directional_wave_features(df, wave_config)


def test_create_directional_wave_features_default_config():
    """Test that default configuration values work."""
    df = create_test_dataframe()

    # Minimal config - should use defaults
    wave_config = {
        "enabled": True
    }

    df_result, feat_names = create_directional_wave_features(df, wave_config)

    # Should use default 8 directions
    assert len([f for f in feat_names if "lag" not in f and "velocity" not in f]) == 9  # 8 directions + avg

    # Should have lag1 and lag2 (default temporal_lags=[1, 2])
    assert any("lag1" in f for f in feat_names)
    assert any("lag2" in f for f in feat_names)


def test_create_directional_wave_features_feature_count():
    """Test that the correct number of features is generated."""
    df = create_test_dataframe()

    wave_config = {
        "enabled": True,
        "directions": ["N", "S", "E", "W"],  # 4 directions
        "temporal_lags": [1, 2],  # 2 lags
        "max_distance_km": 5000,
        "include_velocity": True,  # Add velocity
        "include_aggregate": True  # Add aggregate
    }

    df_result, feat_names = create_directional_wave_features(df, wave_config)

    # Expected:
    # - 4 base directional features
    # - 1 aggregate feature
    # - (4 + 1) * 2 lags = 10 lag features
    # - 5 velocity features (4 directions + 1 aggregate)
    # Total: 4 + 1 + 10 + 5 = 20
    assert len(feat_names) == 20


def test_create_directional_wave_features_no_neighbors():
    """Test behavior when locations have no neighbors in a direction."""
    # Create single location
    df = pd.DataFrame([
        {
            "location": "01",
            "wk_end_date": pd.Timestamp("2024-01-01"),
            "inc_trans_cs": 0.5,
            "agg_level": "state",
            "source": "nhsn"
        }
    ])

    wave_config = {
        "enabled": True,
        "directions": ["N"],
        "temporal_lags": [],
        "max_distance_km": 10,  # Very small distance - no neighbors
        "include_velocity": False,
        "include_aggregate": False
    }

    df_result, feat_names = create_directional_wave_features(df, wave_config)

    # Feature should exist but be NaN (no neighbors)
    assert "inc_trans_cs_wave_N" in feat_names
    assert pd.isna(df_result.loc[0, "inc_trans_cs_wave_N"])
