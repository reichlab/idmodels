"""Unit tests for directional wave feature generation."""

import time

import numpy as np
import pandas as pd
import pytest

from idmodels.features import DirectionalWaveFeature
from idmodels.spatial_utils import get_location_centroids


def create_test_dataframe():
    """Create a simple test dataframe with synthetic data."""
    # Create 3 locations over 5 time points
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

    # Should have base features + lag1 for each direction
    expected_feats = ["inc_trans_cs_wave_N",
                      "inc_trans_cs_wave_S",
                      "inc_trans_cs_wave_N_lag1",
                      "inc_trans_cs_wave_S_lag1"]

    assert set(feat_names) == set(expected_feats)

    # Check that features were added to dataframe
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

    # Should have 8 base features (one per direction)
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

    # Should have N, S, and avg
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

    # Should have base + lag1 + lag2
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

    # Should have base + lag1 + velocity
    assert "inc_trans_cs_wave_N" in feat_names
    assert "inc_trans_cs_wave_N_lag1" in feat_names
    assert "inc_trans_cs_wave_N_velocity" in feat_names

    # Check that velocity is computed correctly (current - lag1)
    for loc in df_result["location"].unique():
        loc_data = df_result[df_result["location"] == loc].reset_index(drop=True)
        # For locations with enough history
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

    # Check lag semantics for one location
    loc_data = df_result[df_result["location"] == "01"].sort_values("wk_end_date").reset_index(drop=True)

    # At index i, lag1 should equal base value at index i-1
    for i in range(1, len(loc_data)):
        base_prev = loc_data.loc[i - 1, "inc_trans_cs_wave_N"]
        lag1_curr = loc_data.loc[i, "inc_trans_cs_wave_N_lag1"]

        # If both exist and are not NaN, they should match
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

    # Index should be preserved
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


def test_non_state_rows_get_nan_and_national_is_not_a_neighbor():
    """Only state rows are featurized; national rows get NaN and never feed state averages."""
    df = create_test_dataframe()
    dates = df["wk_end_date"].unique()
    national = pd.DataFrame({"location": "US",
                             "wk_end_date": dates,
                             "inc_trans_cs": 1e6,
                             "agg_level": "national",
                             "source": "nhsn"})
    df = pd.concat([df, national], ignore_index=True)

    df_result, _ = DirectionalWaveFeature(directions=["N"],
                                          temporal_lags=[1],
                                          max_distance_km=5000,
                                          include_velocity=True,
                                          include_aggregate=True,
                                          ).apply(df, [])

    is_national = df_result["agg_level"] == "national"
    wave_cols = [c for c in df_result.columns if "wave" in c]
    assert df_result.loc[is_national, wave_cols].isna().all().all()
    # The huge national value must not leak into any state's neighbor average
    assert (df_result.loc[~is_national, "inc_trans_cs_wave_avg"].dropna().abs() < 1e3).all()
    assert df_result.loc[~is_national, "inc_trans_cs_wave_avg"].notna().any()


def test_sources_are_featurized_independently():
    """Neighbor averages and lags must stay within a source, not mix e.g. NHSN into NSSP rows."""
    dates = pd.date_range("2024-01-06", periods=4, freq="W-SAT")
    df = pd.DataFrame([
        {"source": src, "location": loc, "wk_end_date": d, "agg_level": "state", "inc_trans_cs": scale * (i + 1)}
        for src, scale in [("nhsn", 1.0), ("nssp", 100.0)]
        for loc in ["25", "33", "50", "09", "44"]  # New England states
        for i, d in enumerate(dates)
    ])

    df_result, _ = DirectionalWaveFeature(directions=["N"],
                                          temporal_lags=[1],
                                          max_distance_km=1000,
                                          include_velocity=False,
                                          include_aggregate=True,
                                          ).apply(df, [])

    # Every location in a source has the same value on a given date, so the neighbor average equals it
    np.testing.assert_allclose(df_result["inc_trans_cs_wave_avg"], df_result["inc_trans_cs"])
    # lag1 is the previous week's value within the same source
    lagged = df_result.dropna(subset=["inc_trans_cs_wave_avg_lag1"])
    assert len(lagged) == 2 * 5 * 3
    np.testing.assert_allclose(lagged["inc_trans_cs_wave_avg_lag1"],
                               lagged["inc_trans_cs"] - np.where(lagged["source"] == "nhsn", 1.0, 100.0))


def test_group_columns_separate_series_within_a_source():
    """Extra group columns (e.g. SMH season/trajectory) define separate series within a source."""
    df = pd.concat([create_test_dataframe().assign(season="a", inc_trans_cs=1.0),
                    create_test_dataframe().assign(season="b", inc_trans_cs=5.0)], ignore_index=True)
    feature = DirectionalWaveFeature(directions=["N"],
                                     temporal_lags=[1],
                                     max_distance_km=5000,
                                     include_velocity=False,
                                     include_aggregate=True,
                                     group_columns=["source", "location", "season"])
    df_result, _ = feature.apply(df, [])
    np.testing.assert_allclose(df_result["inc_trans_cs_wave_avg"], df_result["inc_trans_cs"])
    lagged = df_result.dropna(subset=["inc_trans_cs_wave_avg_lag1"])
    np.testing.assert_allclose(lagged["inc_trans_cs_wave_avg_lag1"], lagged["inc_trans_cs"])

    # Without season in group_columns the rows aren't unique per series
    with pytest.raises(ValueError, match="duplicate rows"):
        DirectionalWaveFeature(directions=["N"], temporal_lags=[], max_distance_km=5000).apply(df, [])


def test_preserves_non_default_index_and_row_order():
    """The caller's index and row order are returned unchanged, even when not sorted by location/date."""
    df = create_test_dataframe().sample(frac=1, random_state=0)
    df.index = df.index + 1000

    df_result, _ = DirectionalWaveFeature(directions=["N"],
                                          temporal_lags=[1],
                                          max_distance_km=5000,
                                          ).apply(df, [])

    assert df_result.index.tolist() == df.index.tolist()
    pd.testing.assert_frame_equal(df_result[df.columns], df)


def test_create_directional_wave_features_missing_coordinates():
    """Test that missing coordinates raise ValueError."""
    df = create_test_dataframe()

    # Add a location without coordinates
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
                                                   max_distance_km=10,  # Very small distance - no neighbors
                                                   include_velocity=False,
                                                   include_aggregate=False,
                                                   ).apply(df, [])

    assert "inc_trans_cs_wave_N" in feat_names
    # Feature should exist but be NaN (no neighbors)
    assert pd.isna(df_result.loc[0, "inc_trans_cs_wave_N"])


def test_create_directional_wave_features_realistic_scale_is_fast():
    """Regression test: computing wave features over a realistic amount of training
    history (all states, ~2 seasons of weekly data) must stay fast. The original
    implementation scanned the full dataframe with a boolean mask for every
    (row, neighbor, direction) combination, which took tens of minutes at this scale;
    the vectorized implementation should take well under a second.
    """
    locations = [loc for loc in get_location_centroids(agg_level="state") if loc != "US"]
    dates = pd.date_range("2023-10-01", periods=100, freq="W")

    rng = np.random.default_rng(0)
    df = pd.DataFrame([
        {
            "location": loc,
            "wk_end_date": date,
            "inc_trans_cs": rng.normal(),
            "agg_level": "state",
            "source": "nhsn",
        }
        for loc in locations
        for date in dates
    ])

    start = time.perf_counter()
    df_result, feat_names = DirectionalWaveFeature(
        directions=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        temporal_lags=[1, 2],
        max_distance_km=1500,
        include_velocity=True,
        include_aggregate=True,
    ).apply(df, [])
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"directional wave feature computation took {elapsed:.1f}s at realistic scale"
    for feat in feat_names:
        assert feat in df_result.columns
