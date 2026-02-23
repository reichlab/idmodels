"""Integration test for GBQR model with directional wave features."""


import numpy as np
import pandas as pd

from idmodels.config import DataSource, GBQRModelConfig, PowerTransform
from idmodels.preprocess import create_directional_wave_features, create_features_and_targets


def create_realistic_test_data():
    """Create realistic test data mimicking the structure from DiseaseDataLoader."""
    # Create data for several states over multiple weeks
    np.random.seed(42)

    states = ["01", "06", "13", "36", "42", "48"]  # AL, CA, GA, NY, PA, TX
    dates = pd.date_range("2023-10-01", periods=20, freq="W")

    data = []
    for state in states:
        for i, date in enumerate(dates):
            # Create somewhat realistic progression of incidence
            base_inc = 0.5 + 0.3 * np.sin(i / 10) + np.random.randn() * 0.1

            data.append({
                "agg_level": "state",
                "location": state,
                "season": "2023-24",
                "season_week": (i % 52) + 1,
                "wk_end_date": date,
                "inc": max(0, base_inc),
                "source": "nhsn",
                "pop": 1000000 + int(state) * 100000,
                "log_pop": np.log(1000000 + int(state) * 100000),
                "inc_trans": base_inc,
                "inc_trans_scale_factor": 0.5,
                "inc_trans_cs": base_inc * 0.5,
                "inc_trans_center_factor": 0.1
            })

    return pd.DataFrame(data)


def test_gbqr_preprocessing_without_waves():
    """Test that GBQR preprocessing works without wave features (backwards compatibility)."""
    df = create_realistic_test_data()

    init_feats = ["inc_trans_cs", "log_pop"]

    # This should work without wave features (backwards compatibility)
    df_result, feat_names = create_features_and_targets(
        df=df,
        incl_level_feats=True,
        max_horizon=3,
        curr_feat_names=init_feats
    )

    # Check that basic features are present
    assert "inc_trans_cs" in feat_names
    assert "log_pop" in feat_names

    # Check that no wave features are present
    wave_feats = [f for f in feat_names if "wave" in f]
    assert len(wave_feats) == 0


def test_gbqr_preprocessing_with_waves_enabled():
    """Test that GBQR preprocessing works with wave features enabled."""
    df = create_realistic_test_data()

    # Create directional wave features
    wave_config = {
        "enabled": True,
        "directions": ["N", "S", "E", "W"],
        "temporal_lags": [1, 2],
        "max_distance_km": 2000,
        "include_velocity": False,
        "include_aggregate": True
    }

    df_with_waves, wave_feat_names = create_directional_wave_features(df, wave_config)

    # Check that wave features were created
    assert len(wave_feat_names) > 0
    assert "inc_trans_cs_wave_N" in wave_feat_names
    assert "inc_trans_cs_wave_S" in wave_feat_names
    assert "inc_trans_cs_wave_E" in wave_feat_names
    assert "inc_trans_cs_wave_W" in wave_feat_names
    assert "inc_trans_cs_wave_avg" in wave_feat_names
    assert "inc_trans_cs_wave_N_lag1" in wave_feat_names
    assert "inc_trans_cs_wave_N_lag2" in wave_feat_names

    # Now pass through the full feature creation pipeline
    init_feats = ["inc_trans_cs", "log_pop"] + wave_feat_names

    df_result, feat_names = create_features_and_targets(
        df=df_with_waves,
        incl_level_feats=True,
        max_horizon=3,
        curr_feat_names=init_feats
    )

    # Check that all wave features are in the final feature list
    for wave_feat in wave_feat_names:
        assert wave_feat in feat_names

    # Check that basic features are still present
    assert "inc_trans_cs" in feat_names
    assert "log_pop" in feat_names

    # Check that targets were created
    assert "delta_target" in df_result.columns


def test_gbqr_preprocessing_with_all_wave_options():
    """Test GBQR preprocessing with all wave feature options enabled."""
    df = create_realistic_test_data()

    # Create directional wave features with all options
    wave_config = {
        "enabled": True,
        "directions": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        "temporal_lags": [1, 2],
        "max_distance_km": 2000,
        "include_velocity": True,
        "include_aggregate": True
    }

    df_with_waves, wave_feat_names = create_directional_wave_features(df, wave_config)

    # Expected features:
    # - 8 directions
    # - 1 aggregate
    # - Each has: base + lag1 + lag2 + velocity
    # Total: 9 * 4 = 36 features
    expected_feature_count = 9 * 4
    assert len(wave_feat_names) == expected_feature_count

    # Check that velocity features exist
    assert "inc_trans_cs_wave_N_velocity" in wave_feat_names
    assert "inc_trans_cs_wave_avg_velocity" in wave_feat_names

    # Pass through full pipeline
    init_feats = ["inc_trans_cs", "log_pop"] + wave_feat_names

    df_result, feat_names = create_features_and_targets(
        df=df_with_waves,
        incl_level_feats=True,
        max_horizon=3,
        curr_feat_names=init_feats
    )

    # All wave features should be in final feature list
    for wave_feat in wave_feat_names:
        assert wave_feat in feat_names


def test_gbqr_wave_features_no_nan_for_valid_data():
    """Test that wave features produce valid values for locations with neighbors."""
    df = create_realistic_test_data()

    wave_config = {
        "enabled": True,
        "directions": ["N", "S"],
        "temporal_lags": [],
        "max_distance_km": 3000,
        "include_velocity": False,
        "include_aggregate": True
    }

    df_with_waves, wave_feat_names = create_directional_wave_features(df, wave_config)

    # For aggregate feature, most locations should have some neighbors
    # Check that we have at least some non-NaN values
    avg_feature = df_with_waves["inc_trans_cs_wave_avg"]
    non_nan_count = (~avg_feature.isna()).sum()

    # At least half of the values should be non-NaN (locations have neighbors)
    assert non_nan_count > len(df_with_waves) * 0.5


def test_gbqr_wave_features_with_model_config_pattern():
    """Test wave features using the model_config pattern from GBQR."""
    df = create_realistic_test_data()

    # Simulate model_config with wave feature settings
    model_config = GBQRModelConfig(
        model_name="gbqr_wave_test",
        sources=[DataSource.NHSN],
        fit_locations_separately=False,
        power_transform=PowerTransform.FOURTH_ROOT,
        use_directional_waves=True,
        wave_directions=["N", "S", "E", "W"],
        wave_temporal_lags=[1, 2],
        wave_max_distance_km=2000,
        wave_include_velocity=False,
        wave_include_aggregate=True
    )

    # This is how it would be called in GBQR.run()
    init_feats = ["inc_trans_cs", "log_pop"]

    if hasattr(model_config, "use_directional_waves") and model_config.use_directional_waves:
        wave_config = {
            "enabled": True,
            "directions": model_config.wave_directions,
            "temporal_lags": model_config.wave_temporal_lags,
            "max_distance_km": model_config.wave_max_distance_km,
            "include_velocity": model_config.wave_include_velocity,
            "include_aggregate": model_config.wave_include_aggregate
        }
        df, wave_feat_names = create_directional_wave_features(df, wave_config)
        init_feats = init_feats + wave_feat_names

    # Verify wave features were added
    assert len([f for f in init_feats if "wave" in f]) > 0

    # Continue with normal preprocessing
    df_result, feat_names = create_features_and_targets(
        df=df,
        incl_level_feats=True,
        max_horizon=3,
        curr_feat_names=init_feats
    )

    # Verify everything worked
    assert len(feat_names) > len(["inc_trans_cs", "log_pop"])


def test_gbqr_wave_features_backwards_compatibility():
    """Test that missing wave config attributes don't break GBQR."""
    df = create_realistic_test_data()

    # Model config WITHOUT wave feature settings (backwards compatibility)
    model_config = GBQRModelConfig(
        model_name="gbqr_no_waves",
        sources=[DataSource.NHSN],
        fit_locations_separately=False,
        power_transform=PowerTransform.FOURTH_ROOT,
        # use_directional_waves defaults to False
    )

    init_feats = ["inc_trans_cs", "log_pop"]

    # This check should pass and not add wave features
    if hasattr(model_config, "use_directional_waves") and model_config.use_directional_waves:
        # This block should not execute
        raise AssertionError("Should not execute wave feature code")

    # Normal preprocessing should work
    df_result, feat_names = create_features_and_targets(
        df=df,
        incl_level_feats=model_config.incl_level_feats,
        max_horizon=3,
        curr_feat_names=init_feats
    )

    # No wave features should be present
    wave_feats = [f for f in feat_names if "wave" in f]
    assert len(wave_feats) == 0


def test_gbqr_wave_features_lag_values_are_correct():
    """Test that lag features contain correct time-shifted values."""
    df = create_realistic_test_data()

    wave_config = {
        "enabled": True,
        "directions": ["N"],
        "temporal_lags": [1, 2],
        "max_distance_km": 2000,
        "include_velocity": False,
        "include_aggregate": False
    }

    df_with_waves, wave_feat_names = create_directional_wave_features(df, wave_config)

    # Check lag semantics for one location
    test_location = "06"  # California
    loc_data = df_with_waves[df_with_waves["location"] == test_location] \
        .sort_values("wk_end_date") \
        .reset_index(drop=True)

    # Verify lag1 at time t equals base value at time t-1
    for i in range(1, len(loc_data)):
        base_prev = loc_data.loc[i-1, "inc_trans_cs_wave_N"]
        lag1_curr = loc_data.loc[i, "inc_trans_cs_wave_N_lag1"]

        if not pd.isna(base_prev) and not pd.isna(lag1_curr):
            assert abs(base_prev - lag1_curr) < 1e-6

    # Verify lag2 at time t equals base value at time t-2
    for i in range(2, len(loc_data)):
        base_prev2 = loc_data.loc[i-2, "inc_trans_cs_wave_N"]
        lag2_curr = loc_data.loc[i, "inc_trans_cs_wave_N_lag2"]

        if not pd.isna(base_prev2) and not pd.isna(lag2_curr):
            assert abs(base_prev2 - lag2_curr) < 1e-6
