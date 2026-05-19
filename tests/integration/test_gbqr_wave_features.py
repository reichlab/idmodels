"""Integration test for GBQR model with directional wave features."""

import numpy as np
import pandas as pd
from idmodels.config import GBQRModelConfig, PowerTransform, SourceType
from idmodels.features import (
    DirectionalWaveFeature,
    FeaturePipeline,
    HolidayFeature,
    HorizonTargetFeature,
    LagFeature,
    LevelFeatureFilter,
    OneHotEncodingFeature,
    RollingMeanFeature,
    TaylorFeature,
)


def create_realistic_test_data():
    """Create realistic test data mimicking the structure from DiseaseDataLoader."""
    np.random.seed(42)

    states = ["01", "06", "13", "36", "42", "48"]  # AL, CA, GA, NY, PA, TX
    dates = pd.date_range("2023-10-01", periods=20, freq="W")

    data = []
    for state in states:
        for i, date in enumerate(dates):
            base_inc = 0.5 + 0.3 * np.sin(i / 10) + np.random.randn() * 0.1

            data.append({"agg_level": "state",
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
                         "inc_trans_center_factor": 0.1})

    return pd.DataFrame(data)


def test_gbqr_preprocessing_without_waves():
    """Test that GBQR preprocessing works without wave features (backwards compatibility)."""
    df = create_realistic_test_data()

    init_feats = ["inc_trans_cs", "log_pop"]

    features = [OneHotEncodingFeature(columns=["source", "agg_level", "location"]),
                HolidayFeature(),
                LagFeature(columns=["inc_trans_cs"], lags=[1, 2]),
                TaylorFeature(column="inc_trans_cs", degree=2, window_sizes=[4, 6]),
                TaylorFeature(column="inc_trans_cs", degree=1, window_sizes=[3, 5]),
                RollingMeanFeature(column="inc_trans_cs", window_sizes=[2, 4]),
                LagFeature(columns=None, lags=[1, 2]),
                HorizonTargetFeature(column="inc_trans_cs", max_horizon=3)]
    df_result, feat_names = FeaturePipeline(features=features, initial_feat_names=init_feats).apply(df)

    assert "inc_trans_cs" in feat_names
    assert "log_pop" in feat_names

    wave_feats = [f for f in feat_names if "wave" in f]
    assert len(wave_feats) == 0


def test_gbqr_preprocessing_with_waves_enabled():
    """Test that GBQR preprocessing works with wave features enabled."""
    df = create_realistic_test_data()

    df_with_waves, wave_feat_names = DirectionalWaveFeature(
        directions=["N", "S", "E", "W"],
        temporal_lags=[1, 2],
        max_distance_km=2000,
        include_velocity=False,
        include_aggregate=True,
    ).apply(df, [])

    assert len(wave_feat_names) > 0
    assert "inc_trans_cs_wave_N" in wave_feat_names
    assert "inc_trans_cs_wave_S" in wave_feat_names
    assert "inc_trans_cs_wave_E" in wave_feat_names
    assert "inc_trans_cs_wave_W" in wave_feat_names
    assert "inc_trans_cs_wave_avg" in wave_feat_names
    assert "inc_trans_cs_wave_N_lag1" in wave_feat_names
    assert "inc_trans_cs_wave_N_lag2" in wave_feat_names

    init_feats = ["inc_trans_cs", "log_pop"] + wave_feat_names

    features = [OneHotEncodingFeature(columns=["source", "agg_level", "location"]),
                HolidayFeature(),
                LagFeature(columns=["inc_trans_cs"], lags=[1, 2]),
                TaylorFeature(column="inc_trans_cs", degree=2, window_sizes=[4, 6]),
                TaylorFeature(column="inc_trans_cs", degree=1, window_sizes=[3, 5]),
                RollingMeanFeature(column="inc_trans_cs", window_sizes=[2, 4]),
                LagFeature(columns=None, lags=[1, 2]),
                HorizonTargetFeature(column="inc_trans_cs", max_horizon=3)]
    df_result, feat_names = FeaturePipeline(features=features, initial_feat_names=init_feats).apply(df_with_waves)

    for wave_feat in wave_feat_names:
        assert wave_feat in feat_names

    assert "inc_trans_cs" in feat_names
    assert "log_pop" in feat_names

    assert "delta_target" in df_result.columns


def test_gbqr_preprocessing_with_all_wave_options():
    """Test GBQR preprocessing with all wave feature options enabled."""
    df = create_realistic_test_data()

    df_with_waves, wave_feat_names = DirectionalWaveFeature(
        directions=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        temporal_lags=[1, 2],
        max_distance_km=2000,
        include_velocity=True,
        include_aggregate=True,
    ).apply(df, [])

    # 8 directions + 1 aggregate = 9 base features
    # Each has: base + lag1 + lag2 + velocity = 4
    # Total: 9 * 4 = 36 features
    expected_feature_count = 9 * 4
    assert len(wave_feat_names) == expected_feature_count

    assert "inc_trans_cs_wave_N_velocity" in wave_feat_names
    assert "inc_trans_cs_wave_avg_velocity" in wave_feat_names

    init_feats = ["inc_trans_cs", "log_pop"] + wave_feat_names

    features = [
        OneHotEncodingFeature(columns=["source", "agg_level", "location"]),
        HolidayFeature(),
        LagFeature(columns=["inc_trans_cs"], lags=[1, 2]),
        TaylorFeature(column="inc_trans_cs", degree=2, window_sizes=[4, 6]),
        TaylorFeature(column="inc_trans_cs", degree=1, window_sizes=[3, 5]),
        RollingMeanFeature(column="inc_trans_cs", window_sizes=[2, 4]),
        LagFeature(columns=None, lags=[1, 2]),
        HorizonTargetFeature(column="inc_trans_cs", max_horizon=3),
    ]
    df_result, feat_names = FeaturePipeline(features=features, initial_feat_names=init_feats).apply(df_with_waves)

    for wave_feat in wave_feat_names:
        assert wave_feat in feat_names


def test_gbqr_wave_features_no_nan_for_valid_data():
    """Test that wave features produce valid values for locations with neighbors."""
    df = create_realistic_test_data()

    df_with_waves, _ = DirectionalWaveFeature(directions=["N", "S"],
                                              temporal_lags=[],
                                              max_distance_km=3000,
                                              include_velocity=False,
                                              include_aggregate=True,
                                              ).apply(df, [])

    avg_feature = df_with_waves["inc_trans_cs_wave_avg"]
    non_nan_count = (~avg_feature.isna()).sum()

    assert non_nan_count > len(df_with_waves) * 0.5


def test_gbqr_wave_features_with_model_config_pattern():
    """Test wave features using the model_config pattern from GBQR."""
    df = create_realistic_test_data()

    model_config = GBQRModelConfig(model_name="gbqr_wave_test",
                                   sources=[SourceType.NHSN],
                                   fit_locations_separately=False,
                                   power_transform=PowerTransform.FOURTH_ROOT,
                                   use_directional_waves=True,
                                   wave_directions=["N", "S", "E", "W"],
                                   wave_temporal_lags=[1, 2],
                                   wave_max_distance_km=2000,
                                   wave_include_velocity=False,
                                   wave_include_aggregate=True)

    init_feats = ["inc_trans_cs", "log_pop"]

    if hasattr(model_config, "use_directional_waves") and model_config.use_directional_waves:
        df, wave_feat_names = DirectionalWaveFeature(
            directions=model_config.wave_directions,
            temporal_lags=model_config.wave_temporal_lags,
            max_distance_km=model_config.wave_max_distance_km,
            include_velocity=model_config.wave_include_velocity,
            include_aggregate=model_config.wave_include_aggregate,
        ).apply(df, [])
        init_feats = init_feats + wave_feat_names

    assert len([f for f in init_feats if "wave" in f]) > 0

    features = [OneHotEncodingFeature(columns=["source", "agg_level", "location"]),
                HolidayFeature(),
                LagFeature(columns=["inc_trans_cs"], lags=[1, 2]),
                TaylorFeature(column="inc_trans_cs", degree=2, window_sizes=[4, 6]),
                TaylorFeature(column="inc_trans_cs", degree=1, window_sizes=[3, 5]),
                RollingMeanFeature(column="inc_trans_cs", window_sizes=[2, 4]),
                LagFeature(columns=None, lags=[1, 2]),
                HorizonTargetFeature(column="inc_trans_cs", max_horizon=3)]
    df_result, feat_names = FeaturePipeline(features=features, initial_feat_names=init_feats).apply(df)

    assert len(feat_names) > len(["inc_trans_cs", "log_pop"])


def test_gbqr_wave_features_backwards_compatibility():
    """Test that missing wave config attributes don't break GBQR."""
    df = create_realistic_test_data()

    model_config = GBQRModelConfig(model_name="gbqr_no_waves",
                                   sources=[SourceType.NHSN],
                                   fit_locations_separately=False,
                                   power_transform=PowerTransform.FOURTH_ROOT,
                                   # use_directional_waves defaults to False
                                   )

    init_feats = ["inc_trans_cs", "log_pop"]

    if hasattr(model_config, "use_directional_waves") and model_config.use_directional_waves:
        raise AssertionError("Should not execute wave feature code")

    features = [
        OneHotEncodingFeature(columns=["source", "agg_level", "location"]),
        HolidayFeature(),
        LagFeature(columns=["inc_trans_cs"], lags=[1, 2]),
        TaylorFeature(column="inc_trans_cs", degree=2, window_sizes=[4, 6]),
        TaylorFeature(column="inc_trans_cs", degree=1, window_sizes=[3, 5]),
        RollingMeanFeature(column="inc_trans_cs", window_sizes=[2, 4]),
        LagFeature(columns=None, lags=[1, 2]),
        HorizonTargetFeature(column="inc_trans_cs", max_horizon=3),
    ]
    if not model_config.incl_level_feats:
        features.append(LevelFeatureFilter())
    df_result, feat_names = FeaturePipeline(features=features, initial_feat_names=init_feats).apply(df)

    wave_feats = [f for f in feat_names if "wave" in f]
    assert len(wave_feats) == 0


def test_gbqr_wave_features_lag_values_are_correct():
    """Test that lag features contain correct time-shifted values."""
    df = create_realistic_test_data()

    df_with_waves, _ = DirectionalWaveFeature(
        directions=["N"],
        temporal_lags=[1, 2],
        max_distance_km=2000,
        include_velocity=False,
        include_aggregate=False,
    ).apply(df, [])

    test_location = "06"  # California
    loc_data = df_with_waves[df_with_waves["location"] == test_location] \
        .sort_values("wk_end_date") \
        .reset_index(drop=True)

    for i in range(1, len(loc_data)):
        base_prev = loc_data.loc[i - 1, "inc_trans_cs_wave_N"]
        lag1_curr = loc_data.loc[i, "inc_trans_cs_wave_N_lag1"]

        if not pd.isna(base_prev) and not pd.isna(lag1_curr):
            assert abs(base_prev - lag1_curr) < 1e-6

    for i in range(2, len(loc_data)):
        base_prev2 = loc_data.loc[i - 2, "inc_trans_cs_wave_N"]
        lag2_curr = loc_data.loc[i, "inc_trans_cs_wave_N_lag2"]

        if not pd.isna(base_prev2) and not pd.isna(lag2_curr):
            assert abs(base_prev2 - lag2_curr) < 1e-6
