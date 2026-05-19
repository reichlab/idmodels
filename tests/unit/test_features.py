"""Unit tests for feature classes in idmodels.features."""

import numpy as np
import pandas as pd
import pytest
from idmodels.features import (
    FeaturePipeline,
    HolidayFeature,
    HorizonTargetFeature,
    LagFeature,
    LevelFeatureFilter,
    OneHotEncodingFeature,
    RollingMeanFeature,
    TaylorFeature,
)


def make_df(n_weeks=20, locations=("01", "06", "36"), seed=0, season="2023-24"):
    rng = np.random.default_rng(seed)
    rows = []
    for loc in locations:
        for week in range(1, n_weeks + 1):
            rows.append({"source": "nhsn",
                         "location": loc,
                         "season": season,
                         "season_week": week,
                         "wk_end_date": pd.Timestamp("2023-10-01") + pd.Timedelta(weeks=week - 1),
                         "inc_trans_cs": rng.normal(0.0, 0.5), })
    return pd.DataFrame(rows)


class TestOneHotEncodingFeature:
    def test_adds_dummy_columns(self):
        df = make_df()
        feat = OneHotEncodingFeature(columns=["source"])
        df_out, feat_names = feat.apply(df.copy(), [])
        assert "source_nhsn" in feat_names
        assert "source_nhsn" in df_out.columns


    def test_updates_feat_names(self):
        df = make_df()
        initial = ["inc_trans_cs"]
        feat = OneHotEncodingFeature(columns=["source"])
        _, feat_names = feat.apply(df.copy(), list(initial))
        assert "inc_trans_cs" in feat_names
        assert "source_nhsn" in feat_names


    def test_multiple_columns_both_appear_in_feat_names_and_df(self):
        df = make_df(season="2023-24")
        feat = OneHotEncodingFeature(columns=["source", "season"])
        df_out, feat_names = feat.apply(df.copy(), [])
        assert "source_nhsn" in feat_names
        assert "source_nhsn" in df_out.columns
        assert "season_2023-24" in feat_names
        assert "season_2023-24" in df_out.columns


class TestLagFeature:
    def test_creates_lag_columns(self):
        df = make_df()
        feat = LagFeature(columns=["inc_trans_cs"], lags=[1, 2])
        df_out, feat_names = feat.apply(df.copy(), ["inc_trans_cs"])
        assert "inc_trans_cs_lag1" in df_out.columns
        assert "inc_trans_cs_lag2" in df_out.columns
        assert "inc_trans_cs_lag1" in feat_names
        assert "inc_trans_cs_lag2" in feat_names


    def test_lag_semantics(self):
        df = make_df(n_weeks=10, locations=("01",))
        feat = LagFeature(columns=["inc_trans_cs"], lags=[1])
        df_out, _ = feat.apply(df.copy(), ["inc_trans_cs"])
        df_out = df_out.sort_values("wk_end_date").reset_index(drop=True)
        for i in range(1, len(df_out)):
            orig = df_out.loc[i - 1, "inc_trans_cs"]
            lagged = df_out.loc[i, "inc_trans_cs_lag1"]
            if not pd.isna(lagged):
                assert abs(orig - lagged) < 1e-12


    def test_raises_if_columns_none(self):
        feat = LagFeature(columns=None, lags=[1])
        with pytest.raises(ValueError, match="must be resolved"):
            feat.apply(pd.DataFrame(), [])


class TestHorizonTargetFeature:
    def test_adds_horizon_column(self):
        df = make_df()
        feat = HorizonTargetFeature(column="inc_trans_cs", max_horizon=3)
        df_out, feat_names = feat.apply(df.copy(), ["inc_trans_cs"])
        assert "horizon" in df_out.columns
        assert "horizon" in feat_names


    def test_adds_delta_target(self):
        df = make_df()
        feat = HorizonTargetFeature(column="inc_trans_cs", max_horizon=2)
        df_out, _ = feat.apply(df.copy(), ["inc_trans_cs"])
        assert "delta_target" in df_out.columns


    def test_expands_rows(self):
        n_rows = len(make_df())
        df = make_df()
        feat = HorizonTargetFeature(column="inc_trans_cs", max_horizon=4)
        df_out, _ = feat.apply(df.copy(), ["inc_trans_cs"])
        assert len(df_out) >= n_rows


class TestLevelFeatureFilter:
    def test_removes_inc_trans_cs(self):
        feat_names = ["inc_trans_cs", "season_week", "log_pop"]
        feat = LevelFeatureFilter()
        _, out = feat.apply(pd.DataFrame(), list(feat_names))
        assert "inc_trans_cs" not in out
        assert "season_week" in out


    def test_removes_rollmean(self):
        feat_names = ["inc_trans_cs_rollmean_w2", "inc_trans_cs_rollmean_w4", "log_pop"]
        feat = LevelFeatureFilter()
        _, out = feat.apply(pd.DataFrame(), list(feat_names))
        assert "inc_trans_cs_rollmean_w2" not in out
        assert "log_pop" in out


    def test_removes_taylor_c0_not_c1(self):
        feat_names = ["inc_trans_cs_taylor_d2_c0_w4t_sNone",
                      "inc_trans_cs_taylor_d2_c1_w4t_sNone", ]
        feat = LevelFeatureFilter()
        _, out = feat.apply(pd.DataFrame(), list(feat_names))
        assert "inc_trans_cs_taylor_d2_c0_w4t_sNone" not in out
        assert "inc_trans_cs_taylor_d2_c1_w4t_sNone" in out


class TestFeaturePipeline:
    def test_accumulator_resolved_for_null_lag(self):
        """LagFeature(columns=None) should resolve to columns added since last LagFeature."""
        df = make_df()
        pipeline = FeaturePipeline(
            features=[
                OneHotEncodingFeature(columns=["source"]),
                LagFeature(columns=None, lags=[1]),
            ],
            initial_feat_names=["inc_trans_cs"],
        )
        df_out, feat_names = pipeline.apply(df.copy())
        assert "source_nhsn_lag1" in feat_names


    def test_initial_feats_not_in_accumulator(self):
        """Columns in initial_feat_names should not be lagged by LagFeature(columns=None)."""
        df = make_df()
        pipeline = FeaturePipeline(
            features=[LagFeature(columns=None, lags=[1])],
            initial_feat_names=["inc_trans_cs"],
        )
        df_out, feat_names = pipeline.apply(df.copy())
        # No new columns added before LagFeature, so no lags created
        assert "inc_trans_cs_lag1" not in feat_names


    def test_accumulator_resets_after_lag(self):
        """After a LagFeature step, accumulator resets; next LagFeature only picks up new cols."""
        df = make_df()
        pipeline = FeaturePipeline(
            features=[OneHotEncodingFeature(columns=["source"]),
                      LagFeature(columns=None, lags=[1]),
                      OneHotEncodingFeature(columns=["season"]),
                      LagFeature(columns=None, lags=[1])],
            initial_feat_names=["inc_trans_cs"],
        )
        df_out, feat_names = pipeline.apply(df.copy())
        # First batch lagged: source_nhsn → source_nhsn_lag1
        assert "source_nhsn_lag1" in feat_names
        # Second batch lagged: only season dummy (added after first lag step)
        # source_nhsn should NOT be re-lagged
        source_lag_count = sum(1 for f in feat_names if f.startswith("source_nhsn_lag"))
        assert source_lag_count == 1


    def test_explicit_columns_lag(self):
        """LagFeature with explicit columns= ignores accumulator."""
        df = make_df()
        pipeline = FeaturePipeline(
            features=[LagFeature(columns=["inc_trans_cs"], lags=[1, 2])],
            initial_feat_names=["inc_trans_cs"],
        )
        df_out, feat_names = pipeline.apply(df.copy())
        assert "inc_trans_cs_lag1" in feat_names
        assert "inc_trans_cs_lag2" in feat_names


    def test_empty_features_list(self):
        """An empty pipeline should return df and initial_feat_names unchanged."""
        df = make_df()
        pipeline = FeaturePipeline(features=[], initial_feat_names=["inc_trans_cs"])
        df_out, feat_names = pipeline.apply(df.copy())
        assert feat_names == ["inc_trans_cs"]
        assert len(df_out) == len(df)


class TestHolidayFeature:
    def test_adds_delta_xmas_column(self):
        df = make_df(season="2022/23")
        df_out, _ = HolidayFeature().apply(df.copy(), [])
        assert "delta_xmas" in df_out.columns


    def test_adds_to_feat_names(self):
        df = make_df(season="2022/23")
        _, feat_names = HolidayFeature().apply(df.copy(), ["inc_trans_cs"])
        assert "delta_xmas" in feat_names
        assert "inc_trans_cs" in feat_names


    def test_delta_xmas_value_is_season_week_minus_xmas_week(self):
        from iddata.utils import get_holidays
        season = "2022/23"
        xmas_week = (get_holidays()
                     .query("holiday == 'Christmas Day' and season == @season")
                     ["season_week"].iloc[0])
        test_week = 10
        df = pd.DataFrame([{"source": "nhsn", "location": "01", "season": season,
                             "season_week": test_week,
                             "wk_end_date": pd.Timestamp("2023-01-07"),
                             "inc_trans_cs": 0.5}])
        df_out, _ = HolidayFeature().apply(df, [])
        assert df_out["delta_xmas"].iloc[0] == test_week - xmas_week


    def test_does_not_add_xmas_spike(self):
        df = make_df(season="2022/23")
        df_out, _ = HolidayFeature().apply(df.copy(), [])
        assert "xmas_spike" not in df_out.columns


class TestTaylorFeature:
    def test_adds_columns_to_df(self):
        df = make_df()
        feat = TaylorFeature(column="inc_trans_cs", degree=2, window_sizes=[4])
        cols_before = set(df.columns)
        df_out, _ = feat.apply(df.copy(), [])
        assert len(set(df_out.columns) - cols_before) > 0


    def test_updates_feat_names(self):
        df = make_df()
        feat = TaylorFeature(column="inc_trans_cs", degree=2, window_sizes=[4])
        _, feat_names = feat.apply(df.copy(), ["inc_trans_cs"])
        taylor_feats = [f for f in feat_names if "taylor" in f]
        # degree 2 produces 3 coefficients (c0, c1, c2) for one window size
        assert len(taylor_feats) == 3


    def test_multiple_window_sizes_produce_more_features(self):
        df = make_df()
        feat_one = TaylorFeature(column="inc_trans_cs", degree=1, window_sizes=[3])
        feat_two = TaylorFeature(column="inc_trans_cs", degree=1, window_sizes=[3, 5])
        _, names_one = feat_one.apply(df.copy(), [])
        _, names_two = feat_two.apply(df.copy(), [])
        assert len(names_two) == 2 * len(names_one)


class TestRollingMeanFeature:
    def test_adds_rollmean_columns(self):
        df = make_df()
        feat = RollingMeanFeature(column="inc_trans_cs", window_sizes=[2, 4])
        df_out, _ = feat.apply(df.copy(), [])
        assert "inc_trans_cs_rollmean_w2" in df_out.columns
        assert "inc_trans_cs_rollmean_w4" in df_out.columns


    def test_updates_feat_names(self):
        df = make_df()
        feat = RollingMeanFeature(column="inc_trans_cs", window_sizes=[2, 4])
        _, feat_names = feat.apply(df.copy(), ["inc_trans_cs"])
        assert "inc_trans_cs_rollmean_w2" in feat_names
        assert "inc_trans_cs_rollmean_w4" in feat_names
        assert "inc_trans_cs" in feat_names


    def test_default_group_columns(self):
        feat = RollingMeanFeature(column="inc_trans_cs", window_sizes=[2])
        assert feat.group_columns == ["location"]


    def test_produces_valid_non_nan_values(self):
        df = make_df(n_weeks=20, locations=("01",))
        feat = RollingMeanFeature(column="inc_trans_cs", window_sizes=[2])
        df_out, _ = feat.apply(df.copy(), [])
        non_nan = df_out["inc_trans_cs_rollmean_w2"].notna()
        assert non_nan.sum() > len(df_out) * 0.8
