"""Unit tests for feature classes in idmodels.features."""

import numpy as np
import pandas as pd
import pytest

from idmodels.features import (
    FeaturePipeline,
    HorizonTargetFeature,
    LagFeature,
    LevelFeatureFilter,
    OneHotEncodingFeature,
)


def make_df(n_weeks=20, locations=("01", "06", "36"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for loc in locations:
        for week in range(1, n_weeks + 1):
            rows.append({"source": "nhsn",
                         "location": loc,
                         "season": "2023-24",
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
