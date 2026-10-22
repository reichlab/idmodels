import datetime
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from iddata.enums import Disease

from idmodels.config import RunConfig, SampleIdScheme, SourceType
from idmodels.model import IDModel


class _FakeModel(IDModel):
    """Minimal concrete IDModel for testing base-class behavior in isolation.

    Only exists so IDModel (an ABC) can be instantiated to call base-class methods like
    _raise_if_incomplete() and _format_output() directly. These hooks are never meant to run;
    NotImplementedError makes it loud and immediate if a test ever exercises them by accident
    instead of silently misbehaving.
    """

    def _build_sources(self, run_config):
        raise NotImplementedError

    def _build_feature_pipeline(self, run_config):
        raise NotImplementedError

    def _fit_and_predict(self, df, feat_names, run_config):
        raise NotImplementedError


class _SourcesReached(Exception):
    pass


class _FakeSampleModel(_FakeModel):
    """Supports sample forecasts; signals when run() gets past the sample check to _build_sources()."""

    supports_sample_forecasts = True

    def _build_sources(self, run_config):
        raise _SourcesReached


def test_run_rejects_samples_before_loading_data_when_unsupported():
    run_config = Mock(num_sample_forecasts=5)
    with pytest.raises(NotImplementedError, match="_FakeModel does not support sample forecasts"):
        _FakeModel(model_config=Mock()).run(run_config)


def test_run_allows_samples_when_supported():
    run_config = Mock(num_sample_forecasts=5)
    with pytest.raises(_SourcesReached):
        _FakeSampleModel(model_config=Mock()).run(run_config)


def test_run_validates_run_config_changed_after_construction():
    run_config = RunConfig(disease=Disease.FLU, ref_date=datetime.date(2024, 1, 6), output_root=Path("model-output"),
                           artifact_store_root=None, max_horizon=3, states=["US"], hsas=[], q_levels=[0.5],
                           q_labels=["0.5"], num_sample_forecasts=5)
    run_config.num_sample_forecasts = 101  # over the STATE_ABBREVIATION limit, bypassing __post_init__
    with pytest.raises(ValueError, match="at most 100"):
        _FakeSampleModel(model_config=Mock()).run(run_config)


def test_raise_if_incomplete_raises_on_any_nan_value():
    """A single NaN in the value column must not be silently written to a submission CSV --
    this is the guard that would have caught the GBQR failure at its source.
    """
    model = _FakeModel(model_config=Mock())
    run_config = Mock(ref_date="2026-08-15")
    preds_df = pd.DataFrame({"value": [1.0, float("nan"), 3.0]})

    with pytest.raises(ValueError, match=r"1/3 rows have NaN value"):
        model._raise_if_incomplete(preds_df, run_config, save_path="out.csv")


def test_raise_if_incomplete_raises_when_all_nan():
    """The exact incident scenario: every row's value is NaN (all-NA population join)."""
    model = _FakeModel(model_config=Mock())
    run_config = Mock(ref_date="2026-08-15")
    preds_df = pd.DataFrame({"value": [float("nan")] * 4})

    with pytest.raises(ValueError, match=r"4/4 rows have NaN value"):
        model._raise_if_incomplete(preds_df, run_config, save_path="out.csv")


def test_raise_if_incomplete_raises_when_empty():
    """An empty prediction set (e.g. the main source lags every supplementary source, so nothing
    is left to predict) must not be written as a header-only submission CSV.
    """
    model = _FakeModel(model_config=Mock())
    run_config = Mock(ref_date="2026-08-15")
    preds_df = pd.DataFrame({"value": pd.Series([], dtype=float)})

    with pytest.raises(ValueError, match=r"No predictions"):
        model._raise_if_incomplete(preds_df, run_config, save_path="out.csv")


def test_raise_if_incomplete_passes_when_no_nan():
    model = _FakeModel(model_config=Mock())
    run_config = Mock(ref_date="2026-08-15")
    preds_df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    model._raise_if_incomplete(preds_df, run_config, save_path="out.csv")  # should not raise


def _mixed_preds_df(locations, agg_levels, num_samples):
    """Horizon-1 quantile and sample rows (sample_index 0..num_samples-1) for each location, in scrambled order."""
    rows = []
    for loc, agg_level in zip(locations, agg_levels):
        rows += [{"location": loc, "agg_level": agg_level, "output_type": "quantile", "output_type_id": q,
                  "sample_index": float("nan")} for q in ["0.975", "0.5"]]
        rows += [{"location": loc, "agg_level": agg_level, "output_type": "sample", "output_type_id": float("nan"),
                  "sample_index": float(i)} for i in reversed(range(num_samples))]
    preds_df = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    preds_df["wk_end_date"] = pd.Timestamp("2024-01-06")
    preds_df["horizon"] = 1
    preds_df["value"] = 1.0
    return preds_df


def test_format_output_labels_state_abbreviation_samples_after_quantiles():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NHSN))
    run_config = Mock(disease=Disease.FLU, ref_date=datetime.date(2024, 1, 6),
                      sample_id_scheme=SampleIdScheme.STATE_ABBREVIATION)
    preds_df = _mixed_preds_df(locations=["US", "01"], agg_levels=["national", "state"], num_samples=2)

    out = model._format_output(preds_df, run_config)

    assert out["output_type"].tolist() == ["quantile"] * 4 + ["sample"] * 4
    assert out["output_type_id"].tolist() == ["0.5", "0.5", "0.975", "0.975", "AL00", "US00", "AL01", "US01"]
    assert "sample_index" not in out.columns


def test_format_output_sorts_draw_number_samples_numerically():
    """DRAW_NUMBER labels aren't zero-padded, so sorting by label would put "10" before "2"."""
    model = _FakeModel(model_config=Mock(main_source=SourceType.NHSN))
    run_config = Mock(disease=Disease.FLU, ref_date=datetime.date(2024, 1, 6),
                      sample_id_scheme=SampleIdScheme.DRAW_NUMBER)
    preds_df = _mixed_preds_df(locations=["US"], agg_levels=["national"], num_samples=12)

    out = model._format_output(preds_df, run_config)

    assert out.loc[out["output_type"] == "sample", "output_type_id"].tolist() == [str(i) for i in range(1, 13)]


def test_format_output_without_samples_keeps_quantile_order():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NHSN))
    run_config = Mock(disease=Disease.FLU, ref_date=datetime.date(2024, 1, 6),
                      sample_id_scheme=SampleIdScheme.STATE_ABBREVIATION)
    preds_df = _mixed_preds_df(locations=["US", "01"], agg_levels=["national", "state"], num_samples=0)
    preds_df = preds_df.drop(columns="sample_index")  # GBQR never produces a sample_index column

    out = model._format_output(preds_df, run_config)

    assert out["output_type_id"].tolist() == ["0.5", "0.5", "0.975", "0.975"]
    assert out["location"].tolist() == ["01", "US", "01", "US"]


def _two_week_df(latest_inc_by_uid):
    """
    nssp rows for two weeks per unique_id; the latest week's inc comes from latest_inc_by_uid. Only nssp rows are
    built because the tests use main_source=NSSP, and _drop_incomplete_locations only checks main-source rows.
    """
    rows = []
    for uid, latest_inc in latest_inc_by_uid.items():
        rows.append({"unique_id": uid, "source": "nssp", "wk_end_date": pd.Timestamp("2026-10-10"), "inc": 1.0})
        rows.append({"unique_id": uid, "source": "nssp", "wk_end_date": pd.Timestamp("2026-10-17"),
                     "inc": latest_inc})
    return pd.DataFrame(rows)


def test_drop_incomplete_locations_drops_and_warns():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": float("nan"), "hsa143": float("nan"), "state25": 2.0})
    # a non-main-source row for a dropped location is dropped with it
    df = pd.concat([df, pd.DataFrame([{"unique_id": "state13", "source": "nhsn",
                                       "wk_end_date": pd.Timestamp("2026-10-17"), "inc": 3.0}])])

    with pytest.warns(UserWarning, match=r"Dropping 2 location\(s\).*2026-10-17.*\['hsa143', 'state13'\]"):
        out = model._drop_incomplete_locations(df)

    assert set(out["unique_id"]) == {"state25"}


def test_drop_incomplete_locations_keeps_complete_data_unchanged():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": 1.0, "state25": 2.0})
    pd.testing.assert_frame_equal(model._drop_incomplete_locations(df), df)


def test_drop_incomplete_locations_raises_when_all_incomplete():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": float("nan"), "state25": float("nan")})
    with pytest.raises(ValueError, match="Every location is missing a value for the latest week"):
        model._drop_incomplete_locations(df)
