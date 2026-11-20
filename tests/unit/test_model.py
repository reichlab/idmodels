from unittest.mock import Mock

import pandas as pd
import pytest
from iddata.enums import Disease

from idmodels.config import RunConfig, SourceType
from idmodels.model import IDModel


class _FakeModel(IDModel):
    """Minimal concrete IDModel for testing base-class behavior in isolation.

    Only exists so IDModel (an ABC) can be instantiated to call base-class methods like
    _raise_if_incomplete() and _drop_incomplete_locations() directly. These hooks are
    never meant to run; NotImplementedError makes it loud and immediate if a test ever
    exercises them by accident instead of silently misbehaving.
    """

    def _build_sources(self, run_config):
        raise NotImplementedError

    def _build_feature_pipeline(self, run_config):
        raise NotImplementedError

    def _fit_and_predict(self, df, feat_names, run_config):
        raise NotImplementedError


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
    assert model.dropped_unique_ids == ["hsa143", "state13"]


def test_drop_incomplete_locations_keeps_complete_data_unchanged():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": 1.0, "state25": 2.0})
    pd.testing.assert_frame_equal(model._drop_incomplete_locations(df), df)
    assert model.dropped_unique_ids == []


def test_drop_incomplete_locations_raises_when_all_incomplete():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": float("nan"), "state25": float("nan")})
    with pytest.raises(ValueError, match="Every location is missing a value for the latest week"):
        model._drop_incomplete_locations(df)


def test_drop_incomplete_locations_uses_given_latest_week():
    """With an explicit latest week, one location reporting a newer week doesn't cause the others to be dropped."""
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": 1.0, "state25": 2.0})
    df = pd.concat([df, pd.DataFrame([{"unique_id": "state13", "source": "nssp",
                                       "wk_end_date": pd.Timestamp("2026-10-24"), "inc": 3.0}])])
    pd.testing.assert_frame_equal(model._drop_incomplete_locations(df, latest=pd.Timestamp("2026-10-17")), df)
    assert model.dropped_unique_ids == []


def _run_config(ref_date="2026-10-24", max_data_lag_weeks=1):
    return Mock(ref_date=pd.Timestamp(ref_date).date(), max_data_lag_weeks=max_data_lag_weeks)


@pytest.mark.parametrize("max_data_lag_weeks, expected", [
    (1, pd.Timestamp("2026-10-17")),
    (2, pd.Timestamp("2026-10-10")),
    (None, None),
])
def test_expected_week(max_data_lag_weeks, expected):
    assert IDModel._expected_week(_run_config(max_data_lag_weeks=max_data_lag_weeks)) == expected


def test_raise_if_stale_passes_when_current():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": 1.0, "state25": 2.0})  # latest week 2026-10-17 = ref_date - 1 week
    model._raise_if_stale(df, _run_config())  # should not raise


def test_raise_if_stale_passes_when_data_are_newer_than_expected():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": 1.0, "state25": 2.0})
    model._raise_if_stale(df, _run_config(max_data_lag_weeks=2))  # should not raise


def test_raise_if_stale_raises_when_latest_week_missing():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": 1.0, "state25": 2.0})
    with pytest.raises(ValueError, match=r"expected a value for the week ending 2026-10-24.*latest week with data is "
                                         r"2026-10-17"):
        model._raise_if_stale(df, _run_config(ref_date="2026-10-31"))


def test_raise_if_stale_raises_when_latest_week_all_nan():
    """Rows for the expected week exist but every value is NaN, so the data are still stale."""
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": float("nan"), "state25": float("nan")})
    with pytest.raises(ValueError, match="latest week with data is 2026-10-10"):
        model._raise_if_stale(df, _run_config())


def test_raise_if_stale_ignores_other_sources():
    """A newer week from a non-main source doesn't satisfy the check."""
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": 1.0})
    df = pd.concat([df, pd.DataFrame([{"unique_id": "state13", "source": "nhsn",
                                       "wk_end_date": pd.Timestamp("2026-10-24"), "inc": 3.0}])])
    with pytest.raises(ValueError, match="not current"):
        model._raise_if_stale(df, _run_config(ref_date="2026-10-31"))


def test_raise_if_stale_skipped_when_off():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": 1.0})
    model._raise_if_stale(df, _run_config(ref_date="2027-01-02", max_data_lag_weeks=None))  # should not raise


@pytest.mark.parametrize("max_data_lag_weeks", [-1, 1.5, True])
def test_run_config_rejects_invalid_max_data_lag_weeks(tmp_path, max_data_lag_weeks):
    with pytest.raises(ValueError, match="max_data_lag_weeks must be an int of at least 0"):
        RunConfig(disease=Disease.FLU, ref_date=pd.Timestamp("2026-10-24").date(), output_root=tmp_path,
                  artifact_store_root=None, max_horizon=3, states=["US"], hsas=[], q_levels=[0.5], q_labels=["0.5"],
                  max_data_lag_weeks=max_data_lag_weeks)
