from unittest.mock import Mock

import pandas as pd
import pytest

from idmodels.config import SourceType
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


def test_drop_incomplete_locations_keeps_complete_data_unchanged():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": 1.0, "state25": 2.0})
    pd.testing.assert_frame_equal(model._drop_incomplete_locations(df), df)


def test_drop_incomplete_locations_raises_when_all_incomplete():
    model = _FakeModel(model_config=Mock(main_source=SourceType.NSSP))
    df = _two_week_df({"state13": float("nan"), "state25": float("nan")})
    with pytest.raises(ValueError, match="Every location is missing a value for the latest week"):
        model._drop_incomplete_locations(df)
