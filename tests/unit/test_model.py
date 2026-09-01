from unittest.mock import Mock

import pandas as pd
import pytest

from idmodels.model import IDModel


class _FakeModel(IDModel):
    """Minimal concrete IDModel for testing base-class behavior in isolation.

    Only exists so IDModel (an ABC) can be instantiated to call _raise_if_incomplete()
    directly. These hooks are never meant to run; NotImplementedError makes it loud and
    immediate if a test ever exercises them by accident instead of silently misbehaving.
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


def test_raise_if_incomplete_passes_when_no_nan():
    model = _FakeModel(model_config=Mock())
    run_config = Mock(ref_date="2026-08-15")
    preds_df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    model._raise_if_incomplete(preds_df, run_config, save_path="out.csv")  # should not raise
