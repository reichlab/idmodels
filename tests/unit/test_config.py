import datetime
from pathlib import Path

import pytest
from iddata.enums import Disease

from idmodels.config import RunConfig


def _run_config(num_sample_forecasts, states=("US",), hsas=()):
    return RunConfig(
        disease=Disease.FLU,
        ref_date=datetime.date(2024, 1, 6),
        output_root=Path("model-output"),
        artifact_store_root=None,
        max_horizon=3,
        states=list(states),
        hsas=list(hsas),
        q_levels=[0.5],
        q_labels=["0.5"],
        num_sample_forecasts=num_sample_forecasts,
    )


def test_run_config_defaults_to_no_samples():
    run_config = RunConfig(disease=Disease.FLU, ref_date=datetime.date(2024, 1, 6), output_root=Path("model-output"),
                           artifact_store_root=None, max_horizon=3, states=["US"], hsas=[], q_levels=[0.5],
                           q_labels=["0.5"])
    assert run_config.num_sample_forecasts == 0


def test_run_config_rejects_negative_num_sample_forecasts():
    with pytest.raises(ValueError, match="must be at least 0"):
        _run_config(-1)


def test_run_config_allows_hsas_and_many_samples():
    _run_config(500, states=["US", "66"], hsas=["1"])  # should not raise


def test_run_config_validate_catches_changes_after_construction():
    run_config = _run_config(5)
    run_config.validate()  # should not raise

    run_config.num_sample_forecasts = -1
    with pytest.raises(ValueError, match="must be at least 0"):
        run_config.validate()


def test_run_config_allows_samples_only():
    run_config = _run_config(5)
    run_config.q_levels, run_config.q_labels = [], []
    run_config.validate()  # should not raise


def test_run_config_rejects_no_quantiles_and_no_samples():
    run_config = _run_config(0)
    run_config.q_levels, run_config.q_labels = [], []
    with pytest.raises(ValueError, match="requests no output"):
        run_config.validate()


def test_run_config_rejects_mismatched_quantile_levels_and_labels():
    run_config = _run_config(0)
    run_config.q_labels = ["0.5", "0.9"]
    with pytest.raises(ValueError, match="same length"):
        run_config.validate()
