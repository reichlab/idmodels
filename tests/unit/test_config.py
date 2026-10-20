import datetime
from pathlib import Path

import pytest
from iddata.enums import Disease

from idmodels.config import RunConfig, SampleIdScheme


def _run_config(num_sample_forecasts, sample_id_scheme=SampleIdScheme.STATE_ABBREVIATION, states=("US",), hsas=()):
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
        sample_id_scheme=sample_id_scheme,
    )


def test_run_config_defaults_to_no_samples():
    run_config = RunConfig(disease=Disease.FLU, ref_date=datetime.date(2024, 1, 6), output_root=Path("model-output"),
                           artifact_store_root=None, max_horizon=3, states=["US"], hsas=[], q_levels=[0.5],
                           q_labels=["0.5"])
    assert run_config.num_sample_forecasts == 0
    assert run_config.sample_id_scheme == SampleIdScheme.STATE_ABBREVIATION


@pytest.mark.parametrize("scheme", list(SampleIdScheme))
def test_run_config_rejects_negative_num_sample_forecasts(scheme):
    with pytest.raises(ValueError, match="must be at least 0"):
        _run_config(-1, scheme)


@pytest.mark.parametrize("num", [0, 1, 100])
def test_run_config_state_abbreviation_accepts_up_to_100(num):
    _run_config(num, SampleIdScheme.STATE_ABBREVIATION)  # should not raise


def test_run_config_state_abbreviation_rejects_over_100():
    with pytest.raises(ValueError, match="at most 100"):
        _run_config(101, SampleIdScheme.STATE_ABBREVIATION)


def test_run_config_state_abbreviation_rejects_hsas_when_sampling():
    with pytest.raises(ValueError, match="only supports states and national"):
        _run_config(5, SampleIdScheme.STATE_ABBREVIATION, hsas=["1"])


def test_run_config_state_abbreviation_rejects_unmapped_states_when_sampling():
    """Fails when the config is created rather than after fitting, when sample rows are labeled."""
    with pytest.raises(ValueError, match=r"No state abbreviation for state\(s\) \['66'\]"):
        _run_config(5, SampleIdScheme.STATE_ABBREVIATION, states=["US", "66"])


@pytest.mark.parametrize("num, scheme", [(0, SampleIdScheme.STATE_ABBREVIATION), (5, SampleIdScheme.DRAW_NUMBER)])
def test_run_config_allows_unmapped_states_when_not_labeling_by_abbreviation(num, scheme):
    _run_config(num, scheme, states=["US", "66"])  # should not raise


@pytest.mark.parametrize("scheme", list(SampleIdScheme))
def test_run_config_allows_hsas_without_samples(scheme):
    _run_config(0, scheme, hsas=["1"])  # should not raise


def test_run_config_draw_number_allows_hsas_and_over_100():
    _run_config(500, SampleIdScheme.DRAW_NUMBER, hsas=["1"])  # should not raise


def test_run_config_validate_catches_changes_after_construction():
    run_config = _run_config(5, SampleIdScheme.STATE_ABBREVIATION)
    run_config.validate()  # should not raise

    run_config.hsas = ["1"]
    with pytest.raises(ValueError, match="only supports states and national"):
        run_config.validate()
