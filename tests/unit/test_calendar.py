import datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from idmodels.calendar import calendar_labels, holiday_week_end, holiday_week_ends, parse_holiday_spec
from idmodels.config import PoolingStrategy, PowerTransform, RunConfig, SARIXModelConfig, SourceType
from idmodels.sarix import SARIXModel

D = datetime.date


@pytest.mark.parametrize("holiday, expected", [
    (D(2023, 12, 25), D(2023, 12, 30)),  # Monday
    (D(2022, 12, 25), D(2022, 12, 31)),  # Sunday: first day of its week
    (D(2021, 12, 25), D(2021, 12, 25)),  # Saturday: last day of its week
])
def test_holiday_week_end(holiday, expected):
    assert holiday_week_end(holiday) == expected


def test_holiday_week_ends():
    assert holiday_week_ends("thanksgiving", [2022, 2023]) == [D(2022, 11, 26), D(2023, 11, 25)]
    assert holiday_week_ends("christmas", [2023]) == [D(2023, 12, 30)]
    assert holiday_week_ends("new_year", [2024]) == [D(2024, 1, 6)]
    assert holiday_week_ends("christmas+1", [2023]) == [D(2024, 1, 6)]
    assert holiday_week_ends("thanksgiving-1", [2023]) == [D(2023, 11, 18)]


@pytest.mark.parametrize("spec", ["easter", "christmas+", "Christmas", "christmas+1.5"])
def test_parse_holiday_spec_invalid(spec):
    with pytest.raises(ValueError, match="invalid holiday spec"):
        parse_holiday_spec(spec)


def test_calendar_labels():
    dates = pd.date_range("2023-11-04", "2024-01-20", freq="W-SAT")
    labels = calendar_labels(dates, ["thanksgiving", "christmas", "new_year"])
    expected = pd.Series(0, index=dates)
    expected[pd.Timestamp("2023-11-25")] = 1
    expected[pd.Timestamp("2023-12-30")] = 2
    expected[pd.Timestamp("2024-01-06")] = 3
    np.testing.assert_array_equal(labels, expected.values)


def test_calendar_labels_overlap():
    dates = pd.date_range("2023-11-04", "2024-01-20", freq="W-SAT")
    with pytest.raises(ValueError, match="same week"):
        calendar_labels(dates, ["new_year", "christmas+1"])


def test_calendar_labels_not_saturdays():
    with pytest.raises(ValueError, match="Saturdays"):
        calendar_labels(pd.date_range("2023-11-05", periods=3, freq="W-SUN"), ["christmas"])


def _config(**kwargs):
    return SARIXModelConfig(model_name="test", main_source=SourceType.NHSN, fit_locations_separately=False,
                            power_transform=PowerTransform.FOURTH_ROOT, p=1, **kwargs)


def test_config_validation():
    with pytest.raises(ValueError, match="calendar_pooling must be specified"):
        _config(calendar_holidays=["christmas"])
    with pytest.raises(ValueError, match="should not be specified"):
        _config(calendar_pooling=PoolingStrategy.SHARED)
    with pytest.raises(ValueError, match="invalid holiday spec"):
        _config(calendar_holidays=["easter"], calendar_pooling=PoolingStrategy.SHARED)
    with pytest.raises(ValueError, match="distinct"):
        _config(calendar_holidays=["christmas", "christmas"], calendar_pooling=PoolingStrategy.SHARED)


def _run_config(max_horizon):
    return RunConfig(disease=None, ref_date=D(2023, 12, 23), output_root=None, artifact_store_root=None,
                     max_horizon=max_horizon, states=["US"], hsas=[], q_levels=[0.5], q_labels=["0.5"])


def _df():
    # two locations in contiguous blocks, as reshaped into SARIX's xy
    dates = pd.date_range("2023-11-04", "2023-12-16", freq="W-SAT")
    return pd.concat([
        pd.DataFrame({"unique_id": uid, "wk_end_date": dates, "inc_trans_cs": 0.0, "inc_trans_center_factor": c})
        for uid, c in [("stateUS", 0.7), ("state01", 0.4)]
    ], ignore_index=True)


def test_extra_sarix_params_none():
    assert SARIXModel(_config())._get_extra_sarix_params(_df(), _run_config(4)) == {}


def test_extra_sarix_params_calendar():
    model = SARIXModel(_config(calendar_holidays=["thanksgiving", "christmas", "new_year"],
                               calendar_pooling=PoolingStrategy.PARTIAL))
    params = model._get_extra_sarix_params(_df(), _run_config(4))

    # 7 training weeks (11/4 - 12/16) followed by 4 forecast weeks (12/23 - 1/13)
    expected = np.zeros((11, 1), dtype=int)
    expected[3, 0] = 1  # 2023-11-25
    expected[8, 0] = 2  # 2023-12-30, forecast horizon 2
    expected[9, 0] = 3  # 2024-01-06, forecast horizon 3
    np.testing.assert_array_equal(params["calendar_labels"], expected)
    assert params["calendar_pooling"] == "partial"
    assert params["input_power"] == 0.25
    np.testing.assert_array_equal(params["input_center"], [[0.7], [0.4]])  # batch order of xy


def test_extra_sarix_params_misaligned_dates():
    df = _df().iloc[:-1]  # state01 is missing its last week
    model = SARIXModel(_config(calendar_holidays=["christmas"], calendar_pooling=PoolingStrategy.SHARED))
    with pytest.raises(ValueError, match="same weeks"):
        model._get_extra_sarix_params(df, _run_config(4))


def test_extra_sarix_params_no_power_transform():
    cfg = _config(calendar_holidays=["christmas"], calendar_pooling=PoolingStrategy.SHARED)
    cfg.power_transform = PowerTransform.NONE
    assert SARIXModel(cfg)._get_extra_sarix_params(_df(), _run_config(4))["input_power"] == 1.0


def test_extra_sarix_params_with_covariates():
    # covariates come before the target in xy and get no calendar effects
    model = SARIXModel(_config(calendar_holidays=["christmas"], calendar_pooling=PoolingStrategy.SHARED, x=["cov"]))
    params = model._get_extra_sarix_params(_df().assign(cov=1.0), _run_config(4))
    assert params["calendar_labels"].shape == (11, 2)
    assert not params["calendar_labels"][:, 0].any() and params["calendar_labels"][:, 1].any()
    np.testing.assert_array_equal(params["input_center"], [[np.nan, 0.7], [np.nan, 0.4]])


def _fake_sarix(**kwargs):
    xy = kwargs["xy"]
    return SimpleNamespace(predictions=np.zeros((5, xy.shape[0], kwargs["forecast_horizon"], xy.shape[-1])))


def test_fit_and_predict_passes_calendar_params_to_sarix():
    model = SARIXModel(_config(calendar_holidays=["christmas"], calendar_pooling=PoolingStrategy.SHARED,
                               calendar_effect_sd_prior_scale=0.25, calendar_effect_mu_prior_scale=0.5))
    with patch("idmodels.sarix.sarix.SARIX", side_effect=_fake_sarix) as m:
        model._fit_and_predict(_df().assign(agg_level="state"), ["inc_trans_cs"], _run_config(4))
    kw = m.call_args.kwargs
    assert kw["calendar_labels"].shape == (11, 1) and kw["calendar_labels"][8, 0] == 1  # 2023-12-30
    assert kw["calendar_pooling"] == PoolingStrategy.SHARED
    assert kw["calendar_effect_sd_prior_scale"] == 0.25
    assert kw["calendar_effect_mu_prior_scale"] == 0.5
    assert kw["input_power"] == 0.25
    np.testing.assert_array_equal(kw["input_center"], [[0.7], [0.4]])


@pytest.mark.parametrize("holidays", [
    ["christmas", "new_year", "christmas+1"],  # new_year is always the week after christmas
    ["new_year+1", "christmas+2"],
    ["christmas", "christmas+0"],
    ["new_year-1", "christmas"],
])
def test_config_rejects_aliased_holidays(holidays):
    with pytest.raises(ValueError, match="same week"):
        _config(calendar_holidays=holidays, calendar_pooling=PoolingStrategy.SHARED)


def test_fit_and_predict_misaligned_dates_fails_before_fit():
    # equal row counts, so the reshape succeeds, but state01's weeks are shifted by one
    df = _df().assign(agg_level="state")
    df.loc[df["unique_id"] == "state01", "wk_end_date"] += pd.Timedelta(weeks=1)
    with patch("idmodels.sarix.sarix.SARIX", side_effect=_fake_sarix) as m:
        with pytest.raises(ValueError, match="same weeks"):
            SARIXModel(_config())._fit_and_predict(df, ["inc_trans_cs"], _run_config(4))
    m.assert_not_called()


def test_fit_and_predict_interleaved_locations():
    df = _df().assign(agg_level="state").sort_values(["wk_end_date", "unique_id"], ignore_index=True)
    with patch("idmodels.sarix.sarix.SARIX", side_effect=_fake_sarix) as m:
        with pytest.raises(ValueError, match="contiguous"):
            SARIXModel(_config())._fit_and_predict(df, ["inc_trans_cs"], _run_config(4))
    m.assert_not_called()


def test_fit_and_predict_maps_none_calendar_pooling_to_unshared():
    # only calendar_pooling is mapped; theta and sigma pooling are passed to SARIX unchanged
    model = SARIXModel(_config(calendar_holidays=["christmas"], calendar_pooling=PoolingStrategy.NONE))
    with patch("idmodels.sarix.sarix.SARIX", side_effect=_fake_sarix) as m:
        model._fit_and_predict(_df().assign(agg_level="state"), ["inc_trans_cs"], _run_config(4))
    kw = m.call_args.kwargs
    assert kw["calendar_pooling"] == "unshared"
    assert kw["theta_pooling"] == kw["sigma_pooling"] == PoolingStrategy.NONE
