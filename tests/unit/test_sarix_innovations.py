import datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from idmodels.config import (
    PoolingStrategy,
    PowerTransform,
    RunConfig,
    SARIXFourierModelConfig,
    SARIXModelConfig,
    SourceType,
)
from idmodels.sarix import SARIXFourierModel, SARIXModel


def _config(config_class=SARIXModelConfig, **kwargs):
    return config_class(model_name="test", main_source=SourceType.NHSN, fit_locations_separately=False,
                        power_transform=PowerTransform.FOURTH_ROOT, p=1, **kwargs)


def _run_config():
    return RunConfig(disease=None, ref_date=datetime.date(2023, 12, 23), output_root=None, artifact_store_root=None,
                     max_horizon=4, states=["US"], hsas=[], q_levels=[0.5], q_labels=["0.5"])


def _df():
    dates = pd.date_range("2023-11-04", "2023-12-16", freq="W-SAT")
    return pd.concat([
        pd.DataFrame({"unique_id": uid, "location": uid[-2:], "wk_end_date": dates, "inc_trans_cs": 0.0,
                      "inc_trans_center_factor": c, "agg_level": "state"})
        for uid, c in [("stateUS", 0.7), ("state01", 0.4)]
    ], ignore_index=True)


def _fake_sarix(**kwargs):
    xy = kwargs["xy"]
    return SimpleNamespace(predictions=np.zeros((5, xy.shape[0], kwargs["forecast_horizon"], xy.shape[-1])))


def test_config_validation():
    with pytest.raises(ValueError, match="innovation_dist must be"):
        _config(innovation_dist="cauchy")
    with pytest.raises(ValueError, match="innovation_df_pooling must be specified"):
        _config(innovation_dist="t")
    with pytest.raises(ValueError, match="should not be specified"):
        _config(innovation_df_pooling=PoolingStrategy.SHARED)


def test_extra_sarix_params_normal():
    assert SARIXModel(_config())._get_extra_sarix_params(_df(), _run_config()) == {}


@pytest.mark.parametrize("pooling, expected", [
    (PoolingStrategy.NONE, "unshared"),
    (PoolingStrategy.SHARED, "shared"),
    (PoolingStrategy.PARTIAL, "partial"),
])
def test_extra_sarix_params_t(pooling, expected):
    model = SARIXModel(_config(innovation_dist="t", innovation_df_pooling=pooling, innovation_df_prior_scale=5.0,
                               innovation_df_sd_prior_scale=0.5))
    assert model._get_extra_sarix_params(_df(), _run_config()) == {
        "innovation_dist": "t", "innovation_df_pooling": expected, "innovation_df_prior_scale": 5.0,
        "innovation_df_sd_prior_scale": 0.5}


def test_fit_and_predict_passes_t_with_calendar_and_fourier_params():
    """The Fourier model passes the innovation, calendar and Fourier parameters together."""
    cfg = _config(SARIXFourierModelConfig, fourier_K=2, fourier_pooling=PoolingStrategy.SHARED,
                  calendar_holidays=["christmas"], calendar_pooling=PoolingStrategy.SHARED,
                  innovation_dist="t", innovation_df_pooling=PoolingStrategy.PARTIAL)
    with patch("idmodels.sarix.sarix.SARIX", side_effect=_fake_sarix) as m:
        SARIXFourierModel(cfg)._fit_and_predict(_df(), ["inc_trans_cs"], _run_config())
    kw = m.call_args.kwargs
    assert kw["innovation_dist"] == "t"
    assert kw["innovation_df_pooling"] == "partial"
    assert kw["innovation_df_prior_scale"] == 10.0
    assert kw["innovation_df_sd_prior_scale"] == 1.0
    assert kw["calendar_pooling"] == "shared" and kw["calendar_labels"].shape == (11, 1)
    assert kw["fourier_K"] == 2


def test_fit_and_predict_normal_passes_no_innovation_params():
    with patch("idmodels.sarix.sarix.SARIX", side_effect=_fake_sarix) as m:
        SARIXModel(_config())._fit_and_predict(_df(), ["inc_trans_cs"], _run_config())
    assert not any(k.startswith("innovation") for k in m.call_args.kwargs)
