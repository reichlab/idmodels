import datetime
from pathlib import Path
from unittest.mock import patch

import lightgbm
import numpy
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from idmodels.config import GBQRModelConfig, PowerTransform, SourceType
from idmodels.gbqr import GBQRModel


def test_gbqr_nhsn(make_run_config):
    date = datetime.date.fromisoformat("2024-01-06")
    fips_codes = ["US", "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17", "18", "19",
                  "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
                  "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
                  "56", "72"]
    model_config = create_test_gbqr_model_config(sources=[SourceType.FLUSURVNET, SourceType.NHSN, SourceType.ILINET])
    run_config = make_run_config(ref_date=date, states=fips_codes, hsas=[])

    # patch lgb.LGBMRegressor's `predict()` to return the same values to make the tests reproducible across OSs
    with patch.object(lightgbm.sklearn.LGBMModel, "predict", return_value=_predictions_val()):
        model = GBQRModel(model_config)
        model.run(run_config)
    actual_df = pd.read_csv(run_config.output_root / f"UMass-{model_config.model_name}" /
                            f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv")
    expected_df = pd.read_csv(Path("tests") / "integration" / "data" /
                              f"UMass-{model_config.model_name}" /
                              f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv")
    assert_frame_equal(actual_df, expected_df)


@pytest.mark.parametrize("fips_codes, nci_ids", [
    (["US", "01", "25"], []),  # states only (US national counts as a state)
    ([], ["1", "25", "99"]),  # hsas only
    (["US", "01", "25"], ["1", "25", "99"])  # both
])
def test_gbqr_nssp(make_run_config, fips_codes, nci_ids):
    date = datetime.date.fromisoformat("2025-11-22")
    model_config = create_test_gbqr_model_config(sources=[SourceType.NSSP])
    run_config = make_run_config(ref_date=date, states=fips_codes, hsas=nci_ids)

    # patch the `_np_percentile()` helper function return the same values to make the tests reproducible across OSs
    if (fips_codes != []) & (nci_ids == []):
        locs_len = 3  # only forecast for 3 states
        agg_level = "state"
    elif (fips_codes == []) & (nci_ids != []):
        locs_len = 3  # only forecast for 3 hsas
        agg_level = "hsa"
    else:
        locs_len = 6  # only forecast for 6 locs
        agg_level = "both"

    # patch lgb.LGBMRegressor's `predict()` to return the same values to make the tests reproducible across OSs
    with patch.object(lightgbm.sklearn.LGBMModel, "predict",
                      return_value=_predictions_val()[0:(locs_len * 3)]):  # x3 quantiles
        model = GBQRModel(model_config)
        model.run(run_config)
    actual_df = pd.read_csv(run_config.output_root / f"UMass-{model_config.model_name}" /
                            f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv")
    expected_df = pd.read_csv(Path("tests") / "integration" / "data" /
                              f"UMass-{model_config.model_name}" /
                              f"{str(run_config.ref_date)}-UMass-{model_config.model_name}-{agg_level}.csv")
    assert_frame_equal(actual_df, expected_df)


def create_test_gbqr_model_config(sources):
    if SourceType.NHSN in sources:
        main_source = SourceType.NHSN
    elif SourceType.NSSP in sources:
        main_source = SourceType.NSSP
    else:
        main_source = None

    model_config = GBQRModelConfig(
        model_name="gbqr_" + main_source.value + "_no_reporting_adj",

        incl_level_feats=True,

        # bagging setup
        num_bags=10,
        bag_frac_samples=0.7,

        # adjustments to reporting
        reporting_adj=False,

        # data sources and adjustments for reporting issues
        sources=sources,

        # fit locations separately or jointly
        fit_locations_separately=False,

        # power transform applied to surveillance signals
        power_transform=PowerTransform.FOURTH_ROOT,
    )
    return model_config



def _predictions_val():
    return numpy.array([
        -0.10884266, -0.11411782, -0.17619509, -0.08364025, -0.10244736, -0.16727379, -0.09546074, -0.17045369,
        -0.17278568, -0.11298913, -0.19227807, -0.14617675, -0.13368925, -0.09696042, -0.1027716, -0.08390366,
        -0.08712261, -0.09499113, -0.15597696, -0.13101825, -0.12856413, -0.12029619, -0.10139341, -0.10120798,
        -0.12944103, -0.10346755, -0.13421479, -0.07137864, -0.11986316, -0.130491, -0.12085533, -0.1315757,
        -0.10771773, -0.17399641, -0.11908955, -0.10019343, -0.05442041, -0.05877476, -0.10446433, -0.15300424,
        -0.18010623, -0.09650918, -0.13475974, -0.0964629, -0.13605704, -0.14251085, -0.12972634, -0.07168675,
        -0.09557477, -0.09840103, -0.21644938, -0.09870431, -0.0764911, -0.1494211, -0.12459323, -0.23314011,
        -0.09794522, -0.15779808, -0.22165089, -0.12907891, -0.20806452, -0.21627292, -0.15593362, -0.24470388,
        -0.21334364, -0.17598979, -0.13845846, -0.14435533, -0.11597898, -0.1207362, -0.13959495, -0.19600365,
        -0.16658358, -0.16277678, -0.1638527, -0.13557105, -0.13412814, -0.16258698, -0.13523057, -0.17245702,
        -0.08568361, -0.16011519, -0.1864427, -0.15969408, -0.18746014, -0.14952648, -0.23039219, -0.15232162,
        -0.1327803, -0.06263264, -0.07307973, -0.1361057, -0.22058937, -0.23823502, -0.10644479, -0.16530044,
        -0.13205234, -0.16920299, -0.18925207, -0.16477677, -0.10021285, -0.11413695, -0.13060149, -0.27361888,
        -0.12021192, -0.1158, -0.18091993, -0.1305779, -0.31868126, -0.11249511, -0.2020846, -0.30728416, -0.15950042,
        -0.23065431, -0.24428418, -0.19267856, -0.33024503, -0.25543255, -0.21543295, -0.17017334, -0.18436581,
        -0.13610372, -0.14678262, -0.16043744, -0.24600945, -0.18013397, -0.19303495, -0.20147989, -0.16378134,
        -0.16170607, -0.19378843, -0.15906578, -0.21970134, -0.10316246, -0.20314399, -0.21403258, -0.20106399,
        -0.2313942, -0.19520809, -0.32762028, -0.15957368, -0.16449517, -0.06741025, -0.09077963, -0.16827476,
        -0.2490845, -0.33546311, -0.12326175, -0.1939618, -0.16376722, -0.19949919, -0.19650414, -0.20602138,
        -0.1203376, -0.13177956, -0.15881178, -0.33819272, -0.12741436, -0.15506679])
