import datetime
from pathlib import Path
from unittest.mock import patch

import lightgbm
import numpy
import pandas as pd
import pytest
from iddata.sources.flusurvnet import FluSurvNetDataSource
from iddata.sources.nhsn import NHSNDataSource
from pandas.testing import assert_frame_equal

from idmodels.config import GBQRModelConfig, PowerTransform, SourceType
from idmodels.gbqr import GBQRModel


def test_gbqr_nhsn(make_run_config):
    date = datetime.date.fromisoformat("2024-01-06")
    fips_codes = ["US", "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17", "18", "19",
                  "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
                  "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
                  "56", "72"]
    model_config = create_test_gbqr_model_config(main_source=SourceType.NHSN, supplementary_sources=[SourceType.FLUSURVNET, SourceType.ILINET])
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


def test_gbqr_nhsn_smh(make_run_config):
    date = datetime.date.fromisoformat("2024-12-07")
    fips_codes = ["US", "01", "06", "25", "48", "36", "56", "72"]
    model_config = create_test_gbqr_model_config(main_source=SourceType.NHSN, supplementary_sources=[SourceType.SMH], smh_model=["NotreDame-FRED"], smh_otid=["010100010100"], custom_name="nhsn_smh")
    run_config = make_run_config(ref_date=date, states=fips_codes, hsas=[])

    # patch lgb.LGBMRegressor's `predict()` to return the same values to make the tests reproducible across OSs
    with patch.object(lightgbm.sklearn.LGBMModel, "predict",
                      return_value=_predictions_val()[0:33]):
        model = GBQRModel(model_config)
        model.run(run_config)
    actual_df = pd.read_csv(run_config.output_root / f"UMass-{model_config.model_name}" /
                            f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv")
    expected_df = pd.read_csv(Path("tests") / "integration" / "data" /
                              f"UMass-{model_config.model_name}" /
                              f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv")
    assert_frame_equal(actual_df, expected_df)



# tests that synthetic locations aren't dropped
def test_filter_locations(make_run_config):
    """
    Unit test for GBQRModel._filter_locations: particularly that the SMH rows are filtered correctly
    """
    model_config = create_test_gbqr_model_config(
        main_source=SourceType.NHSN,
        supplementary_sources=[SourceType.NSSP, SourceType.SMH],
    )
    run_config = make_run_config(ref_date=datetime.date.fromisoformat("2024-12-07"), states=["US"], hsas=["9"])

    rows = [
        # kept: matches selected surveillance location (state)
        {"id": "surveillance", "agg_level": "national", "location": "US", "source": "nhsn", "season": "2024/25"},
        # kept: matches selected surveillance location (hsa)
        {"id": "surveillance", "agg_level": "hsa", "location": "9", "source": "nssp", "season": "2024/25"},
        # kept: matches selected location
        {"id": "smh_match", "agg_level": "national", "location": "syn-US", "source": "smh-NotreDame-FRED", "season": "2024/25A-010100010100"},
        # dropped: wrong location
        {"id": "smh_wrong_location", "agg_level": "state", "location": "syn-01", "source": "smh-NotreDame-FRED", "season": "2024/25A-010100010100"}
    ]
    df = pd.DataFrame(rows)

    model = GBQRModel(model_config)
    filtered_df = model._filter_locations(df, run_config)

    assert set(filtered_df["id"]) == {"surveillance", "surveillance", "smh_match"}


@pytest.mark.parametrize("model_id, otid, row_ids", [
    ([], [], {"surveillance", "smh_match", "smh_wrong_model", "smh_wrong_otid"}), # all SMH models and otids
    (["NotreDame-FRED"], [], {"surveillance", "smh_match", "smh_wrong_otid"}), # Restrict to single SMH model
    ([], ["010100010100"], {"surveillance", "smh_match", "smh_wrong_model"}), # Restrict to a single SMH otid
    (["NotreDame-FRED"], ["010100010100"], {"surveillance", "smh_match"}) # Restrict to a single SMH model-otid combo
])
def test_gbqr_filter_smh(make_run_config, model_id, otid, row_ids):
    """
    Unit test for GBQRModel._filter_smh: non-SMH rows always pass through, and SMH rows are
    kept only when they match the configured smh_model and smh_otid, and have wk_end_date
    strictly before run_config.ref_date.
    """
    model_config = create_test_gbqr_model_config(
        main_source=SourceType.NHSN,
        supplementary_sources=[SourceType.SMH],
        smh_model=model_id,
        smh_otid=otid,
    )
    run_config = make_run_config(ref_date=datetime.date.fromisoformat("2024-12-07"), states=["US"], hsas=[])

    rows = [
        # kept: non-SMH surveillance source, unaffected by any SMH filter
        {"id": "surveillance", "source": "nhsn", "season": "2024/25", "wk_end_date": pd.Timestamp("2024-12-14")},
        # kept: matches configured model + output_type_id, strictly before ref_date
        {"id": "smh_match", "source": "smh-NotreDame-FRED", "season": "2024/25A-010100010100",
         "wk_end_date": pd.Timestamp("2024-11-30"), "round": 5, "location": "syn-US"},
        # dropped: wrong model_id
        {"id": "smh_wrong_model", "source": "smh-OtherModel", "season": "2024/25A-010100010100",
         "wk_end_date": pd.Timestamp("2024-11-30"), "round": 5, "location": "syn-US"},
        # dropped: wrong output_type_id
        {"id": "smh_wrong_otid", "source": "smh-NotreDame-FRED", "season": "2024/25A-999999999999",
         "wk_end_date": pd.Timestamp("2024-11-30"), "round": 5, "location": "syn-US"},
        # dropped: wk_end_date not strictly before ref_date
        {"id": "smh_not_before_ref_date", "source": "smh-NotreDame-FRED", "season": "2024/25A-010100010100",
         "wk_end_date": pd.Timestamp("2024-12-07"), "round": 5, "location": "syn-US"},
    ]
    df = pd.DataFrame(rows)

    model = GBQRModel(model_config)
    filtered_df = model._filter_smh(df, model_config, run_config)

    assert set(filtered_df["id"]) == row_ids


def _smh_otid_rows(otids, model_id="NotreDame-FRED", smh_round=5, locations=("syn-US",)):
    """
    Synthetic SMH rows spanning `locations` (default: a single location), each seeing the full
    set of `otids`. output_type_id is scoped per (round, location) in real SMH data (see
    iddata.sources.smh), so otid sampling/filtering is done within those groups.
    """
    rows = [
        {"id": f"smh_{loc}_{otid}", "source": f"smh-{model_id}", "season": f"2024/25A-{otid}",
         "wk_end_date": pd.Timestamp("2024-11-30"), "round": smh_round, "location": loc}
        for loc in locations
        for otid in otids
    ]
    return rows + [{"id": "surveillance", "source": "nhsn", "season": "2024/25", "wk_end_date": pd.Timestamp("2024-12-14")}]


def test_gbqr_filter_smh_num_otid_samples_available_ids(make_run_config):
    """
    Unit test for GBQRModel._filter_smh: when smh_num_otid is set (instead of an explicit
    smh_otid list), the requested number of ids is randomly sampled independently within each
    (round, location) group, from the ids actually present in that group's (already
    model-filtered) SMH rows.
    """
    model_config = create_test_gbqr_model_config(
        main_source=SourceType.NHSN,
        supplementary_sources=[SourceType.SMH],
        smh_model=["NotreDame-FRED"],
    )
    model_config.smh_num_otid = 2
    model_config.smh_otid_seed = 42
    run_config = make_run_config(ref_date=datetime.date.fromisoformat("2024-12-07"), states=["US"], hsas=[])

    df = pd.DataFrame(_smh_otid_rows(["a", "b", "c", "d"]))

    model = GBQRModel(model_config)
    filtered_df = model._filter_smh(df, model_config, run_config)

    kept_otids = {row_id.rsplit("_", 1)[-1] for row_id in filtered_df["id"] if row_id.startswith("smh_")}
    assert len(kept_otids) == 2
    assert kept_otids <= {"a", "b", "c", "d"}
    assert "surveillance" in set(filtered_df["id"])
    # a single-location group's sampled ids should never spill over onto smh_otid
    # (that field is reserved for the explicit-list path)
    assert model_config.smh_otid == []


def test_gbqr_filter_smh_num_otid_samples_independently_per_location(make_run_config):
    """Each (round, location) group draws its own independent sample of smh_num_otid ids."""
    model_config = create_test_gbqr_model_config(
        main_source=SourceType.NHSN, supplementary_sources=[SourceType.SMH], smh_model=["NotreDame-FRED"],
    )
    model_config.smh_num_otid = 2
    model_config.smh_otid_seed = 42
    run_config = make_run_config(ref_date=datetime.date.fromisoformat("2024-12-07"), states=["US"], hsas=[])

    df = pd.DataFrame(_smh_otid_rows(["a", "b", "c", "d"], locations=("syn-US", "syn-01")))

    model = GBQRModel(model_config)
    filtered_df = model._filter_smh(df, model_config, run_config)

    for loc in ("syn-US", "syn-01"):
        kept = {row_id.rsplit("_", 1)[-1] for row_id in filtered_df["id"]
                if row_id.startswith(f"smh_{loc}_")}
        assert len(kept) == 2
        assert kept <= {"a", "b", "c", "d"}


def test_gbqr_filter_smh_num_otid_samples_independently_per_model(make_run_config):
    """
    Regression test: output_type_id is assigned independently per SMH model and is NOT a globally
    unique identifier within a (round, location) group -- two different models can reuse the same
    otid string. When no smh_model filter narrows the source down to one model (smh_model=[], the
    "all models" case), sampling must draw smh_num_otid ids *per model*, not from a single pool
    shared across models: pooling before sampling (and then joining back on (round, location, otid)
    alone) would let a sampled id incidentally match rows from every model that happens to reuse
    that id string, and could just as easily fail to match a given model's rows at all, starving
    that model even though smh_num_otid ids were "sampled".
    """
    model_config = create_test_gbqr_model_config(
        main_source=SourceType.NHSN, supplementary_sources=[SourceType.SMH], smh_model=[],
    )
    model_config.smh_num_otid = 2
    model_config.smh_otid_seed = 42
    run_config = make_run_config(ref_date=datetime.date.fromisoformat("2024-12-07"), states=["US"], hsas=[])

    # model_a has 4 ids to sample from; model_b reuses 2 of the SAME id strings and has no others.
    # A pooled (round, location)-only sample of 2 ids could easily miss both of model_b's ids
    # entirely (e.g. sampling "c"/"d"), which would incorrectly starve model_b of any SMH rows.
    rows = (
        _smh_otid_rows(["a", "b", "c", "d"], model_id="model_a")
        + _smh_otid_rows(["a", "b"], model_id="model_b")
    )
    df = pd.DataFrame(rows)

    model = GBQRModel(model_config)
    filtered_df = model._filter_smh(df, model_config, run_config)

    def kept_otids_for(model_id):
        model_rows = filtered_df[filtered_df["source"] == f"smh-{model_id}"]
        return {row_id.rsplit("_", 1)[-1] for row_id in model_rows["id"]}

    kept_a = kept_otids_for("model_a")
    kept_b = kept_otids_for("model_b")

    # model_b only has 2 ids available, so with smh_num_otid=2 it must always retain exactly both
    assert kept_b == {"a", "b"}
    # model_a independently samples 2 of its own 4 ids, regardless of what was drawn for model_b
    assert len(kept_a) == 2
    assert kept_a <= {"a", "b", "c", "d"}


def test_gbqr_filter_smh_num_otid_is_reproducible_with_seed(make_run_config):
    """Same smh_otid_seed produces the same sampled ids across separate calls."""
    run_config = make_run_config(ref_date=datetime.date.fromisoformat("2024-12-07"), states=["US"], hsas=[])
    df = pd.DataFrame(_smh_otid_rows(["a", "b", "c", "d", "e"]))

    def sample():
        model_config = create_test_gbqr_model_config(
            main_source=SourceType.NHSN, supplementary_sources=[SourceType.SMH], smh_model=["NotreDame-FRED"],
        )
        model_config.smh_num_otid = 3
        model_config.smh_otid_seed = 7
        filtered_df = GBQRModel(model_config)._filter_smh(df.copy(), model_config, run_config)
        return set(filtered_df["id"])

    assert sample() == sample()


def test_gbqr_filter_smh_num_otid_raises_if_more_than_available(make_run_config):
    model_config = create_test_gbqr_model_config(
        main_source=SourceType.NHSN, supplementary_sources=[SourceType.SMH], smh_model=["NotreDame-FRED"],
    )
    model_config.smh_num_otid = 5
    run_config = make_run_config(ref_date=datetime.date.fromisoformat("2024-12-07"), states=["US"], hsas=[])
    df = pd.DataFrame(_smh_otid_rows(["a", "b"]))

    model = GBQRModel(model_config)
    with pytest.raises(ValueError, match="exceeds"):
        model._filter_smh(df, model_config, run_config)


def test_gbqr_model_config_rejects_both_smh_otid_and_smh_num_otid():
    with pytest.raises(ValueError, match="at most one"):
        GBQRModelConfig(
            model_name="gbqr_bad_config",
            main_source=SourceType.NHSN,
            fit_locations_separately=False,
            power_transform=PowerTransform.FOURTH_ROOT,
            smh_otid=["010100010100"],
            smh_num_otid=2,
        )


@pytest.mark.parametrize("fips_codes, nci_ids", [
    (["US", "01", "25"], []),  # states only (US national counts as a state)
    ([], ["1", "25", "99"]),  # hsas only
    (["US", "01", "25"], ["1", "25", "99"])  # both
])
def test_gbqr_nssp(make_run_config, fips_codes, nci_ids):
    date = datetime.date.fromisoformat("2025-11-22")
    model_config = create_test_gbqr_model_config(main_source=SourceType.NSSP)
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


def test_gbqr_invalid_main_source_raises(make_run_config):
    model_config = create_test_gbqr_model_config(main_source=SourceType.ILINET)
    run_config = make_run_config(ref_date=datetime.date.fromisoformat("2024-01-06"), states=["US"], hsas=[])

    with pytest.raises(ValueError, match="GBQRModel only supports NHSN and NSSP as main source."):
        GBQRModel(model_config)._build_sources(run_config)


def test_gbqr_build_sources_dedupes_main_source_in_supplementary_sources(make_run_config):
    model_config = create_test_gbqr_model_config(
        main_source=SourceType.NHSN,
        supplementary_sources=[SourceType.NHSN, SourceType.FLUSURVNET],
    )
    run_config = make_run_config(ref_date=datetime.date.fromisoformat("2024-01-06"), states=["US"], hsas=[])

    sources = GBQRModel(model_config)._build_sources(run_config)

    assert len(sources) == 2
    assert {type(s) for s in sources} == {NHSNDataSource, FluSurvNetDataSource}


def test_gbqr_test_set_predictions_filter_to_main_source(make_run_config):
    """
    Regression test: _train_gbq_and_predict() used to keep test-set rows for source in
    {"nhsn", "nssp"} unconditionally. That was only safe because a model's sources could not
    include both NHSN and NSSP at once. Now that NHSN can be a training_source alongside NSSP
    as main_source, the filter must be scoped to main_source -- otherwise both sources' rows
    survive into the output, producing duplicate (location, wk_end_date, horizon) rows.
    """
    model_config = create_test_gbqr_model_config(main_source=SourceType.NSSP, supplementary_sources=[SourceType.NHSN])
    model_config.num_bags = 2
    model_config.bag_frac_samples = 1.0
    date = datetime.date.fromisoformat("2024-01-06")
    run_config = make_run_config(ref_date=date, states=["01"], hsas=[])

    dates = pd.to_datetime(["2023-12-16", "2023-12-23", "2023-12-30", "2024-01-06"])
    rows = []
    for source in ["nssp", "nhsn"]:
        for i, wk_end_date in enumerate(dates):
            is_test = wk_end_date == dates.max()
            rows.append({
                "source": source,
                "agg_level": "state",
                "location": "01",
                "wk_end_date": wk_end_date,
                "pop": 1_000_000,
                "inc_trans_cs": 0.1 * (i + 1),
                "horizon": 1,
                "inc_trans_center_factor": 0.0,
                "inc_trans_scale_factor": 1.0,
                "season": "2023/24",
                "season_week": 10,
                "delta_target": None if is_test else 0.05 * (i + 1),
                "feat1": float(i),
            })
    df = pd.DataFrame(rows)

    model = GBQRModel(model_config)
    with patch.object(lightgbm.sklearn.LGBMModel, "predict", return_value=numpy.array([0.1, 0.1])):
        preds_df = model._fit_and_predict(df, feat_names=["feat1"], run_config=run_config)

    assert set(preds_df["source"].unique()) == {"nssp"}
    key_cols = ["location", "wk_end_date", "horizon", "output_type_id"]
    assert not preds_df.duplicated(subset=key_cols).any()


def create_test_gbqr_model_config(main_source, supplementary_sources=[], smh_model=[], smh_otid=[], custom_name=None):
    name = custom_name if custom_name is not None else main_source.value
    model_config = GBQRModelConfig(
        model_name="gbqr_" + name + "_no_reporting_adj",

        incl_level_feats=True,

        # bagging setup
        num_bags=10,
        bag_frac_samples=0.7,

        # adjustments to reporting
        reporting_adj=False,

        # data sources and adjustments for reporting issues
        main_source=main_source,
        supplementary_sources=supplementary_sources,

        # fit locations separately or jointly
        fit_locations_separately=False,

        # power transform applied to surveillance signals
        power_transform=PowerTransform.FOURTH_ROOT,

        # smh trajectory filters
        smh_model = smh_model,
        smh_otid = smh_otid
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
