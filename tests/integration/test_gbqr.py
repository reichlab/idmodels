import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pandas.testing import assert_frame_equal

from idmodels.gbqr import GBQRModel


def test_gbqr(tmp_path):
    model_config = SimpleNamespace(
        model_class = "gbqr",
        model_name = "gbqr_no_reporting_adj",
        
        incl_level_feats = True,

        # bagging setup
        num_bags = 10,
        bag_frac_samples = 0.7,

        # adjustments to reporting
        reporting_adj = False,

        # data sources and adjustments for reporting issues
        sources = ["flusurvnet", "nhsn", "ilinet"],

        # fit locations separately or jointly
        fit_locations_separately = False,

        # power transform applied to surveillance signals
        power_transform = "4rt"
    )


    run_config = SimpleNamespace(
        disease="flu",
        ref_date=datetime.date.fromisoformat("2024-01-06"),
        output_root=tmp_path / "model-output",
        artifact_store_root=tmp_path / "artifact-store",
        save_feat_importance=False,
        locations=["US", "01", "02", "04", "05", "06", "08", "09", "10", "11",
                   "12", "13", "15", "16", "17", "18", "19", "20", "21", "22",
                   "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
                   "33", "34", "35", "36", "37", "38", "39", "40", "41", "42",
                   "44", "45", "46", "47", "48", "49", "50", "51", "53", "54",
                   "55", "56", "72"],
        max_horizon=3,
        q_levels = [0.025, 0.50, 0.975],
        q_labels = ["0.025", "0.5", "0.975"],
        num_bags = 10
    )
    
    model = GBQRModel(model_config)
    model.run(run_config)
    
    actual_df = pd.read_csv(
        run_config.output_root / "UMass-gbqr_no_reporting_adj" /
        "2024-01-06-UMass-gbqr_no_reporting_adj.csv"
    )
    expected_df = pd.read_csv(
        Path("tests") / "integration" / "data" /
        "UMass-gbqr_no_reporting_adj" /
        "2024-01-06-UMass-gbqr_no_reporting_adj.csv"
    )
    assert_frame_equal(actual_df, expected_df)
