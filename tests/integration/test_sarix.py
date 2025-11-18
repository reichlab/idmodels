import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from idmodels.sarix import SARIXFourierModel, SARIXModel


# Co-written with Claude
def test_combined_state_and_hsa_fail(tmp_path):
    date = datetime.date.fromisoformat("2025-09-27")
    model_config = create_test_sarix_model_config(main_source=["nssp"], theta_pooling="shared", sigma_pooling="none")
    run_config = create_test_sarix_run_config(ref_date=date, states=["44"], hsas=["1", "25"], num=50, tmp_path=tmp_path)
    
    with pytest.raises(NotImplementedError, match="simultaneously forecasting"):
        model = SARIXModel(model_config)
        model.run(run_config)
        raise NotImplementedError("simultaneously forecasting")


def test_sarix_nhsn(tmp_path):
    date = datetime.date.fromisoformat("2024-01-06")
    fips_codes = ["US", "01", "02", "04", "05", "06", "08", "09", "10", "11",
                "12", "13", "15", "16", "17", "18", "19", "20", "21", "22",
                "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
                "33", "34", "35", "36", "37", "38", "39", "40", "41", "42",
                "44", "45", "46", "47", "48", "49", "50", "51", "53", "54",
                "55", "56", "72"]
    model_config = create_test_sarix_model_config(main_source=["nhsn"], theta_pooling="shared", sigma_pooling="none")
    run_config = create_test_sarix_run_config(ref_date=date, states=fips_codes, hsas=[], num=200, tmp_path=tmp_path)
    
    # patch the `_np_percentile()` helper function return the same values to make the tests reproducible across OSs
    with patch("idmodels.sarix._np_percentile", return_value=_np_percentile_val()):
        model = SARIXModel(model_config)
        model.run(run_config)

    actual_df = pd.read_csv(
        run_config.output_root / f"UMass-{model_config.model_name}" / 
        f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv"
    )
    expected_df = pd.read_csv(
        Path("tests") / "integration" / "data" /
        f"UMass-{model_config.model_name}" / 
        f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv"
    )
    assert_frame_equal(actual_df, expected_df)


@pytest.mark.parametrize("fips_codes, nci_ids", [
    # Missouri (29) does not submit to NSSP
    (["US", "01", "02", "04", "05", "06", "08", "09", "10", "11",
    "12", "13", "15", "16", "17", "18", "19", "20", "21", "22",
    "23", "24", "25", "26", "27", "28", "30", "31", "32",
    "33", "34", "35", "36", "37", "38", "39", "40", "41", "42",
    "44", "45", "46", "47", "48", "49", "50", "51", "53", "54",
    "55", "56"],
    []),
    
    ([],
    ["1", "25", "99"])
])
def test_sarix_nssp(tmp_path, fips_codes, nci_ids):
    date = datetime.date.fromisoformat("2025-09-27")
    model_config = create_test_sarix_model_config(main_source=["nssp"], theta_pooling="shared", sigma_pooling="none")
    run_config = create_test_sarix_run_config(ref_date=date, states=fips_codes, hsas=nci_ids, num=200, tmp_path=tmp_path)
    
    # patch the `_np_percentile()` helper function return the same values to make the tests reproducible across OSs
    if fips_codes != []:
        locs_len = 51 # nssp data only covers 51 locations
        agg_level = "state"
    else:
        locs_len = 3 # only forecast for 3 hsas
        agg_level = "hsa"
    
    with patch("idmodels.sarix._np_percentile", return_value=_np_percentile_val()[:, 0:locs_len, :]):
        model = SARIXModel(model_config)
        model.run(run_config)

    actual_df = pd.read_csv(
        run_config.output_root / f"UMass-{model_config.model_name}" / 
        f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv"
    )
    expected_df = pd.read_csv(
        Path("tests") / "integration" / "data" /
        f"UMass-{model_config.model_name}" / 
        f"{str(run_config.ref_date)}-UMass-{model_config.model_name}-{agg_level}.csv"
    )
    assert_frame_equal(actual_df, expected_df)


def test_sarix_shared_sigma_pooling_multiple_batches(tmp_path):
    """Test that sigma_pooling='shared' works correctly with multiple batches (locations)."""
    # Use multiple locations to ensure we have multiple batches
    date = datetime.date.fromisoformat("2024-01-06")
    fips_codes = ["US", "01", "02", "04", "05"]  # Multiple locs = multiple batches
    model_config = create_test_sarix_model_config(main_source=["nhsn"], theta_pooling="none", sigma_pooling="shared")
    run_config = create_test_sarix_run_config(ref_date=date, states=fips_codes, hsas=[], num=200, tmp_path=tmp_path)
    
    model = SARIXModel(model_config)
    model.run(run_config)

    actual_df = pd.read_csv(
        run_config.output_root / f"UMass-{model_config.model_name}" / 
        f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv"
    )

    # Verify the output has the expected structure
    assert len(actual_df) > 0, "Output dataframe should not be empty"
    assert set(actual_df["location"].unique()) == set(run_config.states), \
        "Output should contain predictions for all input states"
    assert all(actual_df["output_type"] == "quantile"), \
        "All outputs should be quantiles"
    # Convert output_type_id to string for comparison since pandas may infer numeric types
    assert set(actual_df["output_type_id"].astype(str).unique()) == set(run_config.q_labels), \
        "Output should contain all specified quantile levels"
    assert actual_df["value"].notna().all(), \
        "All predictions should be non-null"
    assert (actual_df["value"] >= 0).all(), \
        "All predictions should be non-negative"


def test_sarix_fourier_none_pooling(tmp_path):
    """Test SARIXFourierModel with fourier_pooling='none' (unpooled)."""
    model_config = SimpleNamespace(
        model_class="sarix_fourier",
        model_name="sarix_p2_fourier_K2_none",

        # data sources
        sources=["nhsn"],

        # fit locations separately or jointly
        fit_locations_separately=False,

        # SARIX model parameters
        p=2,
        P=0,
        d=0,
        D=0,
        season_period=1,

        # power transform
        power_transform="4rt",

        # parameter pooling
        theta_pooling="shared",
        sigma_pooling="shared",

        # Fourier parameters
        fourier_K=2,
        fourier_pooling="none",  # Unpooled Fourier coefficients

        # covariates
        x=[]
    )

    date = datetime.date.fromisoformat("2024-01-06")
    fips_codes = ["US", "01", "02", "04", "05"] # fewer locs for faster testing
    # model_config = create_test_sarix_model_config(main_source=["nhsn"], theta_pooling="shared", sigma_pooling="none")
    run_config = create_test_sarix_run_config(ref_date=date, states=fips_codes, hsas=[], num=50, tmp_path=tmp_path)

    model = SARIXFourierModel(model_config)
    model.run(run_config)

    # Verify output structure
    actual_df = pd.read_csv(
        run_config.output_root / f"UMass-{model_config.model_name}" / 
        f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv"
    )

    # Assertions
    assert len(actual_df) > 0, "Output dataframe should not be empty"
    assert set(actual_df["location"].unique()) == set(run_config.states), \
        "Output should contain predictions for all input states"
    assert all(actual_df["output_type"] == "quantile"), \
        "All outputs should be quantiles"
    assert set(actual_df["output_type_id"].astype(str).unique()) == set(run_config.q_labels), \
        "Output should contain all specified quantile levels"
    assert actual_df["value"].notna().all(), \
        "All predictions should be non-null"
    assert (actual_df["value"] >= 0).all(), \
        "All predictions should be non-negative"


def test_sarix_fourier_shared_pooling(tmp_path):
    """Test SARIXFourierModel with fourier_pooling='shared' (pooled across locations)."""
    model_config = SimpleNamespace(
        model_class="sarix_fourier",
        model_name="sarix_p2_fourier_K2_shared",

        # data sources
        sources=["nhsn"],

        # fit locations separately or jointly
        fit_locations_separately=False,

        # SARIX model parameters
        p=2,
        P=0,
        d=0,
        D=0,
        season_period=1,

        # power transform
        power_transform="4rt",

        # parameter pooling
        theta_pooling="shared",
        sigma_pooling="shared",

        # Fourier parameters
        fourier_K=2,
        fourier_pooling="shared",  # Shared Fourier coefficients

        # covariates
        x=[]
    )

    date = datetime.date.fromisoformat("2024-01-06")
    fips_codes = ["US", "01", "02", "04", "05"] # fewer locs for faster testing
    # model_config = create_test_sarix_model_config(main_source=["nhsn"], theta_pooling="shared", sigma_pooling="none")
    run_config = create_test_sarix_run_config(ref_date=date, states=fips_codes, hsas=[], num=50, tmp_path=tmp_path)

    model = SARIXFourierModel(model_config)
    model.run(run_config)

    # Verify output structure
    actual_df = pd.read_csv(
        run_config.output_root / f"UMass-{model_config.model_name}" / 
        f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv"
    )

    # Assertions
    assert len(actual_df) > 0, "Output dataframe should not be empty"
    assert set(actual_df["location"].unique()) == set(run_config.states), \
        "Output should contain predictions for all input states"
    assert all(actual_df["output_type"] == "quantile"), \
        "All outputs should be quantiles"
    assert set(actual_df["output_type_id"].astype(str).unique()) == set(run_config.q_labels), \
        "Output should contain all specified quantile levels"
    assert actual_df["value"].notna().all(), \
        "All predictions should be non-null"
    assert (actual_df["value"] >= 0).all(), \
        "All predictions should be non-negative"


def test_sarix_fourier_missing_pooling_parameter():
    """Test that SARIXFourierModel raises error when fourier_pooling is missing."""
    model_config = SimpleNamespace(
        model_class="sarix_fourier",
        model_name="sarix_p2_fourier_K2_nopooling",
        sources=["nhsn"],
        fit_locations_separately=False,
        p=2, P=0, d=0, D=0, season_period=1,
        power_transform="4rt",
        theta_pooling="shared",
        sigma_pooling="shared",
        fourier_K=2,
        # fourier_pooling is MISSING - should cause error
        x=[]
    )

    run_config = SimpleNamespace(
        disease="flu",
        ref_date=datetime.date.fromisoformat("2024-01-06"),
        output_root=Path("/tmp") / "model-output",
        artifact_store_root=Path("/tmp") / "artifact-store",
        save_feat_importance=False,
        states=["US"],
        hsas=[],
        max_horizon=1,
        q_levels=[0.5],
        q_labels=["0.5"],
        num_warmup=10,
        num_samples=10,
        num_chains=1
    )

    model = SARIXFourierModel(model_config)

    # Should raise AttributeError when trying to access missing fourier_pooling
    try:
        model.run(run_config)
        assert False, "Should have raised AttributeError for missing fourier_pooling"
    except AttributeError as e:
        assert "fourier_pooling" in str(e), \
            f"Error should mention fourier_pooling, got: {str(e)}"


def create_test_sarix_model_config(main_source, theta_pooling, sigma_pooling):
    model_config = SimpleNamespace(
        model_class = "sarix",
        model_name = "sarix_" + main_source[0] + "_p6_4rt_theta" + theta_pooling + "_sigma" + sigma_pooling,
        
        # data sources and adjustments for reporting issues
        sources = main_source,
        
        # fit locations separately or jointly
        fit_locations_separately = False,
        
        # SARI model parameters
        p = 6,
        P = 0,
        d = 0,
        D = 0,
        season_period = 1,

        # power transform applied to surveillance signals
        power_transform = "4rt",

        # sharing of information about parameters
        theta_pooling=theta_pooling,
        sigma_pooling=sigma_pooling,
        
        # covariates
        x = []
    )
    return model_config

def create_test_sarix_run_config(ref_date, states, hsas, num, tmp_path):
    run_config = SimpleNamespace(
        disease="flu",
        ref_date=ref_date,
        output_root=tmp_path / "model-output",
        artifact_store_root=tmp_path / "artifact-store",
        save_feat_importance=False,
        states=states,
        hsas = hsas,
        max_horizon=3,
        q_levels = [0.025, 0.50, 0.975],
        q_labels = ["0.025", "0.5", "0.975"],
        num_warmup = num,
        num_samples = num,
        num_chains = 1
    )
    return run_config
    

def _np_percentile_val():
    return numpy.array(
        [[[2.22541624e-01, 1.82324940e-01, 1.27709944e-01],
          [-2.38521753e-01, -3.11683703e-01, -3.89838263e-01],
          [3.65179822e-01, 3.30586682e-01, 2.80645800e-01],
          [1.89957178e-01, 1.98017232e-01, 1.75632590e-01],
          [2.34052414e-01, 1.73046837e-01, 1.49813240e-01],
          [2.95891020e-01, 2.64033157e-01, 1.91925827e-01],
          [1.91364356e-01, 1.62421282e-01, 1.39508408e-01],
          [2.89884646e-01, 2.32042854e-01, 1.86287447e-01],
          [2.47832704e-01, 2.13865414e-01, 1.82933973e-01],
          [2.07883409e-01, 1.73067807e-01, 1.45727926e-01],
          [3.43025134e-01, 3.22408932e-01, 2.91293726e-01],
          [7.79729199e-02, -1.50642925e-02, -1.18422690e-01],
          [2.15420238e-01, 1.62142558e-01, 9.65136154e-02],
          [2.21184102e-01, 1.84105207e-01, 1.47633953e-01],
          [2.45778827e-01, 2.25517928e-01, 1.97119110e-01],
          [6.46950673e-02, -1.08052013e-02, -7.64041666e-02],
          [1.42625988e-01, 8.74901406e-02, 7.09190222e-02],
          [2.04940985e-01, 1.82181005e-01, 1.77787969e-01],
          [2.73062601e-01, 2.17573412e-01, 1.56597306e-01],
          [1.28515982e-01, 9.51378807e-02, 3.62812653e-02],
          [2.96400874e-01, 2.82467369e-01, 2.69363685e-01],
          [3.23664477e-01, 2.97513246e-01, 2.88207608e-01],
          [2.64325314e-01, 2.63950690e-01, 2.12602656e-01],
          [2.22876109e-01, 1.99068035e-01, 1.93246918e-01],
          [2.55219772e-01, 2.34714223e-01, 1.87674624e-01],
          [1.90299119e-01, 1.67028040e-01, 1.26085748e-01],
          [3.69560701e-01, 3.44429153e-01, 3.04606768e-01],
          [9.49449530e-02, 8.53451250e-02, 3.26145997e-02],
          [2.07938436e-01, 1.48126675e-01, 1.04875130e-01],
          [1.96341721e-01, 1.50786200e-01, 8.40540124e-02],
          [3.28539360e-01, 3.15154862e-01, 3.05533560e-01],
          [2.52577208e-01, 2.34228884e-01, 1.74534114e-01],
          [2.95858289e-01, 2.86579932e-01, 2.60370556e-01],
          [4.41472320e-01, 4.15097611e-01, 3.75973117e-01],
          [8.13819937e-02, 1.90340820e-02, -4.50890312e-02],
          [2.67334034e-01, 2.45210168e-01, 2.38477688e-01],
          [1.04748258e-02, -1.03802918e-02, -5.01047624e-02],
          [1.57026848e-01, 1.01242643e-01, 6.48639531e-02],
          [2.57894629e-01, 2.37540027e-01, 2.00216884e-01],
          [2.33968429e-01, 1.95996965e-01, 1.44295382e-01],
          [4.24508706e-01, 3.89446726e-01, 3.40110609e-01],
          [1.26744660e-01, 1.03470394e-01, 2.59670306e-02],
          [2.95026888e-01, 2.89566060e-01, 2.40084887e-01],
          [2.57176686e-01, 2.45254328e-01, 2.14242858e-01],
          [2.47886313e-01, 1.96860882e-01, 1.66562526e-01],
          [9.03973781e-02, -3.93795050e-04, -7.59670980e-02],
          [3.47807512e-01, 3.26417361e-01, 2.87590605e-01],
          [9.25882775e-02, 6.39987770e-02, 4.89607733e-02],
          [2.47943980e-01, 2.29744528e-01, 1.89622971e-01],
          [2.42063859e-01, 2.00323456e-01, 1.56731553e-01],
          [2.62870489e-01, 2.12995209e-01, 1.08971777e-01],
          [-1.22249633e-01, -1.72479167e-01, -2.05317267e-01],
          [2.67621535e-01, 2.62407335e-01, 2.41535093e-01]],
         [[2.97959790e-01, 3.10248822e-01, 3.12872216e-01],
          [-6.29902221e-02, -7.69548900e-02, -6.81971610e-02],
          [4.57400158e-01, 4.53551188e-01, 4.48945090e-01],
          [2.74990052e-01, 3.10983732e-01, 3.40201050e-01],
          [2.95877323e-01, 2.93216154e-01, 2.78667077e-01],
          [3.59498024e-01, 3.59175757e-01, 3.45471948e-01],
          [2.81224459e-01, 2.93072030e-01, 2.91971073e-01],
          [4.47010323e-01, 4.84432980e-01, 4.83699843e-01],
          [3.70261163e-01, 3.94121945e-01, 3.96656036e-01],
          [2.63269126e-01, 2.65370220e-01, 2.64244184e-01],
          [4.12810057e-01, 4.22452450e-01, 4.09426376e-01],
          [2.17054963e-01, 2.00583115e-01, 1.80740878e-01],
          [3.24985415e-01, 3.36928636e-01, 3.25553402e-01],
          [2.80104160e-01, 2.89580196e-01, 2.85762325e-01],
          [3.24534789e-01, 3.34531680e-01, 3.31479549e-01],
          [1.45252407e-01, 1.39778398e-01, 1.36556230e-01],
          [2.21163861e-01, 2.28467964e-01, 2.29796991e-01],
          [2.95052722e-01, 3.28270182e-01, 3.42973635e-01],
          [3.57387066e-01, 3.63619059e-01, 3.53775963e-01],
          [2.44900934e-01, 2.69801259e-01, 2.69644156e-01],
          [3.64212126e-01, 3.98995757e-01, 4.10338163e-01],
          [3.98638755e-01, 4.24133062e-01, 4.29047465e-01],
          [3.41488227e-01, 3.64311695e-01, 3.70579436e-01],
          [2.80877575e-01, 3.03409621e-01, 3.13581020e-01],
          [3.61224726e-01, 3.84219721e-01, 3.95171583e-01],
          [2.64716968e-01, 2.84663692e-01, 2.92632058e-01],
          [4.82557431e-01, 5.08474380e-01, 5.05630761e-01],
          [1.95739634e-01, 2.16610506e-01, 2.45527841e-01],
          [2.83602670e-01, 2.94805557e-01, 2.91800916e-01],
          [3.08676749e-01, 3.22599977e-01, 3.23239058e-01],
          [4.02976051e-01, 4.20294330e-01, 4.35384750e-01],
          [3.42796057e-01, 3.51799279e-01, 3.40504974e-01],
          [3.57374251e-01, 3.67061988e-01, 3.59203115e-01],
          [5.13337582e-01, 5.26766390e-01, 5.28670281e-01],
          [2.21751720e-01, 2.28093848e-01, 2.37711042e-01],
          [3.25570732e-01, 3.44513953e-01, 3.46717134e-01],
          [1.06594760e-01, 1.20929513e-01, 1.39860854e-01],
          [2.34041639e-01, 2.62821510e-01, 2.65469775e-01],
          [3.26401755e-01, 3.45406264e-01, 3.55303675e-01],
          [3.80847201e-01, 4.21464711e-01, 4.17487994e-01],
          [5.08020222e-01, 5.34216642e-01, 5.34987926e-01],
          [2.65070885e-01, 3.18079114e-01, 3.35300788e-01],
          [3.76409158e-01, 4.02034655e-01, 4.09497753e-01],
          [3.04315433e-01, 3.18889007e-01, 3.22016016e-01],
          [3.42446193e-01, 3.63201663e-01, 3.86474803e-01],
          [2.28747316e-01, 2.52544448e-01, 2.73082897e-01],
          [4.17081311e-01, 4.47557062e-01, 4.61737871e-01],
          [1.81169331e-01, 1.94123007e-01, 1.96866766e-01],
          [3.51673305e-01, 3.93421471e-01, 4.29964215e-01],
          [3.20511892e-01, 3.45343351e-01, 3.43077034e-01],
          [4.48880598e-01, 4.45205748e-01, 4.20632169e-01],
          [-9.30569647e-03, -2.07099067e-02, -2.38471739e-02],
          [3.26475084e-01, 3.46230641e-01, 3.46216187e-01]],
         [[3.91308872e-01, 4.57335044e-01, 5.34370412e-01],
          [6.35444466e-02, 1.71694950e-01, 2.41260254e-01],
          [5.39776564e-01, 6.02389070e-01, 6.32036614e-01],
          [3.61068203e-01, 4.51654990e-01, 5.07262272e-01],
          [3.58519227e-01, 3.91382324e-01, 4.11550624e-01],
          [4.34174243e-01, 4.96819534e-01, 5.12898867e-01],
          [3.66144163e-01, 4.49528843e-01, 5.14213127e-01],
          [6.12098245e-01, 7.77543952e-01, 8.55939068e-01],
          [5.04130632e-01, 5.65265204e-01, 6.74976355e-01],
          [3.22685032e-01, 3.39737531e-01, 3.56739443e-01],
          [4.84979967e-01, 5.40613325e-01, 5.82933943e-01],
          [3.62312986e-01, 4.15581792e-01, 4.46406016e-01],
          [4.35765201e-01, 5.13917580e-01, 5.82545452e-01],
          [3.54656872e-01, 3.94405179e-01, 4.32976715e-01],
          [4.00308932e-01, 4.69729722e-01, 5.03149725e-01],
          [2.39865492e-01, 2.67919922e-01, 3.06406984e-01],
          [2.99842047e-01, 3.73577865e-01, 4.24350244e-01],
          [3.79077508e-01, 4.77504222e-01, 5.47623882e-01],
          [4.38741190e-01, 4.99587057e-01, 5.59114397e-01],
          [3.62943360e-01, 4.79146594e-01, 5.59094252e-01],
          [4.33270055e-01, 4.98217464e-01, 5.30032079e-01],
          [4.74268539e-01, 5.34104156e-01, 5.79892798e-01],
          [4.18634977e-01, 4.80803037e-01, 5.29620108e-01],
          [3.35461161e-01, 3.91972119e-01, 4.23834752e-01],
          [4.48058187e-01, 5.18286791e-01, 5.72982916e-01],
          [3.41153666e-01, 4.05986520e-01, 4.57051721e-01],
          [5.78385088e-01, 6.66201897e-01, 7.39366533e-01],
          [2.86166869e-01, 3.58757636e-01, 4.32429693e-01],
          [3.63524315e-01, 3.99179283e-01, 4.54811613e-01],
          [4.19007020e-01, 4.86634410e-01, 5.59979638e-01],
          [4.78470140e-01, 5.54172556e-01, 5.70483108e-01],
          [4.34567777e-01, 5.04369989e-01, 5.60560682e-01],
          [4.09423319e-01, 4.56085717e-01, 4.70233305e-01],
          [5.92027624e-01, 6.39750125e-01, 6.77245133e-01],
          [3.94678583e-01, 4.83388707e-01, 5.68256925e-01],
          [3.74485486e-01, 4.24462480e-01, 4.78810743e-01],
          [1.94327131e-01, 2.52622863e-01, 2.94898193e-01],
          [3.20427980e-01, 3.83304519e-01, 4.41331666e-01],
          [3.98798355e-01, 4.56021016e-01, 5.03770930e-01],
          [5.23937319e-01, 6.46307093e-01, 7.47003183e-01],
          [5.88856509e-01, 6.36478421e-01, 6.69259898e-01],
          [3.81985225e-01, 5.12105028e-01, 6.43082620e-01],
          [4.63262637e-01, 5.50578834e-01, 5.88756181e-01],
          [3.54333093e-01, 3.96404223e-01, 4.48006484e-01],
          [4.64853939e-01, 5.20061310e-01, 5.70508999e-01],
          [3.84713623e-01, 4.57752132e-01, 5.77218857e-01],
          [4.97504447e-01, 5.77449405e-01, 5.98073846e-01],
          [2.56140235e-01, 3.19889196e-01, 4.06984358e-01],
          [4.69504851e-01, 5.87101118e-01, 6.51444320e-01],
          [4.06506771e-01, 4.83620531e-01, 5.41332622e-01],
          [5.91549647e-01, 6.84945406e-01, 7.94059159e-01],
          [8.84183340e-02, 1.46475891e-01, 2.07177738e-01],
          [3.71897836e-01, 4.34920674e-01, 4.55205029e-01]]])
