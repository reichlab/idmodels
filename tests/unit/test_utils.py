from importlib import resources

import pandas as pd
import pytest

from idmodels.config import SampleIdScheme
from idmodels.data import load_state_abbreviations
from idmodels.utils import build_sample_output_type_ids


def test_state_abbreviations_cover_same_locations_as_state_centroids():
    """Catch a location added to one vendored file but not the other."""
    with resources.files("idmodels.data").joinpath("state_centroids.csv").open("r") as f:
        centroid_fips = set(pd.read_csv(f, dtype={"fips": str})["fips"])
    assert set(load_state_abbreviations()) == centroid_fips


def test_state_abbreviations_spot_check():
    abbreviations = load_state_abbreviations()
    assert abbreviations["01"] == "AL"
    assert abbreviations["11"] == "DC"
    assert abbreviations["72"] == "PR"
    assert abbreviations["US"] == "US"


def test_build_sample_output_type_ids_state_abbreviation():
    location = pd.Series(["US", "01", "01", "56"])
    sample_index = pd.Series([0, 1, 99, 5])
    actual = build_sample_output_type_ids(location, sample_index, SampleIdScheme.STATE_ABBREVIATION)
    assert actual.tolist() == ["US00", "AL01", "AL99", "WY05"]


def test_build_sample_output_type_ids_state_abbreviation_raises_on_unmapped_location():
    location = pd.Series(["US", "123"])
    sample_index = pd.Series([0, 0])
    with pytest.raises(KeyError, match="123"):
        build_sample_output_type_ids(location, sample_index, SampleIdScheme.STATE_ABBREVIATION)


def test_build_sample_output_type_ids_draw_number_is_one_based_and_unpadded():
    location = pd.Series(["US", "01", "1", "25"])  # states and HSAs alike
    sample_index = pd.Series([0, 1, 9, 99])
    actual = build_sample_output_type_ids(location, sample_index, SampleIdScheme.DRAW_NUMBER)
    assert actual.tolist() == ["1", "2", "10", "100"]
