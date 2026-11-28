import pandas as pd

from idmodels.utils import build_sample_output_type_ids


def test_build_sample_output_type_ids_is_unique_id_and_one_based_index():
    unique_id = pd.Series(["nationalUS", "state01", "hsa1", "hsa25"])
    sample_index = pd.Series([0, 1, 9, 99])
    actual = build_sample_output_type_ids(unique_id, sample_index)
    assert actual.tolist() == ["nationalUS-1", "state01-2", "hsa1-10", "hsa25-100"]
