import pytest
from iddata.enums import Disease

from idmodels.config import RunConfig, SampleIdScheme


@pytest.fixture
def make_run_config(tmp_path):
    def _make(ref_date, states, hsas, num_sample_forecasts=0, sample_id_scheme=SampleIdScheme.STATE_ABBREVIATION):
        return RunConfig(
            disease=Disease.FLU,
            ref_date=ref_date,
            output_root=tmp_path / "model-output",
            artifact_store_root=tmp_path / "artifact-store",
            states=states,
            hsas=hsas,
            max_horizon=3,
            q_levels=[0.025, 0.50, 0.975],
            q_labels=["0.025", "0.5", "0.975"],
            num_sample_forecasts=num_sample_forecasts,
            sample_id_scheme=sample_id_scheme,
        )
    return _make
