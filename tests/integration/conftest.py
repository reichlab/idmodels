import pytest
from iddata.enums import Disease

from idmodels.config import RunConfig


@pytest.fixture
def make_run_config(tmp_path):
    def _make(ref_date, states, hsas):
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
        )
    return _make
