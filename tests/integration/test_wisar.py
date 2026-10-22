import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

from idmodels.config import PowerTransform, SourceType, WISARModelConfig
from idmodels.wisar import WISARModel


def _synthetic_nhsn(ref_date: datetime.date, locations: list[str]) -> pd.DataFrame:
    """Three seasons of seasonal weekly rates per location, ending the week before ref_date."""
    dates = pd.date_range(end=pd.Timestamp(ref_date) - pd.Timedelta(weeks=1), periods=156, freq="W-SAT")
    rng = np.random.default_rng(42)
    frames = []
    for i, loc in enumerate(locations):
        seasonal = 1 + 0.9 * np.cos(2 * np.pi * (dates.dayofyear.values - 20) / 365.25)
        frames.append(pd.DataFrame({
            "source": "nhsn",
            "agg_level": "national" if loc == "US" else "state",
            "location": loc,
            "season": "x",
            "season_week": 1,
            "wk_end_date": dates,
            "inc": (1 + i) * seasonal * rng.lognormal(0, 0.1, len(dates)),
            "pop": 1e6 * (1 + i),
        }))
    df = pd.concat(frames, ignore_index=True)
    df.loc[(df["location"] == locations[0]) & (df["wk_end_date"] == dates[100]), "inc"] = np.nan  # interior gap
    return df


def test_wisar_run_produces_hub_formatted_quantiles(make_run_config):
    ref_date = datetime.date.fromisoformat("2025-01-11")
    states = ["US", "01", "25"]
    run_config = make_run_config(ref_date=ref_date, states=states, hsas=[])
    model_config = WISARModelConfig(model_name="wisar_test", main_source=SourceType.NHSN,
                                    fit_locations_separately=False, power_transform=PowerTransform.FOURTH_ROOT,
                                    train_start=datetime.date(2022, 1, 1), num_warmup=100, num_samples=100,
                                    num_chains=1, n_param_draws=50, n_innov_per_param=2)

    with patch("idmodels.model.DiseaseDataLoader") as loader:
        loader.return_value.load.return_value = _synthetic_nhsn(ref_date, states)
        WISARModel(model_config).run(run_config)

    out = pd.read_csv(run_config.output_root / "UMass-wisar_test" / "2025-01-11-UMass-wisar_test.csv",
                      dtype={"location": str, "output_type_id": str})
    assert list(out.columns) == ["location", "reference_date", "horizon", "target_end_date", "target", "output_type",
                                 "output_type_id", "value"]
    assert len(out) == len(states) * run_config.max_horizon * len(run_config.q_levels)
    assert set(out["location"]) == set(states)
    assert sorted(out["horizon"].unique()) == [0, 1, 2]  # last observation is the week before ref_date
    assert (out["target"] == "wk inc flu hosp").all()
    assert (out["reference_date"] == "2025-01-11").all()
    assert out["value"].notna().all() and (out["value"] >= 0).all()

    # quantiles are non-decreasing within each location/horizon
    wide = out.pivot_table(index=["location", "horizon"], columns="output_type_id", values="value")
    assert (np.diff(wide[run_config.q_labels].values, axis=1) >= 0).all()


def test_wisar_run_is_reproducible_given_seeds(make_run_config, tmp_path):
    ref_date = datetime.date.fromisoformat("2025-01-11")
    states = ["US", "01", "25"]
    data = _synthetic_nhsn(ref_date, states)

    def run(name: str, seed_fit: int) -> pd.DataFrame:
        run_config = make_run_config(ref_date=ref_date, states=states, hsas=[])
        run_config.output_root = tmp_path / name
        model_config = WISARModelConfig(model_name="wisar_test", main_source=SourceType.NHSN,
                                        fit_locations_separately=False, power_transform=PowerTransform.FOURTH_ROOT,
                                        train_start=datetime.date(2022, 1, 1), num_warmup=100, num_samples=100,
                                        num_chains=2, n_param_draws=50, n_innov_per_param=2, seed_fit=seed_fit)
        with patch("idmodels.model.DiseaseDataLoader") as loader:
            loader.return_value.load.return_value = data.copy()
            WISARModel(model_config).run(run_config)
        return pd.read_csv(run_config.output_root / "UMass-wisar_test" / "2025-01-11-UMass-wisar_test.csv")

    first, second = run("a", seed_fit=9001), run("b", seed_fit=9001)
    pd.testing.assert_frame_equal(first, second)  # bit-identical, not just close

    other_seed = run("c", seed_fit=1)
    assert not np.array_equal(first["value"].values, other_seed["value"].values)
