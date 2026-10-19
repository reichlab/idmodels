import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from idmodels.config import NowcastConfig, ScaleByCompletenessConfig, SourceType
from idmodels.model import IDModel
from idmodels.nowcast import Nowcaster, ScaleByCompletenessNowcaster, build_nowcaster

AS_OF = datetime.date(2026, 1, 14)  # a Wednesday release
NEWEST = pd.Timestamp("2026-01-10")  # the newest week in that release
WEEKS = pd.date_range(end=NEWEST, periods=8, freq="7D")


def _data() -> pd.DataFrame:
    """Long-format iddata-like output: NHSN for two states and the US, plus ILINet for one state."""
    frames = []
    for source, location, agg in [
        ("nhsn", "01", "state"),
        ("nhsn", "02", "state"),
        ("nhsn", "US", "national"),
        ("ilinet", "01", "state"),
    ]:
        frames.append(
            pd.DataFrame(
                {
                    "source": source,
                    "agg_level": agg,
                    "location": location,
                    "wk_end_date": WEEKS,
                    "inc": np.arange(1.0, 9.0),
                    "pop": 1e6,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _reporting(tmp_path, rows) -> str:
    """
    Write an as-of reporting archive: 100 hospitals reporting every week in every location, published 4 days after
    the week ended, except for the (location, week) pairs in `rows`, which have the values given there instead.
    """
    base = [
        {"location": loc, "wk_end_date": w, "report_date": w + pd.Timedelta(days=4), "hosp_reporting": 100}
        for loc in ["01", "02", "US"]
        for w in WEEKS
    ]
    given = {(r["location"], pd.Timestamp(r["wk_end_date"])) for r in rows}
    archive = pd.DataFrame([r for r in base if (r["location"], r["wk_end_date"]) not in given] + rows)
    archive["report_date"] = pd.to_datetime(archive["report_date"]).dt.date
    archive["wk_end_date"] = pd.to_datetime(archive["wk_end_date"]).dt.date
    path = tmp_path / "reporting.csv"
    archive.to_csv(path, index=False)
    return path


def _nowcaster(path, **kwargs) -> ScaleByCompletenessNowcaster:
    return ScaleByCompletenessNowcaster(ScaleByCompletenessConfig(reporting_path=path, **kwargs))


def _inc(df, source, location, week=NEWEST):
    return df.loc[(df["source"] == source) & (df["location"] == location) & (df["wk_end_date"] == week), "inc"].item()


def test_scales_newest_week_by_completeness(tmp_path):
    # state 01: only 50 of the usual 100 hospitals had reported the newest week by the release
    path = _reporting(
        tmp_path, [{"location": "01", "wk_end_date": NEWEST, "report_date": "2026-01-13", "hosp_reporting": 50}]
    )
    df = _data()
    out = _nowcaster(path).apply(df, AS_OF)
    assert _inc(out, "nhsn", "01") == pytest.approx(2 * 8.0)
    # complete locations, earlier weeks and other sources are unchanged
    assert _inc(out, "nhsn", "02") == 8.0 and _inc(out, "nhsn", "US") == 8.0
    assert _inc(out, "ilinet", "01") == 8.0
    pd.testing.assert_frame_equal(out.loc[out["wk_end_date"] < NEWEST], df.loc[df["wk_end_date"] < NEWEST])


def test_uses_only_values_published_by_the_reference_date(tmp_path):
    # the newest week was first published with 50 hospitals, then revised to 100 after the release
    path = _reporting(
        tmp_path,
        [
            {"location": "01", "wk_end_date": NEWEST, "report_date": "2026-01-13", "hosp_reporting": 50},
            {"location": "01", "wk_end_date": NEWEST, "report_date": "2026-01-20", "hosp_reporting": 100},
        ],
    )
    out = _nowcaster(path).apply(_data(), AS_OF)
    assert _inc(out, "nhsn", "01") == pytest.approx(16.0)
    # as of a later date, the revised (complete) value is used
    later = _nowcaster(path).apply(_data(), datetime.date(2026, 1, 21))
    assert _inc(later, "nhsn", "01") == pytest.approx(8.0)


def test_num_weeks_and_completeness_bounds(tmp_path):
    path = _reporting(
        tmp_path,
        [
            {"location": "01", "wk_end_date": NEWEST, "report_date": "2026-01-13", "hosp_reporting": 1},
            {
                "location": "01",
                "wk_end_date": NEWEST - pd.Timedelta(weeks=1),
                "report_date": "2026-01-13",
                "hosp_reporting": 80,
            },
            # more hospitals than usual: never scaled down
            {"location": "02", "wk_end_date": NEWEST, "report_date": "2026-01-13", "hosp_reporting": 120},
        ],
    )
    out = _nowcaster(path, num_weeks=2, min_completeness=0.1).apply(_data(), AS_OF)
    assert _inc(out, "nhsn", "01") == pytest.approx(8.0 / 0.1)  # completeness 0.01, floored at 0.1
    assert _inc(out, "nhsn", "01", NEWEST - pd.Timedelta(weeks=1)) == pytest.approx(7.0 / 0.8)
    assert _inc(out, "nhsn", "02") == 8.0


def test_locations_without_reporting_data_are_unchanged(tmp_path):
    path = _reporting(tmp_path, [])
    archive = pd.read_csv(path, dtype={"location": str})
    archive.loc[archive["location"] != "02"].to_csv(path, index=False)
    df = _data()
    pd.testing.assert_frame_equal(_nowcaster(path).apply(df, AS_OF), df)


def test_contract_is_enforced():
    class _ChangesOtherSource(Nowcaster):
        def _adjust(self, df, as_of):
            df.loc[df["source"] == "ilinet", "inc"] += 1
            return df

    class _DropsRows(Nowcaster):
        def _adjust(self, df, as_of):
            return df.iloc[1:]

    cfg = Mock(source=SourceType.NHSN, num_weeks=1)
    with pytest.raises(ValueError, match="changed values other than"):
        _ChangesOtherSource(cfg).apply(_data(), AS_OF)
    with pytest.raises(ValueError, match="changed the rows or columns"):
        _DropsRows(cfg).apply(_data(), AS_OF)


def test_config_validation_and_factory(tmp_path):
    with pytest.raises(TypeError, match="abstract"):
        NowcastConfig()
    with pytest.raises(ValueError, match="reporting_path"):
        ScaleByCompletenessConfig()
    assert isinstance(
        build_nowcaster(ScaleByCompletenessConfig(reporting_path=tmp_path / "x.csv")), ScaleByCompletenessNowcaster
    )


class _FakeModel(IDModel):
    def _build_sources(self, run_config):
        return []

    def _build_feature_pipeline(self, run_config):
        raise NotImplementedError

    def _fit_and_predict(self, df, feat_names, run_config):
        raise NotImplementedError


@pytest.mark.parametrize("with_nowcast", [False, True])
def test_idmodel_applies_nowcast_to_loaded_data(tmp_path, with_nowcast):
    path = _reporting(
        tmp_path, [{"location": "01", "wk_end_date": NEWEST, "report_date": "2026-01-13", "hosp_reporting": 50}]
    )
    nowcast = ScaleByCompletenessConfig(reporting_path=path) if with_nowcast else None
    model = _FakeModel(model_config=Mock(nowcast=nowcast))
    run_config = Mock(ref_date=AS_OF, states=["01", "02", "US"], hsas=[])
    with patch("idmodels.model.DiseaseDataLoader") as loader:
        loader.return_value.load.return_value = _data()
        df = model._load_data(run_config)
    # without a nowcast the loaded data are used as they are
    assert _inc(df, "nhsn", "01") == (16.0 if with_nowcast else 8.0)
    assert _inc(df, "nhsn", "02") == 8.0
