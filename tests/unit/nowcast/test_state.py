import datetime

import numpy as np
import pandas as pd
import pytest

from idmodels.config import ScaleByStateConfig
from idmodels.nowcast import ScaleByStateNowcaster, build_nowcaster
from idmodels.nowcast.state import season_start

AS_OF = datetime.date(2026, 1, 17)  # a Saturday; the data come from the Wednesday 2026-01-14 release
NEWEST = pd.Timestamp("2026-01-10")
WEEKS = pd.date_range("2025-09-06", NEWEST, freq="7D")
SETTLED_WEEKS = WEEKS[(WEEKS >= "2025-10-01") & (WEEKS <= NEWEST - pd.Timedelta(weeks=4))]


def _archives(tmp_path):
    """
    Each week is first published in the Wednesday release after it ends and revised a week later: state 01 from 50
    to 100 admissions (scaling by completeness under-corrects it, because all its hospitals had reported), state 02
    stays at 100. 100 hospitals report every week in both states.
    """
    admissions, reporting = [], []
    for w in WEEKS:
        first_release, next_release = w + pd.Timedelta(days=4), w + pd.Timedelta(days=11)
        for loc, first, revised in [("01", 50, 100), ("02", 100, 100)]:
            admissions.append({"location": loc, "wk_end_date": w, "report_date": first_release, "admissions": first})
            admissions.append({"location": loc, "wk_end_date": w, "report_date": next_release, "admissions": revised})
            reporting.append({"location": loc, "wk_end_date": w, "report_date": first_release, "hosp_reporting": 100})
    paths = {}
    for name, rows in [("admissions", admissions), ("reporting", reporting)]:
        # only values published by the release the data come from
        archive = pd.DataFrame(rows).query("report_date <= '2026-01-14'")
        paths[name] = tmp_path / f"{name}.parquet"
        archive.to_parquet(paths[name])
    return paths


def _nowcaster(tmp_path, **kwargs) -> ScaleByStateNowcaster:
    paths = _archives(tmp_path)
    return ScaleByStateNowcaster(
        ScaleByStateConfig(reporting_path=paths["reporting"], admissions_path=paths["admissions"], **kwargs)
    )


def test_season_start():
    assert season_start(datetime.date(2026, 1, 17), 10) == pd.Timestamp("2025-10-01")
    assert season_start(datetime.date(2025, 10, 4), 10) == pd.Timestamp("2025-10-01")
    assert season_start(datetime.date(2025, 9, 27), 10) == pd.Timestamp("2024-10-01")


def test_training_weeks_use_values_from_each_weeks_own_release(tmp_path):
    weeks = _nowcaster(tmp_path).training_weeks(AS_OF)
    state01 = weeks.loc[weeks["location"] == "01"]
    # this season's weeks at least settle_weeks older than the newest week
    assert list(state01["wk_end_date"]) == list(SETTLED_WEEKS)
    # each week was the newest in the Wednesday release after it ended
    assert (state01["release"] == state01["wk_end_date"] + pd.Timedelta(days=4)).all()
    assert (state01["first"] == 50).all() and (state01["settled"] == 100).all()
    assert (state01["completeness"] == 1).all()


def test_state_effects_are_shrunk_mean_log_errors(tmp_path):
    effects = _nowcaster(tmp_path, shrink=10).state_effects(AS_OF)
    n = len(SETTLED_WEEKS)
    assert effects["01"] == pytest.approx(n * np.log(101 / 51) / (n + 10))
    assert effects["02"] == pytest.approx(0.0)


def test_adjusts_newest_week_on_the_count_scale(tmp_path):
    nowcaster = _nowcaster(tmp_path)
    effect = nowcaster.state_effects(AS_OF)["01"]
    df = pd.DataFrame(
        {
            "source": "nhsn",
            "agg_level": "state",
            "location": ["01"] * 5 + ["02"] * 5,
            "wk_end_date": list(WEEKS[-5:]) * 2,
            "inc": 5.0,  # rate per 100,000: 50 admissions
            "pop": 1e6,
        }
    )
    out = nowcaster.apply(df, AS_OF)
    newest01 = (out["location"] == "01") & (out["wk_end_date"] == NEWEST)
    assert out.loc[newest01, "inc"].item() == pytest.approx(np.expm1(np.log1p(50) + effect) / 10)
    # no correction for state 02, and earlier weeks are unchanged
    assert (out.loc[~newest01, "inc"] == 5.0).all()


def test_no_correction_without_settled_weeks_this_season(tmp_path):
    # early in the season no week is old enough to have settled
    nowcaster = _nowcaster(tmp_path, settle_weeks=20)
    assert nowcaster.state_effects(AS_OF).empty


def test_factory():
    assert isinstance(build_nowcaster(ScaleByStateConfig()), ScaleByStateNowcaster)
