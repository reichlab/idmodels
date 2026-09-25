import datetime

import numpy as np
import pandas as pd
import pytest
from iddata.enums import Disease

from idmodels.config import (
    PeakBaselineModelConfig,
    PeakGBQRModelConfig,
    PeakKCDEModelConfig,
    RunConfig,
    SourceType,
)
from idmodels.peak import PeakBaselineModel, PeakGBQRModel, PeakKCDEModel
from idmodels.peak.base import PeakInputs, season_of, season_week_to_date
from idmodels.peak.revision import RevisionModel, apply_revisions
from idmodels.peak.series import (
    build_replay_rows,
    build_season_arrays,
    running_max,
    season_peaks,
    state_features,
)

LOCS = ["US", "01", "02"]
Q_LEVELS = [0.025, 0.25, 0.5, 0.75, 0.975]


def _curve(peak_week, height, n=53, width=4.0, base=0.2):
    sw = np.arange(1, n + 1)
    return base + height * np.exp(-0.5 * ((sw - peak_week) / width) ** 2)


def _long(source, location, season, values, agg_level):
    sw = np.arange(1, len(values) + 1)
    start = season_week_to_date(season, 1)
    return pd.DataFrame(
        {
            "source": source,
            "agg_level": agg_level,
            "location": location,
            "season": season,
            "season_week": sw.astype(float),
            "inc": values,
            "wk_end_date": [pd.Timestamp(start) + pd.Timedelta(weeks=int(w) - 1) for w in sw],
        }
    )


def _synthetic_inputs(ref_date: datetime.date) -> PeakInputs:
    rng = np.random.default_rng(1)
    frames = []
    seasons = [f"{y}/{str(y + 1)[2:]}" for y in range(2010, 2026)]
    for i, season in enumerate(seasons):
        for j, loc in enumerate(LOCS):
            agg = "national" if loc == "US" else "state"
            pk, h = 20 + (i * 3 + j) % 12, 5 + (i % 5)
            frames.append(_long("ilinet", loc, season, _curve(pk, h / 5) * rng.lognormal(0, 0.05, 53), agg))
            if season >= "2022/23":
                frames.append(_long("nhsn", loc, season, _curve(pk, h) * rng.lognormal(0, 0.05, 53), agg))
    data = pd.concat(frames, ignore_index=True)
    pops = {"US": 3.3e8, "01": 5e6, "02": 7e5}
    data["pop"] = data["location"].map(pops)

    # current season (2026/27) NHSN counts, observed through the week before ref_date, released weekly
    season = season_of(ref_date)
    cur = _curve(24, 8.0)
    vint = []
    for weeks_back in range(12, -1, -1):
        as_of = ref_date - datetime.timedelta(days=3 + 7 * weeks_back)
        last_sw = (as_of - season_week_to_date(season, 1)).days // 7
        if last_sw < 1:
            continue
        for loc in LOCS:
            counts = cur[:last_sw] * pops[loc] / 1e5
            counts = counts * np.concatenate([np.ones(max(last_sw - 1, 0)), [0.9]])  # last week under-reported
            df = _long("nhsn", loc, season, counts, "state")[
                ["location", "season", "season_week", "wk_end_date", "inc"]
            ]
            df["as_of"] = pd.Timestamp(as_of)
            vint.append(df)
    vintages = pd.concat(vint, ignore_index=True)
    cur_rows = _long("nhsn", "US", season, cur[:10], "national").assign(pop=pops["US"])
    data = pd.concat([data, cur_rows], ignore_index=True)
    return PeakInputs(data=data, vintages=vintages)


def _run_config(ref_date, tmp_path):
    return RunConfig(
        disease=Disease.FLU,
        ref_date=ref_date,
        output_root=tmp_path,
        artifact_store_root=None,
        max_horizon=4,
        states=LOCS,
        hsas=[],
        q_levels=Q_LEVELS,
        q_labels=[str(q) for q in Q_LEVELS],
    )


class TestSeriesHelpers:
    def test_season_week_to_date(self):
        assert season_week_to_date("2026/27", 10) == datetime.date(2026, 10, 10)
        assert season_week_to_date("2026/27", 43) == datetime.date(2027, 5, 29)
        assert season_of(datetime.date(2026, 10, 10)) == "2026/27"

    def test_peaks_and_running_max(self):
        y = np.full((1, 53), np.nan)
        y[0, :30] = np.arange(30)
        y[0, 30:] = 0
        peak, week = season_peaks(y, 10, 43)
        assert peak[0] == 29 and week[0] == 30
        m, m_week = running_max(y, 20, 10)
        assert m[0] == 19 and m_week[0] == 20
        # before the window opens the running max is the latest value
        m, m_week = running_max(y, 5, 10)
        assert m[0] == 4 and m_week[0] == 5

    def test_gap_filling_only_short_interior_gaps(self):
        vals = np.full(53, np.nan)
        vals[:7] = [1.0, np.nan, np.nan, np.nan, 5.0, np.nan, 7.0]
        arrays = build_season_arrays(_long("nhsn", "US", "2022/23", vals, "national"))
        np.testing.assert_array_equal(arrays.y[0, :7], [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0])
        assert np.isnan(arrays.y[0, 7:]).all()

    def test_features_do_not_look_ahead(self):
        y = _curve(25, 10.0)[None, :]
        y2 = y.copy()
        y2[0, 20:] = 100.0  # change the future
        f1 = state_features(y, 20, np.array([0.01]), 10)
        f2 = state_features(y2, 20, np.array([0.01]), 10)
        pd.testing.assert_frame_equal(f1, f2)

    def test_replay_targets(self):
        df = _long("nhsn", "US", "2022/23", _curve(25, 10.0), "national")
        rows = build_replay_rows(build_season_arrays(df), 10, 43, 5, 25)
        assert rows["season_week"].min() == 5 and rows["season_week"].max() == 43
        assert (rows["peak_week"] == 25).all()
        before = rows.loc[rows["season_week"] >= 10].query("season_week < 25")
        after = rows.query("season_week >= 25")
        assert (before["z"] > 0).all() and (before["k"] > 0).all()
        assert np.allclose(after["z"], 0.0) and (after["k"] <= 0).all()


class TestRevision:
    def test_fallback_and_apply(self):
        rng = np.random.default_rng(0)
        model = RevisionModel(max_lag=3)
        rho = model.sample(np.array([10.0, 1000.0]), 50, rng)
        assert rho.shape == (2, 50, 3)
        counts = np.array([[1.0, 2.0, 3.0, 4.0, np.nan], [10.0, 20.0, 30.0, np.nan, np.nan]])
        revised = apply_revisions(counts, np.array([3, 2]), rho)
        assert revised.shape == (2, 50, 5)
        # weeks older than max_lag are untouched
        assert np.all(revised[0, :, 0] == 1.0)
        assert np.all(np.isnan(revised[1, :, 3]))

    def test_fit_recovers_constant_revision(self):
        rows = []
        for a in range(20):
            as_of = pd.Timestamp("2025-01-01") + pd.Timedelta(weeks=a)
            for w in range(a + 1):
                wk = pd.Timestamp("2024-12-28") + pd.Timedelta(weeks=w)
                final_like = 100.0
                rep = final_like if a - w >= 1 else (final_like + 1) / np.exp(0.2) - 1
                rows.append({"location": "01", "wk_end_date": wk, "inc": rep, "as_of": as_of})
        model = RevisionModel(max_lag=2, min_vectors=1).fit(pd.DataFrame(rows))
        assert len(model.vectors) > 0
        assert np.allclose(model.vectors[:, 0], 0.2) and np.allclose(model.vectors[:, 1], 0.0)


@pytest.mark.parametrize(
    "model",
    [
        PeakBaselineModel(
            PeakBaselineModelConfig(
                model_name="peak_baseline", supplementary_sources=[SourceType.ILINET], num_revision_draws=20
            )
        ),
        PeakGBQRModel(
            PeakGBQRModelConfig(
                model_name="peak_gbqr",
                supplementary_sources=[SourceType.ILINET],
                num_revision_draws=20,
                num_bags=2,
                n_estimators=10,
            )
        ),
        PeakGBQRModel(
            PeakGBQRModelConfig(
                model_name="peak_gbqr_offset",
                supplementary_sources=[SourceType.ILINET],
                num_revision_draws=20,
                num_bags=2,
                n_estimators=10,
                size_offset=True,
            )
        ),
        PeakKCDEModel(
            PeakKCDEModelConfig(
                model_name="peak_kcde",
                supplementary_sources=[SourceType.ILINET],
                num_revision_draws=20,
                num_tuning_rows=100,
                max_tuning_iter=5,
            )
        ),
    ],
)
def test_forecast_is_valid_submission(model, tmp_path):
    ref_date = datetime.date(2026, 12, 19)  # season week 21, before the synthetic current-season peak
    run_config = _run_config(ref_date, tmp_path)
    df = model.forecast(_synthetic_inputs(ref_date), run_config)

    pmf = df.loc[df["output_type"] == "pmf"]
    assert set(pmf["target"]) == {"peak week inc flu hosp"}
    assert pmf.groupby("location").size().eq(34).all()
    assert set(pmf["output_type_id"]) == {str(season_week_to_date("2026/27", w)) for w in range(10, 44)}
    assert np.allclose(pmf.groupby("location")["value"].sum(), 1.0)
    # every window date, including past weeks that cannot become the peak, has positive probability
    assert pmf["value"].min() > 0
    assert pmf["value"].min() >= model.model_config.pmf_floor / (1 + 34 * model.model_config.pmf_floor) - 1e-12

    q = df.loc[df["output_type"] == "quantile"]
    assert set(q["target"]) == {"peak inc flu hosp"}
    assert q.groupby("location").size().eq(len(Q_LEVELS)).all()
    for _, g in q.groupby("location"):
        vals = g.sort_values("output_type_id", key=lambda s: s.astype(float))["value"].to_numpy()
        assert np.all(np.diff(vals) >= 0) and np.all(vals == np.round(vals))
    assert df["horizon"].isna().all() and df["target_end_date"].isna().all()

    # deterministic given the reference date
    pd.testing.assert_frame_equal(df, model.forecast(_synthetic_inputs(ref_date), run_config))
