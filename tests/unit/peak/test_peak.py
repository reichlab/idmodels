import datetime
import warnings

import numpy as np
import pandas as pd
import pytest
from iddata.enums import Disease

from idmodels.config import (
    PeakBaselineModelConfig,
    PeakGBQRModelConfig,
    RunConfig,
    SourceType,
)
from idmodels.peak import PeakBaselineModel, PeakGBQRModel
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


def _synthetic_strain(seasons) -> dict:
    """Weekly A, B, A(H1), A(H3) positives (4, 53) for each location and the HHS regions of LOCS (Regions 4 and 10)."""
    rng = np.random.default_rng(2)
    geos = LOCS + ["Region 4", "Region 10"]
    return {(g, s): rng.poisson(30, (4, 53)).astype(float) for g in geos for s in seasons}


def _run_config(ref_date, tmp_path, disease=Disease.FLU, states=LOCS):
    return RunConfig(
        disease=disease,
        ref_date=ref_date,
        output_root=tmp_path,
        artifact_store_root=None,
        max_horizon=4,
        states=states,
        hsas=[],
        q_levels=Q_LEVELS,
        q_labels=[str(q) for q in Q_LEVELS],
    )


class TestSeriesHelpers:
    def test_season_week_to_date(self):
        assert season_week_to_date("2026/27", 10) == datetime.date(2026, 10, 10)
        assert season_week_to_date("2026/27", 43) == datetime.date(2027, 5, 29)
        assert season_of(datetime.date(2026, 10, 10)) == "2026/27"

    def test_default_window_matches_flusight_every_season(self):
        # FluSight's peak-week dates for each season; the default window (season weeks 10-43) gives them all
        cfg = PeakBaselineModelConfig(model_name="b")
        flusight = {
            "2023/24": ("2023-10-07", "2024-05-25"),
            "2024/25": ("2024-10-05", "2025-05-24"),
            "2025/26": ("2025-10-04", "2026-05-23"),
            "2026/27": ("2026-10-10", "2027-05-29"),
        }
        for season, (start, end) in flusight.items():
            assert str(season_week_to_date(season, cfg.window_start_week)) == start
            assert str(season_week_to_date(season, cfg.window_end_week)) == end

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

    def test_sync_burden_features_do_not_look_ahead(self):
        rng = np.random.default_rng(0)
        y = np.stack([_curve(20 + i, 5.0 + i) for i in range(4)]) * rng.lognormal(0, 0.05, (4, 53))
        y2 = y.copy()
        y2[:, 22:] = y2[:, 22:] * 100 + 5  # change the future of every series
        kw = dict(
            group=np.array([0, 0, 0, 1]),
            pool=np.array([True, True, False, True]),
            hist_total=np.full(4, 50.0),
            hist_cum=np.tile(np.linspace(0, 50, 53), (4, 1)),
        )
        f1 = state_features(y, 22, np.full(4, 0.01), 10, **kw)
        f2 = state_features(y2, 22, np.full(4, 0.01), 10, **kw)
        pd.testing.assert_frame_equal(f1, f2)
        # group 0 pools rows 0 and 1 (row 2 is not in the pool); group 1 is row 3 alone
        assert np.isclose(f1.loc[0, "sync_med_rel_max"], f1.loc[[0, 1], "rel_max"].median())
        assert np.isclose(f1.loc[2, "sync_med_rel_max"], f1.loc[0, "sync_med_rel_max"])
        assert np.isclose(f1.loc[3, "sync_med_rel_max"], f1.loc[3, "rel_max"])
        cum = np.nansum(y[:, 4:22], axis=1)
        assert np.allclose(f1["cum_vs_hist_total"], np.log(cum + 0.01) - np.log(50.01))

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
        PeakGBQRModel(
            PeakGBQRModelConfig(
                model_name="peak_gbqr",
                supplementary_sources=[SourceType.ILINET],
                num_revision_draws=20,
                num_bags=2,
                n_estimators=10,
                sync_burden_features=True,
            )
        ),
        PeakGBQRModel(
            PeakGBQRModelConfig(
                model_name="peak_gbqr",
                supplementary_sources=[SourceType.ILINET],
                num_revision_draws=20,
                num_bags=2,
                n_estimators=10,
                sync_reported_only=True,
                size_offset=True,
                size_feature_groups=["core", "sb", "trend", "bshare"],
                timing_feature_groups=["core", "recession", "latlon", "holiday", "h3"],
            )
        ),
    ],
)
def test_forecast_is_valid_submission(model, tmp_path):
    ref_date = datetime.date(2026, 12, 19)  # season week 21, before the synthetic current-season peak
    run_config = _run_config(ref_date, tmp_path)
    inputs = _synthetic_inputs(ref_date)
    cfg = model.model_config
    groups = set(getattr(cfg, "size_feature_groups", None) or []) | set(
        getattr(cfg, "timing_feature_groups", None) or []
    )
    if {"bshare", "h3"} & groups:
        # the type/subtype features need type/subtype counts
        inputs.strain = _synthetic_strain(inputs.data["season"].unique())
    df = model.forecast(inputs, run_config)

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
    pd.testing.assert_frame_equal(df, model.forecast(inputs, run_config))


def test_gbqr_oob_predictions_exclude_own_season(tmp_path):
    ref_date = datetime.date(2026, 12, 19)
    model = PeakGBQRModel(
        PeakGBQRModelConfig(
            model_name="g", supplementary_sources=[SourceType.ILINET], num_bags=4, n_estimators=10, progress_bar=False
        )
    )
    model.forecast(_synthetic_inputs(ref_date), _run_config(ref_date, tmp_path))
    rows = model.train_rows_
    timing, size = model.oob_predict(rows)
    # raw GBQR timing may sum to < 1 where tail classes fall outside the window (renormalized downstream)
    assert timing.shape == (len(rows), model.kmax + 1) and np.all(timing.sum(axis=1) <= 1 + 1e-9)
    assert np.all(np.diff(size, axis=1) >= 0)
    # a season in only some bags is predicted by the others only
    season = rows["season"].iloc[0]
    use = np.array([season not in b for b in model.bag_seasons_])
    if use.any() and not use.all():
        sel = rows["season"] == season
        t_oob, _ = model._predict(rows.loc[sel], bag_mask=np.tile(use, (sel.sum(), 1)))
        assert np.allclose(t_oob, timing[sel.to_numpy()])


def test_extra_features_do_not_look_ahead():
    from idmodels.peak.extra_features import FEATURE_GROUPS, holiday_matrix

    rng = np.random.default_rng(0)
    n = 3
    y = np.stack([_curve(20 + i, 5.0 + i) for i in range(n)]) * rng.lognormal(0, 0.05, (n, 53))
    types = {k: rng.poisson(30, (n, 53)).astype(float) for k in ["A", "B", "H1", "H3", "H1n", "H3n"]}
    kw = dict(holiday=holiday_matrix(["2018/19"] * n), latlon=(np.array([40.0, 35.0, np.nan]), np.zeros(n)))
    f1 = state_features(y, 24, np.full(n, 0.01), 10, types=types, **kw)
    y2, types2 = y.copy(), {k: v.copy() for k, v in types.items()}
    y2[:, 24:] = y2[:, 24:] * 100 + 5
    for v in types2.values():
        v[:, 24:] = v[:, 24:] * 100 + 5
    f2 = state_features(y2, 24, np.full(n, 0.01), 10, types=types2, **kw)
    pd.testing.assert_frame_equal(f1, f2)
    for cols in FEATURE_GROUPS.values():
        assert set(cols) <= set(f1.columns)
    assert f1["b_share_3wk"].notna().all() and f1["lat"].isna().sum() == 1


def test_gbqr_size_and_timing_features_are_independent(tmp_path):
    """Changing only the timing feature groups leaves the peak-size forecast unchanged, and vice versa."""
    ref_date = datetime.date(2026, 12, 19)
    run_config = _run_config(ref_date, tmp_path)

    def forecast(size_groups, timing_groups):
        cfg = PeakGBQRModelConfig(
            model_name="peak_gbqr",
            supplementary_sources=[SourceType.ILINET],
            num_revision_draws=20,
            num_bags=2,
            n_estimators=10,
            progress_bar=False,
            size_feature_groups=size_groups,
            timing_feature_groups=timing_groups,
        )
        return PeakGBQRModel(cfg).forecast(_synthetic_inputs(ref_date), run_config)

    a = forecast(["core", "sb"], ["core"])
    b = forecast(["core", "sb"], ["core", "recession", "latlon"])
    c = forecast(["core", "trend"], ["core"])
    size, pmf = a["target"] == "peak inc flu hosp", a["output_type"] == "pmf"
    assert np.array_equal(a.loc[size, "value"], b.loc[size, "value"])
    assert np.allclose(a.loc[pmf, "value"], c.loc[pmf, "value"])


def _baseline_model(**kwargs):
    cfg = PeakBaselineModelConfig(
        model_name="peak_baseline", supplementary_sources=[SourceType.ILINET], num_revision_draws=20, **kwargs
    )
    return PeakBaselineModel(cfg)


def test_type_subtype_features_use_strain(tmp_path):
    ref_date = datetime.date(2026, 12, 19)
    inputs = _synthetic_inputs(ref_date)
    inputs.strain = _synthetic_strain(inputs.data["season"].unique())
    model = PeakGBQRModel(
        PeakGBQRModelConfig(
            model_name="g",
            supplementary_sources=[SourceType.ILINET],
            num_revision_draws=20,
            num_bags=2,
            n_estimators=10,
            progress_bar=False,
            size_feature_groups=["core", "bshare"],
            timing_feature_groups=["core", "h3"],
        )
    )
    model.forecast(inputs, _run_config(ref_date, tmp_path))
    rows = model.train_rows_
    assert rows["b_share_3wk"].notna().any()
    assert rows.filter(like="h3").notna().any().any()


def test_non_flu_disease_is_rejected(tmp_path):
    ref_date = datetime.date(2026, 12, 19)
    with pytest.raises(ValueError, match="only influenza"):
        _baseline_model().forecast(_synthetic_inputs(ref_date), _run_config(ref_date, tmp_path, disease=Disease.RSV))


def test_locations_without_data(tmp_path):
    ref_date = datetime.date(2026, 12, 19)
    inputs = _synthetic_inputs(ref_date)
    # by default, a requested location without current-season data is an error
    with pytest.raises(ValueError, match=r"locations \['06'\]"):
        _baseline_model().forecast(inputs, _run_config(ref_date, tmp_path, states=LOCS + ["06"]))
    # with drop_missing_locations, it is omitted with a warning
    with pytest.warns(UserWarning, match="omitting them"):
        df = _baseline_model(drop_missing_locations=True).forecast(
            inputs, _run_config(ref_date, tmp_path, states=LOCS + ["06"])
        )
    assert set(df["location"]) == set(LOCS)
    # if no requested location has data, it is an error either way
    with pytest.raises(ValueError, match=r"locations \['06'\]"):
        _baseline_model(drop_missing_locations=True).forecast(inputs, _run_config(ref_date, tmp_path, states=["06"]))


def test_reusing_fit_with_different_data_warns(tmp_path):
    ref_date = datetime.date(2026, 12, 19)
    inputs = _synthetic_inputs(ref_date)
    model = _baseline_model()
    model.forecast(inputs, _run_config(ref_date, tmp_path))
    # the same training series (e.g. the next reference date's data): the fit is reused without a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(inputs.data, "2026/27", Q_LEVELS)
    # different training series: the fit is still reused, with a warning
    with pytest.warns(UserWarning, match="reusing the fit"):
        model.fit(inputs.data.loc[inputs.data["location"] != "02"], "2026/27", Q_LEVELS)
