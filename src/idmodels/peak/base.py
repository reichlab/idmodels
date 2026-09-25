"""
Base class for direct models of the seasonal peak week and peak size of NHSN influenza hospital admissions.

A concrete model supplies _fit() and _predict(). _predict() maps state features (see idmodels.peak.series) to
  - timing probabilities over classes c = 0, 1, ..., kmax, where c = 0 means "the peak has already occurred"
    (k <= 0) and c = k >= 1 means "the peak is k weeks after the most recent observed week";
  - quantiles of z = log(peak + eps) - log(M_t + eps) at levels self.size_levels.
The base class turns these into hub submissions, propagating uncertainty about revisions to recent data by
recomputing features and predictions on Monte Carlo draws of the revised series (see idmodels.peak.revision).
"""

import calendar
import datetime
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from iddata.ancillary.population import PopulationData
from iddata.loader import DiseaseDataLoader
from iddata.sources.flusurvnet import FluSurvNetDataSource
from iddata.sources.ilinet import ILINetDataSource
from iddata.sources.nhsn import NHSNDataSource
from iddata.utils import add_season_columns

from idmodels.config import PeakModelConfig, RunConfig, SourceType
from idmodels.peak.revision import RevisionModel, apply_revisions, load_nhsn_vintages
from idmodels.peak.series import (
    LOG_EPS,
    N_SEASON_WEEKS,
    SOURCE_CODES,
    SeasonArrays,
    build_replay_rows,
    build_season_arrays,
    historical_log_peaks,
    peak_week_climatology,
    state_features,
)
from idmodels.utils import build_save_path

PEAK_WEEK_TARGET = "peak week inc {disease} hosp"
PEAK_SIZE_TARGET = "peak inc {disease} hosp"

_DATA_CACHE: dict[tuple, pd.DataFrame] = {}


@dataclass
class PeakInputs:
    """Data available on the reference date."""

    data: pd.DataFrame  # iddata output for NHSN and supplementary sources (rates), with pop
    vintages: pd.DataFrame  # all NHSN vintages (counts) released on or before the reference date


def season_of(date: datetime.date) -> str:
    return add_season_columns(pd.DataFrame({"wk_end_date": [str(date)], "location": ["US"]}))["season"].iloc[0]


def season_week_to_date(season: str, season_week: int) -> datetime.date:
    """The Saturday ending the given season week of the given season."""
    start_year = int(season[:4])
    # season week 1 is MMWR week 31; find the Saturday ending it by scanning late July / early August
    for day in range(20, 40):
        d = datetime.date(start_year, 7, 1) + datetime.timedelta(days=day)
        if d.weekday() == 5:
            sw = add_season_columns(pd.DataFrame({"wk_end_date": [str(d)], "location": ["US"]}))["season_week"].iloc[0]
            if sw == 1:
                return d + datetime.timedelta(weeks=season_week - 1)
    raise ValueError(f"could not locate season week 1 of {season}")


class PeakModel(ABC):
    """Abstract base for direct peak models. See module docstring."""

    def __init__(self, model_config: PeakModelConfig):
        self.model_config = model_config
        self._fitted_key: tuple | None = None

    # ---------------------------------------------------------------------------------------------------------------
    # public entry points

    def run(self, run_config: RunConfig) -> pd.DataFrame:
        """Load data available on run_config.ref_date, forecast, and save a hub-formatted csv. Returns the forecast."""
        inputs = self.load_inputs(run_config.ref_date)
        preds_df = self.forecast(inputs, run_config)
        save_path = build_save_path(root=run_config.output_root, run_config=run_config, model_config=self.model_config)
        if preds_df["value"].isna().any():
            raise ValueError(
                f"NaN forecast values for {self.model_config.model_name} at {run_config.ref_date}; "
                f"refusing to write {save_path}"
            )
        preds_df.to_csv(save_path, index=False, na_rep="NA")
        return preds_df

    def load_inputs(self, ref_date: datetime.date) -> PeakInputs:
        sources = [SourceType.NHSN] + list(self.model_config.supplementary_sources)
        key = (tuple(s.value for s in sources), ref_date)
        if key not in _DATA_CACHE:
            src_objs = []
            for s in sources:
                if s == SourceType.NHSN:
                    src_objs.append(NHSNDataSource())
                elif s == SourceType.ILINET:
                    src_objs.append(ILINetDataSource())
                elif s == SourceType.FLUSURVNET:
                    src_objs.append(FluSurvNetDataSource())
                else:
                    raise ValueError(f"unsupported source for peak models: {s}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _DATA_CACHE[key] = DiseaseDataLoader().load(
                    sources=src_objs, as_of=ref_date, ancillary=[PopulationData()]
                )
        return PeakInputs(data=_DATA_CACHE[key], vintages=load_nhsn_vintages(ref_date))

    def forecast(self, inputs: PeakInputs, run_config: RunConfig) -> pd.DataFrame:
        """Hub-formatted peak week pmf and peak size quantile forecasts for run_config.ref_date."""
        cfg = self.model_config
        season = season_of(run_config.ref_date)
        self.size_levels = np.asarray(run_config.q_levels, dtype=float)
        self.kmax = cfg.window_end_week - 1

        self._fit_if_needed(inputs, season, run_config)
        rng = np.random.default_rng(int(calendar.timegm(run_config.ref_date.timetuple())))
        pmf, quantiles, locations = self._predict_current_season(inputs, season, run_config, rng)
        return self._format_output(pmf, quantiles, locations, season, run_config)

    # ---------------------------------------------------------------------------------------------------------------
    # model-specific pieces

    @abstractmethod
    def _fit(self, rows: pd.DataFrame) -> None:
        """Fit to season-replay training rows (see idmodels.peak.series.build_replay_rows)."""
        ...

    @abstractmethod
    def _predict(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (timing, size): timing has shape (n, kmax + 1) with rows summing to 1 (classes as described in the
        module docstring); size has shape (n, len(self.size_levels)) of non-decreasing quantiles of z.
        """
        ...

    # ---------------------------------------------------------------------------------------------------------------
    # training

    def _training_arrays(self, inputs: PeakInputs, season: str) -> SeasonArrays:
        data = inputs.data.loc[inputs.data["season"] < season]
        return build_season_arrays(data)

    def _fit_if_needed(self, inputs: PeakInputs, season: str, run_config: RunConfig) -> None:
        """
        Training uses only seasons before the current one, so the fit only changes when the season does (up to minor
        revisions to earlier seasons' data). Refit once per (season, quantile levels).
        """
        key = (season, tuple(self.size_levels))
        if self._fitted_key == key:
            return
        cfg = self.model_config
        arrays = self._training_arrays(inputs, season)
        self.hist_ = historical_log_peaks(arrays, cfg.window_start_week, cfg.window_end_week, cfg.min_window_obs)
        self.clim_ = peak_week_climatology(
            self.hist_, cfg.window_start_week, cfg.window_end_week, smoothing_sd=cfg.timing_smoothing_sd
        )
        rows = build_replay_rows(
            arrays, cfg.window_start_week, cfg.window_end_week, cfg.replay_start_week, cfg.min_window_obs
        )
        rows["timing_class"] = np.clip(rows["k"], 0, None).astype(int)
        self.train_rows_ = rows
        self._fit(rows)
        self._fitted_key = key

    def _spread_climatology(self, t: np.ndarray, first_k: int) -> np.ndarray:
        """
        For each current week t, climatological probabilities over classes c = first_k..kmax (peak at week t + c),
        normalized to sum to 1 (zeros if no such in-window week exists). Shape (len(t), kmax + 1).
        """
        cfg = self.model_config
        t = np.asarray(t, dtype=int)
        out = np.zeros((len(t), self.kmax + 1))
        classes = np.arange(self.kmax + 1)
        for ti in np.unique(t):
            weeks = ti + classes
            ok = (classes >= first_k) & (weeks >= cfg.window_start_week) & (weeks <= cfg.window_end_week)
            probs = np.zeros(self.kmax + 1)
            probs[ok] = self.clim_[weeks[ok] - cfg.window_start_week]
            if probs.sum() > 0:
                probs /= probs.sum()
            out[t == ti] = probs
        return out

    # ---------------------------------------------------------------------------------------------------------------
    # prediction for the current season

    def _current_counts(self, inputs: PeakInputs, season: str, locations: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Season-aligned counts (n_loc, N_SEASON_WEEKS) from the most recent vintage, and population per location."""
        v = inputs.vintages
        latest = v.loc[(v["as_of"] == v["as_of"].max()) & (v["season"] == season)]
        wide = latest.pivot_table(index="location", columns="season_week", values="inc", aggfunc="last")
        wide = wide.reindex(index=locations, columns=np.arange(1, N_SEASON_WEEKS + 1, dtype=float))
        counts = wide.to_numpy(dtype=float)

        pop = inputs.data.loc[inputs.data["source"] == "nhsn", ["location", "season", "pop"]].dropna()
        pop = pop.sort_values("season").groupby("location")["pop"].last()  # most recent season with a population
        cur = (
            inputs.data.loc[(inputs.data["source"] == "nhsn") & (inputs.data["season"] == season), ["location", "pop"]]
            .dropna()
            .groupby("location")["pop"]
            .last()
        )
        pop.update(cur)
        return counts, pop.reindex(locations).to_numpy(dtype=float)

    def _predict_current_season(
        self, inputs: PeakInputs, season: str, run_config: RunConfig, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        cfg = self.model_config
        w0, w1 = cfg.window_start_week, cfg.window_end_week
        locations = [loc for loc in run_config.states]
        counts, pop = self._current_counts(inputs, season, locations)
        has_data = ~np.all(np.isnan(counts), axis=1) & ~np.isnan(pop)
        if not has_data.all():
            dropped = [loc for loc, ok in zip(locations, has_data) if not ok]
            warnings.warn(f"no current-season data or population for {dropped}; omitting them")
        locations = [loc for loc, ok in zip(locations, has_data) if ok]
        counts, pop = counts[has_data], pop[has_data]
        n_loc, n_draws = len(locations), cfg.num_revision_draws

        last_week = np.array([np.flatnonzero(~np.isnan(row)).max() for row in counts])
        rev_model = RevisionModel(max_lag=cfg.revision_max_lag).fit(inputs.vintages)
        rho = rev_model.sample(counts[np.arange(n_loc), last_week], n_draws, rng)
        revised = apply_revisions(counts, last_week, rho)  # (n_loc, n_draws, weeks)
        rates = (revised * 1e5 / pop[:, None, None]).reshape(n_loc * n_draws, N_SEASON_WEEKS)

        # series row r = loc * n_draws + draw; national features come from the US series of the same draw
        loc_of_row = np.repeat(np.arange(n_loc), n_draws)
        draw_of_row = np.tile(np.arange(n_draws), n_loc)
        nat_idx = np.full(n_loc * n_draws, -1)
        if "US" in locations:
            nat_idx = locations.index("US") * n_draws + draw_of_row

        nhsn_hist = self.hist_.loc[self.hist_["source"] == "nhsn"].groupby("location")["log_peak"].mean()
        hist_peak = nhsn_hist.reindex(locations).to_numpy()[loc_of_row]
        eps = np.full(n_loc * n_draws, LOG_EPS["nhsn"])

        t_of_row = (last_week + 1)[loc_of_row]
        feats = pd.DataFrame(index=np.arange(n_loc * n_draws))
        for t in np.unique(t_of_row):
            f = state_features(rates, int(t), eps, w0, nat_idx=nat_idx, hist_peak=hist_peak)
            sel = t_of_row == t
            for col in f.columns.drop("observed"):
                if col not in feats:
                    feats[col] = np.nan
                feats.loc[sel, col] = f.loc[sel, col].to_numpy()
        feats["src_code"] = SOURCE_CODES["nhsn"]
        feats["source"] = "nhsn"

        timing, size = self._predict(feats)

        # ---- peak week pmf over window weeks w0..w1
        timing = timing.copy()
        classes = np.arange(self.kmax + 1)
        target_week = t_of_row[:, None] + classes[None, :]
        allowed = (target_week >= w0) & (target_week <= w1)
        allowed[:, 0] = t_of_row >= w0  # "already peaked" requires an observed in-window week
        timing[~allowed] = 0.0
        tot = timing.sum(axis=1, keepdims=True)
        timing = np.where(tot > 0, timing / np.where(tot > 0, tot, 1.0), 0.0)

        n_weeks = w1 - w0 + 1
        pmf_rows = np.zeros((n_loc * n_draws, n_weeks))
        for c in range(1, self.kmax + 1):
            wk = t_of_row + c
            ok = (wk >= w0) & (wk <= w1)
            pmf_rows[np.flatnonzero(ok), wk[ok] - w0] += timing[ok, c]
        past_week = feats["max_week"].to_numpy().astype(int)
        ok = (past_week >= w0) & (past_week <= w1)
        pmf_rows[np.flatnonzero(ok), past_week[ok] - w0] += timing[ok, 0]
        pmf = pmf_rows.reshape(n_loc, n_draws, n_weeks).mean(axis=1)
        pmf = np.maximum(pmf, cfg.pmf_floor)
        pmf = pmf / pmf.sum(axis=1, keepdims=True)

        # ---- peak size quantiles (counts)
        n_u = cfg.num_size_levels
        u = (np.arange(n_u) + 0.5) / n_u
        z = np.stack([np.interp(u, self.size_levels, np.sort(row)) for row in size])  # (rows, n_u)
        z = np.where((t_of_row >= w0)[:, None], np.maximum(z, 0.0), z)
        lm = feats["lm"].to_numpy()[:, None]
        peak_rate = np.maximum(np.exp(lm + z) - LOG_EPS["nhsn"], 0.0)
        peak_counts = peak_rate * pop[loc_of_row][:, None] / 1e5
        peak_counts = peak_counts.reshape(n_loc, n_draws * n_u)
        quantiles = np.quantile(peak_counts, run_config.q_levels, axis=1).T  # (n_loc, n_q)
        quantiles = np.maximum.accumulate(np.round(np.maximum(quantiles, 0.0)), axis=1)

        return pmf, quantiles, locations

    def _format_output(
        self, pmf: np.ndarray, quantiles: np.ndarray, locations: list[str], season: str, run_config: RunConfig
    ) -> pd.DataFrame:
        cfg = self.model_config
        disease = run_config.disease.value
        weeks = np.arange(cfg.window_start_week, cfg.window_end_week + 1)
        dates = [str(season_week_to_date(season, int(w))) for w in weeks]
        pmf_df = pd.DataFrame(pmf, columns=dates)
        pmf_df["location"] = locations
        pmf_df = pmf_df.melt(id_vars="location", var_name="output_type_id", value_name="value")
        pmf_df["target"] = PEAK_WEEK_TARGET.format(disease=disease)
        pmf_df["output_type"] = "pmf"

        q_df = pd.DataFrame(quantiles, columns=run_config.q_labels)
        q_df["location"] = locations
        q_df = q_df.melt(id_vars="location", var_name="output_type_id", value_name="value")
        q_df["target"] = PEAK_SIZE_TARGET.format(disease=disease)
        q_df["output_type"] = "quantile"
        q_df["value"] = q_df["value"].astype(int)

        out = pd.concat([pmf_df, q_df], ignore_index=True)
        out["reference_date"] = str(run_config.ref_date)
        out["horizon"] = np.nan
        out["target_end_date"] = np.nan
        cols = [
            "reference_date",
            "target",
            "horizon",
            "target_end_date",
            "location",
            "output_type",
            "output_type_id",
            "value",
        ]
        return out[cols].sort_values(["target", "location", "output_type_id"], kind="stable").reset_index(drop=True)
