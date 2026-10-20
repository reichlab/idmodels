"""
Base class for direct models of the seasonal peak week and peak size of NHSN influenza hospital admissions.

Notation. For a season observed through season week t, M_t is the running maximum of the in-window weeks observed so
far, and k = peak week - t is the number of weeks until the season's peak (k <= 0: the peak has already occurred).
A concrete model supplies _fit() and _predict(). _predict() maps state features (see idmodels.peak.series) to
  - timing probabilities over classes c = 0, 1, ..., kmax, where c = 0 means "the peak has already occurred"
    (k <= 0) and c = k >= 1 means "the peak is k weeks after the most recent observed week";
  - quantiles of z = log(peak + eps) - log(M_t + eps), the log growth still to come in the running maximum, at levels
    self.size_levels.
The base class turns these into hub submissions, propagating uncertainty about revisions to recent data by
recomputing features and predictions on Monte Carlo draws of the revised series (see idmodels.peak.revision).

PeakModel does not subclass idmodels.model.IDModel: IDModel's workflow (transform the weekly series, build lagged
features, predict each horizon) does not fit targets that are a single value per season, which are learned instead
from historical seasons replayed week by week (idmodels.peak.series.build_replay_rows).
"""

import calendar
import datetime
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from iddata.ancillary.population import PopulationData
from iddata.enums import Disease
from iddata.loader import DiseaseDataLoader
from iddata.sources.flusurvnet import FluSurvNetDataSource
from iddata.sources.ilinet import ILINetDataSource
from iddata.sources.nhsn import NHSNDataSource

from idmodels.config import PeakModelConfig, RunConfig, SourceType
from idmodels.peak.dates import season_of, season_week_to_date
from idmodels.peak.extra_features import holiday_matrix
from idmodels.peak.revision import RevisionModel, apply_revisions, load_nhsn_vintages
from idmodels.peak.series import (
    LOG_EPS,
    N_SEASON_WEEKS,
    SOURCE_CODES,
    build_replay_rows,
    build_season_arrays,
    cumulative_curves,
    historical_log_peaks,
    peak_week_climatology,
    state_features,
)
from idmodels.utils import build_save_path

PEAK_WEEK_TARGET = "peak week inc {disease} hosp"
PEAK_SIZE_TARGET = "peak inc {disease} hosp"

# input data loaded by load_inputs(), keyed by (sources, reference date). Bounded so that a long loop over reference
# dates does not keep every date's data in memory; only the most recently used entries are kept.
_DATA_CACHE: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
_DATA_CACHE_SIZE = 4


@dataclass
class PeakInputs:
    """Data available on the reference date."""

    data: pd.DataFrame  # iddata output for NHSN and supplementary sources (rates), with pop
    vintages: pd.DataFrame  # all NHSN vintages (counts) released on or before the reference date


class PeakModel(ABC):
    """Abstract base for direct peak models. See module docstring."""

    def __init__(self, model_config: PeakModelConfig):
        self.model_config = model_config
        self._fitted_key: tuple | None = None
        self._fitted_series: frozenset = frozenset()

    # ---------------------------------------------------------------------------------------------------------------
    # public entry points

    def run(self, run_config: RunConfig) -> pd.DataFrame:
        """
        Load data available on run_config.ref_date, forecast, and save a hub-formatted csv. Returns the forecast.
        Peak models use run_config's disease, ref_date, output_root, states (the locations to forecast), q_levels and
        q_labels; artifact_store_root, max_horizon and hsas are ignored, and num_sample_forecasts must be 0.
        """
        inputs = self.load_inputs(run_config.ref_date)
        preds_df = self.forecast(inputs, run_config)
        save_path = build_save_path(root=run_config.output_root, run_config=run_config, model_config=self.model_config)
        preds_df.to_csv(save_path, index=False, na_rep="NA")
        return preds_df

    def load_inputs(self, ref_date: datetime.date) -> PeakInputs:
        sources = [SourceType.NHSN] + list(self.model_config.supplementary_sources)
        key = (tuple(s.value for s in sources), ref_date)
        if key in _DATA_CACHE:
            _DATA_CACHE.move_to_end(key)
        else:
            source_map = {
                SourceType.NHSN: NHSNDataSource(),
                SourceType.ILINET: ILINetDataSource(),
                SourceType.FLUSURVNET: FluSurvNetDataSource(),
            }
            unsupported = [s for s in sources if s not in source_map]
            if unsupported:
                raise ValueError(f"unsupported sources for peak models: {unsupported}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = DiseaseDataLoader().load(
                    sources=[source_map[s] for s in sources], as_of=ref_date, ancillary=[PopulationData()]
                )
            _DATA_CACHE[key] = data
            while len(_DATA_CACHE) > _DATA_CACHE_SIZE:
                _DATA_CACHE.popitem(last=False)
        return PeakInputs(data=_DATA_CACHE[key], vintages=load_nhsn_vintages(ref_date))

    def forecast(self, inputs: PeakInputs, run_config: RunConfig) -> pd.DataFrame:
        """Hub-formatted peak week pmf and peak size quantile forecasts for run_config.ref_date."""
        # the data loaded by load_inputs() and the revision model's vintages are influenza-only
        if run_config.disease != Disease.FLU:
            raise ValueError(f"peak models support only influenza, not {Disease(run_config.disease).value!r}")
        # sample output is trajectories of weekly incidence across forecast horizons, which peak targets (one
        # season-level peak week and peak size per location, with no horizons) don't have; raise rather than
        # silently write a file without the requested samples
        if run_config.num_sample_forecasts > 0:
            raise NotImplementedError("peak models do not support sample forecasts; set num_sample_forecasts=0.")
        season = season_of(run_config.ref_date)
        self.fit(inputs.data, season, run_config.q_levels)
        rng = np.random.default_rng(int(calendar.timegm(run_config.ref_date.timetuple())))
        pmf, quantiles, locations = self._predict_current_season(inputs, season, run_config, rng)
        # fail loudly rather than return (or write) an incomplete submission
        if np.isnan(pmf).any() or np.isnan(quantiles).any():
            bad = [loc for loc, p, q in zip(locations, pmf, quantiles) if np.isnan(p).any() or np.isnan(q).any()]
            raise ValueError(f"NaN forecast values for {self.model_config.model_name} at {run_config.ref_date}: {bad}")
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

    def fit(self, data: pd.DataFrame, season: str, q_levels) -> None:
        """
        Fit to all seasons of `data` (long-format iddata output, any sources) strictly before `season`. The fit only
        changes when the season does (up to minor revisions to earlier seasons' data), so it is repeated only when
        the season or the quantile levels change. To fit the same configuration to different data in the same season,
        use a new model object.
        """
        cfg = self.model_config
        self.size_levels = np.asarray(q_levels, dtype=float)
        self.kmax = cfg.window_end_week - 1
        key = (season, tuple(self.size_levels))
        train = data.loc[data["season"] < season]
        excluded = pd.MultiIndex.from_tuples(cfg.exclude_training_series) if cfg.exclude_training_series else None
        if excluded is not None:
            train = train.loc[~pd.MultiIndex.from_arrays([train["source"], train["location"]]).isin(excluded)]
        # the training series (source, location, season): unchanged by revisions to earlier seasons' values, but
        # different if `data` has other sources, locations or seasons than the data of the existing fit
        series = frozenset(train[["source", "location", "season"]].drop_duplicates().itertuples(index=False, name=None))
        if self._fitted_key == key:
            if series != self._fitted_series:
                warnings.warn(
                    f"reusing the fit for season {season}, although `data` has different training series than the "
                    "data it was fit to; use a new model object to fit to these data"
                )
            return
        arrays = build_season_arrays(train)
        self.hist_ = historical_log_peaks(arrays, cfg.window_start_week, cfg.window_end_week, cfg.min_window_obs)
        # mean season total and cumulative curve per (source, location) over the complete training series
        complete = arrays.subset(
            arrays.keys.set_index(["source", "location", "season"]).index.isin(
                self.hist_.set_index(["source", "location", "season"]).index
            )
        )
        total, curves = cumulative_curves(complete.y, cfg.window_end_week)
        src_loc = (complete.keys["source"] + "|" + complete.keys["location"]).to_numpy()
        self.hist_total_ = pd.Series(total).groupby(src_loc).mean()
        self.hist_cum_ = pd.DataFrame(curves).groupby(src_loc).mean()
        self.clim_ = peak_week_climatology(
            self.hist_, cfg.window_start_week, cfg.window_end_week, smoothing_sd=cfg.timing_smoothing_sd
        )
        rows = build_replay_rows(
            arrays,
            cfg.window_start_week,
            cfg.window_end_week,
            cfg.replay_start_week,
            cfg.min_window_obs,
        )
        rows["timing_class"] = np.clip(rows["k"], 0, None).astype(int)
        self.train_rows_ = rows
        self._fit(rows)
        self._fitted_key = key
        self._fitted_series = series

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
        locations = list(run_config.states)
        counts, pop = self._current_counts(inputs, season, locations)
        has_data = ~np.all(np.isnan(counts), axis=1) & ~np.isnan(pop)
        if not has_data.all():
            dropped = [loc for loc, ok in zip(locations, has_data) if not ok]
            msg = f"no current-season NHSN data or population on {run_config.ref_date} for locations {dropped}"
            if not cfg.drop_missing_locations or not has_data.any():
                raise ValueError(msg)
            warnings.warn(f"{msg}; omitting them")
        locations = [loc for loc, ok in zip(locations, has_data) if ok]
        counts, pop = counts[has_data], pop[has_data]
        n_loc, n_draws = len(locations), cfg.num_revision_draws

        last_week = np.array([np.flatnonzero(~np.isnan(row)).max() for row in counts])
        rev_model = RevisionModel(max_lag=cfg.revision_max_lag).fit(inputs.vintages)
        rho = rev_model.sample(counts[np.arange(n_loc), last_week], n_draws, rng)
        revised = apply_revisions(counts, last_week, rho)  # (n_loc, n_draws, weeks)
        rates = revised * 1e5 / pop[:, None, None]

        pmf, peak_rate = self._peak_distributions(
            rates, counts * 1e5 / pop[:, None], locations, nat_loc="US" if "US" in locations else None, season=season
        )
        peak_counts = peak_rate * pop[:, None] / 1e5
        quantiles = np.quantile(peak_counts, run_config.q_levels, axis=1).T  # (n_loc, n_q)
        quantiles = np.maximum.accumulate(np.round(np.maximum(quantiles, 0.0)), axis=1)
        return pmf, quantiles, locations

    def _peak_distributions(
        self,
        rates: np.ndarray,
        reported: np.ndarray,
        locations: list[str],
        nat_loc: str | None,
        season: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the model to Monte Carlo draws of the final current-season NHSN series.

        rates: (n_loc, n_draws, N_SEASON_WEEKS) plausible final series; reported: (n_loc, N_SEASON_WEEKS) the series
        as currently reported. The current week of each location is its last reported week + 1 (1-based season week).
        Returns the peak week pmf over window weeks (n_loc, n_weeks), averaged over draws and floored, and samples of
        the peak size in the units of `rates` (n_loc, n_draws * num_size_levels).
        """
        source = "nhsn"
        cfg = self.model_config
        w0, w1 = cfg.window_start_week, cfg.window_end_week
        n_loc, n_draws = rates.shape[:2]
        rates = rates.reshape(n_loc * n_draws, N_SEASON_WEEKS)
        last_week = np.array([np.flatnonzero(~np.isnan(row)).max() for row in reported])

        # series row r = loc * n_draws + draw; national features come from the national series of the same draw
        loc_of_row = np.repeat(np.arange(n_loc), n_draws)
        draw_of_row = np.tile(np.arange(n_draws), n_loc)
        nat_idx = np.full(n_loc * n_draws, -1)
        if nat_loc is not None:
            nat_idx = locations.index(nat_loc) * n_draws + draw_of_row

        src_hist = self.hist_.loc[self.hist_["source"] == source].groupby("location")["log_peak"].mean()
        hist_peak_loc = src_hist.reindex(locations).to_numpy()
        hist_peak = hist_peak_loc[loc_of_row]
        src_keys = [f"{source}|{loc}" for loc in locations]
        hist_total_loc = self.hist_total_.reindex(src_keys).to_numpy()
        hist_cum_loc = self.hist_cum_.reindex(src_keys).to_numpy()
        pool = np.asarray([loc != nat_loc for loc in locations])
        eps = np.full(n_loc * n_draws, LOG_EPS[source])

        t_of_row = (last_week + 1)[loc_of_row]
        holiday = holiday_matrix([season] * (n_loc * n_draws))
        feats = pd.DataFrame(index=np.arange(n_loc * n_draws))
        for t in np.unique(t_of_row):
            # synchrony features are computed across the locations of each revision draw
            f = state_features(
                rates,
                int(t),
                eps,
                w0,
                nat_idx=nat_idx,
                hist_peak=hist_peak,
                group=draw_of_row,
                pool=pool[loc_of_row],
                hist_total=hist_total_loc[loc_of_row],
                hist_cum=hist_cum_loc[loc_of_row],
                holiday=holiday,
            )
            sel = t_of_row == t
            for col in f.columns.drop("observed"):
                if col not in feats:
                    feats[col] = np.nan
                feats.loc[sel, col] = f.loc[sel, col].to_numpy()
        feats["src_code"] = SOURCE_CODES[source]
        feats["source"] = source
        feats["location"] = np.asarray(locations)[loc_of_row]

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

        # ---- peak size samples
        n_u = cfg.num_size_levels
        u = (np.arange(n_u) + 0.5) / n_u
        z = np.stack([np.interp(u, self.size_levels, np.sort(row)) for row in size])  # (rows, n_u)
        z = np.where((t_of_row >= w0)[:, None], np.maximum(z, 0.0), z)
        lm = feats["lm"].to_numpy()[:, None]
        peak = np.maximum(np.exp(lm + z) - LOG_EPS[source], 0.0)
        return pmf, peak.reshape(n_loc, n_draws * n_u)

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
