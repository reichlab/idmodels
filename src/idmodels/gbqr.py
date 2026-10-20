import calendar
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from iddata.enums import Disease
from iddata.sources.flusurvnet import FluSurvNetDataSource
from iddata.sources.ilinet import ILINetDataSource
from iddata.sources.nhsn import NHSNDataSource
from iddata.sources.nssp import NSSPDataSource
from iddata.sources.smh import SMHDataSource
from joblib import Parallel, delayed

from idmodels.config import SMH_UNUSED_WHEN_FITTING_SEPARATELY, GBQRModelConfig, RunConfig, SourceType
from idmodels.features import (
    DirectionalWaveFeature,
    Feature,
    FeaturePipeline,
    HolidayFeature,
    HorizonTargetFeature,
    LagFeature,
    LevelFeatureFilter,
    OneHotEncodingFeature,
    RollingMeanFeature,
    TaylorFeature,
)
from idmodels.model import IDModel
from idmodels.utils import build_save_path


def _smh_otid(season: pd.Series) -> pd.Series:
    """
    Extract the output_type_id (trajectory sample id) from SMH season strings, which are formatted
    "{season}{scenario_letter}-{output_type_id}" (see iddata.sources.smh.SMHDataSource.load).
    """
    return season.str.split("-", n=1).str[1]


def _ref_date_seed(run_config: RunConfig) -> int:
    """Seed for random number generation, based on reference date, so reruns for a date are reproducible."""
    return int(calendar.timegm(run_config.ref_date.timetuple()))


class GBQRModel(IDModel):
    """Gradient Boosted Quantile Regression forecast model."""

    def __init__(self, model_config: GBQRModelConfig):
        # Narrow self.model_config from ModelConfig to GBQRModelConfig so that
        # type checkers resolve GBQRModelConfig-specific attributes in this class.
        super().__init__(model_config)
        self.model_config: GBQRModelConfig = model_config

    def _resolve_smh_otid(
        self, df_smh: pd.DataFrame, model_config: GBQRModelConfig, run_config: RunConfig
    ) -> pd.DataFrame | None:
        """
        Resolve the SMH output_type_ids (trajectory sample ids) to filter to, as a
        ["round", "location", "source", "otid"] DataFrame of keys to join against, or None if no
        trajectory filter applies (neither smh_otid nor smh_num_otid is set). If smh_otid was set
        explicitly, keep the keys whose otid is in it (applied globally, across every round/location
        alike). If smh_num_otid was set instead, randomly sample that many ids independently within
        each (round, location, source) group: output_type_id is NOT a globally consistent identifier across
        rounds -- round 4 shares each id across every location (a true trajectory sample), but
        rounds 5+ scope each id to a single location (an arbitrary per-location index, disjoint
        from other locations' ids). Sampling from the pooled, round/location-agnostic set of ids
        would mostly select ids that only exist for one location each, starving every other
        location of SMH rows. Grouping must also include `source` (the SMH model): otid values are
        independently assigned per model and are not unique across models within a (round,
        location) group, so pooling across models before sampling -- or joining the sampled ids
        back on (round, location) alone -- would let one sampled id match unrelated rows from every
        other model that happens to reuse that same id string.
        """
        if not model_config.smh_otid and not model_config.smh_num_otid:
            return None

        df_keys = df_smh.assign(otid=_smh_otid(df_smh["season"]))[
            ["round", "location", "source", "otid"]
        ].drop_duplicates()
        if model_config.smh_otid:
            return df_keys.loc[df_keys["otid"].isin(model_config.smh_otid)]

        # default to the same ref_date-based seed as the rest of GBQR, so a rerun for the same date samples the
        # same trajectories
        seed = model_config.smh_otid_seed if model_config.smh_otid_seed is not None else _ref_date_seed(run_config)
        rng = np.random.default_rng(seed)

        sampled_frames = []
        for (r, loc, src), group in df_keys.groupby(["round", "location", "source"]):
            available = sorted(group["otid"].unique())
            if model_config.smh_num_otid > len(available):
                raise ValueError(
                    f"smh_num_otid={model_config.smh_num_otid} exceeds the {len(available)} "
                    f"output_type_ids available for round={r}, location={loc}, source={src}, "
                    f"smh_model={model_config.smh_model}."
                )
            sampled = rng.choice(available, size=model_config.smh_num_otid, replace=False)
            sampled_frames.append(pd.DataFrame({"round": r, "location": loc, "source": src, "otid": sampled}))

        # when there are no SMH rows to sample from, return an empty frame that keeps df_smh's key
        # dtypes: a bare pd.DataFrame(columns=...) has object-dtype columns, which pandas refuses to
        # merge against the numeric `round` column in _filter_smh
        return pd.concat(sampled_frames, ignore_index=True) if sampled_frames else df_keys.iloc[0:0]

    def _filter_smh(self, df: pd.DataFrame, model_config: GBQRModelConfig, run_config: RunConfig) -> pd.DataFrame:
        # SMH source values are formatted "smh-{model_id}"; season values are formatted
        # "{season}{scenario_letter}-{output_type_id}" (see iddata.sources.smh.SMHDataSource.load).
        df_surveillance = df.loc[df["source"].str[:4] != "smh-"]
        df_smh = df.loc[df["source"].str[:4] == "smh-"]

        # SMH was explicitly requested as a source, so if no SMH rows survive filtering, raise rather
        # than silently fitting a surveillance-only model whose output is still labeled as using SMH.
        # Each filter is checked separately so the error names the one that removed the last rows.
        # only filter for model and otid if included in the config file
        if model_config.smh_model:
            df_smh = df_smh.loc[df_smh["source"].isin([f"smh-{m}" for m in model_config.smh_model])]
        if df_smh.empty:
            raise ValueError(f"No SMH rows found for smh_model={model_config.smh_model}.")

        df_smh = df_smh.loc[df_smh["wk_end_date"] < pd.Timestamp(run_config.ref_date)]
        if df_smh.empty:
            raise ValueError(
                f"No SMH rows for smh_model={model_config.smh_model} have wk_end_date before "
                f"ref_date={run_config.ref_date}."
            )

        df_keys = self._resolve_smh_otid(df_smh, model_config, run_config)
        if model_config.smh_num_otid and run_config.artifact_store_root is not None:
            # record the randomly sampled trajectories so a run can be traced back to them. Sampling covers every
            # SMH location (so a location's sampled ids don't depend on which states were requested), but this
            # runs before _filter_locations, so only save the keys for requested locations, matched the same way
            # (SMH locations are state/national "syn-XX")
            save_path = build_save_path(
                root=run_config.artifact_store_root, run_config=run_config, model_config=model_config, subdir="smh_otid"
            )
            df_keys.loc[df_keys["location"].str[-2:].isin(run_config.states)].to_csv(save_path, index=False)
        if df_keys is not None:
            df_smh = (
                df_smh.assign(otid=_smh_otid(df_smh["season"]))
                .merge(df_keys, on=["round", "location", "source", "otid"], how="inner")
                .drop(columns=["otid"])
            )
            if df_smh.empty:
                raise ValueError(
                    f"No SMH rows for smh_model={model_config.smh_model} before ref_date={run_config.ref_date} "
                    f"match smh_otid={model_config.smh_otid}."
                )

        df_smh = df_smh.drop(columns=["round"])
        return pd.concat([df_surveillance, df_smh], join="inner", axis=0)

    def _build_sources(self, run_config: RunConfig):
        source_map = {
            SourceType.NHSN: NHSNDataSource(disease=run_config.disease),
            SourceType.NSSP: NSSPDataSource(disease=run_config.disease),
            SourceType.ILINET: ILINetDataSource(scale_to_positive=self.model_config.reporting_adj),
            SourceType.FLUSURVNET: FluSurvNetDataSource(burden_adj=self.model_config.reporting_adj),
            SourceType.SMH: SMHDataSource(
                disease=run_config.disease,
                model_id=self.model_config.smh_model,
                output_type_id=self.model_config.smh_otid,
            ),
        }
        # concatenate + dedupe sources while preserving order so main_source is always first
        all_sources = list(dict.fromkeys([self.model_config.main_source] + self.model_config.supplementary_sources))

        # Check if both nhsn and nssp data are included as sources
        if self.model_config.main_source not in [SourceType.NHSN, SourceType.NSSP]:
            raise ValueError("GBQRModel only supports NHSN and NSSP as main source.")

        return [source_map[s] for s in all_sources]

    def _uses_smh(self) -> bool:
        return SourceType.SMH in [self.model_config.main_source] + self.model_config.supplementary_sources

    def _filter_sources_df(self, df: pd.DataFrame, run_config: RunConfig) -> pd.DataFrame:
        if self._uses_smh():
            df = self._filter_smh(df, self.model_config, run_config)
        return df

    def _build_feature_pipeline(self, run_config: RunConfig) -> FeaturePipeline:
        if run_config.disease in (Disease.FLU, Disease.RSV):
            initial_feats = ["inc_trans_cs", "season_week", "log_pop"]
        else:
            initial_feats = ["inc_trans_cs", "log_pop"]

        group_cols = ["source", "location"]
        if self._uses_smh():
            # SMH trajectories share (source, location) and differ only in the season string (which encodes
            # scenario + output_type_id), so scope lag/rolling/Taylor/horizon-target features to season to keep
            # them from treating every trajectory as one interleaved series. This applies to every source in the run,
            # not just SMH, so the surveillance sources are featurized differently than in the same config without
            # SMH. To measure SMH's effect, compare against a baseline that also groups by season (see issue #36).
            group_cols = group_cols + ["season"]

        features: list[Feature] = []

        # Create directional wave features if enabled
        if self.model_config.use_directional_waves:
            features.append(
                DirectionalWaveFeature(
                    directions=self.model_config.wave_directions,
                    temporal_lags=self.model_config.wave_temporal_lags,
                    max_distance_km=self.model_config.wave_max_distance_km,
                    include_velocity=self.model_config.wave_include_velocity,
                    include_aggregate=self.model_config.wave_include_aggregate,
                )
            )

        features += [
            OneHotEncodingFeature(columns=["source", "agg_level", "location"]),
            HolidayFeature(),
            LagFeature(columns=["inc_trans_cs"], lags=[1, 2], group_columns=group_cols),
            TaylorFeature(column="inc_trans_cs", degree=2, window_sizes=[4, 6], group_columns=group_cols),
            TaylorFeature(column="inc_trans_cs", degree=1, window_sizes=[3, 5], group_columns=group_cols),
            RollingMeanFeature(column="inc_trans_cs", window_sizes=[2, 4], group_columns=group_cols),
            LagFeature(columns=None, lags=[1, 2], group_columns=group_cols),
            HorizonTargetFeature(column="inc_trans_cs", max_horizon=run_config.max_horizon, group_columns=group_cols),
        ]

        if not self.model_config.incl_level_feats:
            features.append(LevelFeatureFilter())

        return FeaturePipeline(features=features, initial_feat_names=initial_feats)

    def _fit_and_predict(self, df: pd.DataFrame, feat_names: list[str], run_config: RunConfig) -> pd.DataFrame:
        """Fit bagged LightGBM and return long-format predictions in inc_trans_cs space."""
        # keep only rows that are in-season
        if run_config.disease in (Disease.FLU, Disease.RSV):
            df = df.query("season_week >= 5 and season_week <= 45")

        # "test set" df used to generate look-ahead predictions: the main source's latest week. Taking
        # the max date across all sources would select only supplementary-source rows whenever the
        # main source lags another source (e.g. SMH always runs up to ref_date), leaving nothing to predict
        df_main = df.loc[df["source"] == self.model_config.main_source.value]
        df_test = df_main.loc[df_main.wk_end_date == df_main.wk_end_date.max()].copy()
        # "train set" df for model fitting; target value non-missing
        df_train = df.loc[~df["delta_target"].isna().values]

        # train model and obtain test set predictions
        if self.model_config.fit_locations_separately:
            # one fit per main-source location (df_test holds only main-source rows). By design, SMH rows
            # don't contribute to these fits: their synthetic locations ("syn-06") give them unique_ids
            # ("statesyn-06") that never match a real location's ("state06"), so SMH only informs jointly fit models
            if self._uses_smh():
                warnings.warn(SMH_UNUSED_WHEN_FITTING_SEPARATELY)
            unique_ids = df_test["unique_id"].unique()
            preds_df = pd.concat(
                [self._train_gbq_and_predict(run_config, df_train, df_test, feat_names, uid) for uid in unique_ids],
                axis=0,
            )
        else:
            preds_df = self._train_gbq_and_predict(run_config, df_train, df_test, feat_names)

        return preds_df

    def _train_gbq_and_predict(self, run_config, df_train, df_test, feat_names, unique_id=None):
        # filter to location if necessary
        if unique_id is not None:
            df_test = df_test.query(f'unique_id == "{unique_id}"')
            df_train = df_train.query(f'unique_id == "{unique_id}"')

        # get x and y
        x_test = df_test[feat_names]
        x_train = df_train[feat_names]
        y_train = df_train["delta_target"]

        # test set predictions:
        # same number of rows as df_test, one column per quantile level
        test_pred_qs_df = self._get_test_quantile_predictions(run_config, df_train, x_train, y_train, x_test)

        # add predictions to original test df
        df_test.reset_index(drop=True, inplace=True)
        df_test_w_preds = pd.concat([df_test, test_pred_qs_df], axis=1)

        # melt to get columns into rows, keeping only the things we need to invert data
        # transforms later on
        cols_to_keep = [
            "source",
            "agg_level",
            "location",
            "wk_end_date",
            "pop",
            "inc_trans_cs",
            "horizon",
            "inc_trans_center_factor",
            "inc_trans_scale_factor",
        ]
        preds_df = df_test_w_preds[cols_to_keep + run_config.q_labels]
        preds_df = preds_df.loc[preds_df["source"] == self.model_config.main_source.value]
        preds_df = pd.melt(preds_df, id_vars=cols_to_keep, var_name="output_type_id", value_name="delta_hat")

        # value in inc_trans_cs space (before inverse transform)
        preds_df["value"] = preds_df["inc_trans_cs"] + preds_df["delta_hat"]
        preds_df = preds_df.drop(columns=["delta_hat"])

        # Sort quantiles to prevent crossing (in transformed space, which is monotone)
        gcols = ["source", "agg_level", "location", "wk_end_date", "horizon"]
        preds_df = self._quantile_noncrossing(preds_df, gcols=gcols)

        return preds_df

    def _get_test_quantile_predictions(self, run_config, df_train, x_train, y_train, x_test):
        rng = np.random.default_rng(seed=_ref_date_seed(run_config))
        # seeds for lgb model fits, one per combination of bag and quantile level
        lgb_seeds = rng.integers(1e8, size=(self.model_config.num_bags, len(run_config.q_levels)))

        train_seasons = df_train["season"].unique()

        # indices of observations in each bag; drawn from `rng` sequentially by bag number so the
        # random sequence (and thus the resulting bags) doesn't depend on how fits are parallelized below
        bag_obs_inds_by_bag = [
            df_train["season"].isin(
                rng.choice(
                    train_seasons, size=int(len(train_seasons) * self.model_config.bag_frac_samples), replace=False
                )
            )
            for _ in range(self.model_config.num_bags)
        ]

        def _fit_bag_quantile(b, q_ind, q_level):
            bag_obs_inds = bag_obs_inds_by_bag[b]
            # n_jobs=1: fits run concurrently across bags/quantiles via the Parallel call below, so each
            # individual lgb fit is kept single-threaded to avoid oversubscribing CPU cores
            model = lgb.LGBMRegressor(
                verbosity=-1, objective="quantile", alpha=q_level, random_state=lgb_seeds[b, q_ind], n_jobs=1
            )
            model.fit(X=x_train.loc[bag_obs_inds, :], y=y_train.loc[bag_obs_inds])
            feat_importance_df = pd.DataFrame(
                {"feat": x_train.columns, "importance": model.feature_importances_, "b": b, "q_level": q_level}
            )
            return model.predict(X=x_test), feat_importance_df

        # fits are independent across (bag, quantile level) pairs, so run them concurrently; "threads"
        # avoids pickling/copying x_train/y_train per task, and lightgbm releases the GIL during fitting
        results = Parallel(n_jobs=-1, prefer="threads", verbose=1)(
            delayed(_fit_bag_quantile)(b, q_ind, q_level)
            for b in range(self.model_config.num_bags)
            for q_ind, q_level in enumerate(run_config.q_levels)
        )

        test_preds_by_bag = np.empty((x_test.shape[0], self.model_config.num_bags, len(run_config.q_levels)))
        feat_importance = []
        result_iter = iter(results)
        for b in range(self.model_config.num_bags):
            for q_ind in range(len(run_config.q_levels)):
                preds, feat_importance_df = next(result_iter)
                test_preds_by_bag[:, b, q_ind] = preds
                feat_importance.append(feat_importance_df)

        # combine and save feature importance scores
        if self.model_config.save_feat_importance:
            feat_importance_df = pd.concat(feat_importance, axis=0)
            save_path = build_save_path(
                root=run_config.artifact_store_root,
                run_config=run_config,
                model_config=self.model_config,
                subdir="feat_importance",
            )
            feat_importance_df.to_csv(save_path, index=False)

        # combined predictions across bags: median
        test_pred_qs = np.median(test_preds_by_bag, axis=1)
        # test predictions as a data frame, one column per quantile level
        test_pred_qs_df = pd.DataFrame(test_pred_qs)
        test_pred_qs_df.columns = run_config.q_labels
        return test_pred_qs_df

    def _quantile_noncrossing(self, preds_df: pd.DataFrame, gcols: list[str]) -> pd.DataFrame:
        # Sort rows so quantile labels are ascending within each group, then sort values the same way. Positional
        # alignment after reset_index pairs the smallest value with the smallest quantile label, fixing any crossings.
        # All non-gcol columns (inc_trans_center_factor, inc_trans_scale_factor, pop, …) are preserved — only "value" is
        # reassigned.
        preds_df = preds_df.sort_values(gcols + ["output_type_id"]).reset_index(drop=True)
        preds_df["value"] = preds_df.groupby(gcols)["value"].transform(np.sort)
        return preds_df
