import calendar

import lightgbm as lgb
import numpy as np
import pandas as pd
from iddata.enums import Disease
from iddata.sources.flusurvnet import FluSurvNetDataSource
from iddata.sources.ilinet import ILINetDataSource
from iddata.sources.nhsn import NHSNDataSource
from iddata.sources.nssp import NSSPDataSource
from tqdm.autonotebook import tqdm

from idmodels.config import GBQRModelConfig, RunConfig, SourceType
from idmodels.features import (
    DirectionalWaveFeature,
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


class GBQRModel(IDModel):
    """Gradient Boosted Quantile Regression forecast model."""


    def __init__(self, model_config: GBQRModelConfig):
        # Narrow self.model_config from ModelConfig to GBQRModelConfig so that
        # type checkers resolve GBQRModelConfig-specific attributes in this class.
        super().__init__(model_config)
        self.model_config: GBQRModelConfig = model_config


    def _build_sources(self, run_config: RunConfig):
        source_map = {SourceType.NHSN: NHSNDataSource(disease=run_config.disease),
                      SourceType.NSSP: NSSPDataSource(disease=run_config.disease),
                      SourceType.ILINET: ILINetDataSource(scale_to_positive=self.model_config.reporting_adj),
                      SourceType.FLUSURVNET: FluSurvNetDataSource(burden_adj=self.model_config.reporting_adj)}
        # concatenate + dedupe sources while preserving order so main_source is always first
        all_sources = list(dict.fromkeys([self.model_config.main_source] + self.model_config.training_sources))

        # Check if both nhsn and nssp data are included as sources
        if self.model_config.main_source not in [SourceType.NHSN, SourceType.NSSP]:
            raise ValueError("GBQRModel only supports NHSN and NSSP as main source.")

        return [source_map[s] for s in all_sources]


    def _build_feature_pipeline(self, run_config: RunConfig) -> FeaturePipeline:
        if run_config.disease in (Disease.FLU, Disease.RSV):
            initial_feats = ["inc_trans_cs", "season_week", "log_pop"]
        else:
            initial_feats = ["inc_trans_cs", "log_pop"]

        features = []

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
            LagFeature(columns=["inc_trans_cs"], lags=[1, 2]),
            TaylorFeature(column="inc_trans_cs", degree=2, window_sizes=[4, 6]),
            TaylorFeature(column="inc_trans_cs", degree=1, window_sizes=[3, 5]),
            RollingMeanFeature(column="inc_trans_cs", window_sizes=[2, 4]),
            LagFeature(columns=None, lags=[1, 2]),
            HorizonTargetFeature(column="inc_trans_cs", max_horizon=run_config.max_horizon),
        ]

        if not self.model_config.incl_level_feats:
            features.append(LevelFeatureFilter())

        return FeaturePipeline(features=features, initial_feat_names=initial_feats)


    def _fit_and_predict(self, df: pd.DataFrame, feat_names: list[str], run_config: RunConfig) -> pd.DataFrame:
        """Fit bagged LightGBM and return long-format predictions in inc_trans_cs space."""
        # keep only rows that are in-season
        if run_config.disease in (Disease.FLU, Disease.RSV):
            df = df.query("season_week >= 5 and season_week <= 45")

        # "test set" df used to generate look-ahead predictions
        df_test = df.loc[df.wk_end_date == df.wk_end_date.max()].copy()
        # "train set" df for model fitting; target value non-missing
        df_train = df.loc[~df["delta_target"].isna().values]

        # train model and obtain test set predictions
        if self.model_config.fit_locations_separately:
            unique_ids = df_test["unique_id"].unique()
            preds_df = pd.concat(
                [self._train_gbq_and_predict(run_config, df_train, df_test, feat_names, uid)
                 for uid in unique_ids],
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
        cols_to_keep = ["source", "agg_level", "location", "wk_end_date", "pop", "inc_trans_cs", "horizon",
                        "inc_trans_center_factor", "inc_trans_scale_factor"]
        preds_df = df_test_w_preds[cols_to_keep + run_config.q_labels]
        preds_df = preds_df.loc[preds_df["source"] == self.model_config.main_source.value]
        preds_df = pd.melt(preds_df,
                           id_vars=cols_to_keep,
                           var_name="output_type_id",
                           value_name="delta_hat")

        # value in inc_trans_cs space (before inverse transform)
        preds_df["value"] = preds_df["inc_trans_cs"] + preds_df["delta_hat"]
        preds_df = preds_df.drop(columns=["delta_hat"])

        # Sort quantiles to prevent crossing (in transformed space, which is monotone)
        gcols = ["source", "agg_level", "location", "wk_end_date", "horizon"]
        preds_df = self._quantile_noncrossing(preds_df, gcols=gcols)

        return preds_df


    def _get_test_quantile_predictions(self, run_config, df_train, x_train, y_train, x_test):
        # seed for random number generation, based on reference date
        rng_seed = int(calendar.timegm(run_config.ref_date.timetuple()))
        rng = np.random.default_rng(seed=rng_seed)
        # seeds for lgb model fits, one per combination of bag and quantile level
        lgb_seeds = rng.integers(1e8, size=(self.model_config.num_bags, len(run_config.q_levels)))

        test_preds_by_bag = np.empty((x_test.shape[0], self.model_config.num_bags, len(run_config.q_levels)))
        train_seasons = df_train["season"].unique()
        feat_importance = []

        # training loop over bags
        for b in tqdm(range(self.model_config.num_bags), "Bag number"):
            # get indices of observations that are in bag
            bag_seasons = rng.choice(train_seasons,
                                     size=int(len(train_seasons) * self.model_config.bag_frac_samples),
                                     replace=False)
            bag_obs_inds = df_train["season"].isin(bag_seasons)

            for q_ind, q_level in enumerate(run_config.q_levels):
                # fit to bag
                model = lgb.LGBMRegressor(verbosity=-1,
                                          objective="quantile",
                                          alpha=q_level,
                                          random_state=lgb_seeds[b, q_ind])
                model.fit(X=x_train.loc[bag_obs_inds, :], y=y_train.loc[bag_obs_inds])

                feat_importance.append(pd.DataFrame({"feat": x_train.columns,
                                                     "importance": model.feature_importances_,
                                                     "b": b,
                                                     "q_level": q_level}))
                # test set predictions
                test_preds_by_bag[:, b, q_ind] = model.predict(X=x_test)

        # combine and save feature importance scores
        if self.model_config.save_feat_importance:
            feat_importance_df = pd.concat(feat_importance, axis=0)
            save_path = build_save_path(root=run_config.artifact_store_root,
                                        run_config=run_config,
                                        model_config=self.model_config,
                                        subdir="feat_importance")
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
