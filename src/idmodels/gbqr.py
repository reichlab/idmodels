import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from iddata.loader import DiseaseDataLoader
from tqdm.autonotebook import tqdm

from idmodels.preprocess import create_features_and_targets
from idmodels.utils import build_save_path


class GBQRModel():
    def __init__(self, model_config):
        self.model_config = model_config
    
    
    def run(self, run_config):
        """
        Load flu data, generate predictions from a gbqr model, and save them as a csv file.
        
        Parameters
        ----------
        run_config: configuration object with settings for the run
        """
        # load flu data
        if self.model_config.reporting_adj:
            ilinet_kwargs = None
            flusurvnet_kwargs = None
        else:
            ilinet_kwargs = {"scale_to_positive": False}
            flusurvnet_kwargs = {"burden_adj": False}
        
        fdl = DiseaseDataLoader()
        df = fdl.load_data(nhsn_kwargs={"as_of": run_config.ref_date, "disease": run_config.disease},
                           ilinet_kwargs=ilinet_kwargs,
                           flusurvnet_kwargs=flusurvnet_kwargs,
                           sources=self.model_config.sources,
                           power_transform=self.model_config.power_transform)
        if run_config.locations is not None:
            df = df.loc[df["location"].isin(run_config.locations)]
        
        # augment data with features and target values
        if run_config.disease == "flu":
            init_feats = ["inc_trans_cs", "season_week", "log_pop"]
        elif run_config.disease == "covid":
            init_feats = ["inc_trans_cs", "log_pop"]
        
        df, feat_names = create_features_and_targets(
            df = df,
            incl_level_feats=self.model_config.incl_level_feats,
            max_horizon=run_config.max_horizon,
            curr_feat_names=init_feats)
        
        # keep only rows that are in-season
        if run_config.disease == "flu":
            df = df.query("season_week >= 5 and season_week <= 45")
        
        # "test set" df used to generate look-ahead predictions
        df_test = df.loc[df.wk_end_date == df.wk_end_date.max()] \
            .copy()
        
        # "train set" df for model fitting; target value non-missing
        df_train = df.loc[~df["delta_target"].isna().values]
        
        # train model and obtain test set predictinos
        if self.model_config.fit_locations_separately:
            locations = df_test["location"].unique()
            preds_df = [
                self._train_gbq_and_predict(
                    run_config,
                    df_train, df_test, feat_names, location
                ) for location in locations
            ]
            preds_df = pd.concat(preds_df, axis=0)
        else:
            preds_df = self._train_gbq_and_predict(
                run_config,
                df_train, df_test, feat_names
            )
        
        # save
        save_path = build_save_path(
            root=run_config.output_root,
            run_config=run_config,
            model_config=self.model_config
        )
        preds_df.to_csv(save_path, index=False)


    def _train_gbq_and_predict(self, run_config,
                               df_train, df_test, feat_names, location = None):
        """
        Train gbq model and get predictions on the original target scale,
        formatted in the FluSight hub format.
        
        Parameters
        ----------
        run_config: configuration object with settings for the run
        df_train: data frame with training data
        df_test: data frame with test data
        feat_names: list of names of columns with features
        location: optional string of location to fit to. Default, None, fits to all locations
        
        Returns
        -------
        Pandas data frame with test set predictions in FluSight hub format
        """
        # filter to location if necessary
        if location is not None:
            df_test = df_test.query(f'location == "{location}"')
            df_train = df_train.query(f'location == "{location}"')
        
        # get x and y
        x_test = df_test[feat_names]
        x_train = df_train[feat_names]
        y_train = df_train["delta_target"]
        
        # test set predictions:
        # same number of rows as df_test, one column per quantile level
        test_pred_qs_df = self._get_test_quantile_predictions(
            run_config,
            df_train, x_train, y_train, x_test
        )
        
        # add predictions to original test df
        df_test.reset_index(drop=True, inplace=True)
        df_test_w_preds = pd.concat([df_test, test_pred_qs_df], axis=1)
        
        # melt to get columns into rows, keeping only the things we need to invert data
        # transforms later on
        cols_to_keep = ["source", "location", "wk_end_date", "pop",
                        "inc_trans_cs", "horizon",
                        "inc_trans_center_factor", "inc_trans_scale_factor"]
        preds_df = df_test_w_preds[cols_to_keep + run_config.q_labels]
        preds_df = preds_df.loc[(preds_df["source"] == "nhsn")]
        preds_df = pd.melt(preds_df,
                        id_vars=cols_to_keep,
                        var_name="quantile",
                        value_name = "delta_hat")
        
        # build data frame with predictions on the original scale
        preds_df["inc_trans_cs_target_hat"] = preds_df["inc_trans_cs"] + preds_df["delta_hat"]
        preds_df["inc_trans_target_hat"] = (preds_df["inc_trans_cs_target_hat"] + preds_df["inc_trans_center_factor"]) * (preds_df["inc_trans_scale_factor"] + 0.01)
        if self.model_config.power_transform == "4rt":
            inv_power = 4
        elif self.model_config.power_transform is None:
            inv_power = 1
        else:
            raise ValueError('unsupported power_transform: must be "4rt" or None')
        
        preds_df["value"] = (np.maximum(preds_df["inc_trans_target_hat"], 0.0) ** inv_power - 0.01 - 0.75**4) * preds_df["pop"] / 100000
        preds_df["value"] = np.maximum(preds_df["value"], 0.0)
        
        # get predictions into the format needed for FluSight hub submission
        preds_df = self._format_as_flusight_output(preds_df, run_config.ref_date)
        
        # sort quantiles to avoid quantile crossing
        preds_df = self._quantile_noncrossing(
            preds_df,
            gcols = ["location", "reference_date", "horizon", "target_end_date",
                    "target", "output_type"]
        )
        
        return preds_df


    def _get_test_quantile_predictions(self, run_config,
                                       df_train, x_train, y_train, x_test):
        """
        Train the model on bagged subsets of the training data and obtain
        quantile predictions. This is the heart of the method.
        
        Parameters
        ----------
        run_config: configuration object with settings for the run
        df_train: Pandas data frame with training data
        x_train: numpy array with training instances in rows, features in columns
        y_train: numpy array with target values
        x_test: numpy array with test instances in rows, features in columns
        
        Returns
        -------
        Pandas data frame with test set predictions. The number of rows matches
        the number of rows of `x_test`. The number of columns matches the number
        of quantile levels for predictions as specified in the `run_config`.
        Column names are given by `run_config.q_labels`.
        """
        # seed for random number generation, based on reference date
        rng_seed = int(time.mktime(run_config.ref_date.timetuple()))
        rng = np.random.default_rng(seed=rng_seed)
        # seeds for lgb model fits, one per combination of bag and quantile level
        lgb_seeds = rng.integers(1e8, size=(self.model_config.num_bags, len(run_config.q_levels)))
        
        # training loop over bags
        test_preds_by_bag = np.empty((x_test.shape[0], self.model_config.num_bags, len(run_config.q_levels)))
        
        train_seasons = df_train["season"].unique()
        
        feat_importance = list()
        
        for b in tqdm(range(self.model_config.num_bags), "Bag number"):
            # get indices of observations that are in bag
            bag_seasons = rng.choice(
                train_seasons,
                size = int(len(train_seasons) * self.model_config.bag_frac_samples),
                replace=False)
            bag_obs_inds = df_train["season"].isin(bag_seasons)
            
            for q_ind, q_level in enumerate(run_config.q_levels):
                # fit to bag
                model = lgb.LGBMRegressor(
                    verbosity=-1,
                    objective="quantile",
                    alpha=q_level,
                    random_state=lgb_seeds[b, q_ind])
                model.fit(X=x_train.loc[bag_obs_inds, :], y=y_train.loc[bag_obs_inds])

                feat_importance.append(
                    pd.DataFrame({
                        "feat": x_train.columns,
                        "importance": model.feature_importances_,
                        "b": b,
                        "q_level": q_level
                    })
                )
                
                # test set predictions
                test_preds_by_bag[:, b, q_ind] = model.predict(X=x_test)
        
        # combine and save feature importance scores
        if run_config.save_feat_importance:
            feat_importance = pd.concat(feat_importance, axis=0)
            save_path = build_save_path(
                root=run_config.artifact_store_root,
                run_config=run_config,
                model_config=self.model_config,
                subdir="feat_importance")
            feat_importance.to_csv(save_path, index=False)
        
        # combined predictions across bags: median
        test_pred_qs = np.median(test_preds_by_bag, axis=1)
        
        # test predictions as a data frame, one column per quantile level
        test_pred_qs_df = pd.DataFrame(test_pred_qs)
        test_pred_qs_df.columns = run_config.q_labels
        
        return test_pred_qs_df


    def _format_as_flusight_output(self, preds_df, ref_date):
        # keep just required columns and rename to match hub format
        preds_df = preds_df[["location", "wk_end_date", "horizon", "quantile", "value"]] \
            .rename(columns={"quantile": "output_type_id"})
        
        preds_df["target_end_date"] = preds_df["wk_end_date"] + pd.to_timedelta(7*preds_df["horizon"], unit="days")
        preds_df["reference_date"] = ref_date
        preds_df["horizon"] = (pd.to_timedelta(preds_df["target_end_date"].dt.date - ref_date).dt.days / 7).astype(int)
        preds_df["target"] = "wk inc flu hosp"
        
        preds_df["output_type"] = "quantile"
        preds_df.drop(columns="wk_end_date", inplace=True)
        
        return preds_df


    def _quantile_noncrossing(self, preds_df, gcols):
        """
        Sort predictions to be in alignment with quantile levels, to prevent
        quantile crossing.
        
        Parameters
        ----------
        preds_df: data frame with quantile predictions
        gcols: columns to group by; predictions will be sorted within those groups
        
        Returns
        -------
        Sorted version of preds_df, guaranteed not to have quantile crossing
        """
        g = preds_df.set_index(gcols).groupby(gcols)
        preds_df = g[["output_type_id", "value"]] \
            .transform(lambda x: x.sort_values()) \
            .reset_index()
        
        return preds_df
