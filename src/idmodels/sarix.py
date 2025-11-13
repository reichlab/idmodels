
import numpy as np
import pandas as pd
from iddata.loader import DiseaseDataLoader
from iddata.utils import get_holidays
from sarix import sarix

from idmodels.utils import build_save_path


class SARIXModel():
    def __init__(self, model_config):
        self.model_config = model_config

    def run(self, run_config):
        valid_sources = np.array(["nhsn", "nssp"])
        if not np.isin(np.array(self.model_config.sources), valid_sources).all():
            raise ValueError("For SARIX, the only supported data sources are 'nhsn' or 'nssp'.")
        
        # Check if both nhsn and nssp data are included as sources
        if all(src in self.model_config.sources for src in ["nhsn", "nssp"]):
            raise ValueError("Only one of 'nhsn' or 'nssp' may be selected as a data source.")

        fdl = DiseaseDataLoader()
        if "nhsn" in self.model_config.sources:
            df = fdl.load_data(nhsn_kwargs={"as_of": run_config.ref_date, "disease": run_config.disease},
                               sources=self.model_config.sources,
                               power_transform=self.model_config.power_transform)
            target_name = "wk inc " + run_config.disease + " hosp"
        elif "nssp" in self.model_config.sources:
            df = fdl.load_data(nssp_kwargs={"as_of": None, "disease": run_config.disease},
                               sources=self.model_config.sources,
                               power_transform=self.model_config.power_transform)
            target_name = "wk inc " + run_config.disease + " prop ed visits"

        if (run_config.states == []) & (run_config.hsas == []):
            raise ValueError("User must request a non-empty set of locations to forecast for.")

        if (run_config.states != []) & (run_config.hsas != []):
            raise NotImplementedError("Functionality for simultaneously forecasting state- and hsa-level locations is not yet implemented.")
        
        df_states = df.loc[(df["location"].isin(run_config.states)) & (df["agg_level"] != "hsa")]
        df_hsas = df.loc[(df["location"].isin(run_config.hsas)) & (df["agg_level"] == "hsa")]
        df = pd.concat([df_states, df_hsas], join = "inner", axis = 0)

        # season week relative to christmas
        df = df.merge(
            get_holidays() \
                .query("holiday == 'Christmas Day'") \
                .drop(columns=["holiday", "date"]) \
                .rename(columns={"season_week": "xmas_week"}),
            how="left",
            on="season") \
        .assign(delta_xmas = lambda x: x["season_week"] - x["xmas_week"])
        df["xmas_spike"] = np.maximum(3 - np.abs(df["delta_xmas"]), 0)
   
        xy_colnames = self.model_config.x + ["inc_trans_cs"]
        df = df.query("wk_end_date >= '2022-10-01'").interpolate()
        unique_locations = len(run_config.states) + len(run_config.hsas)
        batched_xy = df[xy_colnames].values.reshape(unique_locations, -1, len(xy_colnames))
        
        sarix_fit_all_locs_theta_pooled = sarix.SARIX(
            xy = batched_xy,
            p = self.model_config.p,
            d = self.model_config.d,
            P = self.model_config.P,
            D = self.model_config.D,
            season_period = self.model_config.season_period,
            transform="none", # transformations are handled outside of SARIX
            theta_pooling=self.model_config.theta_pooling,
            sigma_pooling=self.model_config.sigma_pooling,
            forecast_horizon = run_config.max_horizon,
            num_warmup = run_config.num_warmup,
            num_samples = run_config.num_samples,
            num_chains = run_config.num_chains
        )

        pred_qs = _np_percentile(sarix_fit_all_locs_theta_pooled.predictions[..., :, :, 0],
                                 np.array(run_config.q_levels) * 100, axis=0)
        
        df_data_last_obs = df.groupby(["location", "agg_level"]).tail(1)
        
        preds_df = pd.concat([
            pd.DataFrame(pred_qs[i, :, :]) \
            .set_axis(df_data_last_obs["location"], axis="index") \
            .set_axis(np.arange(1, run_config.max_horizon+1), axis="columns") \
            .assign(output_type_id = q_label) \
            for i, q_label in enumerate(run_config.q_labels)
        ]) \
        .reset_index() \
        .melt(["location", "output_type_id"], var_name="horizon") \
        .merge(df_data_last_obs, on="location", how="left")
        
        # build data frame with predictions on the original scale
        preds_df["value"] = (preds_df["value"] + preds_df["inc_trans_center_factor"]) * preds_df["inc_trans_scale_factor"]
        if self.model_config.power_transform == "4rt":
            preds_df["value"] = np.maximum(preds_df["value"], 0.0) ** 4
        else:
            preds_df["value"] = np.maximum(preds_df["value"], 0.0) ** 2
        
        preds_df["value"] = (preds_df["value"] - 0.01 - 0.75**4) * preds_df["pop"] / 100000
        preds_df["value"] = np.maximum(preds_df["value"], 0.0)
        
        # keep just required columns and rename to match hub format
        preds_df = preds_df[["location", "wk_end_date", "horizon", "output_type_id", "value"]]
        
        preds_df["target_end_date"] = preds_df["wk_end_date"] + pd.to_timedelta(7*preds_df["horizon"], unit="days")
        preds_df["reference_date"] = run_config.ref_date
        preds_df["horizon"] = (pd.to_timedelta(preds_df["target_end_date"].dt.date - run_config.ref_date).dt.days / 7).astype(int)
        preds_df["output_type"] = "quantile"
        preds_df["target"] = target_name
        preds_df.drop(columns="wk_end_date", inplace=True)
        
        if target_name == "wk inc " + run_config.disease + " prop ed visits":
            preds_df["value"] = preds_df["value"] / 100 # percentage to proportion
            preds_df["value"] = np.minimum(preds_df["value"], 1.0)

        # save
        save_path = build_save_path(
            root=run_config.output_root,
            run_config=run_config,
            model_config=self.model_config
        )
        preds_df.to_csv(save_path, index=False)


def _np_percentile(predictions, q_levels, axis):
    """
    Simple helper function to ease patching from unit tests.
    """
    return np.percentile(predictions, q_levels, axis)
