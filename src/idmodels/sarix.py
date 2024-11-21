
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
        fdl = DiseaseDataLoader()
        df = fdl.load_data(nhsn_kwargs={"as_of": run_config.ref_date, "disease": run_config.disease},
                           sources=self.model_config.sources,
                           power_transform=self.model_config.power_transform)
        if run_config.locations is not None:
            df = df.loc[df["location"].isin(run_config.locations)]
        
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
        
        xy_colnames = ["inc_trans_cs"] + self.model_config.x
        df = df.query("wk_end_date >= '2022-10-01'").interpolate()
        batched_xy = df[xy_colnames].values.reshape(len(df["location"].unique()), -1, len(xy_colnames))
        
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
        
        pred_qs = np.percentile(sarix_fit_all_locs_theta_pooled.predictions[..., :, :, 0],
                                np.array(run_config.q_levels) * 100, axis=0)
        
        df_nhsn_last_obs = df.groupby(["location"]).tail(1)
        
        preds_df = pd.concat([
            pd.DataFrame(pred_qs[i, :, :]) \
            .set_axis(df_nhsn_last_obs["location"], axis="index") \
            .set_axis(np.arange(1, run_config.max_horizon+1), axis="columns") \
            .assign(output_type_id = q_label) \
            for i, q_label in enumerate(run_config.q_labels)
        ]) \
        .reset_index() \
        .melt(["location", "output_type_id"], var_name="horizon") \
        .merge(df_nhsn_last_obs, on="location", how="left")
        
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
        preds_df["target"] = "wk inc flu hosp"
        preds_df.drop(columns="wk_end_date", inplace=True)
        
        # save
        save_path = build_save_path(
            root=run_config.output_root,
            run_config=run_config,
            model_config=self.model_config
        )
        preds_df.to_csv(save_path, index=False)
