import numpy as np
import pandas as pd
from iddata.loader import DiseaseDataLoader
from iddata.utils import get_holidays
from sarix import sarix

from idmodels.config import DataSource, PowerTransform, SARIXFourierModelConfig
from idmodels.utils import build_save_path


class SARIXModel():
    def __init__(self, model_config):
        self.model_config = model_config

    def _get_extra_sarix_params(self, df):
        """Return extra parameters to pass to SARIX constructor. Returns empty dict by default."""
        extra_params = {}

        # Add innovation distribution parameters if specified
        if hasattr(self.model_config, "innovation_dist"):
            extra_params["innovation_dist"] = self.model_config.innovation_dist
        if hasattr(self.model_config, "innovation_df_prior_scale"):
            extra_params["innovation_df_prior_scale"] = self.model_config.innovation_df_prior_scale

        return extra_params

    def run(self, run_config):
        valid_sources = {DataSource.NHSN, DataSource.NSSP}
        if not set(self.model_config.sources) <= valid_sources:
            raise ValueError("For SARIX, the only supported data sources are 'nhsn' or 'nssp'.")

        # Check if both nhsn and nssp data are included as sources
        if (DataSource.NHSN in self.model_config.sources) and (DataSource.NSSP in self.model_config.sources):
            raise ValueError("Only one of 'nhsn' or 'nssp' may be selected as a data source.")

        fdl = DiseaseDataLoader()
        if DataSource.NHSN in self.model_config.sources:
            df = fdl.load_data(nhsn_kwargs={"as_of": run_config.ref_date, "disease": run_config.disease},
                               sources=self.model_config.sources,
                               power_transform=self.model_config.power_transform)
            target_name = "wk inc " + run_config.disease + " hosp"
        elif DataSource.NSSP in self.model_config.sources:
            df = fdl.load_data(nssp_kwargs={"as_of": run_config.ref_date, "disease": run_config.disease},
                               sources=self.model_config.sources,
                               power_transform=self.model_config.power_transform)
            target_name = "wk inc " + run_config.disease + " prop ed visits"

        if (run_config.states == []) & (run_config.hsas == []):
            raise ValueError("User must request a non-empty set of locations to forecast for.")

        df_states = df.loc[(df["location"].isin(run_config.states)) & (df["agg_level"] != "hsa")]
        df_hsas = df.loc[(df["location"].isin(run_config.hsas)) & (df["agg_level"] == "hsa")]
        df = pd.concat([df_states, df_hsas], join = "inner", axis = 0)
        df["unique_id"] = df["agg_level"] + df["location"]

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
   
        # missing values are interpolated when possible
        xy_colnames = self.model_config.x + ["inc_trans_cs"]

        df = df.query("wk_end_date >= '2022-10-01'").interpolate()
        batched_xy = df[xy_colnames].values.reshape(len(df["unique_id"].unique()), -1, len(xy_colnames))

        # Get any extra parameters for the SARIX constructor
        extra_params = self._get_extra_sarix_params(df)

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
            forecast_horizon=run_config.max_horizon,
            num_warmup=self.model_config.num_warmup,
            num_samples=self.model_config.num_samples,
            num_chains=self.model_config.num_chains,
            **extra_params
        )

        pred_qs = _np_percentile(sarix_fit_all_locs_theta_pooled.predictions[..., :, :, 0],
                                 np.array(run_config.q_levels) * 100, axis=0)
        
        df_data_last_obs = df.groupby(["unique_id", "agg_level"]).tail(1)
        
        preds_df = pd.concat([
            pd.DataFrame(pred_qs[i, :, :]) \
            .set_axis(df_data_last_obs["unique_id"], axis="index") \
            .set_axis(np.arange(1, run_config.max_horizon+1), axis="columns") \
            .assign(output_type_id = q_label) \
            for i, q_label in enumerate(run_config.q_labels)
        ]) \
        .reset_index() \
        .melt(["unique_id", "output_type_id"], var_name="horizon") \
        .merge(df_data_last_obs, on="unique_id", how="left")
        
        # build data frame with predictions on the original scale
        preds_df["value"] = (preds_df["value"] + preds_df["inc_trans_center_factor"]) * preds_df["inc_trans_scale_factor"]
        if self.model_config.power_transform == PowerTransform.FOURTH_ROOT:
            preds_df["value"] = np.maximum(preds_df["value"], 0.0) ** 4
        else:
            preds_df["value"] = np.maximum(preds_df["value"], 0.0) ** 2
        
        preds_df["value"] = (preds_df["value"] - 0.01 - 0.75**4)
        preds_df["value"] = np.maximum(preds_df["value"], 0.0)
        
        if "nhsn" in preds_df["source"].unique():
            # turn nhsn rates back into counts
            preds_df["value"] = preds_df["value"] * preds_df["pop"] / 100000
        
        if target_name == "wk inc " + run_config.disease + " prop ed visits":
            preds_df["value"] = preds_df["value"] / 100 # percentage to proportion
            preds_df["value"] = np.minimum(preds_df["value"], 1.0)
        
        # keep just required columns and rename to match hub format
        req_cols = ["location", "wk_end_date", "horizon", "output_type_id", "value"]
        
        # we count national as state since it is coded using the same 2-digit fips code
        preds_df["geo_level"] = np.where(preds_df["agg_level"] == "national", "state", preds_df["agg_level"])
        if len(preds_df["geo_level"].unique()) > 1:
            req_cols.insert(0, "agg_level")
        
        preds_df = preds_df[req_cols]
        
        preds_df["target_end_date"] = preds_df["wk_end_date"] + pd.to_timedelta(7*preds_df["horizon"], unit="days")
        preds_df["reference_date"] = run_config.ref_date
        preds_df["horizon"] = (pd.to_timedelta(preds_df["target_end_date"].dt.date - run_config.ref_date).dt.days / 7).astype(int)
        preds_df["output_type"] = "quantile"
        preds_df["target"] = target_name
        preds_df.drop(columns="wk_end_date", inplace=True)

        # save
        save_path = build_save_path(
            root=run_config.output_root,
            run_config=run_config,
            model_config=self.model_config
        )
        # Ensure output_type_id is string to avoid pandas inferring it as float when reading
        preds_df["output_type_id"] = preds_df["output_type_id"].astype(str)
        preds_df.to_csv(save_path, index=False)


class SARIXFourierModel(SARIXModel):
    """
    SARIX model with Fourier seasonality terms.

    Adds annual seasonal patterns using Fourier harmonics to the base SARIX model.

    Required model_config parameters:
    - fourier_K: Number of Fourier harmonic pairs (int)
    - fourier_pooling: How to share Fourier coefficients across locations ('none' or 'shared')
    """
    def __init__(self, model_config):
        if not isinstance(model_config, SARIXFourierModelConfig):
            raise TypeError(
                f"SARIXFourierModel requires a SARIXFourierModelConfig, got {type(model_config).__name__}"
            )
        super().__init__(model_config)

    def _get_extra_sarix_params(self, df):
        """Return Fourier-specific parameters for SARIX constructor."""
        # Get base parameters (includes innovation_dist if specified)
        extra_params = super()._get_extra_sarix_params(df)

        # Extract day-of-year from dates for Fourier features
        # Take the first location's dates (same for all locations after reshaping)
        day_of_year = df.groupby("location")["wk_end_date"].apply(lambda x: x.dt.dayofyear.values).iloc[0]

        # Add Fourier-specific parameters
        extra_params.update({
            "day_of_year": day_of_year,
            "fourier_K": self.model_config.fourier_K,
            "fourier_pooling": self.model_config.fourier_pooling
        })

        return extra_params


def _np_percentile(predictions, q_levels, axis):
    """
    Simple helper function to ease patching from unit tests.
    """
    return np.percentile(predictions, q_levels, axis)
