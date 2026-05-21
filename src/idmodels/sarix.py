import numpy as np
import pandas as pd
from iddata.sources.nhsn import NHSNDataSource
from iddata.sources.nssp import NSSPDataSource
from sarix import sarix

from idmodels.config import RunConfig, SARIXFourierModelConfig, SARIXModelConfig, SourceType
from idmodels.features import FeaturePipeline
from idmodels.model import IDModel


class SARIXModel(IDModel):
    """SARIX (Bayesian ARIMA with exogenous covariates) forecast model."""


    def __init__(self, model_config: SARIXModelConfig):
        # Narrow self.model_config from ModelConfig to SARIXModelConfig so that
        # type checkers resolve SARIXModelConfig-specific attributes in this class.
        super().__init__(model_config)
        self.model_config: SARIXModelConfig = model_config


    def _build_sources(self, run_config: RunConfig):
        sources_map = {SourceType.NHSN: NHSNDataSource(disease=run_config.disease),
                       SourceType.NSSP: NSSPDataSource(disease=run_config.disease)}
        if not set(self.model_config.sources) <= sources_map.keys():
            raise ValueError("SARIXModel only supports NHSN and NSSP sources.")

        # Check if both nhsn and nssp data are included as sources
        if SourceType.NHSN in self.model_config.sources and SourceType.NSSP in self.model_config.sources:
            raise ValueError("Only one of NHSN or NSSP may be selected.")

        return [sources_map[s] for s in self.model_config.sources]


    def _build_feature_pipeline(self, run_config: RunConfig) -> FeaturePipeline:
        return FeaturePipeline(features=[],
                               initial_feat_names=["inc_trans_cs"] + self.model_config.x)


    def _fit_and_predict(self, df: pd.DataFrame, feat_names: list[str], run_config: RunConfig) -> pd.DataFrame:
        """Fit SARIX and return quantile predictions in long format (inc_trans_cs space)."""
        xy_colnames = self.model_config.x + ["inc_trans_cs"]

        # missing values are interpolated when possible
        df = df.query("wk_end_date >= '2022-10-01'").interpolate()
        batched_xy = df[xy_colnames].values.reshape(
            len(df["unique_id"].unique()), -1, len(xy_colnames))

        # Get any extra parameters for the SARIX constructor
        extra_params = self._get_extra_sarix_params(df)

        sarix_fit = sarix.SARIX(xy=batched_xy,
                                p=self.model_config.p,
                                d=self.model_config.d,
                                P=self.model_config.P,
                                D=self.model_config.D,
                                season_period=self.model_config.season_period,
                                transform="none",  # transformations are handled outside of SARIX
                                theta_pooling=self.model_config.theta_pooling,
                                sigma_pooling=self.model_config.sigma_pooling,
                                forecast_horizon=run_config.max_horizon,
                                num_warmup=self.model_config.num_warmup,
                                num_samples=self.model_config.num_samples,
                                num_chains=self.model_config.num_chains,
                                **extra_params)

        pred_qs = _np_percentile(
            sarix_fit.predictions[..., :, :, 0],
            np.array(run_config.q_levels) * 100,
            axis=0,
        )

        df_data_last_obs = df.groupby(["unique_id", "agg_level"]).tail(1)

        preds_df = pd.concat(
            [
                pd.DataFrame(pred_qs[i, :, :])
                .set_axis(df_data_last_obs["unique_id"], axis="index")
                .set_axis(np.arange(1, run_config.max_horizon + 1), axis="columns")
                .assign(output_type_id=q_label)
                for i, q_label in enumerate(run_config.q_labels)
            ]
        ).reset_index() \
            .melt(["unique_id", "output_type_id"], var_name="horizon") \
            .merge(df_data_last_obs, on="unique_id", how="left")

        # value is already in inc_trans_cs space (SARIX predicts inc_trans_cs directly)
        return preds_df


    def _get_extra_sarix_params(self, df: pd.DataFrame) -> dict:
        """Hook for subclasses. Returns {} by default."""
        return {}


class SARIXFourierModel(SARIXModel):
    """Extends SARIXModel with Fourier seasonality terms."""


    def __init__(self, model_config: SARIXFourierModelConfig):
        if not isinstance(model_config, SARIXFourierModelConfig):
            raise TypeError(f"SARIXFourierModel requires a SARIXFourierModelConfig, got {type(model_config).__name__}")

        super().__init__(model_config)
        # Narrow self.model_config from SARIXModelConfig to SARIXFourierModelConfig so
        # that type checkers resolve fourier-specific attributes in this subclass.
        self.model_config: SARIXFourierModelConfig = model_config


    def _get_extra_sarix_params(self, df: pd.DataFrame) -> dict:
        # Extract day-of-year from dates for Fourier features
        # Take the first location's dates (same for all locations after reshaping)
        day_of_year = (
            df.groupby("location")["wk_end_date"]
            .apply(lambda x: x.dt.dayofyear.values)
            .iloc[0]
        )
        return {"day_of_year": day_of_year,
                "fourier_K": self.model_config.fourier_K,
                "fourier_pooling": self.model_config.fourier_pooling}


def _np_percentile(predictions, q_levels, axis):
    """Helper to ease patching from unit tests."""
    return np.percentile(predictions, q_levels, axis)
