import numpy as np
import pandas as pd
from iddata.sources.nhsn import NHSNDataSource
from iddata.sources.nssp import NSSPDataSource
from sarix import sarix

from idmodels.config import RunConfig, SARIXFourierModelConfig, SARIXModelConfig, SourceType
from idmodels.features import FeaturePipeline
from idmodels.model import IDModel
from idmodels.utils import ref_date_seed


class SARIXModel(IDModel):
    """SARIX (Bayesian ARIMA with exogenous covariates) forecast model."""

    supports_sample_forecasts = True


    def __init__(self, model_config: SARIXModelConfig):
        # Narrow self.model_config from ModelConfig to SARIXModelConfig so that
        # type checkers resolve SARIXModelConfig-specific attributes in this class.
        super().__init__(model_config)
        self.model_config: SARIXModelConfig = model_config


    def _build_sources(self, run_config: RunConfig):
        sources_map = {SourceType.NHSN: NHSNDataSource(disease=run_config.disease),
                       SourceType.NSSP: NSSPDataSource(disease=run_config.disease)}
        if self.model_config.main_source not in sources_map:
            raise ValueError("SARIXModel only supports NHSN and NSSP as main source.")

        return [sources_map[self.model_config.main_source]]


    def _build_feature_pipeline(self, run_config: RunConfig) -> FeaturePipeline:
        return FeaturePipeline(features=[],
                               initial_feat_names=["inc_trans_cs"] + self.model_config.x)


    def _fit_and_predict(self, df: pd.DataFrame, feat_names: list[str], run_config: RunConfig) -> pd.DataFrame:
        """Fit SARIX and return quantile (and optionally sample) predictions in long format (inc_trans_cs space)."""
        xy_colnames = self.model_config.x + ["inc_trans_cs"]

        # missing values are interpolated when possible
        df = _interpolate_by_location(df.query("wk_end_date >= '2022-10-01'"), xy_colnames)
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

        # posterior predictive draws, shape (num_samples * num_chains, n_locations, max_horizon)
        draws = np.asarray(sarix_fit.predictions[..., :, :, 0])
        pred_qs = _np_percentile(
            draws,
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
            .assign(output_type="quantile") \
            .merge(df_data_last_obs, on="unique_id", how="left")

        if run_config.num_sample_forecasts > 0:
            draw_inds = _select_draw_indices(draws.shape[0], run_config.num_sample_forecasts, ref_date_seed(run_config))
            samples_df = _draws_to_samples_df(draws, df_data_last_obs["unique_id"].values, draw_inds) \
                .assign(output_type="sample") \
                .merge(df_data_last_obs, on="unique_id", how="left")
            preds_df = pd.concat([preds_df, samples_df], ignore_index=True)

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


def _select_draw_indices(num_draws: int, num_sample_forecasts: int, seed: int) -> np.ndarray:
    """
    Choose which posterior predictive draws to output as samples, without replacement. Chosen once
    for all locations, so sample k in every location comes from the same posterior draw.
    """
    if num_sample_forecasts > num_draws:
        raise ValueError(f"num_sample_forecasts ({num_sample_forecasts}) exceeds the {num_draws} available SARIX "
                         f"posterior draws (num_samples * num_chains).")
    return np.random.default_rng(seed=seed).choice(num_draws, size=num_sample_forecasts, replace=False)


def _draws_to_samples_df(draws: np.ndarray, unique_ids: np.ndarray, draw_inds: np.ndarray) -> pd.DataFrame:
    """
    Reshape the selected draws (draws has shape (num_draws, n_locations, max_horizon), with locations in
    unique_ids order) into long format with columns unique_id, horizon (1-based), sample_index, value.
    sample_index is the 0-based position of each draw in draw_inds.
    """
    sampled = draws[draw_inds]
    n_samples, n_locations, max_horizon = sampled.shape
    return pd.DataFrame({
        "unique_id": np.tile(np.repeat(unique_ids, max_horizon), n_samples),
        "horizon": np.tile(np.arange(1, max_horizon + 1), n_samples * n_locations),
        "sample_index": np.repeat(np.arange(n_samples), n_locations * max_horizon),
        "value": sampled.reshape(-1),
    })


def _interpolate_by_location(df: pd.DataFrame, xy_colnames: list[str]) -> pd.DataFrame:
    """Interpolate missing values per location.

    Interpolating across the concatenated frame directly would blend values across location
    boundaries (a single-row gap at the end of one location's block would be filled from the
    next location's first row). limit_area="inside" leaves a genuine trailing gap (e.g. missing
    population for the latest week) as NaN rather than forward-filling it, so it can still be
    caught downstream instead of silently propagating a fabricated value.
    """
    df = df.copy()
    df[xy_colnames] = df.groupby("unique_id")[xy_colnames].transform(
        lambda g: g.interpolate(limit_area="inside")
    )
    return df
