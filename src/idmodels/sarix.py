import numpy as np
import pandas as pd
from iddata.sources.nhsn import NHSNDataSource
from iddata.sources.nssp import NSSPDataSource
from sarix import sarix

from idmodels.calendar import calendar_labels
from idmodels.config import (
    PoolingStrategy,
    PowerTransform,
    RunConfig,
    SARIXFourierModelConfig,
    SARIXModelConfig,
    SourceType,
)
from idmodels.features import FeaturePipeline
from idmodels.model import IDModel


class SARIXModel(IDModel):
    """SARIX (Bayesian ARIMA with exogenous covariates) forecast model."""


    def __init__(self, model_config: SARIXModelConfig):
        # Narrow self.model_config from ModelConfig to SARIXModelConfig so that
        # type checkers resolve SARIXModelConfig-specific attributes in this class.
        super().__init__(model_config)
        self.model_config: SARIXModelConfig = model_config
        # set by the last run(), for inspecting fits (e.g. posterior samples of calendar effects):
        # the fitted sarix.SARIX object, the unique_ids in batch order and the training weeks' end dates
        self.sarix_fit: sarix.SARIX | None = None
        self.fit_unique_ids: list[str] = []
        self.fit_wk_end_dates: pd.DatetimeIndex | None = None


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
        """Fit SARIX and return quantile predictions in long format (inc_trans_cs space)."""
        xy_colnames = self.model_config.x + ["inc_trans_cs"]

        # missing values are interpolated when possible
        df = _interpolate_by_location(df.query("wk_end_date >= '2022-10-01'"), xy_colnames)
        # checks that the reshape aligns weeks across locations, before the (slow) fit
        wk_end_dates = _batch_wk_end_dates(df)
        batched_xy = df[xy_colnames].values.reshape(
            len(df["unique_id"].unique()), -1, len(xy_colnames))

        # Get any extra parameters for the SARIX constructor
        extra_params = self._get_extra_sarix_params(df, run_config)

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
        self.sarix_fit = sarix_fit
        self.fit_unique_ids = list(df["unique_id"].unique())
        self.fit_wk_end_dates = wk_end_dates

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


    def _get_extra_sarix_params(self, df: pd.DataFrame, run_config: RunConfig) -> dict:
        """
        Extra parameters for the SARIX constructor. The base class returns calendar-effect parameters when
        model_config.calendar_holidays is non-empty and innovation-distribution parameters when
        model_config.innovation_dist is "t", and {} otherwise. Subclasses extend this.
        """
        cfg = self.model_config
        params = {}
        if cfg.innovation_dist == "t":
            params = {"innovation_dist": "t",
                      "innovation_df_pooling": _sarix_pooling(cfg.innovation_df_pooling),
                      "innovation_df_prior_scale": cfg.innovation_df_prior_scale,
                      "innovation_df_sd_prior_scale": cfg.innovation_df_sd_prior_scale}
        if not cfg.calendar_holidays:
            return params

        # labels for the training weeks followed by the forecast weeks, shared across locations;
        # only the target (last column of xy) gets calendar effects
        train_dates = _batch_wk_end_dates(df)
        fcast_dates = train_dates[-1] + pd.to_timedelta(7 * np.arange(1, run_config.max_horizon + 1), unit="D")
        target_labels = calendar_labels(train_dates.append(fcast_dates), cfg.calendar_holidays)
        labels = np.zeros((len(target_labels), len(cfg.x) + 1), dtype=int)
        labels[:, -1] = target_labels

        # xy's target is inc_trans_cs = a * u**power - center for a positive u on the original scale (incidence
        # plus floors/offsets) and a = 1 / (scale_factor + 0.01) > 0, with center and a constant within each
        # location. Tell SARIX the power and center so that calendar effects act as multipliers on u
        power = 0.25 if cfg.power_transform == PowerTransform.FOURTH_ROOT else 1.0
        center = df.groupby("unique_id", sort=False)["inc_trans_center_factor"].agg(["min", "max"])
        if not np.allclose(center["min"], center["max"]):
            raise ValueError("inc_trans_center_factor is not constant within each location")
        input_center = np.full((len(center), len(cfg.x) + 1), np.nan)  # x columns have no calendar effects
        input_center[:, -1] = center["min"].values

        return {**params,
                "calendar_labels": labels,
                "calendar_pooling": _sarix_pooling(cfg.calendar_pooling),
                "calendar_effect_sd_prior_scale": cfg.calendar_effect_sd_prior_scale,
                "calendar_effect_mu_prior_scale": cfg.calendar_effect_mu_prior_scale,
                "input_power": power,
                "input_center": input_center}


class SARIXFourierModel(SARIXModel):
    """Extends SARIXModel with Fourier seasonality terms."""


    def __init__(self, model_config: SARIXFourierModelConfig):
        if not isinstance(model_config, SARIXFourierModelConfig):
            raise TypeError(f"SARIXFourierModel requires a SARIXFourierModelConfig, got {type(model_config).__name__}")

        super().__init__(model_config)
        # Narrow self.model_config from SARIXModelConfig to SARIXFourierModelConfig so
        # that type checkers resolve fourier-specific attributes in this subclass.
        self.model_config: SARIXFourierModelConfig = model_config


    def _get_extra_sarix_params(self, df: pd.DataFrame, run_config: RunConfig) -> dict:
        # Extract day-of-year from dates for Fourier features
        # Take the first location's dates (same for all locations after reshaping)
        day_of_year = (
            df.groupby("location")["wk_end_date"]
            .apply(lambda x: x.dt.dayofyear.values)
            .iloc[0]
        )
        return {**super()._get_extra_sarix_params(df, run_config),
                "day_of_year": day_of_year,
                "fourier_K": self.model_config.fourier_K,
                "fourier_pooling": self.model_config.fourier_pooling}


def _batch_wk_end_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Week-ending dates of the time points of SARIX's xy, which is df reshaped to (unique_id, time, variable): the
    dates of the first unique_id's rows, checked to be the same for every unique_id, with each unique_id's rows
    contiguous in df.
    """
    if (df["unique_id"] != df["unique_id"].shift()).sum() != df["unique_id"].nunique():
        raise ValueError("each location's rows must be contiguous in df, so that they can be reshaped into xy")
    dates_by_id = df.groupby("unique_id", sort=False)["wk_end_date"].apply(lambda x: tuple(pd.to_datetime(x)))
    if dates_by_id.nunique() != 1:
        raise ValueError("locations do not all have the same weeks, so their dates cannot be aligned in xy")
    return pd.DatetimeIndex(dates_by_id.iloc[0])


def _sarix_pooling(value: PoolingStrategy) -> str:
    """
    The sarix calendar_pooling or innovation_df_pooling option for a PoolingStrategy. sarix deprecates "none" in
    favour of "unshared".
    """
    return {PoolingStrategy.NONE: "unshared",
            PoolingStrategy.SHARED: "shared",
            PoolingStrategy.PARTIAL: "partial"}[PoolingStrategy(value)]


def _np_percentile(predictions, q_levels, axis):
    """Helper to ease patching from unit tests."""
    return np.percentile(predictions, q_levels, axis)


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
