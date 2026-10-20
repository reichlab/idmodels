from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from iddata.ancillary.population import PopulationData
from iddata.loader import DiseaseDataLoader
from iddata.sources.base import DataSource as IdDataSource

# Import SourceType from iddata via the re-export in config
from idmodels.config import ModelConfig, PowerTransform, RunConfig, SourceType
from idmodels.constants import FLUSURVNET_FLOOR, FLUSURVNET_SCALE, ILINET_FLOOR, ILINET_SCALE, NHSN_FLOOR
from idmodels.features import FeaturePipeline
from idmodels.transforms import (
    CenterScaleTransform,
    ComposedTransform,
    FourthRootTransform,
    IdentityTransform,
    SourceScaleTransform,
    Transform,
)
from idmodels.utils import build_sample_output_type_ids, build_save_path


class IDModel(ABC):
    """
    Abstract base class for infectious disease forecast models. Subclasses implement _build_sources(),
    _build_feature_pipeline(), and _fit_and_predict(). run() orchestrates the full workflow.
    """

    # whether _fit_and_predict can produce sample rows; models that can't refuse runs that request samples
    supports_sample_forecasts: bool = False


    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config


    def run(self, run_config: RunConfig) -> None:
        """Load data, generate predictions, and save to file."""
        run_config.validate()
        # raise before loading any data, rather than silently write a file without the requested samples
        if run_config.num_sample_forecasts > 0 and not self.supports_sample_forecasts:
            raise NotImplementedError(f"{type(self).__name__} does not support sample forecasts yet; "
                                      f"set num_sample_forecasts=0.")

        sources = self._build_sources(run_config)
        df = DiseaseDataLoader().load(sources=sources, as_of=run_config.ref_date, ancillary=[PopulationData()])
        df = self._filter_sources_df(df, run_config)
        df = self._filter_locations(df, run_config)
        df["unique_id"] = df["agg_level"] + df["location"]

        transform = self._build_transform()
        df = transform.apply(df)

        pipeline = self._build_feature_pipeline(run_config)
        df, feat_names = pipeline.apply(df)

        preds_df = self._fit_and_predict(df, feat_names, run_config)

        preds_df = self._invert_and_scale(preds_df, transform, run_config)

        preds_df = self._format_output(preds_df, run_config)
        save_path = build_save_path(root=run_config.output_root,
                                    run_config=run_config,
                                    model_config=self.model_config)

        self._raise_if_incomplete(preds_df, run_config, save_path)

        # Ensure output_type_id is string to avoid pandas inferring it as float when reading
        preds_df["output_type_id"] = preds_df["output_type_id"].astype(str)
        preds_df.to_csv(save_path, index=False)


    @abstractmethod
    def _build_sources(self, run_config: RunConfig) -> list[IdDataSource]:
        """Instantiate iddata DataSource objects for this model."""
        ...


    @abstractmethod
    def _build_feature_pipeline(self, run_config: RunConfig) -> FeaturePipeline:
        """Return the FeaturePipeline for this model."""
        ...


    @abstractmethod
    def _fit_and_predict(self, df: pd.DataFrame, feat_names: list[str], run_config: RunConfig) -> pd.DataFrame:
        """
        Fit model and generate quantile predictions, plus sample predictions if
        run_config.num_sample_forecasts > 0.

        Returns long-format DataFrame with columns:
            source, agg_level, location, wk_end_date, pop, horizon,
            inc_trans_cs, inc_trans_center_factor, inc_trans_scale_factor,
            output_type ("quantile" or "sample"), output_type_id, value (in inc_trans_cs space)
        Quantile rows' output_type_id is the quantile label. Sample rows instead carry an integer
        sample_index (0-based draw number, consistent across locations) and are labeled later, by
        _format_output, according to run_config.sample_id_scheme.
        """
        ...


    def _build_transform(self) -> Transform:
        """Default: ComposedTransform([SourceScaleTransform, power_transform, CenterScaleTransform()])."""
        # process response variable:
        # - source-specific floor and scale (NHSN: +0.316; ILINet: *4; FluSurvNet: /2.5)
        # - fourth root transform to stabilize variability
        # - divide by location- and source-specific 95th percentile
        # - center relative to location- and source-specific mean
        #   (note non-standard order of center/scale)
        _SOURCE_SCALE_PARAMS = {
            SourceType.NHSN: (NHSN_FLOOR, 1.0),
            SourceType.ILINET: (ILINET_FLOOR, ILINET_SCALE),
            SourceType.FLUSURVNET: (FLUSURVNET_FLOOR, FLUSURVNET_SCALE),
        }
        all_sources = [self.model_config.main_source] + getattr(self.model_config, "supplementary_sources", [])
        source_scale_params = {
            src.value: params
            for src, params in _SOURCE_SCALE_PARAMS.items()
            if src in all_sources
        }
        if self.model_config.power_transform == PowerTransform.FOURTH_ROOT:
            power_t: Transform = FourthRootTransform()
        else:
            power_t = IdentityTransform()
        return ComposedTransform([SourceScaleTransform(source_scale_params), power_t, CenterScaleTransform()])


    def _filter_sources_df(self, df: pd.DataFrame, run_config: RunConfig) -> pd.DataFrame:
        """Optional hook for subclasses to filter/adjust the combined, loaded sources DataFrame."""
        return df


    def _filter_locations(self, df: pd.DataFrame, run_config: RunConfig) -> pd.DataFrame:
        if not run_config.states and not run_config.hsas:
            raise ValueError("RunConfig must specify at least one state or HSA.")

        # Restrict to state/national rows before matching on the location suffix: some sources
        # (e.g. ILINet's HHS region rows, "Region 1".."Region 10") have location strings whose last
        # two characters can coincidentally match a requested state FIPS code (e.g. "Region 10" vs.
        # Delaware's "10"), which would otherwise let non-state/national rows leak through.
        df_states = df.loc[
            (df["location"].str[-2:].isin(run_config.states)) & (df["agg_level"].isin(["state", "national"]))
        ]
        df_hsas = df.loc[(df["location"].isin(run_config.hsas)) & (df["agg_level"] == "hsa")]
        return pd.concat([df_states, df_hsas], join="inner", axis=0)


    def _invert_and_scale(self, preds_df: pd.DataFrame, transform: Transform, run_config: RunConfig) -> pd.DataFrame:
        """Inverse transform predictions then convert to original units."""
        preds_df["value"] = transform.invert(preds_df["value"].values, context=preds_df)
        preds_df["value"] = np.maximum(preds_df["value"], 0.0)

        if self.model_config.main_source == SourceType.NHSN:
            # turn nhsn rates back into counts
            preds_df["value"] = preds_df["value"] * preds_df["pop"] / 100000
        elif self.model_config.main_source == SourceType.NSSP:
            preds_df["value"] = np.minimum(preds_df["value"] / 100, 1.0)  # percentage to proportion

        return preds_df


    def _format_output(self, preds_df: pd.DataFrame, run_config: RunConfig) -> pd.DataFrame:
        """Reshape to FluSight hub submission format."""
        if self.model_config.main_source == SourceType.NHSN:
            target_name = f"wk inc {run_config.disease.value} hosp"
        elif self.model_config.main_source == SourceType.NSSP:
            target_name = f"wk inc {run_config.disease.value} prop ed visits"

        preds_df["target_end_date"] = preds_df["wk_end_date"] + pd.to_timedelta(
            7 * preds_df["horizon"], unit="days"
        )
        preds_df["reference_date"] = run_config.ref_date
        preds_df["horizon"] = (
                pd.to_timedelta(preds_df["target_end_date"].dt.date - run_config.ref_date).dt.days / 7
        ).astype(int)
        preds_df["target"] = target_name
        preds_df.drop(columns="wk_end_date", inplace=True)

        req_cols = ["location", "reference_date", "horizon", "target_end_date",
                    "target", "output_type", "output_type_id", "value"]

        # we count national as state since it is coded using the same 2-digit fips code
        preds_df["geo_level"] = np.where(
            preds_df["agg_level"] == "national", "state", preds_df["agg_level"]
        )
        if len(preds_df["geo_level"].unique()) > 1:
            req_cols = ["agg_level"] + req_cols

        # quantile rows first, then sample rows. Samples sort by their numeric index rather than their
        # label, since DRAW_NUMBER labels aren't zero-padded ("10" sorts before "2" as a string)
        sort_cols = ["horizon", "location", "agg_level"]
        ascending = [True, True, False]
        # either block may be empty: a run can be quantiles only or samples only (and samples-only rows have no
        # output_type_id column until they are labeled here)
        is_sample = preds_df["output_type"] == "sample"
        blocks = []
        # an empty preds_df passes through as-is, for _raise_if_incomplete() to report
        if (~is_sample).any() or preds_df.empty:
            blocks.append(preds_df.loc[~is_sample].sort_values(["output_type_id"] + sort_cols,
                                                               ascending=[True] + ascending))
        if is_sample.any():
            samples_df = preds_df.loc[is_sample].copy()
            samples_df["sample_index"] = samples_df["sample_index"].astype(int)
            samples_df["output_type_id"] = build_sample_output_type_ids(samples_df["location"],
                                                                        samples_df["sample_index"],
                                                                        run_config.sample_id_scheme)
            blocks.append(samples_df.sort_values(["sample_index"] + sort_cols, ascending=[True] + ascending))

        return pd.concat(blocks).reset_index(drop=True)[req_cols]


    def _raise_if_incomplete(self, preds_df: pd.DataFrame, run_config: RunConfig, save_path) -> None:
        """Raise if there are no predictions or any predicted value is NaN, refusing to silently write
        an incomplete submission.

        Missing upstream data (e.g. population for the latest week) can propagate as NaN all the
        way through a model without raising -- LightGBM and SARIX both happily predict from NaN
        features -- so this is the last point at which a bad run can still fail loudly instead of
        writing a CSV that looks complete but has an all-NaN or partially-NaN value column, or no
        rows at all.
        """
        if preds_df.empty:
            raise ValueError(
                f"No predictions for {self.model_config} at ref_date={run_config.ref_date}; "
                f"refusing to write {save_path}"
            )
        if preds_df["value"].isna().any():
            n_na = preds_df["value"].isna().sum()
            raise ValueError(
                f"{n_na}/{len(preds_df)} rows have NaN value for {self.model_config} at "
                f"ref_date={run_config.ref_date}; refusing to write {save_path}"
            )
