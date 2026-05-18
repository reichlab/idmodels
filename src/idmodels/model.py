from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from iddata.ancillary.population import PopulationData
from iddata.loader import DiseaseDataLoader
from iddata.sources.base import DataSource as IdDataSource

# Import SourceType from iddata via the re-export in config
from idmodels.config import ModelConfig, PowerTransform, RunConfig, SourceType
from idmodels.features import FeaturePipeline
from idmodels.transforms import (
    CenterScaleTransform,
    ComposedTransform,
    FourthRootTransform,
    IdentityTransform,
    Transform,
)
from idmodels.utils import build_save_path


class IDModel(ABC):
    """
    Abstract base class for infectious disease forecast models. Subclasses implement _build_sources(),
    _build_feature_pipeline(), and _fit_and_predict(). run() orchestrates the full workflow.
    """


    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config


    def run(self, run_config: RunConfig) -> None:
        """Load data, generate predictions, and save to file."""
        sources = self._build_sources(run_config)
        df = DiseaseDataLoader().load(sources=sources, as_of=run_config.ref_date, ancillary=[PopulationData()])
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
        Fit model and generate quantile predictions.

        Returns long-format DataFrame with columns:
            source, agg_level, location, wk_end_date, pop, horizon,
            inc_trans_cs, inc_trans_center_factor, inc_trans_scale_factor,
            output_type_id, value (in inc_trans_cs space)
        """
        ...


    def _build_transform(self) -> Transform:
        """Default: ComposedTransform([power_transform, CenterScaleTransform()])."""
        if self.model_config.power_transform == PowerTransform.FOURTH_ROOT:
            power_t: Transform = FourthRootTransform()
        else:
            power_t = IdentityTransform()
        return ComposedTransform([power_t, CenterScaleTransform()])


    def _filter_locations(self, df: pd.DataFrame, run_config: RunConfig) -> pd.DataFrame:
        if not run_config.states and not run_config.hsas:
            raise ValueError("RunConfig must specify at least one state or HSA.")

        df_states = df.loc[(df["location"].isin(run_config.states)) & (df["agg_level"] != "hsa")]
        df_hsas = df.loc[(df["location"].isin(run_config.hsas)) & (df["agg_level"] == "hsa")]
        return pd.concat([df_states, df_hsas], join="inner", axis=0)


    def _invert_and_scale(self, preds_df: pd.DataFrame, transform: Transform, run_config: RunConfig) -> pd.DataFrame:
        """Inverse transform predictions then convert to original units."""
        preds_df["value"] = transform.invert(preds_df["value"].values, context=preds_df)
        preds_df["value"] = np.maximum(preds_df["value"], 0.0)

        if SourceType.NHSN in self.model_config.sources:
            preds_df["value"] = preds_df["value"] * preds_df["pop"] / 100000
        elif SourceType.NSSP in self.model_config.sources:
            preds_df["value"] = np.minimum(preds_df["value"] / 100, 1.0)

        return preds_df


    def _format_output(self, preds_df: pd.DataFrame, run_config: RunConfig) -> pd.DataFrame:
        """Reshape to FluSight hub submission format."""
        if SourceType.NHSN in self.model_config.sources:
            target_name = f"wk inc {run_config.disease.value} hosp"
        else:
            target_name = f"wk inc {run_config.disease.value} prop ed visits"

        preds_df["target_end_date"] = preds_df["wk_end_date"] + pd.to_timedelta(
            7 * preds_df["horizon"], unit="days"
        )
        preds_df["reference_date"] = run_config.ref_date
        preds_df["horizon"] = (
                pd.to_timedelta(preds_df["target_end_date"].dt.date - run_config.ref_date).dt.days / 7
        ).astype(int)
        preds_df["output_type"] = "quantile"
        preds_df["target"] = target_name
        preds_df.drop(columns="wk_end_date", inplace=True)

        req_cols = ["location", "reference_date", "horizon", "target_end_date",
                    "target", "output_type", "output_type_id", "value"]

        preds_df["geo_level"] = np.where(
            preds_df["agg_level"] == "national", "state", preds_df["agg_level"]
        )
        if len(preds_df["geo_level"].unique()) > 1:
            req_cols = ["agg_level"] + req_cols

        return preds_df.sort_values(
            ["output_type_id", "horizon", "location", "agg_level"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)[req_cols]
