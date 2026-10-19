import datetime
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from iddata.enums import (
    Disease,  # used internally for RunConfig.disease; import from iddata.enums directly in callers
    SourceType,  # re-exported for callers: from idmodels.config import SourceType
)


class PowerTransform(str, Enum):
    FOURTH_ROOT = "4rt"
    NONE = "none"


class PoolingStrategy(str, Enum):
    NONE = "none"
    SHARED = "shared"


@dataclass
class NowcastConfig(ABC):
    """
    Abstract base for nowcast configuration (see idmodels.nowcast). A nowcast replaces the most recent, still
    incomplete values of one source by estimates of their eventual (settled) values before a model is fit.
    """

    # the source whose most recent values are adjusted
    source: SourceType = SourceType.NHSN
    # number of most recent weeks adjusted for each location (1: the newest week only)
    num_weeks: int = 1

    def __post_init__(self):
        if type(self) is NowcastConfig:
            raise TypeError("NowcastConfig is abstract - use a specific nowcast config, e.g. ScaleByCompletenessConfig")


@dataclass
class ScaleByCompletenessConfig(NowcastConfig):
    """
    Divide each adjusted week's value by its completeness: the number of hospitals reporting that week divided by the
    usual number (the median number reporting the weeks `typical_lags` weeks before the newest week), both as known
    on the reference date.
    """

    # as-of archive of hospitals reporting: a csv or parquet file with columns location (FIPS code, or "US"),
    # wk_end_date, report_date (when the value was published) and hosp_reporting. Required.
    reporting_path: Path | None = None
    typical_lags: tuple[int, ...] = (2, 3, 4)
    # completeness is floored at this value, which caps the scale-up at 1 / min_completeness; it is capped at 1, so
    # values are never scaled down
    min_completeness: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        if self.reporting_path is None:
            raise ValueError("ScaleByCompletenessConfig requires reporting_path")


@dataclass
class ModelConfig(ABC):
    """Abstract base for model configuration."""

    model_name: str
    main_source: SourceType
    fit_locations_separately: bool
    power_transform: PowerTransform
    # optional nowcast of the most recent values, applied to the input data before the model is fit
    nowcast: NowcastConfig | None = None

    def __post_init__(self):
        if type(self) is ModelConfig:
            raise TypeError("ModelConfig is abstract - use SARIXModelConfig or GBQRModelConfig")


@dataclass
class RunConfig:
    """Run configuration: disease, locations, output paths, quantile levels."""

    disease: Disease
    ref_date: datetime.date
    output_root: Path
    artifact_store_root: Path | None
    max_horizon: int
    states: list[str]
    hsas: list[str]
    q_levels: list[float]
    q_labels: list[str]


@dataclass
class SARIXModelConfig(ModelConfig):
    p: int = 0
    P: int = 0
    d: int = 0
    D: int = 0
    season_period: int = 1
    theta_pooling: PoolingStrategy = PoolingStrategy.NONE
    sigma_pooling: PoolingStrategy = PoolingStrategy.NONE
    x: list = field(default_factory=list)
    num_warmup: int = 2000
    num_samples: int = 2000
    num_chains: int = 1


@dataclass
class SARIXFourierModelConfig(SARIXModelConfig):
    fourier_K: int = 1
    fourier_pooling: PoolingStrategy = PoolingStrategy.NONE


@dataclass
class GBQRModelConfig(ModelConfig):
    supplementary_sources: list[SourceType] = field(default_factory=list)
    incl_level_feats: bool = True
    num_bags: int = 100
    bag_frac_samples: float = 0.7
    reporting_adj: bool = False
    save_feat_importance: bool = False

    # directional wave features (disabled by default)
    use_directional_waves: bool = False
    wave_directions: list[str] = field(default_factory=lambda: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    wave_temporal_lags: list[int] = field(default_factory=lambda: [1, 2])
    wave_max_distance_km: float = 1000.0
    wave_include_velocity: bool = False
    wave_include_aggregate: bool = True


@dataclass
class PeakModelConfig:
    """
    Configuration shared by the direct seasonal-peak models (see idmodels.peak). These models always forecast the
    NHSN peak; `supplementary_sources` (ILINet and/or FluSurvNet) supply additional historical seasons used only for
    training.
    """

    model_name: str
    supplementary_sources: list[SourceType] = field(default_factory=lambda: [SourceType.ILINET, SourceType.FLUSURVNET])
    # (source, location) pairs never used for training. ILINet for Puerto Rico and the US Virgin Islands is zero for
    # whole seasons or has implausible pre-season values (sparse lab testing), so it is excluded.
    exclude_training_series: list[tuple[str, str]] = field(default_factory=lambda: [("ilinet", "72"), ("ilinet", "78")])
    # season weeks (1 = MMWR week 31) bounding the window in which the peak is defined. The FluSight window is the 34
    # weeks starting MMWR week 40, which is always season weeks 10-43 (e.g. 2025-10-04 through 2026-05-23 for 2025/26
    # and 2026-10-10 through 2027-05-29 for 2026/27), so these do not need to change from season to season
    window_start_week: int = 10
    window_end_week: int = 43
    # earliest season week of the most recent observation for which training rows are built
    replay_start_week: int = 5
    # minimum number of non-missing in-window weeks for a historical season to be used for training
    min_window_obs: int = 25
    # data revision (backfill) Monte Carlo
    num_revision_draws: int = 200
    revision_max_lag: int = 10
    # if False, raise an error when a requested location has no current-season NHSN data or population; if True,
    # omit such locations from the forecast with a warning (an error is still raised if no location has data)
    drop_missing_locations: bool = False
    # floor applied to every peak-week probability before renormalizing; guarantees a finite log score
    pmf_floor: float = 1e-4
    # standard deviation (in weeks) of the Gaussian kernel used to smooth the climatological peak-week distribution
    timing_smoothing_sd: float = 1.5
    # number of stratified probability levels per revision draw used to turn predicted z quantiles into samples
    num_size_levels: int = 200

    def __post_init__(self):
        if type(self) is PeakModelConfig:
            raise TypeError("PeakModelConfig is abstract - use one of the model-specific peak configs")


@dataclass
class PeakBaselineModelConfig(PeakModelConfig):
    # half-width (in season weeks) of the window of historical rows used for the empirical distribution of peak size
    size_week_halfwidth: int = 2


@dataclass
class PeakGBQRModelConfig(PeakModelConfig):
    num_bags: int = 25
    progress_bar: bool = True
    bag_frac_samples: float = 0.7
    # timing classes are {already peaked, 1, ..., max_k weeks ahead, more than max_k weeks ahead}
    max_k: int = 12
    # feature groups for the peak-size quantile regressions and the timing classifier, from "core"
    # (idmodels.peak.series.CORE_FEATURES), "sb" (synchrony and burden, SYNC_BURDEN_FEATURES) and "holiday"
    # (idmodels.peak.extra_features.FEATURE_GROUPS)
    size_feature_groups: list[str] = field(default_factory=lambda: ["core"])
    timing_feature_groups: list[str] = field(default_factory=lambda: ["core"])
    # passed through to lightgbm
    n_estimators: int = 100
    learning_rate: float = 0.05
    min_child_samples: int = 50
