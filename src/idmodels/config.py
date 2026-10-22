import datetime
import warnings
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from iddata.enums import (
    Disease,  # used internally for RunConfig.disease; import from iddata.enums directly in callers
    SourceType,  # re-exported for callers: from idmodels.config import SourceType
)

SMH_UNUSED_WHEN_FITTING_SEPARATELY = (
    "SMH is a source but fit_locations_separately=True: SMH data will not be used for fitting, since "
    "per-location fits only train on each real location's own rows and never include SMH's synthetic "
    "'syn-' locations."
)


class PowerTransform(str, Enum):
    FOURTH_ROOT = "4rt"
    NONE = "none"


class PoolingStrategy(str, Enum):
    NONE = "none"
    SHARED = "shared"


class WISARObjective(str, Enum):
    WIS = "wis"
    LIKELIHOOD = "likelihood"


class InnovationDist(str, Enum):
    GAUSSIAN = "gaussian"
    STUDENT_T = "studentt"


FLUSIGHT_Q_LEVELS = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                     0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]


@dataclass
class ModelConfig(ABC):
    """Abstract base for model configuration."""

    model_name: str
    main_source: SourceType
    fit_locations_separately: bool
    power_transform: PowerTransform

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

    # smh trajectory filters. SMH rows are only used when fit_locations_separately=False: per-location
    # fits train on each real location's own rows, which never include SMH's synthetic "syn-" locations
    smh_model: list[str] = field(default_factory=list)
    # explicit output_type_ids (trajectory sample ids), applied as-is across every round and location.
    # Note that ids only mean "the same trajectory everywhere" in round 4: rounds 5+ scope each id to a
    # single location, so an explicit list there typically keeps SMH rows for just one location. To get
    # trajectories for every location, use smh_num_otid instead.
    smh_otid: list[str] = field(default_factory=list)
    # alternative to smh_otid: randomly sample this many output_type_ids from those available
    # for smh_model, since they're arbitrary per model and not necessarily sequential. Sampled
    # independently within each (round, location, model), so every location gets trajectories
    smh_num_otid: int | None = None
    # seed for smh_num_otid sampling; None uses GBQR's ref_date-based seed. Sampled ids for the requested
    # locations are saved under artifact_store_root/UMass-{model_name}/smh_otid/ when artifact_store_root is set
    smh_otid_seed: int | None = None

    # directional wave features (disabled by default)
    use_directional_waves: bool = False
    wave_directions: list[str] = field(default_factory=lambda: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    wave_temporal_lags: list[int] = field(default_factory=lambda: [1, 2])
    wave_max_distance_km: float = 1000.0
    wave_include_velocity: bool = False
    wave_include_aggregate: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.smh_otid and self.smh_num_otid:
            raise ValueError("Set at most one of smh_otid (an explicit list) and smh_num_otid (a random sample size).")
        # without this, smh_num_otid=0 would be falsy and silently mean "no trajectory filter" (all trajectories)
        if self.smh_num_otid is not None and self.smh_num_otid < 1:
            raise ValueError(f"smh_num_otid must be at least 1 (or None), got {self.smh_num_otid}.")
        # also warned at run time (GBQRModel._fit_and_predict), in case the config is changed after construction
        if self.fit_locations_separately and SourceType.SMH in [self.main_source] + self.supplementary_sources:
            warnings.warn(SMH_UNUSED_WHEN_FITTING_SEPARATELY, stacklevel=3)


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


@dataclass
class WISARModelConfig(ModelConfig):
    """
    Pooled AR(p) + Fourier seasonality model fit with PyMC, optionally under a WIS-loss Gibbs posterior. Defaults
    reproduce the WISAR6_fourthroot_adaptive_t model, except train_start (see below). Requires the `wisar` extra
    (pymc, pytensor).
    """

    objective: WISARObjective = WISARObjective.WIS
    p: int = 6
    fourier_K: int = 2
    theta_pooling: PoolingStrategy = PoolingStrategy.SHARED
    fourier_pooling: PoolingStrategy = PoolingStrategy.SHARED
    innovation_dist: InnovationDist = InnovationDist.STUDENT_T
    t_dof: float = 5.0  # fixed, not sampled: no gradient through the Student-t icdf w.r.t. degrees of freedom
    sigma_prior_scale: float = 1.0
    theta_sd_prior_scale: float = 1.0
    fourier_beta_sd_prior_scale: float = 1.0
    # quantile levels the WIS loss is evaluated at; part of the fitting objective, independent of RunConfig.q_levels
    wis_q_levels: list[float] = field(default_factory=lambda: list(FLUSIGHT_Q_LEVELS))
    # first week of training data; also the window over which the center/scale factors are computed. iddata's NHSN
    # series starts 2022-08-06 (WISAR6_fourthroot_adaptive_t was developed on hub target data starting 2022-02-05).
    train_start: datetime.date = datetime.date(2022, 8, 6)
    num_warmup: int = 2000
    num_samples: int = 2000
    num_chains: int = 2
    num_cores: int = 1
    target_accept: float = 0.99
    n_param_draws: int = 200
    n_innov_per_param: int = 5
    seed_fit: int = 9001
    seed_forecast: int = 9002
