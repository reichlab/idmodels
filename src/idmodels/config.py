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
    NHSN peak; `supplementary_sources` supply additional historical seasons used only for training.
    """

    model_name: str
    supplementary_sources: list[SourceType] = field(default_factory=lambda: [SourceType.ILINET, SourceType.FLUSURVNET])
    # add the synchrony and burden-to-date features (idmodels.peak.series.SYNC_BURDEN_FEATURES); used by GBQR and hier
    sync_burden_features: bool = False
    # (source, location) pairs never used for training. ILINet for Puerto Rico and the US Virgin Islands is zero for
    # whole seasons or has implausible pre-season values (sparse lab testing), so it is excluded.
    exclude_training_series: list[tuple[str, str]] = field(default_factory=lambda: [("ilinet", "72"), ("ilinet", "78")])
    # season weeks (1 = MMWR week 31) bounding the window in which the peak is defined. 10 and 43 correspond to the
    # FluSight 2026/27 peak-week dates (2026-10-10 through 2027-05-29)
    window_start_week: int = 10
    window_end_week: int = 43
    # earliest season week of the most recent observation for which training rows are built
    replay_start_week: int = 5
    # minimum number of non-missing in-window weeks for a historical season to be used for training
    min_window_obs: int = 25
    # data revision (backfill) Monte Carlo
    num_revision_draws: int = 200
    revision_max_lag: int = 10
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
    bag_frac_samples: float = 0.7
    # timing classes are {already peaked, 1, ..., max_k weeks ahead, more than max_k weeks ahead}
    max_k: int = 12
    # boost each peak-size quantile regression from the conditional-climatology baseline's quantile (LightGBM
    # init_score) instead of the unconditional quantile. Removes implausibly wide late-season upper tails, but scored
    # slightly worse overall in the 2023/24-2025/26 hindcasts, so it is off by default.
    size_offset: bool = False
    # passed through to lightgbm
    n_estimators: int = 100
    learning_rate: float = 0.05
    min_child_samples: int = 50


@dataclass
class PeakKCDEModelConfig(PeakModelConfig):
    # weight on the conditional-climatology baseline in the predictive mixture
    baseline_mix: float = 0.05
    # number of training rows used as queries when selecting bandwidths
    num_tuning_rows: int = 2000
    max_tuning_iter: int = 200
    # analogs are drawn only from training rows within this many season weeks of the current week
    sw_radius: int = 4


@dataclass
class PeakHierModelConfig(PeakModelConfig):
    # numbers of cubic B-spline basis functions for the current week (already-peaked probability and size), the
    # calendar week of a candidate peak (hazard) and the log of the offset m (hazard)
    df_t: int = 6
    df_week: int = 8
    df_offset: int = 5
    # power applied to every training row's likelihood: the ~39 replay rows of a series share one outcome, so a
    # weight of about 0.15 lets each series count as roughly 6 observations
    likelihood_weight: float = 0.15
    # use only origins with season_week % origin_stride == 0 in training (speeds up fitting); the likelihood weight of
    # each row is multiplied by origin_stride, so each series keeps the same total weight
    origin_stride: int = 3
    # scale of the half-normal priors on the standard deviations of the location and season effects
    effect_scale: float = 0.5
    # scale on which z | k >= 1 is modeled: "sqrt" (sqrt(z) ~ normal truncated to (0, inf); sqrt(z) is close to
    # symmetric given k in the training data) or "log" (log(max(z, z_floor)) ~ Student-t if student_t, else normal)
    size_scale: str = "sqrt"
    student_t: bool = True
    location_effects: bool = True
    season_effects: bool = True
    # let the feature effects on the already-peaked probability and the hazard vary linearly with the current week
    # (feature x (t - 24) / 10 interactions); what a feature means changes over the season (e.g. weeks since the max)
    time_varying_coefs: bool = False
    # add indicators for 0, 1, 2 and 3 weeks since the running maximum to the features
    wsm_dummies: bool = False
    # learn the current season's effect from right-censored observations of the current season
    current_season_update: bool = True
    # components of the current season's effect (0: already peaked, 1: hazard, 2: size) updated from the censored
    # observations. Those observations all say "not yet peaked" at earlier weeks, which is lopsided evidence for
    # component 0 (in development it lowered the already-peaked probability at the peak itself).
    current_update_components: list[int] = field(default_factory=lambda: [0, 1, 2])
    # z below this is set to it before taking logs
    z_floor: float = 0.01
    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 2
    target_accept_prob: float = 0.85
    max_tree_depth: int = 8
    progress_bar: bool = False
    # sampler geometry: centered (rather than non-centered) random effects, a dense mass matrix for the fixed effects,
    # and the number of Adam steps used to find a MAP starting point for NUTS (0: start from prior medians)
    centered_effects: bool = True
    # LKJ-correlated components of the location and season effects (else independent components). The correlations
    # are barely identified from ~20 seasons and pushed the posterior mode toward +/-1, which made sampling fail.
    correlated_effects: bool = False
    # which components (0: already peaked, 1: hazard, 2: size) have location / season effects, when not correlated.
    # In the development fits the location SDs of components 0 and 2 were near zero (and slowed sampling).
    location_components: list[int] = field(default_factory=lambda: [1])
    season_components: list[int] = field(default_factory=lambda: [0, 1, 2])
    # season-effect components that are non-centered even when centered_effects (the already-peaked component's SD
    # has posterior mass near zero, a funnel for the centered form)
    season_noncentered_components: list[int] = field(default_factory=lambda: [0])
    # decorrelate the standardized features before fitting (see PeakHierModel._setup_design)
    whiten_features: bool = True
    dense_mass: bool = False
    init_map_steps: int = 0
    # start NUTS at the MAP with the inverse Hessian there as a dense mass matrix; with laplace_mass,
    # adapt_mass_matrix = False keeps that metric fixed and adapts only the step size. Did not help in the development
    # fits (the joint mode of a centered hierarchical model is degenerate in the effect SDs), so off by default.
    laplace_mass: bool = False
    adapt_mass_matrix: bool = False
    # posterior draws used for prediction, and joint (k, z) draws per posterior draw and series
    num_posterior_draws: int = 200
    size_samples_per_draw: int = 5
    laplace_iters: int = 12
