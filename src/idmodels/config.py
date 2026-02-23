import datetime
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DataSource(str, Enum):
    NHSN = "nhsn"
    NSSP = "nssp"
    FLUSURVNET = "flusurvnet"
    ILINET = "ilinet"


class Disease(str, Enum):
    FLU = "flu"
    COVID = "covid"


class PowerTransform(str, Enum):
    FOURTH_ROOT = "4rt"


class PoolingStrategy(str, Enum):
    NONE = "none"
    SHARED = "shared"


@dataclass
class ModelConfig(ABC):
    """
    Abstract base for model configuration.

    Holds settings that describe *what* model to run and how it processes data (sources, transforms, pooling).
    Not instantiated directly - use :class:`SARIXModelConfig` or :class:`GBQRModelConfig`.
    """

    model_name: str
    sources: list[DataSource]
    fit_locations_separately: bool
    power_transform: PowerTransform

    def __post_init__(self):
        if type(self) is ModelConfig:
            raise TypeError("ModelConfig is abstract - use SARIXModelConfig or GBQRModelConfig")


@dataclass
class RunConfig(ABC):
    """
    Abstract base for run configuration.

    Holds settings that describe a single execution: which disease, which locations, output paths, quantile levels, etc.
    Not instantiated directly - use :class:`SARIXRunConfig` or :class:`GBQRRunConfig`.
    """

    disease: Disease
    ref_date: datetime.date
    output_root: Path
    artifact_store_root: Path | None
    max_horizon: int
    states: list[str]
    hsas: list[str]
    q_levels: list[float]
    q_labels: list[str]

    def __post_init__(self):
        if type(self) is RunConfig:
            raise TypeError("RunConfig is abstract - use SARIXRunConfig or GBQRRunConfig")


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


@dataclass
class SARIXFourierModelConfig(SARIXModelConfig):
    fourier_K: int = 1
    fourier_pooling: PoolingStrategy = PoolingStrategy.NONE


@dataclass
class GBQRModelConfig(ModelConfig):
    incl_level_feats: bool = True
    num_bags: int = 100
    bag_frac_samples: float = 0.7
    reporting_adj: bool = False
    # directional wave features (disabled by default)
    use_directional_waves: bool = False
    wave_directions: list[str] = field(default_factory=lambda: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    wave_temporal_lags: list[int] = field(default_factory=lambda: [1, 2])
    wave_max_distance_km: float = 1000.0
    wave_include_velocity: bool = False
    wave_include_aggregate: bool = True


@dataclass
class SARIXRunConfig(RunConfig):
    num_warmup: int = 2000
    num_samples: int = 2000
    num_chains: int = 1


@dataclass
class GBQRRunConfig(RunConfig):
    save_feat_importance: bool = False
