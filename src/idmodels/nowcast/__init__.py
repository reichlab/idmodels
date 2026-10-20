from idmodels.config import NowcastConfig, ScaleByCompletenessConfig, ScaleByStateConfig
from idmodels.nowcast.base import Nowcaster
from idmodels.nowcast.completeness import ScaleByCompletenessNowcaster
from idmodels.nowcast.state import ScaleByStateNowcaster

# nowcaster class for each nowcast config class
_NOWCASTERS: dict[type[NowcastConfig], type[Nowcaster]] = {
    ScaleByCompletenessConfig: ScaleByCompletenessNowcaster,
    ScaleByStateConfig: ScaleByStateNowcaster,
}


def build_nowcaster(config: NowcastConfig) -> Nowcaster:
    """The Nowcaster for a nowcast configuration."""
    if type(config) not in _NOWCASTERS:
        raise ValueError(f"no nowcaster for {type(config).__name__}")
    return _NOWCASTERS[type(config)](config)


__all__ = ["Nowcaster", "ScaleByCompletenessNowcaster", "ScaleByStateNowcaster", "build_nowcaster"]
