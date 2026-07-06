import math

POWER_TRANSFORM_OFFSET: float = 0.01

# Source-specific pre-transform parameters: applied as (inc + floor) * scale before the power transform
NHSN_FLOOR: float = 0.75**4
ILINET_FLOOR: float = math.exp(-7)
ILINET_SCALE: float = 4.0
FLUSURVNET_FLOOR: float = math.exp(-3)
FLUSURVNET_SCALE: float = 1.0 / 2.5

IN_SEASON_WEEK_MIN: int = 10
IN_SEASON_WEEK_MAX: int = 45
