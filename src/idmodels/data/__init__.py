"""Geographic reference data for spatial analysis and sample forecast labeling."""

from functools import lru_cache
from importlib import resources

import pandas as pd


@lru_cache(maxsize=1)
def load_state_abbreviations() -> dict[str, str]:
    """Map state/national FIPS codes ("01", ..., "US") to postal abbreviations ("AL", ..., "US")."""
    with resources.files("idmodels.data").joinpath("state_abbreviations.csv").open("r") as f:
        df = pd.read_csv(f, dtype=str)
    return dict(zip(df["fips"], df["abbreviation"]))
