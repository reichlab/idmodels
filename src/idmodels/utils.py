# This is a copy of ../gbq/utils.py
# with updated model_name choices
# In a future refactor, we should consolidate

import calendar
import datetime

import pandas as pd

from idmodels.config import RunConfig, SampleIdScheme
from idmodels.data import load_state_abbreviations


def validate_ref_date(ref_date):
    if ref_date is None:
        today = datetime.date.today()
        
        # next Saturday: weekly forecasts are relative to this date
        ref_date = today - datetime.timedelta((today.weekday() + 2) % 7 - 7)
        
        return ref_date
    elif isinstance(ref_date, datetime.date):
        # check that it's a Saturday
        if ref_date.weekday() != 5:
            raise ValueError("ref_date must be a Saturday")
        
        return ref_date
    else:
        raise TypeError("ref_date must be a datetime.date object")


def build_save_path(root, run_config, model_config, subdir=None):
    save_dir = root / f"UMass-{model_config.model_name}"
    if subdir is not None:
        save_dir = save_dir / subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"{str(run_config.ref_date)}-UMass-{model_config.model_name}.csv"


def ref_date_seed(run_config: RunConfig) -> int:
    """Seed for random number generation, based on reference date, so reruns for a date are reproducible."""
    return int(calendar.timegm(run_config.ref_date.timetuple()))


def build_sample_output_type_ids(location: pd.Series, sample_index: pd.Series,
                                 scheme: SampleIdScheme) -> pd.Series:
    """
    Label sample rows' output_type_ids from their location and 0-based sample_index:
    - STATE_ABBREVIATION: "{abbreviation}{sample_index:02d}", e.g. "US00", "AL99"
    - DRAW_NUMBER: "{sample_index + 1}", e.g. "1", "100"
    """
    if scheme == SampleIdScheme.STATE_ABBREVIATION:
        abbreviations = load_state_abbreviations()
        missing = set(location) - set(abbreviations)
        if missing:
            raise KeyError(f"No state abbreviation for location(s) {sorted(missing)}; "
                           f"sample_id_scheme={scheme.value} only supports states and national.")
        return location.map(abbreviations) + sample_index.map("{:02d}".format)
    elif scheme == SampleIdScheme.DRAW_NUMBER:
        return (sample_index + 1).astype(str)
    else:
        raise ValueError(f"Unsupported sample_id_scheme: {scheme}")
