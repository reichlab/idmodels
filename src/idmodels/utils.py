# This is a copy of ../gbq/utils.py
# with updated model_name choices
# In a future refactor, we should consolidate

import datetime


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
