"""Conversions between dates and influenza seasons / season weeks (season week 1 is MMWR week 31)."""

import datetime

import pandas as pd
from iddata.utils import add_season_columns


def season_of(date: datetime.date) -> str:
    """The season (e.g. "2026/27") containing `date`."""
    return add_season_columns(pd.DataFrame({"wk_end_date": [str(date)], "location": ["US"]}))["season"].iloc[0]


def season_week_to_date(season: str, season_week: int) -> datetime.date:
    """The Saturday ending the given season week of the given season."""
    start_year = int(season[:4])
    # season week 1 is MMWR week 31; find the Saturday ending it by scanning late July / early August
    for day in range(20, 40):
        d = datetime.date(start_year, 7, 1) + datetime.timedelta(days=day)
        if d.weekday() == 5:
            sw = add_season_columns(pd.DataFrame({"wk_end_date": [str(d)], "location": ["US"]}))["season_week"].iloc[0]
            if sw == 1:
                return d + datetime.timedelta(weeks=season_week - 1)
    raise ValueError(f"could not locate season week 1 of {season}")
