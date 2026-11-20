"""
Holiday calendars for SARIX calendar effects.

Weekly data are indexed by week-ending (Saturday) dates. A holiday's week is the week whose Saturday week-ending
date is on or after the holiday and less than 7 days after it, i.e. the Sunday-Saturday week containing it.

Holidays are named by strings of the form "<name>" or "<name>+<k>" / "<name>-<k>", where the suffix shifts the
labelled week by k weeks (e.g. "new_year+1" is the week after the New Year week, to capture rebounds). Each
distinct string gets its own calendar label, 1, 2, ..., in the order given. Jan 1 is always 7 days after Dec 25, so
"new_year" is the same week as "christmas+1", and specs that name the same week cannot be combined.
"""

import datetime
import re

import numpy as np
import pandas as pd


def _thanksgiving(year: int) -> datetime.date:
    """US Thanksgiving: fourth Thursday of November."""
    nov1 = datetime.date(year, 11, 1)
    first_thu = nov1 + datetime.timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + datetime.timedelta(weeks=3)


HOLIDAYS = {
    "thanksgiving": _thanksgiving,
    "christmas": lambda year: datetime.date(year, 12, 25),
    "new_year": lambda year: datetime.date(year, 1, 1),
}

_SPEC_RE = re.compile(r"^(?P<name>[a-z_]+)(?P<shift>[+-]\d+)?$")


def parse_holiday_spec(spec: str) -> tuple[str, int]:
    """Split a holiday spec such as "christmas+1" into ("christmas", 1)."""
    m = _SPEC_RE.match(spec)
    if m is None or m["name"] not in HOLIDAYS:
        raise ValueError(f"invalid holiday spec {spec!r}: expected '<name>' or '<name>+<k>' with name in "
                         f"{sorted(HOLIDAYS)}")
    return m["name"], int(m["shift"] or 0)


def canonical_holiday_spec(spec: str) -> tuple[str, int]:
    """
    (name, shift) for a holiday spec, with specs that always label the same week mapped to the same value, e.g.
    "christmas" and "christmas+0", or "new_year+k" and "christmas+(k+1)" (Jan 1 is always 7 days after Dec 25).
    """
    name, shift = parse_holiday_spec(spec)
    if name == "new_year":
        return "christmas", shift + 1
    return name, shift


def holiday_week_end(holiday_date: datetime.date) -> datetime.date:
    """The Saturday week-ending date of the Sunday-Saturday week containing holiday_date."""
    # date.weekday(): Monday = 0, ..., Saturday = 5, Sunday = 6
    return holiday_date + datetime.timedelta(days=(5 - holiday_date.weekday()) % 7)


def holiday_week_ends(spec: str, years) -> list[datetime.date]:
    """Week-ending dates labelled by a holiday spec, one per year in years."""
    name, shift = parse_holiday_spec(spec)
    return [holiday_week_end(HOLIDAYS[name](y)) + datetime.timedelta(weeks=shift) for y in years]


def calendar_labels(wk_end_dates, holidays: list[str]) -> np.ndarray:
    """
    Integer calendar labels for a sequence of weekly week-ending dates.

    Parameters
    ----------
    wk_end_dates : array_like of dates
        Week-ending dates (all Saturdays), e.g. the training weeks followed by the forecast weeks.
    holidays : list of str
        Holiday specs (see module docstring). The i-th spec gets label i + 1.

    Returns
    -------
    np.ndarray of int with shape (len(wk_end_dates),): 0 for weeks with no holiday, otherwise the holiday's label.
    """
    if len(set(holidays)) != len(holidays):
        raise ValueError(f"holidays must be distinct, got {holidays}")
    dates = pd.to_datetime(pd.Series(wk_end_dates)).dt.date
    if (pd.to_datetime(dates).dt.weekday != 5).any():
        raise ValueError("wk_end_dates must all be Saturdays")

    # cover the years spanned by the dates, with one extra year on each side for shifted specs
    years = range(min(dates).year - 1, max(dates).year + 2)
    labels = np.zeros(len(dates), dtype=int)
    for label, spec in enumerate(holidays, start=1):
        hit = dates.isin(holiday_week_ends(spec, years)).values
        if (labels[hit] != 0).any():
            raise ValueError(f"holiday {spec!r} falls in the same week as another holiday in {holidays}")
        labels[hit] = label
    return labels
