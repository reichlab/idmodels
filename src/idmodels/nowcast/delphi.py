"""
As-of data from the Delphi Epidata archive (https://delphi.cmu.edu/epidata/v5/archive/), which records every published
value of a signal together with the date it was published.

This is a stopgap until iddata reads Delphi data itself; it is used for data that the iddata snapshots do not have,
such as the number of hospitals reporting to NHSN, and for the history of NHSN revisions.
"""

import functools
import io
import os
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
from iddata.utils import load_fips_mappings

DELPHI_ARCHIVE_URL = "https://delphi.cmu.edu/epidata/v5/archive/"
# an API key is optional (requests without one are rate-limited); it is read from this environment variable, the
# one epidatr and epidatpy use
DELPHI_API_KEY_ENV = "DELPHI_EPIDATA_KEY"
NHSN_HOSP_REPORTING_SIGNAL = "hosprep_confirmed_admissions_flu_ew"
NHSN_ADMISSIONS_SIGNAL = "confirmed_admissions_flu_ew"


def fetch_archive(source: str, signal: str, geo_type: str, timeout: float = 300) -> pd.DataFrame:
    """
    Every published value of a Delphi signal for all locations of a geo_type. Returns the archive's columns (signal,
    report_time, geo_type, geo_value, fill_method, reference_time, value).
    """
    query = urllib.parse.urlencode({"source": source, "signal": signal, "geo_type": geo_type, "format": "csv"})
    request = urllib.request.Request(f"{DELPHI_ARCHIVE_URL}?{query}")
    if key := os.environ.get(DELPHI_API_KEY_ENV):
        request.add_header("token", key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            archive = pd.read_csv(io.BytesIO(response.read()), dtype={"geo_value": str})
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Delphi archive request for {source}/{signal}/{geo_type} failed: {e.code} {e.reason}"
        ) from e
    if archive.empty:
        # the archive answers an unknown signal with an empty table rather than an error
        raise RuntimeError(f"Delphi archive returned no data for {source}/{signal}/{geo_type}")
    return archive


@functools.lru_cache(maxsize=4)
def _nhsn_archive(signal: str) -> pd.DataFrame:
    archive = pd.concat(
        [fetch_archive("nhsn", signal, geo_type) for geo_type in ["nation", "state"]],
        ignore_index=True,
    )
    # keep the published values (other fill methods are Delphi's imputations)
    archive = archive.loc[archive["fill_method"] == "source"]
    fips = load_fips_mappings().astype({"location": str}).set_index("abbreviation")["location"]
    location = archive["geo_value"].str.upper().map(fips)
    if location.isna().any():
        unknown = sorted(archive.loc[location.isna(), "geo_value"].unique())
        raise ValueError(f"Delphi locations without a FIPS code: {unknown}")
    return pd.DataFrame(
        {
            "location": location,
            "wk_end_date": pd.to_datetime(archive["reference_time"]),
            "report_date": pd.to_datetime(archive["report_time"]).dt.tz_localize(None).dt.normalize(),
            "value": archive["value"].astype(float),
        }
    ).reset_index(drop=True)


def _nhsn_archive_as(signal: str, name: str) -> pd.DataFrame:
    return _nhsn_archive(signal).rename(columns={"value": name})


def nhsn_hosp_reporting_archive() -> pd.DataFrame:
    """
    As-of archive of the number of hospitals reporting flu admissions to NHSN, by location and week, from the Delphi
    archive (published values from 2024-12 on): columns location (FIPS code, or "US"), wk_end_date, report_date (the
    date the value was published) and hosp_reporting. Fetched once per process.
    """
    return _nhsn_archive_as(NHSN_HOSP_REPORTING_SIGNAL, "hosp_reporting")


def nhsn_admissions_archive() -> pd.DataFrame:
    """
    As-of archive of NHSN flu admissions (counts), by location and week, from the Delphi archive: columns location,
    wk_end_date, report_date and admissions, as for nhsn_hosp_reporting_archive(). Fetched once per process.
    """
    return _nhsn_archive_as(NHSN_ADMISSIONS_SIGNAL, "admissions")
