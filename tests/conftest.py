import time
import urllib.error

import pandas as pd
import pytest

# External hosts that iddata's PopulationData downloads from on every load (SEER's HSA crosswalk and
# Census county estimates). Every model run in the test suite reloads population data, so without
# caching, CI makes dozens of requests to these servers per run, and they intermittently fail or
# refuse connections from CI runners partway through the suite.
_CACHED_URL_PREFIXES = ("https://seer.cancer.gov/", "https://www2.census.gov/")
_DOWNLOAD_ATTEMPTS = 3


def _caching_reader(reader):
    cache = {}

    def read(path, *args, **kwargs):
        if not (isinstance(path, str) and path.startswith(_CACHED_URL_PREFIXES)):
            return reader(path, *args, **kwargs)

        key = (path, repr(args), repr(sorted(kwargs.items())))
        if key not in cache:
            for attempt in range(_DOWNLOAD_ATTEMPTS):
                try:
                    cache[key] = reader(path, *args, **kwargs)
                    break
                except urllib.error.URLError:
                    if attempt == _DOWNLOAD_ATTEMPTS - 1:
                        raise
                    time.sleep(5 * (attempt + 1))
        # copy so callers can't mutate the cached frame
        return cache[key].copy()

    return read


@pytest.fixture(scope="session", autouse=True)
def cache_population_downloads():
    """Download each external population file at most once per test session, with retries."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pd, "read_csv", _caching_reader(pd.read_csv))
        mp.setattr(pd, "read_excel", _caching_reader(pd.read_excel))
        yield
