import io
from unittest.mock import patch

import pandas as pd
import pytest

from idmodels.nowcast import delphi

_ARCHIVE = {
    "nation": """signal,report_time,geo_type,geo_value,fill_method,reference_time,value
hosprep_confirmed_admissions_flu_ew,2026-01-14T00:00:00Z,nation,us,source,2026-01-10,4000
hosprep_confirmed_admissions_flu_ew,2026-01-21T00:00:00Z,nation,us,source,2026-01-10,5000
""",
    "state": """signal,report_time,geo_type,geo_value,fill_method,reference_time,value
hosprep_confirmed_admissions_flu_ew,2026-01-14T00:00:00Z,state,al,source,2026-01-10,80
hosprep_confirmed_admissions_flu_ew,2026-01-14T00:00:00Z,state,ak,imputed,2026-01-10,9
""",
}


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _urlopen(geo_types=_ARCHIVE):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        geo_type = next(g for g in geo_types if f"geo_type={g}" in request.full_url)
        return _Response(geo_types[geo_type].encode())

    return urlopen, requests


@pytest.fixture(autouse=True)
def _fresh_cache():
    delphi._nhsn_archive.cache_clear()
    yield
    delphi._nhsn_archive.cache_clear()


@pytest.fixture
def _fips():
    fips = pd.DataFrame({"abbreviation": ["US", "AL", "AK"], "location": ["US", "01", "02"]})
    with patch("idmodels.nowcast.delphi.load_fips_mappings", return_value=fips):
        yield


def test_nhsn_hosp_reporting_archive(_fips, monkeypatch):
    monkeypatch.setenv(delphi.DELPHI_API_KEY_ENV, "secret")
    urlopen, requests = _urlopen()
    with patch("urllib.request.urlopen", urlopen):
        archive = delphi.nhsn_hosp_reporting_archive()
        delphi.nhsn_hosp_reporting_archive()  # cached: no second fetch
    assert len(requests) == 2
    assert all(r.get_header("Token") == "secret" for r in requests)
    assert all(f"signal={delphi.NHSN_HOSP_REPORTING_SIGNAL}" in r.full_url for r in requests)
    # imputed values are dropped; locations are FIPS codes; dates are naive days
    expected = pd.DataFrame(
        {
            "location": ["US", "US", "01"],
            "wk_end_date": pd.to_datetime(["2026-01-10"] * 3),
            "report_date": pd.to_datetime(["2026-01-14", "2026-01-21", "2026-01-14"]),
            "hosp_reporting": [4000.0, 5000.0, 80.0],
        }
    )
    pd.testing.assert_frame_equal(archive, expected)


def test_no_api_key_header_without_key(_fips, monkeypatch):
    monkeypatch.delenv(delphi.DELPHI_API_KEY_ENV, raising=False)
    urlopen, requests = _urlopen()
    with patch("urllib.request.urlopen", urlopen):
        delphi.nhsn_hosp_reporting_archive()
    assert all(r.get_header("Token") is None for r in requests)


def test_empty_archive_and_unknown_locations_raise(_fips):
    header = _ARCHIVE["state"].splitlines()[0] + "\n"
    with patch("urllib.request.urlopen", _urlopen({"nation": header, "state": header})[0]):
        with pytest.raises(RuntimeError, match="no data"):
            delphi.nhsn_hosp_reporting_archive()
    unknown = {"nation": _ARCHIVE["nation"], "state": _ARCHIVE["state"].replace(",al,", ",zz,")}
    with patch("urllib.request.urlopen", _urlopen(unknown)[0]):
        with pytest.raises(ValueError, match="zz"):
            delphi.nhsn_hosp_reporting_archive()
