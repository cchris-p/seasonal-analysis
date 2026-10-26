import pytest

from seasonal_analysis.data_sources import (
    DUKASCOPY,
    FIRSTRATEDATA,
    UnsupportedDataSource,
    is_decision_use_source,
    is_legacy_source,
    normalize_data_source,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("dukascopy", DUKASCOPY),
        ("Dukascopy", DUKASCOPY),
        ("dukascopy_gmt+0_us-dst", DUKASCOPY),
        ("DUKASCOPY_GMT+0_US-DST", DUKASCOPY),
        ("dukascopy gmt+0 us-dst", DUKASCOPY),
        ("firstratedata", FIRSTRATEDATA),
        ("FRD", FIRSTRATEDATA),
        ("", DUKASCOPY),
        (None, DUKASCOPY),
    ],
)
def test_normalize_data_source(raw, expected):
    assert normalize_data_source(raw) == expected


def test_normalize_rejects_unknown_token():
    with pytest.raises(UnsupportedDataSource):
        normalize_data_source("merged")


def test_source_classification_helpers():
    assert is_decision_use_source("dukascopy_gmt+0_us-dst")
    assert is_legacy_source("firstratedata")
    assert not is_decision_use_source("firstratedata")
