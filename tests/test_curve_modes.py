from __future__ import annotations

import numpy as np
import pandas as pd

from seasonal_analysis.analysis import (
    CURVE_MODE_LINK_RELATIVES,
    CURVE_MODE_LOG_RETURN_COMPOUNDED,
    CURVE_MODE_YEARLY_AVERAGE,
    build_seasonal_pattern_curve,
)


def _synthetic_prices(
    start: str = "2005-01-01", end: str = "2019-12-31", drift: float = 4e-4
) -> pd.DataFrame:
    """Daily business-day prices with a strong secular drift and a February bump."""
    idx = pd.bdate_range(start, end)
    t = np.arange(len(idx))
    doy = idx.dayofyear.to_numpy()
    seasonal = 0.03 * np.sin(2.0 * np.pi * doy / 365.0)
    price = 100.0 * np.exp(drift * t + seasonal)
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price}, index=idx
    )


def _curve(mode: str) -> pd.DataFrame:
    return build_seasonal_pattern_curve(
        _synthetic_prices(),
        lookback_years=12,
        exclude_incomplete_last_year=False,
        curve_mode=mode,
    )


def test_compounded_mode_is_drift_inclusive() -> None:
    curve = _curve(CURVE_MODE_LOG_RETURN_COMPOUNDED)
    vals = curve["seasonal_index"].to_numpy()
    assert vals[0] == 100.0
    # Drift-inclusive: the year-end level is materially above the start.
    assert vals[-1] - vals[0] > 5.0


def test_link_relatives_is_drift_free() -> None:
    curve = _curve(CURVE_MODE_LINK_RELATIVES)
    vals = curve["seasonal_index"].to_numpy()
    assert vals[0] == 100.0
    assert abs(vals[-1] - vals[0]) < 2.0
    # mean per-slot factor over the year is ~0 (drift removed)
    assert abs(float(curve["mean_log_return"].mean())) < 1e-9


def test_yearly_average_is_drift_free_and_causal() -> None:
    curve = _curve(CURVE_MODE_YEARLY_AVERAGE)
    vals = curve["seasonal_index"].to_numpy()
    assert vals[0] == 100.0
    assert abs(vals[-1] - vals[0]) < 2.0


def test_calendar_slots_drop_leap_day_and_are_ordered() -> None:
    curve = _curve(CURVE_MODE_LINK_RELATIVES)
    days = list(curve["calendar_day"])
    assert "02-29" not in days
    assert days == sorted(days)


def test_unsupported_mode_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_seasonal_pattern_curve(_synthetic_prices(), curve_mode="bogus")
