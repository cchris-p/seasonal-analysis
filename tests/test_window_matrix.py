from __future__ import annotations

import numpy as np
import pandas as pd

from seasonal_analysis.analysis import build_window_return_matrix


def _prices() -> pd.DataFrame:
    idx = pd.bdate_range("2005-01-03", "2019-12-31")
    t = np.arange(len(idx))
    doy = idx.dayofyear.to_numpy()
    price = 100.0 * np.exp(2e-4 * t + 0.03 * np.sin(2 * np.pi * doy / 365.0))
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price}, index=idx
    )


def test_window_return_matrix_shape_and_keys() -> None:
    windows = [("02-01", "02-28"), ("06-01", "06-30"), ("10-01", "10-31")]
    mat, years, keys = build_window_return_matrix(_prices(), "EURUSD", windows)
    assert mat.shape[0] == len(years) == 15
    assert mat.shape[1] == len(keys) == 3
    assert all("->" in k for k in keys)
    # returns are fractional (small magnitude)
    assert np.nanmax(np.abs(mat)) < 0.5


def test_empty_windows_returns_empty_matrix() -> None:
    mat, years, keys = build_window_return_matrix(_prices(), "EURUSD", [])
    assert mat.shape == (0, 0)
    assert years == [] and keys == []
