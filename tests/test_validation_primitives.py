from __future__ import annotations

import math

import pytest

from seasonal_analysis.calendar_utils import shift_mmdd_trading_days
from seasonal_analysis.metrics import trade_metrics
from seasonal_analysis.significance import (
    adjust_pvalues_benjamini_hochberg,
    adjust_pvalues_bonferroni,
    bootstrap_ci,
    sign_permutation_p_value,
)
from seasonal_analysis.universes import (
    is_crypto_symbol,
    is_forex_symbol,
    is_futures_symbol,
)


def test_trade_metrics_shape_and_values() -> None:
    m = trade_metrics([0.10, -0.05, 0.02, -0.01])
    assert m["num_trades"] == 4
    assert m["win_rate"] == 0.5
    assert m["avg_return"] == pytest.approx(0.015)
    assert m["median_return"] == pytest.approx(0.005)
    assert m["worst_return"] == -0.05
    assert m["worst_year_return"] == -0.05
    assert m["max_drawdown_proxy"] == pytest.approx(-0.05)
    assert m["profit_factor"] is not None
    assert m["mae"] > 0

    empty = trade_metrics([])
    assert empty["num_trades"] == 0
    assert empty["avg_return"] is None


def test_shift_mmdd_trading_days() -> None:
    assert shift_mmdd_trading_days("01-08", 1) == "01-09"
    assert shift_mmdd_trading_days("01-08", 0) == "01-08"
    assert shift_mmdd_trading_days("12-31", 1) is None


def test_bootstrap_and_significance_are_deterministic_with_seed() -> None:
    values = [0.01, -0.02, 0.03, 0.015, 0.005]
    ci1 = bootstrap_ci(values, n=200, seed=7)
    ci2 = bootstrap_ci(values, n=200, seed=7)
    assert ci1 == ci2
    assert ci1["p2_5"] <= ci1["mean"] <= ci1["p97_5"]

    p1 = sign_permutation_p_value(values, n=200, seed=7)
    p2 = sign_permutation_p_value(values, n=200, seed=7)
    assert p1 == p2
    assert 0.0 < p1["p_value"] <= 1.0


def test_multiple_testing_adjustments() -> None:
    p = [0.01, 0.02, None, 0.5]
    bonf = adjust_pvalues_bonferroni(p)
    assert bonf[0] == 0.03
    assert bonf[2] is None

    bh = adjust_pvalues_benjamini_hochberg(p)
    assert bh[2] is None
    assert bh[0] is not None and bh[1] is not None and bh[3] is not None
    assert math.isclose(bh[0], 0.03, rel_tol=1e-9)


def test_asset_class_helpers() -> None:
    assert is_forex_symbol("EURUSD")
    assert not is_forex_symbol("ES")
    assert is_futures_symbol("ES")
    assert is_futures_symbol("6E")
    assert is_crypto_symbol("BTC")
    assert not is_crypto_symbol("EURUSD")
