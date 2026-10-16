from __future__ import annotations

import numpy as np
import pandas as pd

from seasonal_analysis.analysis import score_seasonal_windows
from seasonal_analysis.costs import CostModel, lookup_spread, trade_cost_frac


def _prices() -> pd.DataFrame:
    idx = pd.bdate_range("2005-01-03", "2019-12-31")
    t = np.arange(len(idx))
    doy = idx.dayofyear.to_numpy()
    price = 100.0 * np.exp(2e-4 * t + 0.05 * np.sin(2 * np.pi * doy / 365.0))
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price}, index=idx
    )


def test_static_cost_frac_and_bps() -> None:
    cost = CostModel(spread_pips=1.0, slippage_pips_per_side=0.5, commission_pips_round_trip=0.6)
    assert cost.round_trip_pips == 2.6
    frac = trade_cost_frac(1.1000, 0.0001, cost)
    assert abs(frac - (2.6 * 0.0001 / 1.1)) < 1e-12

    bps = CostModel(round_trip_bps=5.0)
    assert abs(trade_cost_frac(100.0, 0.0001, bps) - 5.0 / 10000.0) < 1e-12


def test_calibrated_spread_lookup() -> None:
    series = {
        "epoch_seconds": np.array([0, 100, 200], dtype=np.int64),
        "spread": np.array([0.00010, 0.00020, 0.00030]),
    }
    ts = pd.to_datetime([150, 250], unit="s")
    got = lookup_spread(ts, series)
    assert list(got) == [0.0002, 0.0003]


def test_score_reports_net_below_gross() -> None:
    windows = [("02-01", "03-15")]
    gross_stats, _ = score_seasonal_windows(
        _prices(), "EURUSD", windows, direction="long", min_trades=0, min_win_rate=0.0
    )
    cost_stats, _ = score_seasonal_windows(
        _prices(),
        "EURUSD",
        windows,
        direction="long",
        min_trades=0,
        min_win_rate=0.0,
        cost_model=CostModel(spread_pips=1.0, slippage_pips_per_side=0.5, commission_pips_round_trip=0.6),
    )
    assert gross_stats and cost_stats
    g = gross_stats[0]
    c = cost_stats[0]
    assert g.costs_included is False
    assert c.costs_included is True
    assert c.net_avg_profit_points < g.avg_profit_points
    assert c.net_win_rate <= g.win_rate
    assert c.cost_frac_mean > 0.0
