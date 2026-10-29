"""Shared return-metric primitives for seasonal windows.

Pure, JSON-serializable calculations used by the accuracy scorecard and the
TradeSurge validation worker. Keeping them here prevents the app from
re-implementing metric math.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def trade_metrics(returns: Sequence[float]) -> Dict[str, Any]:
    """Summarize a sequence of per-trade fractional returns.

    Returns ``num_trades``, ``win_rate``, ``avg_return``, ``median_return``,
    ``worst_return`` / ``worst_year_return`` (same value), ``max_drawdown_proxy``
    (cumulative-equity drawdown), ``profit_factor``, and ``mae``. Empty/invalid
    input returns the same shape with ``None`` values.
    """
    arr = np.asarray([float(v) for v in returns if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "num_trades": 0,
            "win_rate": None,
            "avg_return": None,
            "median_return": None,
            "worst_return": None,
            "worst_year_return": None,
            "max_drawdown_proxy": None,
            "profit_factor": None,
            "mae": None,
        }

    wins = arr > 0.0
    gains = arr[wins].sum()
    losses = -arr[~wins].sum()
    profit_factor = float(gains / losses) if losses > 0 else None
    cum = np.cumsum(arr)
    dd = cum - np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:]
    mean = float(arr.mean())
    worst = float(arr.min())
    return {
        "num_trades": int(arr.size),
        "win_rate": float(wins.mean()),
        "avg_return": mean,
        "median_return": float(np.median(arr)),
        "worst_return": worst,
        "worst_year_return": worst,
        "max_drawdown_proxy": float(dd.min()) if dd.size else None,
        "profit_factor": profit_factor,
        "mae": float(np.mean(np.abs(arr - mean))),
    }
