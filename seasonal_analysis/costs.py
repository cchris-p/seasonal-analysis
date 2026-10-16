"""Cost-aware (net-of-cost) seasonal-window evaluation.

Ports the established aa-studies per-trade cost model
(``aa-studies/shared/min_backtest/cost.py``, the thin-PoC/``INFRA-039`` model) into
the seasonal engine so window P&L can be reported gross and net:

- :class:`CostModel` — a round-trip per-trade deduction (static spread in pips +
  slippage + commission, or a relative ``round_trip_bps`` for ratio-adjusted
  futures), matching the aa-studies ``CostModel``.
- :func:`lookup_spread` / :func:`trade_cost_frac` — the calibrated path: the
  static spread component is replaced by the per-bar spread looked up at the
  trade's entry time from a calibrated spread series (the AA engine's
  variable-spread lookup, ``INFRA-041``).

The static *values* (e.g. per-symbol spread pips) are caller-supplied so they can
be sourced from the aa-studies tick-derived spread estimates rather than
re-invented here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Round-trip per-trade execution cost.

    FX path: components in pips (``round_trip_pips``). Futures path: set
    ``round_trip_bps`` to a relative round-trip cost in basis points of notional
    (used for ratio-adjusted continuous futures); when set it takes precedence
    over the pip components.
    """

    spread_pips: float = 1.0
    slippage_pips_per_side: float = 0.5
    commission_pips_round_trip: float = 0.6
    round_trip_bps: Optional[float] = None

    @property
    def round_trip_pips(self) -> float:
        return (
            self.spread_pips
            + 2.0 * self.slippage_pips_per_side
            + self.commission_pips_round_trip
        )

    @property
    def non_spread_round_trip_pips(self) -> float:
        """Round-trip cost components other than the spread (slippage + commission)."""
        return 2.0 * self.slippage_pips_per_side + self.commission_pips_round_trip

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["round_trip_pips"] = self.round_trip_pips
        d["non_spread_round_trip_pips"] = self.non_spread_round_trip_pips
        return d


def lookup_spread(entry_times, spread_series: Dict[str, Any]) -> np.ndarray:
    """Per-trade spread (price units) at the entry bar.

    ``spread_series`` maps ``epoch_seconds`` (bar-open UTC epoch seconds) to
    ``spread`` (full bid/ask spread in price units). Each trade maps to the spread
    of the latest bar whose open epoch is ``<=`` its entry time.
    """
    epochs = np.asarray(spread_series["epoch_seconds"], dtype=np.int64)
    spread = np.asarray(spread_series["spread"], dtype=float)
    if epochs.size == 0:
        raise ValueError("spread_series has no epochs")
    entry_ns = pd.DatetimeIndex(pd.to_datetime(entry_times)).asi8
    entry_epoch = (entry_ns // 1_000_000_000).astype(np.int64)
    idx = np.searchsorted(epochs, entry_epoch, side="right") - 1
    idx = np.clip(idx, 0, spread.size - 1)
    return spread[idx]


def trade_cost_frac(
    entry_px: float,
    pip: float,
    cost: CostModel,
    entry_dt=None,
    spread_series: Optional[Dict[str, Any]] = None,
) -> float:
    """Round-trip cost as a fraction of the entry price for one trade.

    Args:
        entry_px: Entry price (must be non-zero).
        pip: Pip size in price units for the symbol.
        cost: The :class:`CostModel`.
        entry_dt: Entry timestamp (used for the calibrated spread lookup).
        spread_series: Optional calibrated spread series; when provided the static
            spread component is replaced by the per-bar spread at ``entry_dt``.

    Returns:
        Cost fraction (e.g. ``0.0002`` = 0.02%).
    """
    if entry_px is None or not np.isfinite(entry_px) or entry_px == 0.0:
        return 0.0
    if cost.round_trip_bps is not None:
        return float(cost.round_trip_bps) / 10000.0
    if spread_series is not None and entry_dt is not None:
        spread_price = float(lookup_spread(pd.Series([entry_dt]), spread_series)[0])
        nonspread = cost.non_spread_round_trip_pips * float(pip)
        return (spread_price + nonspread) / abs(float(entry_px))
    return cost.round_trip_pips * float(pip) / abs(float(entry_px))
