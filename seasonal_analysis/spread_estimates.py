"""Per-symbol tick-derived spread estimates for the seasonal cost model.

The seasonal-window cost model (:class:`seasonal_analysis.costs.CostModel`)
needs a per-trade spread level. Rather than an arbitrary static number, the
default is sourced from the established aa-studies tick-derived estimate: the
**median of the per-bar Dukascopy M5 ``open_spread``** series
(``<DUKASCOPY_ROOT>/<SYMBOL>_tick/<SYMBOL>_GMT+0_US-DST_spread_m5.npz``), i.e.
the same per-bar series the AA engine uses for its variable-spread model
(``INFRA-039``/``INFRA-041``).

This table is a small, committed snapshot so callers (the tradesurge
discovery/validation worker and the UI prefill) do **not** need the multi-GB
tick corpus or the Dukascopy numpy artifacts at runtime. Regenerate it from the
corpus with ``scripts/build_spread_estimates.py``.

The raw Dukascopy spread is wider than a retail MT broker's for the widest
crosses (``INFRA-039``), so these values are conservative (cost-overstated) by
default; the AA MT level calibration (``SPREAD_SERIES_SCALE_BY_SYMBOL``) may be
applied by the caller when a broker-matched level is required.
"""

from __future__ import annotations

from typing import Dict, Optional

from .costs import CostModel

# Median per-bar Dukascopy M5 ``open_spread``, converted to pips
# (0.01 for JPY quotes, 0.0001 otherwise). Computed 2026-10-16 from the
# full-history ``*_spread_m5.npz`` corpus (28/28 symbols).
SPREAD_ESTIMATES_PIPS: Dict[str, float] = {
    "AUDCAD": 2.4,
    "AUDCHF": 2.2,
    "AUDJPY": 1.2,
    "AUDNZD": 2.5,
    "AUDUSD": 1.1,
    "CADCHF": 2.7,
    "CADJPY": 1.4,
    "CHFJPY": 2.0,
    "EURAUD": 2.2,
    "EURCAD": 2.3,
    "EURCHF": 1.4,
    "EURGBP": 1.0,
    "EURJPY": 1.1,
    "EURNZD": 4.1,
    "EURUSD": 0.5,
    "GBPAUD": 3.6,
    "GBPCAD": 3.9,
    "GBPCHF": 2.6,
    "GBPJPY": 2.0,
    "GBPNZD": 6.0,
    "GBPUSD": 1.1,
    "NZDCAD": 3.2,
    "NZDCHF": 2.5,
    "NZDJPY": 2.0,
    "NZDUSD": 1.4,
    "USDCAD": 1.3,
    "USDCHF": 1.2,
    "USDJPY": 0.6,
}

# Provenance token recorded on artifacts that use the default estimate.
SPREAD_ESTIMATES_SOURCE = (
    "dukascopy_tick_series:GMT+0_US-DST:m5:open_spread:median"
)


def estimated_spread_pips(
    symbol: str, default: Optional[float] = None
) -> Optional[float]:
    """Return the tick-derived default spread (pips) for ``symbol``.

    ``symbol`` may be an ``OANDA:``-prefixed or lowercase ticker; it is
    normalized to the bare upper-case pair. Returns ``default`` (``None`` when
    unset) for an unknown symbol.
    """
    key = str(symbol or "").strip().upper()
    if ":" in key:
        key = key.split(":", 1)[1]
    return SPREAD_ESTIMATES_PIPS.get(key, default)


def default_cost_model(
    symbol: str,
    *,
    slippage_pips_per_side: float = 0.5,
    commission_pips_round_trip: float = 0.6,
    round_trip_bps: Optional[float] = None,
    fallback_spread_pips: float = 1.0,
) -> CostModel:
    """Build a :class:`CostModel` prefilled with the per-symbol spread estimate.

    Spread is the tick-derived estimate when known; ``fallback_spread_pips`` is
    used otherwise. Other components keep their established defaults (which the
    caller can override).
    """
    spread = estimated_spread_pips(symbol, default=fallback_spread_pips)
    return CostModel(
        spread_pips=float(spread),
        slippage_pips_per_side=float(slippage_pips_per_side),
        commission_pips_round_trip=float(commission_pips_round_trip),
        round_trip_bps=round_trip_bps,
    )
