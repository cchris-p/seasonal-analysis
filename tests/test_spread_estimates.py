from __future__ import annotations

from seasonal_analysis.spread_estimates import (
    SPREAD_ESTIMATES_PIPS,
    SPREAD_ESTIMATES_SOURCE,
    default_cost_model,
    estimated_spread_pips,
)


def test_table_is_sane() -> None:
    assert len(SPREAD_ESTIMATES_PIPS) == 28
    assert all(v > 0 for v in SPREAD_ESTIMATES_PIPS.values())
    # The majors are tight; a wide minor is wider than a major.
    assert SPREAD_ESTIMATES_PIPS["EURUSD"] < SPREAD_ESTIMATES_PIPS["GBPNZD"]
    assert SPREAD_ESTIMATES_SOURCE.startswith("dukascopy_tick_series")


def test_estimated_spread_pips_normalizes() -> None:
    assert estimated_spread_pips("USDCHF") == SPREAD_ESTIMATES_PIPS["USDCHF"]
    assert estimated_spread_pips("eurusd") == SPREAD_ESTIMATES_PIPS["EURUSD"]
    assert (
        estimated_spread_pips("OANDA:EURUSD") == SPREAD_ESTIMATES_PIPS["EURUSD"]
    )
    assert estimated_spread_pips("NOTAPAIR") is None
    assert estimated_spread_pips("NOTAPAIR", default=9.9) == 9.9


def test_default_cost_model_prefills_symbol_spread() -> None:
    cm = default_cost_model("GBPNZD")
    assert cm.spread_pips == SPREAD_ESTIMATES_PIPS["GBPNZD"]
    assert cm.slippage_pips_per_side == 0.5
    assert cm.commission_pips_round_trip == 0.6

    fallback = default_cost_model("NOTAPAIR", fallback_spread_pips=1.25)
    assert fallback.spread_pips == 1.25
