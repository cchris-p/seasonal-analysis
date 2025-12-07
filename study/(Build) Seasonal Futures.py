# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: jupyter
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Seasonal Futures - Calendar Windows

# %%
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import (
    DEFAULT_SYMBOL,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    LOOKBACK_YEARS,
    MONTHS,
    ENTRY_DAY_RANGE,
    EXIT_DAY_RANGE,
    MIN_LEN_DAYS,
    MAX_LEN_DAYS,
    DIRECTION,
    SMOOTH,
    EXCLUDE_INCOMPLETE_LAST_YEAR,
)
from load_data import load_price_data
from seasonal.seasonal import (
    run_seasonal_analysis,
    plot_seasonal_curve_with_windows,
    plot_per_year_pnl,
    plot_seasonal_stacks_by_lookback,
)

# Core selector (override here for a different futures contract)
symbol = DEFAULT_SYMBOL  # e.g., "ES", "CL", "NG", etc.
start_date = DEFAULT_START_DATE
end_date = DEFAULT_END_DATE

# %%
# Load daily futures data
fut_df: pd.DataFrame = load_price_data(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    granularity="D",
)

# %%
# Run generic seasonal analysis (asset-aware via symbol)
res = run_seasonal_analysis(
    symbol=symbol,
    df=fut_df,
    start=start_date,
    end=end_date,
    lookback_years=LOOKBACK_YEARS,
    months=MONTHS,
    entry_day_range=ENTRY_DAY_RANGE,
    exit_day_range=EXIT_DAY_RANGE,
    min_len_days=MIN_LEN_DAYS,
    max_len_days=MAX_LEN_DAYS,
    direction=DIRECTION,
    min_trades=10,
    min_win_rate=0.80,
    smooth=SMOOTH,
    exclude_incomplete_last_year=EXCLUDE_INCOMPLETE_LAST_YEAR,
)

# %%
print(f"Symbol: {res.symbol}")
print(f"Years available: {res.years_available}")
print("Best windows:")
for w in res.top_windows[:10]:
    print(w)

# %%
# 1) Shade top windows over the seasonal curve
if res.top_windows:
    plot_seasonal_curve_with_windows(
        res.seasonal_curve,
        res.top_windows[:5],
        title=f"{res.symbol} seasonal curve + top windows",
    )

    # 2) Bar chart for one specific window
    w0 = res.top_windows[0]
    plot_per_year_pnl(res.per_year_results, w0.entry_mmdd, w0.exit_mmdd)

# %%
# 3) Plot seasonal stacks for multiple lookbacks on the same futures contract
plot_seasonal_stacks_by_lookback(
    fut_df,
    lookbacks=(5, 10, LOOKBACK_YEARS),
    title=f"{res.symbol} seasonal closes (5/10/{LOOKBACK_YEARS}y)",
)
