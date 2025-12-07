import pandas as pd
import numpy as np


def build_seasonal_filter(
    df_prices: pd.DataFrame,
    seasonal_curve: pd.DataFrame,
    top_windows,  # res["top_windows"]
    win_rate_thr: float = 0.80,
    slope_k: int = 5,
    slope_mag_thr: float = 1.0,  # index points
    pct_q: float = 0.15,  # 15% tails
    multi_lookbacks: tuple = (5, 10, 15),
    price_col: str = "close",
):
    """
    Returns DataFrame with boolean columns:
      allow_long, allow_short, reason (string tag for blocks).
    Policy: union of all blocks; allow_* = True if not blocked.
    """
    # 1) map calendar day to seasonal index
    sc = seasonal_curve.copy()
    mmdd_to_idx = dict(zip(sc["calendar_day"], sc["seasonal_index"]))
    mmdd = df_prices.index.strftime("%m-%d")
    idx_series = mmdd.map(mmdd_to_idx).astype(float)

    # 2) slope filter
    slope = idx_series.diff(slope_k)
    slope_block_long = slope <= -slope_mag_thr  # falling seasonality → block longs
    slope_block_short = slope >= slope_mag_thr  # rising seasonality → block shorts

    # 3) percentile bands
    lo, hi = np.nanpercentile(idx_series.dropna(), [pct_q * 100, (1 - pct_q) * 100])
    pct_block_long = idx_series >= hi  # too seasonally "high" → avoid fresh longs
    pct_block_short = idx_series <= lo  # too seasonally "low"  → avoid fresh shorts

    # 4) window blacklist from high-reliability windows
    win_block_long = pd.Series(False, index=df_prices.index)
    win_block_short = pd.Series(False, index=df_prices.index)
    for w in top_windows:
        if w.win_rate < win_rate_thr:
            continue
        start = pd.to_datetime("2024-" + w.entry_mmdd)
        end = pd.to_datetime("2024-" + w.exit_mmdd)
        # paint every year using calendar match
        mask = (
            mmdd.between(w.entry_mmdd, w.exit_mmdd)
            if end >= start
            else (~mmdd.between(w.exit_mmdd, w.entry_mmdd))
        )
        if w.direction == "long":
            win_block_short |= mask  # block shorts during strong long window
        else:
            win_block_long |= mask  # block longs during strong short window

    # 5) multi-lookback agreement (optional robustness)
    agree_block_long = pd.Series(False, index=df_prices.index)
    agree_block_short = pd.Series(False, index=df_prices.index)
    try:
        from seasonal.seasonal import build_seasonal_pattern_curve

        curves = []
        for lb in multi_lookbacks:
            c = build_seasonal_pattern_curve(
                df_prices, lookback_years=lb, price_col=price_col, smooth=3
            )
            s = mmdd.map(dict(zip(c["calendar_day"], c["seasonal_index"]))).astype(
                float
            )
            curves.append(s)
        slopes = [s.diff(slope_k) for s in curves]
        sign_sum = np.sign(np.column_stack([sl.values for sl in slopes]))
        # majority direction
        maj_up = np.nanmean(sign_sum, axis=1) >= 1 / 3
        maj_down = np.nanmean(sign_sum, axis=1) <= -1 / 3
        agree_block_long = pd.Series(
            maj_down, index=df_prices.index
        )  # majority falling → block longs
        agree_block_short = pd.Series(
            maj_up, index=df_prices.index
        )  # majority rising → block shorts
    except Exception:
        pass  # optional

    # combine blocks
    block_long = slope_block_long | pct_block_long | win_block_long | agree_block_long
    block_short = (
        slope_block_short | pct_block_short | win_block_short | agree_block_short
    )

    out = pd.DataFrame(
        {
            "seasonal_index": idx_series,
            "allow_long": ~block_long.fillna(False),
            "allow_short": ~block_short.fillna(False),
            "block_long": block_long.fillna(False),
            "block_short": block_short.fillna(False),
        },
        index=df_prices.index,
    )
    return out
