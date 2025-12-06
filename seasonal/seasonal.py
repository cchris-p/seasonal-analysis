from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from config_assets import is_futures_asset


# ---------- Utilities ----------
def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _available_years(df: pd.DataFrame) -> List[int]:
    return sorted(pd.Index(df.index.year).unique().tolist())


def _is_year_complete(
    df: pd.DataFrame, year: int, min_days_threshold: int = 250
) -> bool:
    """
    Check if a year has reasonably complete data.

    Args:
        df: DataFrame with datetime index
        year: Year to check
        min_days_threshold: Minimum number of trading days to consider complete
                           (250 is roughly 52 weeks * 5 days - holidays)

    Returns:
        True if year appears to have complete data, False otherwise
    """
    year_data = df[df.index.year == year]
    if year_data.empty:
        return False

    # Count actual trading days in the year
    actual_days = len(year_data)

    # Additional check: make sure we have data in both January and December
    has_january = any(year_data.index.month == 1)
    has_december = any(year_data.index.month == 12)

    return actual_days >= min_days_threshold and has_january and has_december


def _snap_to_trading_day(
    idx: pd.DatetimeIndex, target: pd.Timestamp, policy: str
) -> Optional[pd.Timestamp]:
    """
    Snap a target date to a trading day present in idx.
    policy: 'next' or 'prev'
    Returns None if cannot snap (e.g., out of bounds).
    """
    if target in idx:
        return target
    pos = idx.searchsorted(target)
    if policy == "next":
        if pos < len(idx):
            return idx[pos]
        return None
    else:  # 'prev'
        if pos == 0:
            return None
        return idx[pos - 1]


def _calendar_window_days(
    year: int, entry_mmdd: str, exit_mmdd: str
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    e_month, e_day = map(int, entry_mmdd.split("-"))
    x_month, x_day = map(int, exit_mmdd.split("-"))

    try:
        entry = pd.Timestamp(year=year, month=e_month, day=e_day)
    except ValueError:
        # Handle invalid dates like Feb 29 in non-leap years
        raise ValueError(f"Invalid entry date: {year}-{e_month:02d}-{e_day:02d}")

    try:
        exit_ = pd.Timestamp(year=year, month=x_month, day=x_day)
    except ValueError:
        # Handle invalid dates like Feb 29 in non-leap years
        raise ValueError(f"Invalid exit date: {year}-{x_month:02d}-{x_day:02d}")

    # Allow crossing year-end if exit < entry (rare for FX seasonal grids, but safe):
    if exit_ < entry:
        try:
            exit_ = exit_.replace(year=year + 1)
        except ValueError:
            # Handle case where next year also has invalid date
            raise ValueError(
                f"Invalid exit date in next year: {year+1}-{x_month:02d}-{x_day:02d}"
            )

    return entry, exit_


# ---------- Seasonal index ----------
def compute_year_range_normalized_index(
    df: pd.DataFrame, price_col: str = "close"
) -> pd.Series:
    """
    For each calendar year y and each trading day t in y:
        r_t = 100 * (P_t - min_y) / (max_y - min_y)
    Handles flat-range years by returning 50 for all days in that year.
    Returns a Series aligned to df.index with values in [0,100].
    """
    df = _ensure_dt_index(df)
    out = pd.Series(index=df.index, dtype=float)
    for y, d in df.groupby(df.index.year):
        p = d[price_col]
        pmin, pmax = float(p.min()), float(p.max())
        if np.isclose(pmax, pmin):
            out.loc[d.index] = 50.0
        else:
            out.loc[d.index] = 100.0 * (p - pmin) / (pmax - pmin)
    return out


def build_seasonal_pattern_curve(
    df: pd.DataFrame,
    lookback_years: int = 15,
    price_col: str = "close",
    smooth: int = 0,
    exclude_incomplete_last_year: bool = True,
) -> pd.DataFrame:
    """
    Build a daily seasonal index curve (0-100) by averaging the year-normalized index
    for each calendar 'month-day' across the most recent `lookback_years`.

    Args:
        df: DataFrame with OHLCV data and datetime index
        lookback_years: Number of recent years to include
        price_col: Column name for price data
        smooth: Moving average window for smoothing (0 = no smoothing)
        exclude_incomplete_last_year: If True, exclude the last year if it appears incomplete

    Returns DataFrame with columns: ['calendar_day','seasonal_index'] with 'MM-DD' strings.
    """
    df = _ensure_dt_index(df)
    # Determine years to include
    years = _available_years(df)
    if len(years) == 0:
        raise ValueError("No data.")

    # Optionally exclude the last year if it appears incomplete
    if exclude_incomplete_last_year and len(years) > 1:
        last_year = years[-1]
        if not _is_year_complete(df, last_year):
            print(f"Excluding incomplete last year: {last_year}")
            years = years[:-1]  # Remove the last year

    keep_years = years[-lookback_years:] if len(years) > lookback_years else years

    # Compute normalized index for whole dataset
    norm = compute_year_range_normalized_index(df, price_col=price_col)

    # Assemble by MM-DD
    tmp = pd.DataFrame(
        {
            "idx": df.index,
            "mmdd": df.index.strftime("%m-%d"),
            "year": df.index.year,
            "norm": norm.values,
        }
    )
    tmp = tmp[tmp["year"].isin(keep_years)]
    grp = tmp.groupby("mmdd")["norm"].mean().to_frame("seasonal_index").reset_index()
    grp = grp.sort_values("mmdd").reset_index(drop=True)

    # Optional simple moving average smoothing
    if smooth and smooth > 1:
        si = grp["seasonal_index"].rolling(smooth, min_periods=1, center=True).mean()
        grp["seasonal_index"] = si

    return grp.rename(columns={"mmdd": "calendar_day"})


# ---------- Window grid and scoring ----------
@dataclass
class WindowStats:
    entry_mmdd: str
    exit_mmdd: str
    num_trades: int
    win_rate: float
    avg_profit_points: float
    median_profit_points: float
    worst_loss_points: float
    best_runup_points: float
    worst_drawdown_points: float
    direction: str  # 'long' or 'short'


@dataclass
class SeasonalAnalysisResult:
    symbol: str
    years_available: List[int]
    lookback_years: int
    seasonal_curve: pd.DataFrame
    top_windows: List[WindowStats]
    per_year_results: pd.DataFrame


def _compute_window_pnl_for_year(
    df_year: pd.DataFrame,
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
    direction: str,
    pip_factor: float,
) -> Optional[Dict]:
    """
    Entry/exit on 'Close' at snapped trading dates.
    Computes realized P&L in pips and settlement-path best/worst equity.
    """
    idx = df_year.index
    e = _snap_to_trading_day(idx, entry, "next")
    x = _snap_to_trading_day(idx, exit_, "prev")
    if (e is None) or (x is None) or (x <= e):
        return None

    entry_px = float(df_year.loc[e, "close"])
    exit_px = float(df_year.loc[x, "close"])

    # P&L in pips (positive for profitable trade, given direction)
    if direction == "long":
        pnl_pips = (exit_px - entry_px) * pip_factor
        # path equity
        path = (df_year.loc[e:x, "close"] - entry_px) * pip_factor
    else:
        pnl_pips = (entry_px - exit_px) * pip_factor
        path = (entry_px - df_year.loc[e:x, "close"]) * pip_factor

    # Settlement-path best/worst equity between e..x
    best_runup = float(path.max()) if len(path) else 0.0
    worst_dd = float(path.min()) if len(path) else 0.0

    return {
        "year": int(e.year),
        "entry_dt": e,
        "exit_dt": x,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "pnl_points": float(pnl_pips),
        "best_runup_points": best_runup,  # >= 0
        "worst_drawdown_points": worst_dd,  # <= 0
    }


def _pip_factor_for_symbol(symbol: str) -> float:
    """
    Convert price delta to pips.
    Defaults: 0.0001-base pairs → factor=10000; JPY quote → 100.
    Adjust here if your symbols differ (e.g., metals, indices).
    """
    sym = symbol.upper()
    # Futures and non-FX assets: work in points
    try:
        if is_futures_asset(sym):
            return 1.0
    except Exception:
        pass
    if sym.endswith("JPY"):
        return 100.0
    return 10000.0


def generate_calendar_grid(
    months: List[int],
    entry_day_range: Tuple[int, int],
    exit_day_range: Tuple[int, int],
    min_len_days: int = 5,
    max_len_days: int = 60,
) -> List[Tuple[str, str]]:
    """
    Produce (entry_mmdd, exit_mmdd) pairs like ('01-12','02-03') bounded by length.
    """
    entries = []
    for m in months:
        for d in range(entry_day_range[0], entry_day_range[1] + 1):
            try:
                pd.Timestamp(year=2024, month=m, day=d)  # validate date
                entries.append(f"{m:02d}-{d:02d}")
            except ValueError:
                continue

    exits = []
    for m in months:
        for d in range(exit_day_range[0], exit_day_range[1] + 1):
            try:
                pd.Timestamp(year=2024, month=m, day=d)
                exits.append(f"{m:02d}-{d:02d}")
            except ValueError:
                continue

    grid = []
    # use a leap year consistently to avoid 02-29 errors when computing lengths
    ref_year = 2024
    for e in entries:
        e_m, e_d = map(int, e.split("-"))
        for x in exits:
            x_m, x_d = map(int, x.split("-"))
            # compute length using leap reference year and guard invalid dates
            try:
                e_dt = pd.Timestamp(ref_year, e_m, e_d)
                x_dt = pd.Timestamp(ref_year, x_m, x_d)
            except ValueError:
                continue
            if x_dt < e_dt:
                continue
            L = (x_dt - e_dt).days + 1
            if (L >= min_len_days) and (L <= max_len_days):
                grid.append((e, x))
    # unique and sorted
    grid = sorted(set(grid))
    return grid


def score_seasonal_windows(
    df: pd.DataFrame,
    symbol: str,
    windows: List[Tuple[str, str]],
    lookback_years: int = 15,
    direction: str = "auto",
    min_trades: int = 10,
    min_win_rate: float = 0.80,
) -> Tuple[List[WindowStats], pd.DataFrame]:
    """
    Evaluate each window on the most recent `lookback_years` available.
    direction: 'long', 'short', or 'auto' (choose based on seasonal slope).
    Returns (summary_list, per_year_detail_df)
    """
    df = _ensure_dt_index(df)
    years_all = _available_years(df)
    years_used = (
        years_all[-lookback_years:] if len(years_all) > lookback_years else years_all
    )
    df_used = df[df.index.year.isin(years_used)].copy()
    pip_factor = _pip_factor_for_symbol(symbol)

    # Seasonal slope proxy to guide 'auto'
    curve = build_seasonal_pattern_curve(df_used, lookback_years=len(years_used))
    si_map = dict(zip(curve["calendar_day"], curve["seasonal_index"]))

    per_year_rows = []
    out_stats: List[WindowStats] = []

    for e_mmdd, x_mmdd in windows:
        # decide direction if auto
        if direction in ("long", "short"):
            side = direction
        else:
            si_e = si_map.get(e_mmdd, np.nan)
            si_x = si_map.get(x_mmdd, np.nan)
            side = (
                "long"
                if (np.isfinite(si_e) and np.isfinite(si_x) and si_x >= si_e)
                else "short"
            )

        # year-by-year
        yearly: List[Dict] = []
        for y in years_used:
            # slice that year plus possible spillover to next year for exit snapping
            mask = (df.index >= pd.Timestamp(y, 1, 1)) & (
                df.index < pd.Timestamp(y + 1, 12, 31) + pd.Timedelta(days=2)
            )
            dyy = df.loc[mask]
            if dyy.empty:
                continue
            try:
                entry_dt, exit_dt = _calendar_window_days(y, e_mmdd, x_mmdd)
            except ValueError:
                # Skip invalid date combinations (e.g., Feb 29 in non-leap years)
                continue
            res = _compute_window_pnl_for_year(dyy, entry_dt, exit_dt, side, pip_factor)
            if res:
                res.update(
                    {
                        "entry_mmdd": e_mmdd,
                        "exit_mmdd": x_mmdd,
                        "direction": side,
                        "symbol": symbol,
                    }
                )
                yearly.append(res)

        if not yearly:
            continue

        # aggregate
        pnl = np.array([r["pnl_points"] for r in yearly], dtype=float)
        wins = (pnl > 0).sum()
        num = len(yearly)
        win_rate = wins / num if num else 0.0
        avg_points = float(pnl.mean()) if num else 0.0
        med_points = float(np.median(pnl)) if num else 0.0
        worst_loss = float(pnl.min()) if num else 0.0
        best_runup = (
            float(np.max([r["best_runup_points"] for r in yearly])) if num else 0.0
        )
        worst_dd = (
            float(np.min([r["worst_drawdown_points"] for r in yearly])) if num else 0.0
        )

        # reliability filter
        if (num >= min_trades) and (win_rate >= min_win_rate) and (avg_points > 0):
            out_stats.append(
                WindowStats(
                    entry_mmdd=e_mmdd,
                    exit_mmdd=x_mmdd,
                    num_trades=num,
                    win_rate=win_rate,
                    avg_profit_points=avg_points,
                    median_profit_points=med_points,
                    worst_loss_points=worst_loss,
                    best_runup_points=best_runup,
                    worst_drawdown_points=worst_dd,
                    direction=side,
                )
            )

        per_year_rows.extend(yearly)

    per_year_df = pd.DataFrame(per_year_rows).sort_values(["entry_mmdd", "year"])
    # Rank by simple composite
    out_stats.sort(
        key=lambda w: (w.win_rate * max(w.avg_profit_points, 0.0), w.num_trades),
        reverse=True,
    )
    return out_stats, per_year_df


# ---------- Orchestration ----------
def run_seasonal_analysis(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    lookback_years: int = 15,
    months: Optional[List[int]] = None,
    entry_day_range: Tuple[int, int] = (5, 25),
    exit_day_range: Tuple[int, int] = (10, 31),
    min_len_days: int = 5,
    max_len_days: int = 45,
    direction: str = "auto",
    min_trades: int = 10,
    min_win_rate: float = 0.80,
    smooth: int = 3,
    exclude_incomplete_last_year: bool = True,
) -> SeasonalAnalysisResult:
    """
    Top-level seasonal analysis runner for any asset class.
    If df is None, loads data automatically based on symbol's asset class.
    Returns dict with seasonal curve, top windows, and per-year table.
    """
    if df is None:
        # Load data using centralized loader based on asset class
        from load_data import load_price_data

        df = load_price_data(symbol, start_date=start, end_date=end, granularity="D")
    df = _ensure_dt_index(df)
    df = df.loc[df.index >= pd.to_datetime(start)]
    if months is None:
        months = list(range(1, 13))  # full year grid

    # Build seasonal curve for reporting
    seasonal_curve = build_seasonal_pattern_curve(
        df,
        lookback_years=lookback_years,
        smooth=smooth,
        exclude_incomplete_last_year=exclude_incomplete_last_year,
    )

    # Build grid
    windows = generate_calendar_grid(
        months,
        entry_day_range,
        exit_day_range,
        min_len_days=min_len_days,
        max_len_days=max_len_days,
    )

    # Score
    top_windows, per_year = score_seasonal_windows(
        df=df,
        symbol=symbol,
        windows=windows,
        lookback_years=lookback_years,
        direction=direction,
        min_trades=min_trades,
        min_win_rate=min_win_rate,
    )

    return SeasonalAnalysisResult(
        symbol=symbol,
        years_available=_available_years(df),
        lookback_years=lookback_years,
        seasonal_curve=seasonal_curve,
        top_windows=top_windows,
        per_year_results=per_year,
    )


# ---------- Seasonax-style derived metrics ----------


def _compute_pattern_return_frac(
    direction: str, entry_px: float, exit_px: float
) -> float:
    if entry_px == 0.0:
        raise ValueError("entry_px must be non-zero")
    if direction == "long":
        return (exit_px - entry_px) / entry_px
    if direction == "short":
        return (entry_px - exit_px) / entry_px
    raise ValueError(f"Unsupported direction: {direction!r}")


def enrich_per_year_results_with_returns(
    df: pd.DataFrame,
    per_year_results: pd.DataFrame,
    trading_days_per_year: int,
) -> pd.DataFrame:
    df_local = _ensure_dt_index(df)
    if per_year_results.empty:
        return per_year_results.copy()

    enriched = per_year_results.copy()
    pattern_trading_days: List[int] = []
    return_fracs: List[float] = []
    return_pcts: List[float] = []
    annualised_fracs: List[float] = []
    annualised_pcts: List[float] = []

    for row in enriched.itertuples(index=False):
        entry_dt = getattr(row, "entry_dt")
        exit_dt = getattr(row, "exit_dt")
        direction = getattr(row, "direction")
        entry_px = float(getattr(row, "entry_px"))
        exit_px = float(getattr(row, "exit_px"))

        mask = (df_local.index >= entry_dt) & (df_local.index <= exit_dt)
        days_in_pattern = int(mask.sum())

        r_frac = _compute_pattern_return_frac(direction, entry_px, exit_px)
        if days_in_pattern > 0:
            ann_frac = (1.0 + r_frac) ** (
                float(trading_days_per_year) / float(days_in_pattern)
            ) - 1.0
        else:
            ann_frac = np.nan

        pattern_trading_days.append(days_in_pattern)
        return_fracs.append(r_frac)
        return_pcts.append(r_frac * 100.0)
        annualised_fracs.append(ann_frac)
        annualised_pcts.append(ann_frac * 100.0 if np.isfinite(ann_frac) else np.nan)

    enriched["pattern_trading_days"] = pattern_trading_days
    enriched["return_frac"] = return_fracs
    enriched["return_pct"] = return_pcts
    enriched["annualised_return_frac"] = annualised_fracs
    enriched["annualised_return_pct"] = annualised_pcts
    return enriched


def build_cumulative_profit_series(
    per_year_results: pd.DataFrame,
    entry_mmdd: str,
    exit_mmdd: str,
    as_percent: bool,
) -> pd.Series:
    dfw = per_year_results[
        (per_year_results["entry_mmdd"] == entry_mmdd)
        & (per_year_results["exit_mmdd"] == exit_mmdd)
    ].copy()
    if dfw.empty:
        raise ValueError(f"No trades for window {entry_mmdd} -> {exit_mmdd}")
    dfw = dfw.sort_values("year")

    values: List[float] = []
    if as_percent:
        for row in dfw.itertuples(index=False):
            r_frac = _compute_pattern_return_frac(
                getattr(row, "direction"),
                float(getattr(row, "entry_px")),
                float(getattr(row, "exit_px")),
            )
            values.append(r_frac * 100.0)
        name = "cumulative_return_pct"
    else:
        values = dfw["pnl_points"].astype(float).tolist()
        name = "cumulative_pnl_points"

    cumulative = np.cumsum(np.asarray(values, dtype=float))
    return pd.Series(cumulative, index=dfw["year"].astype(int), name=name)


def summarize_window_kpis(
    df: pd.DataFrame,
    per_year_results: pd.DataFrame,
    entry_mmdd: str,
    exit_mmdd: str,
    trading_days_per_year: int,
) -> Dict[str, Dict[str, float]]:
    dfw = per_year_results[
        (per_year_results["entry_mmdd"] == entry_mmdd)
        & (per_year_results["exit_mmdd"] == exit_mmdd)
    ].copy()
    if dfw.empty:
        raise ValueError(f"No trades for window {entry_mmdd} -> {exit_mmdd}")

    dfw = enrich_per_year_results_with_returns(df, dfw, trading_days_per_year)
    num_trades = int(len(dfw))
    pnl = dfw["pnl_points"].astype(float).to_numpy()
    r_pct = dfw["return_pct"].astype(float).to_numpy()
    ann_pct = dfw["annualised_return_pct"].astype(float).to_numpy()

    wins = r_pct > 0.0
    losses = r_pct < 0.0
    win_rate = float(wins.sum() / num_trades) if num_trades else np.nan

    def _stats(arr: np.ndarray) -> Tuple[float, float, float, float]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan, np.nan, np.nan, np.nan
        return (
            float(arr.mean()),
            float(np.median(arr)),
            float(arr.min()),
            float(arr.max()),
        )

    avg_pnl, med_pnl, min_pnl, max_pnl = _stats(pnl)
    avg_ret, med_ret, min_ret, max_ret = _stats(r_pct)
    avg_ann, med_ann, _, _ = _stats(ann_pct)

    gain_r = r_pct[wins]
    loss_r = r_pct[losses]
    avg_gain, med_gain, _, max_gain = _stats(gain_r)
    avg_loss, med_loss, min_loss, _ = _stats(loss_r)

    cum = pnl.cumsum()
    total_pnl = float(cum[-1]) if cum.size else 0.0
    dd = cum - np.maximum.accumulate(cum)
    max_dd = float(dd.min()) if dd.size else 0.0

    std_ret = float(r_pct.std(ddof=1)) if r_pct.size > 1 else np.nan
    sharpe_like = (
        float(np.nanmean(ann_pct) / std_ret)
        if (np.isfinite(std_ret) and std_ret > 0.0)
        else np.nan
    )

    return {
        "basic": {
            "num_trades": float(num_trades),
            "win_rate": win_rate,
        },
        "returns_pct": {
            "avg": avg_ret,
            "median": med_ret,
            "min": min_ret,
            "max": max_ret,
            "std": std_ret,
            "avg_annualised": avg_ann,
            "median_annualised": med_ann,
        },
        "profit_points": {
            "avg": avg_pnl,
            "median": med_pnl,
            "min": min_pnl,
            "max": max_pnl,
            "total": total_pnl,
        },
        "gains": {
            "num_gains": float(wins.sum()),
            "avg_gain_pct": avg_gain,
            "median_gain_pct": med_gain,
            "best_gain_pct": max_gain,
        },
        "losses": {
            "num_losses": float(losses.sum()),
            "avg_loss_pct": avg_loss,
            "median_loss_pct": med_loss,
            "worst_loss_pct": min_loss,
        },
        "cumulative": {
            "max_drawdown_points": max_dd,
            "end_cumulative_pnl_points": total_pnl,
        },
        "risk": {
            "sharpe_like": sharpe_like,
        },
    }


def compute_pattern_vs_rest_vs_buy_and_hold(
    df: pd.DataFrame,
    per_year_results: pd.DataFrame,
    entry_mmdd: str,
    exit_mmdd: str,
) -> Dict[str, float]:
    df_local = _ensure_dt_index(df)
    dfw = per_year_results[
        (per_year_results["entry_mmdd"] == entry_mmdd)
        & (per_year_results["exit_mmdd"] == exit_mmdd)
    ].copy()
    if dfw.empty:
        raise ValueError(f"No trades for window {entry_mmdd} -> {exit_mmdd}")

    years = sorted(dfw["year"].astype(int).unique().tolist())
    mask_years = df_local.index.year.isin(years)
    px = df_local.loc[mask_years, "close"].astype(float)
    log_ret = np.log(px / px.shift(1))
    log_ret = log_ret[log_ret.notna()]
    if log_ret.empty:
        raise ValueError("Not enough data to compute returns")

    in_pattern = pd.Series(False, index=log_ret.index)
    for row in dfw.itertuples(index=False):
        entry_dt = getattr(row, "entry_dt")
        exit_dt = getattr(row, "exit_dt")
        seg_idx = log_ret.index[
            (log_ret.index >= entry_dt) & (log_ret.index <= exit_dt)
        ]
        if len(seg_idx):
            in_pattern.loc[seg_idx] = True

    pattern_log = log_ret[in_pattern]
    rest_log = log_ret[~in_pattern]

    base_log = float(log_ret.sum())
    pattern_log_sum = float(pattern_log.sum()) if not pattern_log.empty else 0.0
    rest_log_sum = float(rest_log.sum()) if not rest_log.empty else 0.0

    base_ret = float(np.exp(base_log) - 1.0)
    pattern_ret = float(np.exp(pattern_log_sum) - 1.0)
    rest_ret = float(np.exp(rest_log_sum) - 1.0)

    if base_ret != 0.0:
        pattern_share = pattern_ret / base_ret
        rest_share = rest_ret / base_ret
    else:
        pattern_share = np.nan
        rest_share = np.nan

    return {
        "buyhold_return_frac": base_ret,
        "pattern_return_frac": pattern_ret,
        "rest_return_frac": rest_ret,
        "pattern_share_of_buyhold": pattern_share,
        "rest_share_of_buyhold": rest_share,
    }


def plot_seasonal_curve_with_windows(
    seasonal_curve: pd.DataFrame,
    windows: list,
    title: str = "Seasonal index with windows",
):
    """
    seasonal_curve: DataFrame with ['calendar_day','seasonal_index'] from run_seasonal_analysis
    windows: list[WindowStats] or list[(entry_mmdd, exit_mmdd, direction)]
    """
    import matplotlib.pyplot as plt

    sc = seasonal_curve.copy()
    sc = sc.sort_values("calendar_day")
    # Map MM-DD to a common year for plotting
    sc["ts"] = pd.to_datetime("2024-" + sc["calendar_day"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sc["ts"], sc["seasonal_index"], lw=2)
    ax.set_title(title)
    ax.set_ylabel("Seasonal index (0–100)")
    ax.set_xlim(sc["ts"].min(), sc["ts"].max())
    ax.grid(alpha=0.3)

    # Accept WindowStats or tuples
    for w in windows:
        entry = w.entry_mmdd if hasattr(w, "entry_mmdd") else w[0]
        exit_ = w.exit_mmdd if hasattr(w, "exit_mmdd") else w[1]
        side = (
            w.direction if hasattr(w, "direction") else (w[2] if len(w) > 2 else "auto")
        )
        x0 = pd.to_datetime("2024-" + entry)
        x1 = pd.to_datetime("2024-" + exit_)
        if x1 < x0:
            # if a window crosses year-end, split into two spans
            ax.axvspan(
                pd.to_datetime("2024-01-01"),
                x1,
                color=("tab:green" if side == "long" else "tab:red"),
                alpha=0.2,
            )
            ax.axvspan(
                x0,
                pd.to_datetime("2024-12-31"),
                color=("tab:green" if side == "long" else "tab:red"),
                alpha=0.2,
            )
        else:
            ax.axvspan(
                x0, x1, color=("tab:green" if side == "long" else "tab:red"), alpha=0.2
            )

    # nicer x ticks
    ax.set_xticks(pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 13)]))
    ax.set_xticklabels(
        [pd.to_datetime(f"2024-{m:02d}-01").strftime("%b") for m in range(1, 13)]
    )
    plt.tight_layout()
    plt.show()


def plot_per_year_pnl(per_year_results: pd.DataFrame, entry_mmdd: str, exit_mmdd: str):
    """
    per_year_results: DataFrame returned as res['per_year_results']
    Shows realized P&L by year for a specific window.
    """
    import matplotlib.pyplot as plt

    dfw = per_year_results[
        (per_year_results["entry_mmdd"] == entry_mmdd)
        & (per_year_results["exit_mmdd"] == exit_mmdd)
    ].copy()
    if dfw.empty:
        print("No rows for that window.")
        return
    dfw = dfw.sort_values("year")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(dfw["year"].astype(str), dfw["pnl_points"])
    ax.axhline(0, lw=1, color="k")
    ax.set_title(f"P&L by year: {entry_mmdd} → {exit_mmdd}")
    ax.set_ylabel("Points")
    plt.tight_layout()
    plt.show()


def plot_seasonal_stacks_by_lookback(
    df: pd.DataFrame,
    lookbacks=(5, 10, 15),
    smooth: int = 3,
    price_col: str = "close",
    title: str = "Seasonal closes (0–100 index)",
    exclude_incomplete_last_year: bool = True,
):
    """
    Plots seasonal curves for different lookback periods on a single plot with legend.
    Uses build_seasonal_pattern_curve from your module.

    Args:
        df: DataFrame with OHLCV data and datetime index
        lookbacks: Tuple of lookback years to plot
        smooth: Moving average window for smoothing curves
        price_col: Column name for price data
        title: Plot title
        exclude_incomplete_last_year: If True, exclude the last year if it appears incomplete
    """
    import matplotlib.pyplot as plt

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Create single plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors for different lookback periods
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    for i, lb in enumerate(lookbacks):
        curve = build_seasonal_pattern_curve(
            df,
            lookback_years=lb,
            price_col=price_col,
            smooth=smooth,
            exclude_incomplete_last_year=exclude_incomplete_last_year,
        )
        curve = curve.sort_values("calendar_day")
        curve["ts"] = pd.to_datetime("2024-" + curve["calendar_day"])

        # Plot with different color and label
        color = colors[i % len(colors)]
        ax.plot(
            curve["ts"],
            curve["seasonal_index"],
            lw=2,
            color=color,
            label=f"{lb}y lookback",
        )

    ax.set_title(title)
    ax.set_ylabel("Seasonal Index (0-100)")
    ax.set_xlabel("Month")
    ax.grid(alpha=0.3)

    # Set x-axis limits and ticks
    ax.set_xlim(pd.to_datetime("2024-01-01"), pd.to_datetime("2024-12-31"))
    ax.set_xticks(pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 13)]))
    ax.set_xticklabels(
        [pd.to_datetime(f"2024-{m:02d}-01").strftime("%b") for m in range(1, 13)]
    )

    # Add legend
    ax.legend(loc="best", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plt.show()
