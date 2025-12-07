from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from .config_assets import is_futures_asset
from .load_data import load_price_data


# ---------- Data Classes ----------
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
    df: pd.DataFrame
    selected_window: WindowStats  # window originally used for analysis


# ---------- Utilities ----------
def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a sorted DatetimeIndex.

    Args:
        df: Input DataFrame that may or may not have a DatetimeIndex.

    Returns:
        DataFrame with DatetimeIndex, sorted chronologically.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _available_years(df: pd.DataFrame) -> List[int]:
    """Extract sorted list of unique years present in DataFrame index.

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        Sorted list of years (as integers) found in the data.
    """
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
    """Snap a target date to the nearest available trading day.

    Args:
        idx: DatetimeIndex containing available trading days.
        target: Target timestamp to snap.
        policy: Snapping policy, either 'next' (forward) or 'prev' (backward).

    Returns:
        Snapped timestamp if found, None if target is out of bounds.
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
    """Convert calendar window dates to timestamps for a given year.

    Args:
        year: Calendar year for the window.
        entry_mmdd: Entry date in 'MM-DD' format (e.g., '01-15').
        exit_mmdd: Exit date in 'MM-DD' format (e.g., '02-20').

    Returns:
        Tuple of (entry_timestamp, exit_timestamp). If exit < entry, exit is
        moved to the following year to handle year-crossing windows.

    Raises:
        ValueError: If entry or exit dates are invalid (e.g., Feb 29 in non-leap year).
    """
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
    """Compute year-range normalized index for each trading day.

    For each calendar year y and each trading day t in y:
        r_t = 100 * (P_t - min_y) / (max_y - min_y)

    Args:
        df: DataFrame with OHLCV data and datetime index.
        price_col: Name of the price column to normalize (default: 'close').

    Returns:
        Series aligned to df.index with normalized values in [0,100]. Years with
        no price range (flat) return 50 for all days in that year.
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
def _compute_window_pnl_for_year(
    df_year: pd.DataFrame,
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
    direction: str,
    pip_factor: float,
) -> Optional[Dict]:
    """Compute P&L and path metrics for a single year's seasonal window.

    Args:
        df_year: DataFrame with OHLCV data for the year (plus potential spillover).
        entry: Target entry timestamp.
        exit_: Target exit timestamp.
        direction: Trade direction, either 'long' or 'short'.
        pip_factor: Multiplier to convert price delta to pips/points.

    Returns:
        Dictionary containing:
            - year: Calendar year of entry
            - entry_dt, exit_dt: Actual snapped trading timestamps
            - entry_px, exit_px: Entry and exit prices
            - pnl_points: Realized P&L in pips/points
            - best_runup_points: Maximum favorable excursion (>= 0)
            - worst_drawdown_points: Maximum adverse excursion (<= 0)
        Returns None if window cannot be executed (no valid trading days).
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
    """Determine pip conversion factor for a trading symbol.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'USDJPY', 'ES').

    Returns:
        Conversion factor to multiply price delta:
            - 1.0 for futures (points)
            - 100.0 for JPY pairs (0.01 pip)
            - 10000.0 for standard FX pairs (0.0001 pip)
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
    """Generate a grid of seasonal window entry/exit date combinations.

    Args:
        months: List of month numbers to include (1-12).
        entry_day_range: Tuple of (min_day, max_day) for entry dates.
        exit_day_range: Tuple of (min_day, max_day) for exit dates.
        min_len_days: Minimum window length in calendar days (default: 5).
        max_len_days: Maximum window length in calendar days (default: 60).

    Returns:
        List of (entry_mmdd, exit_mmdd) tuples in 'MM-DD' format,
        filtered by window length constraints. Invalid dates (e.g., Feb 29)
        are automatically excluded.
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
    """Score seasonal windows across multiple years and filter by performance.

    Args:
        df: DataFrame with OHLCV data and datetime index.
        symbol: Trading symbol for pip/point conversion.
        windows: List of (entry_mmdd, exit_mmdd) tuples to evaluate.
        lookback_years: Number of recent years to backtest (default: 15).
        direction: Trade direction - 'long', 'short', or 'auto'. Auto determines
            direction based on seasonal curve slope (default: 'auto').
        min_trades: Minimum number of historical trades required (default: 10).
        min_win_rate: Minimum win rate threshold (0-1) to include (default: 0.80).

    Returns:
        Tuple of:
            - List of WindowStats for windows meeting the filters, sorted by
              composite score (win_rate * avg_profit * num_trades).
            - DataFrame with detailed per-year results for all evaluated windows.
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
    """Execute complete seasonal pattern analysis for a trading symbol.

    This is the main entry point that orchestrates seasonal curve generation,
    window grid creation, and window scoring. Data is automatically loaded based
    on the symbol's asset class.

    Args:
        symbol: Trading symbol to analyze (e.g., 'EURUSD', 'ES').
        start: Start date for analysis in 'YYYY-MM-DD' format (default: '2010-01-01').
        end: Optional end date in 'YYYY-MM-DD' format. If None, uses latest available.
        lookback_years: Number of recent years to use for backtesting (default: 15).
        months: List of months (1-12) to include in window grid. None = all months.
        entry_day_range: Tuple of (min_day, max_day) for entry dates (default: (5, 25)).
        exit_day_range: Tuple of (min_day, max_day) for exit dates (default: (10, 31)).
        min_len_days: Minimum window length in days (default: 5).
        max_len_days: Maximum window length in days (default: 45).
        direction: Trade direction - 'long', 'short', or 'auto' (default: 'auto').
        min_trades: Minimum historical trades to qualify a window (default: 10).
        min_win_rate: Minimum win rate (0-1) to qualify a window (default: 0.80).
        smooth: Moving average window for seasonal curve smoothing (default: 3).
        exclude_incomplete_last_year: If True, exclude incomplete final year (default: True).

    Returns:
        SeasonalAnalysisResult containing:
            - symbol: Input symbol
            - years_available: List of years present in data
            - lookback_years: Lookback period used
            - seasonal_curve: DataFrame with daily seasonal index (0-100)
            - top_windows: List of WindowStats for qualifying windows, ranked by score
            - per_year_results: DataFrame with detailed per-year trade results
            - df: Original price data DataFrame
            - selected_window: Top-ranked window (None if no windows qualify)
    """
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
        df=df,
        selected_window=top_windows[0] if top_windows else None,
    )


# ---------- Seasonax-style derived metrics ----------


def _compute_pattern_return_frac(
    direction: str, entry_px: float, exit_px: float
) -> float:
    """Compute fractional return for a pattern trade.

    Args:
        direction: Trade direction, either 'long' or 'short'.
        entry_px: Entry price.
        exit_px: Exit price.

    Returns:
        Fractional return (e.g., 0.05 = 5% gain). Positive for profitable trades.

    Raises:
        ValueError: If entry_px is zero or direction is invalid.
    """
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
    """Enrich per-year results with return metrics and annualized returns.

    Adds columns for pattern trading days, returns (fractional and percentage),
    and annualized returns based on pattern holding period.

    Args:
        df: DataFrame with OHLCV data and datetime index, used to count trading days.
        per_year_results: DataFrame from score_seasonal_windows with per-year trades.
        trading_days_per_year: Number of trading days per year for annualization
            (typically 252 for equities, 250-260 for FX/futures).

    Returns:
        Copy of per_year_results with additional columns:
            - pattern_trading_days: Actual trading days in each pattern window
            - return_frac: Fractional return (e.g., 0.05 = 5%)
            - return_pct: Percentage return
            - annualised_return_frac: Annualized fractional return
            - annualised_return_pct: Annualized percentage return
    """
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
    result: SeasonalAnalysisResult,
    as_percent: bool,
) -> pd.Series:
    """Build cumulative profit series for a specific seasonal window.

    Args:
        result: SeasonalAnalysisResult from run_seasonal_analysis.
        as_percent: If True, compute cumulative returns as percentages. If False,
            compute cumulative P&L in points/pips.

    Returns:
        Series indexed by year with cumulative values. Name is either
        'cumulative_return_pct' or 'cumulative_pnl_points'.

    Raises:
        ValueError: If no trades exist for the specified window.
    """
    window = result.selected_window
    dfw = result.per_year_results[
        (result.per_year_results["entry_mmdd"] == window.entry_mmdd)
        & (result.per_year_results["exit_mmdd"] == window.exit_mmdd)
    ].copy()
    if dfw.empty:
        raise ValueError(
            f"No trades for window {window.entry_mmdd} -> {window.exit_mmdd}"
        )
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
    result: SeasonalAnalysisResult,
    trading_days_per_year: int = 252,
) -> Dict[str, Dict[str, float]]:
    """Compute comprehensive KPI summary for a specific seasonal window.

    Args:
        result: SeasonalAnalysisResult from run_seasonal_analysis.
        trading_days_per_year: Number of trading days per year for annualization.

    Returns:
        Nested dictionary with KPI categories:
            - basic: num_trades, win_rate
            - returns_pct: avg, median, min, max, std, avg_annualised, median_annualised
            - profit_points: avg, median, min, max, total
            - gains: num_gains, avg_gain_pct, median_gain_pct, best_gain_pct
            - losses: num_losses, avg_loss_pct, median_loss_pct, worst_loss_pct
            - cumulative: max_drawdown_points, end_cumulative_pnl_points
            - risk: sharpe_like (annualized return / std return)

    Raises:
        ValueError: If no trades exist for the specified window.
    """
    window = result.selected_window
    dfw = result.per_year_results[
        (result.per_year_results["entry_mmdd"] == window.entry_mmdd)
        & (result.per_year_results["exit_mmdd"] == window.exit_mmdd)
    ].copy()
    if dfw.empty:
        raise ValueError(
            f"No trades for window {window.entry_mmdd} -> {window.exit_mmdd}"
        )

    dfw = enrich_per_year_results_with_returns(result.df, dfw, trading_days_per_year)
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
    result: SeasonalAnalysisResult,
) -> Dict[str, float]:
    """Decompose buy-and-hold returns into pattern vs. rest-of-year performance.

    Computes log returns for pattern windows vs. non-pattern periods to determine
    how much of the total buy-and-hold return is attributable to the seasonal pattern.

    Args:
        result: SeasonalAnalysisResult from run_seasonal_analysis.

    Returns:
        Dictionary containing:
            - buyhold_return_frac: Total buy-and-hold fractional return
            - pattern_return_frac: Fractional return during pattern windows only
            - rest_return_frac: Fractional return during non-pattern periods
            - pattern_share_of_buyhold: Pattern return / buy-and-hold return
            - rest_share_of_buyhold: Rest return / buy-and-hold return

    Raises:
        ValueError: If no trades exist for the specified window or insufficient data.
    """
    window = result.selected_window
    df_local = _ensure_dt_index(result.df)
    dfw = result.per_year_results[
        (result.per_year_results["entry_mmdd"] == window.entry_mmdd)
        & (result.per_year_results["exit_mmdd"] == window.exit_mmdd)
    ].copy()
    if dfw.empty:
        raise ValueError(
            f"No trades for window {window.entry_mmdd} -> {window.exit_mmdd}"
        )

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
