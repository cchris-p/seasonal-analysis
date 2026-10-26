"""Conditional context and TSM baselines for seasonal windows (``SEAS-003D``).

Report-only diagnostics attached to the accuracy artifact:

- conditional splits by causal trend (price vs 200-day MA), volatility (ATR
  percentile), and optional event proximity (requires a caller-supplied event
  calendar; otherwise explicitly ``unavailable``);
- TSM Ch.10 calendar-anomaly baseline (turn-of-month, January, October) with an
  overlap/attribution label;
- TSM Ch.11 periodogram as a **diagnostic only** — never a promotion gate.

Nothing here feeds an execution path or changes promotion.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .analysis import (
    _available_years,
    _calendar_mmdd_range,
    _calendar_window_days,
    _compute_pattern_return_frac,
    _compute_window_pnl_for_year,
    _pip_factor_for_symbol,
    build_seasonal_pattern_curve,
)

MIN_BUCKET_YEARS = 5

_REGIME_ORDER = ("up", "down")
_VOL_ORDER = ("low", "mid", "high")


def _snap_forward(idx: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    if target in idx:
        return target
    pos = int(idx.searchsorted(target))
    if pos >= len(idx):
        return None
    return idx[pos]


def _safe_mean(values: Iterable[Any]) -> Optional[float]:
    arr = np.asarray([float(v) for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def _bucket_stats(outcomes: Sequence[Tuple[str, float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in sorted({k for k, _ in outcomes}):
        vals = [v for k, v in outcomes if k == key]
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "n_years": int(arr.size),
            "adequate": bool(arr.size >= MIN_BUCKET_YEARS),
            "avg_return": float(arr.mean()) if arr.size else None,
            "win_rate": float((arr > 0).mean()) if arr.size else None,
            "suppressed": bool(arr.size < MIN_BUCKET_YEARS),
        }
    return out


def _regimes(
    df: pd.DataFrame,
    symbol: str,
    entry_mmdd: str,
    exit_mmdd: str,
    direction: str,
) -> Dict[str, Any]:
    pip_factor = _pip_factor_for_symbol(symbol)
    close = df["close"].astype(float)
    ma200 = close.rolling(200, min_periods=120).mean()
    tr = pd.concat(
        [
            df["high"].astype(float) - df["low"].astype(float),
            (df["high"].astype(float) - close.shift(1)).abs(),
            (df["low"].astype(float) - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean()
    atr_pct = atr14.rolling(252, min_periods=60).rank(pct=True)

    trend_out: List[Tuple[str, float]] = []
    vol_out: List[Tuple[str, float]] = []
    for y in _available_years(df):
        dyy = df.loc[df.index.year == int(y)]
        if dyy.empty:
            continue
        try:
            entry_dt, exit_dt = _calendar_window_days(int(y), entry_mmdd, exit_mmdd)
        except ValueError:
            continue
        res = _compute_window_pnl_for_year(
            dyy, entry_dt, exit_dt, direction, pip_factor
        )
        if not res:
            continue
        e = _snap_forward(df.index, entry_dt)
        if e is None:
            continue
        ret = _compute_pattern_return_frac(
            direction, float(res["entry_px"]), float(res["exit_px"])
        )
        ma_v = ma200.get(e)
        px = close.get(e)
        if ma_v is not None and px is not None and np.isfinite(ma_v) and np.isfinite(px):
            trend_out.append(("up" if float(px) >= float(ma_v) else "down", ret))
        pct = atr_pct.get(e)
        if pct is not None and np.isfinite(pct):
            p = float(pct)
            bucket = "low" if p <= 0.33 else "mid" if p <= 0.66 else "high"
            vol_out.append((bucket, ret))

    return {
        "trend": {
            "rule": "entry close vs 200-day MA (causal)",
            "min_bucket_years": MIN_BUCKET_YEARS,
            "buckets": _bucket_stats(trend_out),
        },
        "volatility": {
            "rule": "ATR(14) percentile vs trailing 252 days; terciles",
            "min_bucket_years": MIN_BUCKET_YEARS,
            "buckets": _bucket_stats(vol_out),
        },
    }


def _event_proximity(
    event_dates: Optional[Sequence[str]], window_days: int = 2
) -> Dict[str, Any]:
    if not event_dates:
        return {
            "status": "unavailable",
            "reason": "no event calendar supplied; numeric buckets are not reported",
            "window_days": int(window_days),
        }
    return {
        "status": "available",
        "n_events": int(len(event_dates)),
        "window_days": int(window_days),
        "note": "event proximity requires per-year outcomes; supplied events are "
        "recorded but not scored by this artifact without a calendar join",
    }


def _calendar_baseline(
    df: pd.DataFrame,
    entry_mmdd: str,
    exit_mmdd: str,
    direction: str,
) -> Dict[str, Any]:
    """Turn-of-month / January / October baseline and window overlap."""
    close = df["close"].astype(float)
    log_ret = np.log(close / close.groupby(close.index.year).shift(1))
    log_ret = log_ret.dropna()
    if log_ret.empty:
        return {"status": "unavailable", "reason": "no returns"}

    # Turn-of-month: last 3 trading days of month + first 3 of next month.
    idx = log_ret.index
    month_end = idx.to_period("M").to_timestamp("M")
    is_last_3 = (month_end - idx).days <= 4
    next_month_first = idx.day <= 3
    tom_mask = is_last_3 | next_month_first
    jan_mask = idx.month == 1
    oct_mask = idx.month == 10

    tom_mask = np.asarray(tom_mask, dtype=bool)
    jan_mask = np.asarray(jan_mask, dtype=bool)
    oct_mask = np.asarray(oct_mask, dtype=bool)

    def _mean(mask: np.ndarray) -> Optional[float]:
        arr = log_ret[mask]
        return float(arr.mean()) if len(arr) else None

    window_mmdd = set(_calendar_mmdd_range(entry_mmdd, exit_mmdd))
    window_len = len(window_mmdd)
    overlap_tom = sum(1 for d in window_mmdd if _mmdd_in_tom(d)) / window_len if window_len else 0.0
    overlap_jan = sum(1 for d in window_mmdd if d.startswith("01-")) / window_len if window_len else 0.0
    overlap_oct = sum(1 for d in window_mmdd if d.startswith("10-")) / window_len if window_len else 0.0
    max_overlap = max(overlap_tom, overlap_jan, overlap_oct)

    if max_overlap >= 0.8:
        label = "calendar_explained"
    elif max_overlap >= 0.5:
        label = "partly_explained"
    else:
        label = "distinct"

    return {
        "status": "available",
        "baseline_mean_log_return": {
            "turn_of_month": _mean(tom_mask),
            "january": _mean(jan_mask),
            "october": _mean(oct_mask),
            "all_days": float(log_ret.mean()),
        },
        "window_overlap_fraction": {
            "turn_of_month": float(overlap_tom),
            "january": float(overlap_jan),
            "october": float(overlap_oct),
        },
        "attribution_label": label,
        "direction": direction,
        "note": "diagnostic only; a calendar-explainable window is not evidence "
        "of an independent seasonal effect",
    }


def _mmdd_in_tom(mmdd: str) -> bool:
    m, d = map(int, mmdd.split("-"))
    if d <= 3:
        return True
    # Last three calendar days of a month (weekend snapping not applied here).
    last = pd.Timestamp(2024, m, 1) + pd.offsets.MonthEnd(0)
    return d >= int(last.day) - 2


def _cycle_diagnostic(
    df: pd.DataFrame, curve_mode: str
) -> Dict[str, Any]:
    try:
        curve = build_seasonal_pattern_curve(
            df,
            lookback_years=max(len(_available_years(df)), 1),
            smooth=0,
            exclude_incomplete_last_year=False,
            curve_mode=curve_mode,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "unavailable", "reason": str(exc)}
    series = curve["seasonal_index"].astype(float).to_numpy()
    if series.size < 32 or not np.isfinite(series).all():
        return {"status": "unavailable", "reason": "insufficient curve"}
    series = series - series.mean()
    power = np.abs(np.fft.rfft(series)) ** 2
    freqs = np.fft.rfftfreq(series.size, d=1.0)
    if power.size <= 1:
        return {"status": "unavailable", "reason": "insufficient spectrum"}
    power[0] = 0.0  # remove the mean
    k = int(np.argmax(power))
    total = float(power[1:].sum())
    share = float(power[k] / total) if total > 0 else None
    period_days = float(1.0 / freqs[k]) if freqs[k] > 0 else None
    return {
        "status": "diagnostic_only",
        "dominant_period_days": period_days,
        "dominant_power_share": share,
        "label": (
            "periodic_consistent"
            if period_days is not None and 20.0 <= period_days <= 400.0 and (share or 0) >= 0.1
            else "inconclusive"
        ),
        "note": "TSM Ch.11 periodogram is a diagnostic, not a promotion gate",
    }


def build_context_payload(
    df: pd.DataFrame,
    symbol: str,
    entry_mmdd: str,
    exit_mmdd: str,
    direction: str,
    *,
    curve_mode: str = "link_relatives",
    event_dates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "status": "available",
        "trend_and_volatility": _regimes(df, symbol, entry_mmdd, exit_mmdd, direction),
        "event_proximity": _event_proximity(event_dates),
        "calendar_baseline": _calendar_baseline(df, entry_mmdd, exit_mmdd, direction),
        "cycle_diagnostic": _cycle_diagnostic(df, curve_mode),
    }
