"""Universe accuracy evaluation for seasonal windows (``SEAS-003B``).

This module turns the per-candidate validation mechanics into a repeatable,
universe-level accuracy artifact. It is deliberately **report-only**: it never
promotes a window and never gates a decision. The artifact is the source the
read-only insight API (``SEAS-003C``) serves.

Design notes
------------
- Discovery is in-sample and selection-biased. Every row carries a
  ``metric_scope`` split so an in-sample win rate can never be read as a
  probability.
- The data-snooping test is run over the **full candidate grid**, not a
  one-element list. The grid size is pinned before the run and reported.
- Candidate *reporting* windows (the rows written to the artifact) are the
  discovery leaders plus the windows already in use by the journal, so the
  Oct-Nov windows can be reproduced.
- Costs are netted with the ``SEAS-001D`` cost model; gross and net are kept
  side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .analysis import (
    CURVE_MODE_LINK_RELATIVES,
    _available_years,
    _calendar_mmdd_range,
    _calendar_window_days,
    _compute_pattern_return_frac,
    _compute_window_pnl_for_year,
    _pip_factor_for_symbol,
    build_seasonal_pattern_curve,
    build_window_return_matrix,
    generate_calendar_grid,
    score_seasonal_windows,
)
from .costs import CostModel, trade_cost_frac
from .data_sources import normalize_data_source
from .significance import hansen_spa, whites_reality_check
from .config_assets import AssetClass
from .universes import (
    CRYPTO,
    CRYPTO_SYMBOLS,
    FUTURES_CROSS_SECTOR,
    FUTURES_FULL,
    UniverseSpec,
    asset_class_for_symbol,
    crypto_universe_for_profile,
)

FOREX_UNIVERSE_28: Tuple[str, ...] = (
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "NZDUSD",
    "USDJPY",
    "USDCAD",
    "USDCHF",
    "EURGBP",
    "EURAUD",
    "EURJPY",
    "EURCHF",
    "EURCAD",
    "EURNZD",
    "GBPAUD",
    "GBPJPY",
    "GBPCHF",
    "GBPCAD",
    "GBPNZD",
    "AUDJPY",
    "AUDCHF",
    "AUDCAD",
    "AUDNZD",
    "NZDJPY",
    "NZDCHF",
    "NZDCAD",
    "CADJPY",
    "CADCHF",
    "CHFJPY",
)

FOREX_PROFILES: Dict[str, Tuple[str, ...]] = {
    "majors": (
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDJPY",
        "USDCAD",
        "USDCHF",
    ),
    "minors": tuple(s for s in FOREX_UNIVERSE_28 if s not in {
        "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF",
    }),
}
FOREX_PROFILES["all"] = FOREX_UNIVERSE_28

# Frozen futures universes (``SEAS-004``). Profiles are pinned to the frozen
# membership; they are never re-derived at runtime.
FUTURES_UNIVERSE_73: Tuple[str, ...] = FUTURES_FULL.symbols
FUTURES_UNIVERSE_24: Tuple[str, ...] = FUTURES_CROSS_SECTOR.symbols
FUTURES_PROFILES: Dict[str, Tuple[str, ...]] = {
    "all": FUTURES_UNIVERSE_73,
    "cross_sector": FUTURES_UNIVERSE_24,
}

# Frozen crypto universe (``H-006`` / ``SEAS-005``). The crypto IS protocol
# starts in 2018 (coverage varies by symbol), unlike the futures IS start.
CRYPTO_UNIVERSE_25: Tuple[str, ...] = CRYPTO_SYMBOLS
CRYPTO_PROFILES: Dict[str, Tuple[str, ...]] = {"all": CRYPTO_UNIVERSE_25}

PROTOCOL_SPLITS = {
    "is": {"start": FUTURES_FULL.is_start, "end": FUTURES_FULL.is_end},
    "oos_2024": {"start": "2024-01-01", "end": "2024-12-31"},
    "oos_2025": {"start": "2025-01-01", "end": FUTURES_FULL.oos_end},
}

# Crypto IS starts in 2018 and ends in 2023; OOS 2025 is not partial by
# construction (individual symbols may still have short coverage).
CRYPTO_PROTOCOL_SPLITS = {
    "is": {"start": CRYPTO.is_start, "end": CRYPTO.is_end},
    "oos_2024": {"start": "2024-01-01", "end": "2024-12-31"},
    "oos_2025": {"start": "2025-01-01", "end": "2025-12-31"},
}

# Windows already used by the journal / shared-patterns doc. Reproducing these
# from the artifact is an explicit acceptance criterion.
JOURNAL_WINDOWS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "USDJPY": (("10-07", "10-31"), ("10-21", "10-31")),
    "USDCHF": (("10-17", "10-29"),),
    "EURGBP": (("10-20", "11-21"),),
}

DISCOVERY_ENTRY_DAY_RANGE = (5, 25)
DISCOVERY_EXIT_DAY_RANGE = (10, 31)
DISCOVERY_MIN_LEN_DAYS = 5
DISCOVERY_MAX_LEN_DAYS = 45
DISCOVERY_MIN_TRADES = 8
DISCOVERY_MIN_WIN_RATE = 0.75
SAMPLE_MIN_INDEPENDENT_YEARS = 8
SAMPLE_MIN_OOS_TRADES = 5

SCHEMA_VERSION = 1


def full_candidate_family() -> List[Tuple[str, str]]:
    """The pinned candidate family used for data-snooping correction."""
    return generate_calendar_grid(
        list(range(1, 13)),
        DISCOVERY_ENTRY_DAY_RANGE,
        DISCOVERY_EXIT_DAY_RANGE,
        DISCOVERY_MIN_LEN_DAYS,
        DISCOVERY_MAX_LEN_DAYS,
    )


def _as_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _safe_mean(values: Iterable[Any]) -> Optional[float]:
    arr = np.asarray([_as_float(v) for v in values if _as_float(v) is not None], dtype=float)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _metrics(returns: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([float(v) for v in returns if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "num_trades": 0,
            "win_rate": None,
            "avg_return": None,
            "median_return": None,
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
    dd = cum - np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:] if cum.size else np.array([])
    mean = float(arr.mean())
    return {
        "num_trades": int(arr.size),
        "win_rate": float(wins.mean()),
        "avg_return": mean,
        "median_return": float(np.median(arr)),
        "worst_year_return": float(arr.min()),
        "max_drawdown_proxy": float(dd.min()) if dd.size else None,
        "profit_factor": profit_factor,
        "mae": float(np.mean(np.abs(arr - mean))),
    }


def _shift_mmdd_trading_days(mmdd: str, k: int) -> Optional[str]:
    bd = pd.bdate_range("2024-01-01", "2024-12-31")
    m, d = map(int, mmdd.split("-"))
    target = pd.Timestamp(2024, m, d)
    pos = int(bd.searchsorted(target))
    if pos >= len(bd):
        pos = len(bd) - 1
    new = pos + int(k)
    if new < 0 or new >= len(bd):
        return None
    nm = bd[new].strftime("%m-%d")
    if nm == "02-29":
        return None
    return nm


def _year_trade_rows(
    df: pd.DataFrame,
    symbol: str,
    entry_mmdd: str,
    exit_mmdd: str,
    direction: str,
    cost_model: Optional[CostModel] = None,
) -> List[Dict[str, Any]]:
    """Per-independent-year gross/net trade rows for one window."""
    years = _available_years(df)
    pip_factor = _pip_factor_for_symbol(symbol)
    pip = 1.0 / pip_factor if pip_factor else 1.0
    by_year = {int(y): df.loc[df.index.year == int(y)] for y in years}
    rows: List[Dict[str, Any]] = []
    for y in years:
        dyy = by_year.get(int(y))
        if dyy is None or dyy.empty:
            continue
        try:
            entry_dt, exit_dt = _calendar_window_days(int(y), entry_mmdd, exit_mmdd)
        except ValueError:
            continue
        res = _compute_window_pnl_for_year(dyy, entry_dt, exit_dt, direction, pip_factor)
        if not res:
            continue
        gross_frac = _compute_pattern_return_frac(
            direction, float(res["entry_px"]), float(res["exit_px"])
        )
        cost_frac = 0.0
        if cost_model is not None:
            cost_frac = float(
                trade_cost_frac(
                    float(res["entry_px"]),
                    pip,
                    cost_model,
                    res.get("entry_dt"),
                )
            )
        rows.append(
            {
                **res,
                "year": int(y),
                "gross_return": float(gross_frac),
                "cost_frac": cost_frac,
                "net_return": float(gross_frac - cost_frac),
            }
        )
    rows.sort(key=lambda r: int(r["year"]))
    return rows


def _max_drawdown(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return None
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:]
    return float(np.min(cum - peak))


def _walk_forward(rows: List[Dict[str, Any]], train_window: int = 5) -> Dict[str, Any]:
    """Rolling walk-forward folds over independent years."""
    years = [int(r["year"]) for r in rows]
    by_year = {int(r["year"]): r for r in rows}
    folds: List[Dict[str, Any]] = []
    if len(years) < train_window + 1:
        return {"split_mode": "walkforward", "folds": [], "test_metrics": _metrics([])}
    ordered = sorted(years)
    i = 0
    while i + train_window < len(ordered):
        train = ordered[i : i + train_window]
        test = [ordered[i + train_window]]
        test_returns = [
            float(by_year[y]["gross_return"]) for y in test if y in by_year
        ]
        folds.append(
            {
                "train_years": train,
                "test_years": test,
                "test_returns": test_returns,
            }
        )
        i += 1
    all_test = [v for f in folds for v in f["test_returns"]]
    return {
        "split_mode": "walkforward",
        "train_window_years": int(train_window),
        "folds": folds,
        "test_metrics": _metrics(all_test),
        "test_returns": all_test,
    }


def _holdout(rows: List[Dict[str, Any]], test_fraction: float = 0.3) -> Dict[str, Any]:
    years = sorted(int(r["year"]) for r in rows)
    by_year = {int(r["year"]): r for r in rows}
    if len(years) < 4:
        return {
            "split_mode": "percent_holdout",
            "train_years": years,
            "test_years": [],
            "train_metrics": _metrics([by_year[y]["gross_return"] for y in years]),
            "test_metrics": _metrics([]),
        }
    n_test = max(2, int(round(len(years) * test_fraction)))
    n_test = min(n_test, len(years) - 2)
    train_years = years[:-n_test]
    test_years = years[-n_test:]
    train_returns = [float(by_year[y]["gross_return"]) for y in train_years]
    test_returns = [float(by_year[y]["gross_return"]) for y in test_years]
    return {
        "split_mode": "percent_holdout",
        "train_years": train_years,
        "test_years": test_years,
        "train_metrics": _metrics(train_returns),
        "test_metrics": _metrics(test_returns),
        "train_returns": train_returns,
        "test_returns": test_returns,
    }


def _jitter(
    df: pd.DataFrame,
    symbol: str,
    entry_mmdd: str,
    exit_mmdd: str,
    direction: str,
    jitter_days: int = 2,
) -> Dict[str, Any]:
    variants: List[float] = []
    detail: List[Dict[str, Any]] = []
    for de in range(-jitter_days, jitter_days + 1):
        for dx in range(-jitter_days, jitter_days + 1):
            e2 = _shift_mmdd_trading_days(entry_mmdd, de)
            x2 = _shift_mmdd_trading_days(exit_mmdd, dx)
            if e2 is None or x2 is None or e2 > x2:
                continue
            rows = _year_trade_rows(df, symbol, e2, x2, direction)
            if not rows:
                continue
            mean = float(np.mean([r["gross_return"] for r in rows]))
            variants.append(mean)
            detail.append({"entry_shift": de, "exit_shift": dx, "mean_return": mean})
    if not variants:
        return {
            "jitter_days": int(jitter_days),
            "n_variants": 0,
            "min": None,
            "median": None,
            "max": None,
            "mean": None,
            "fraction_positive": None,
            "variants": [],
        }
    arr = np.asarray(variants, dtype=float)
    return {
        "jitter_days": int(jitter_days),
        "n_variants": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "fraction_positive": float((arr > 0).mean()),
        "variants": detail,
    }


def _subperiod(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = [float(r["gross_return"]) for r in rows]
    if not values:
        return {
            "n_subperiods": 2,
            "subperiod_means": [],
            "subperiod_counts": [],
            "overall_mean": None,
            "all_positive": None,
            "sign_consistency": None,
        }
    n = 2
    chunks = np.array_split(np.asarray(values, dtype=float), n)
    means = [float(c.mean()) if c.size else None for c in chunks]
    overall = float(np.mean(values))
    signs = [m for m in means if m is not None]
    same = sum(1 for m in signs if np.sign(m) == np.sign(overall)) if overall != 0 else 0
    return {
        "n_subperiods": n,
        "subperiod_means": means,
        "subperiod_counts": [int(c.size) for c in chunks],
        "overall_mean": overall,
        "all_positive": bool(all(m is not None and m > 0 for m in means)),
        "sign_consistency": float(same / len(signs)) if signs else None,
    }


def _path_profile(
    df: pd.DataFrame,
    symbol: str,
    entry_mmdd: str,
    exit_mmdd: str,
    direction: str,
    cost_model: Optional[CostModel] = None,
) -> List[Dict[str, Any]]:
    """Remaining-window expectancy by calendar day inside the window.

    For each calendar slot in the window, average across independent years the
    return from the close on/after that slot to the window exit. This lets the
    insight API return a remaining-window expectancy for "today" without
    recomputing anything on request.
    """
    slots = _calendar_mmdd_range(entry_mmdd, exit_mmdd)
    if not slots:
        return []
    years = _available_years(df)
    pip_factor = _pip_factor_for_symbol(symbol)
    pip = 1.0 / pip_factor if pip_factor else 1.0
    per_slot: Dict[str, List[float]] = {s: [] for s in slots}
    per_slot_net: Dict[str, List[float]] = {s: [] for s in slots}
    for y in years:
        dyy = df.loc[df.index.year == int(y)]
        if dyy.empty:
            continue
        try:
            entry_dt, exit_dt = _calendar_window_days(int(y), entry_mmdd, exit_mmdd)
        except ValueError:
            continue
        idx = dyy.index
        e = _snap_forward(idx, entry_dt)
        x = _snap_forward(idx, exit_dt)
        if e is None or x is None or x < e:
            continue
        slice_df = dyy.loc[e:x]
        if slice_df.empty:
            continue
        exit_px = float(dyy.loc[x, "close"])
        entry_px = float(dyy.loc[e, "close"])
        cost_frac = 0.0
        if cost_model is not None:
            cost_frac = float(trade_cost_frac(entry_px, pip, cost_model, e))
        seen: set[str] = set()
        for ts, row in slice_df.iterrows():
            mmdd = ts.strftime("%m-%d")
            if mmdd not in per_slot or mmdd in seen:
                continue
            seen.add(mmdd)
            px = float(row["close"])
            if not np.isfinite(px) or px == 0.0:
                continue
            ret = _compute_pattern_return_frac(direction, px, exit_px)
            per_slot[mmdd].append(float(ret))
            per_slot_net[mmdd].append(float(ret - cost_frac))
    out: List[Dict[str, Any]] = []
    for i, slot in enumerate(slots):
        gross = per_slot.get(slot) or []
        net = per_slot_net.get(slot) or []
        out.append(
            {
                "day_index": int(i + 1),
                "mmdd": slot,
                "n_years": int(len(gross)),
                "remaining_expected_return": _safe_mean(gross),
                "remaining_net_expected_return": _safe_mean(net),
                "remaining_win_rate": (
                    float(np.mean(np.asarray(gross) > 0)) if gross else None
                ),
            }
        )
    return out


def _snap_forward(idx: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    if target in idx:
        return target
    pos = int(idx.searchsorted(target))
    if pos >= len(idx):
        return None
    return idx[pos]


def _calibration_label(predicted: Optional[float], oos: Optional[float]) -> Dict[str, Any]:
    if predicted is None:
        bucket = "unknown"
    elif predicted >= 0.8:
        bucket = "high"
    elif predicted >= 0.7:
        bucket = "medium"
    elif predicted >= 0.6:
        bucket = "low"
    else:
        bucket = "very_low"
    gap = None
    if predicted is not None and oos is not None:
        gap = float(oos - predicted)
    return {
        "predicted_win_rate": predicted,
        "oos_hit_rate": oos,
        "confidence_bucket": bucket,
        "calibration_gap": gap,
    }


def _universe_for_config(
    universe: Optional[str],
    asset_class: Optional[str] = None,
    profile: Optional[str] = None,
) -> Optional[UniverseSpec]:
    """Resolve a frozen universe token for a non-forex asset class."""
    key = str(universe or profile or "").strip().lower()
    ac = str(asset_class or "").strip().lower()
    if ac == AssetClass.CRYPTO.value or key in {"crypto", "crypto_full"}:
        if not key or key in {"crypto", "crypto_full", "all"}:
            return crypto_universe_for_profile(profile or "all")
        raise ValueError(f"Unknown crypto universe {universe!r}")
    if not key:
        return None
    if key in {"futures", "futures_full"}:
        return FUTURES_FULL
    if key in {"futures_cross_sector", "cross_sector"}:
        return FUTURES_CROSS_SECTOR
    return None


def _futures_universe_for_config(
    universe: Optional[str], profile: Optional[str] = None
) -> Optional[UniverseSpec]:
    """Backward-compatible futures-only resolver."""
    return _universe_for_config(universe, "futures", profile)


def _point_value(symbol: str) -> Tuple[Optional[float], str]:
    """Contract point value in points, or (None, reason) when not known.

    A missing contract spec is recorded as ``NO DATA``; it is never guessed.
    """
    try:
        from trading_utils.contract_specs import get_futures_point_value

        return float(get_futures_point_value(symbol)), "ok"
    except Exception as exc:
        return None, f"NO DATA: {exc}"


def _protocol_split(
    rows: Sequence[Dict[str, Any]], *, is_start: int = 2010, oos_2025_partial: bool = True
) -> Dict[str, Any]:
    """IS / OOS 2024 / OOS 2025 protocol metrics for one window."""

    def _subset(start_year: int, end_year: int) -> Dict[str, Any]:
        subset = [
            r
            for r in rows
            if start_year <= int(r.get("year", 0)) <= end_year
        ]
        gross = [float(r["gross_return"]) for r in subset]
        net = [float(r["net_return"]) for r in subset]
        return {
            "years": [int(r["year"]) for r in subset],
            "gross": _metrics(gross),
            "net": _metrics(net),
        }

    splits = CRYPTO_PROTOCOL_SPLITS if is_start >= 2018 else PROTOCOL_SPLITS
    return {
        "protocol": {
            "is": dict(splits["is"]),
            "oos_2024": dict(splits["oos_2024"]),
            "oos_2025": dict(splits["oos_2025"]),
            "oos_2025_partial": bool(oos_2025_partial),
        },
        "is": _subset(is_start, 2023),
        "oos_2024": _subset(2024, 2024),
        "oos_2025": _subset(2025, 2025),
    }


@dataclass
class AccuracyConfig:
    curve_mode: str = CURVE_MODE_LINK_RELATIVES
    data_source: str = "dukascopy"
    cost_model: Optional[CostModel] = None
    include_context: bool = False
    spa_n_boot: int = 500
    seed: Optional[int] = 7
    universe: Optional[str] = None
    asset_class: Optional[str] = None
    profile: Optional[str] = None


def evaluate_symbol_accuracy(
    df: pd.DataFrame,
    symbol: str,
    *,
    config: Optional[AccuracyConfig] = None,
    journal_windows: Optional[Sequence[Tuple[str, str]]] = None,
    as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the per-symbol accuracy artifact payload."""
    cfg = config or AccuracyConfig()
    symbol_u = str(symbol).strip().upper()
    asset_class = str(
        cfg.asset_class or asset_class_for_symbol(symbol_u).value
    ).strip().lower()
    is_futures = asset_class == AssetClass.FUTURES.value
    is_crypto = asset_class == AssetClass.CRYPTO.value
    is_non_forex = is_futures or is_crypto
    universe_spec = (
        _universe_for_config(cfg.universe, asset_class, cfg.profile)
        if is_non_forex
        else None
    )
    default_source = "firstratedata" if is_non_forex else "dukascopy"
    requested_source = normalize_data_source(cfg.data_source, default=default_source)
    as_of = as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    years = _available_years(df)
    if not years:
        raise ValueError(f"No usable years for {symbol_u}")

    # Full family for data-snooping. Computed once, reused by both tests.
    family = full_candidate_family()
    matrix, matrix_years, matrix_keys = build_window_return_matrix(
        df,
        symbol_u,
        family,
        direction="auto",
        curve_mode=cfg.curve_mode,
        value_col="profit_pct",
    )
    spa = hansen_spa(matrix, n_boot=cfg.spa_n_boot, seed=cfg.seed) if matrix.size else {}
    rc = whites_reality_check(matrix, n_boot=cfg.spa_n_boot, seed=cfg.seed) if matrix.size else {}
    for payload in (spa, rc):
        if isinstance(payload, dict):
            payload.pop("keep_mask", None)

    # Discovery leaders (reportable rows).
    leaders, _per_year = score_seasonal_windows(
        df=df,
        symbol=symbol_u,
        windows=family,
        lookback_years=max(len(years), 1),
        direction="auto",
        min_trades=DISCOVERY_MIN_TRADES,
        min_win_rate=DISCOVERY_MIN_WIN_RATE,
        curve_mode=cfg.curve_mode,
        cost_model=cfg.cost_model,
    )

    pinned = list(journal_windows if journal_windows is not None else JOURNAL_WINDOWS.get(symbol_u, ()))
    reportable: List[Tuple[str, str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for w in leaders:
        key = (w.entry_mmdd, w.exit_mmdd)
        if key in seen:
            continue
        seen.add(key)
        reportable.append((w.entry_mmdd, w.exit_mmdd, w.direction))
    for key in pinned:
        if key in seen:
            continue
        seen.add(key)
        # Resolve direction using the same drift-free basis as discovery.
        direction = _resolve_direction(df, symbol_u, key[0], key[1], cfg.curve_mode)
        reportable.append((key[0], key[1], direction))

    windows_payload: List[Dict[str, Any]] = []
    for entry_mmdd, exit_mmdd, direction in reportable:
        rows = _year_trade_rows(
            df, symbol_u, entry_mmdd, exit_mmdd, direction, cfg.cost_model
        )
        gross = [float(r["gross_return"]) for r in rows]
        net = [float(r["net_return"]) for r in rows]
        full = _metrics(gross)
        net_metrics = _metrics(net)
        holdout = _holdout(rows)
        wf = _walk_forward(rows)
        oos_metric = wf.get("test_metrics") or {}
        oos_avg = oos_metric.get("avg_return")
        degradation = None
        if full.get("avg_return") is not None and oos_avg is not None:
            degradation = float(full["avg_return"] - oos_avg)
        independent_years = int(len(rows))
        sample_adequate = bool(
            independent_years >= SAMPLE_MIN_INDEPENDENT_YEARS
            and int(oos_metric.get("num_trades") or 0) >= SAMPLE_MIN_OOS_TRADES
        )
        calibration = _calibration_label(
            holdout.get("train_metrics", {}).get("win_rate"),
            holdout.get("test_metrics", {}).get("win_rate"),
        )
        window_payload: Dict[str, Any] = {
            "entry_mmdd": entry_mmdd,
            "exit_mmdd": exit_mmdd,
            "direction": direction,
            "metric_scope": {
                "discovery": "in_sample_discovery",
                "holdout": "out_of_sample",
                "walkforward": "out_of_sample",
            },
            "independent_years": independent_years,
            "years": [int(r["year"]) for r in rows],
            "sample_adequate": sample_adequate,
            "full_window": full,
            "net": {
                **net_metrics,
                "costs_included": cfg.cost_model is not None,
                "avg_cost_frac": _safe_mean([r["cost_frac"] for r in rows]),
            },
            "holdout": holdout,
            "walkforward": wf,
            "oos": {
                "test_years": wf.get("test_metrics", {}).get("num_trades"),
                "avg_return": oos_avg,
                "win_rate": oos_metric.get("win_rate"),
                "degradation": degradation,
            },
            "protocol_split": (
                _protocol_split(
                    rows,
                    is_start=2018 if is_crypto else 2010,
                    oos_2025_partial=not is_crypto,
                )
                if is_non_forex
                else None
            ),
            "robustness": {
                "jitter": _jitter(df, symbol_u, entry_mmdd, exit_mmdd, direction),
                "subperiod_consistency": _subperiod(rows),
            },
            "calibration": calibration,
            "path_profile": _path_profile(
                df, symbol_u, entry_mmdd, exit_mmdd, direction, cfg.cost_model
            ),
        }
        if cfg.include_context:
            try:
                from .context import build_context_payload

                window_payload["context"] = build_context_payload(
                    df,
                    symbol_u,
                    entry_mmdd,
                    exit_mmdd,
                    direction,
                    curve_mode=cfg.curve_mode,
                )
            except Exception as exc:  # pragma: no cover - defensive
                window_payload["context"] = {
                    "status": "unavailable",
                    "reason": f"context evaluation failed: {exc}",
                }
        windows_payload.append(window_payload)

    provenance: Dict[str, Any] = {}
    try:
        provenance = dict(df.attrs.get("data_provenance") or {})
    except Exception:
        provenance = {}
    actual_source = str(provenance.get("data_source") or requested_source)

    universe_name = universe_spec.name if universe_spec is not None else "forex"
    point_value, point_value_status = (
        _point_value(symbol_u) if is_futures else (None, "n/a")
    )
    spec_units = universe_spec.units if universe_spec is not None else "fractional_return"
    spec_annualization = (
        universe_spec.annualization_days if universe_spec is not None else None
    )
    spec_session = universe_spec.session_timezone if universe_spec is not None else None
    protocol_splits = CRYPTO_PROTOCOL_SPLITS if is_crypto else PROTOCOL_SPLITS
    protocol = None
    if is_non_forex:
        protocol = {
            "is": dict(protocol_splits["is"]),
            "oos_2024": dict(protocol_splits["oos_2024"]),
            "oos_2025": dict(protocol_splits["oos_2025"]),
            "oos_2025_partial": not is_crypto,
            "annualization_days": spec_annualization,
            "session_timezone": spec_session,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "seasonality_accuracy",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "as_of_date": as_of,
        "symbol": symbol_u,
        "asset_class": asset_class,
        "universe": universe_name,
        "universe_version": universe_spec.version if universe_spec is not None else None,
        "universe_hash": universe_spec.hash if universe_spec is not None else None,
        "universe_size": len(universe_spec.symbols) if universe_spec is not None else None,
        "data_source": actual_source,
        "requested_data_source": requested_source,
        "fallback_reason": provenance.get("fallback_reason"),
        "data_provenance": provenance,
        "adjustment": (
            provenance.get("adjustment") or (
                universe_spec.adjustment if universe_spec is not None else None
            )
        )
        if is_non_forex
        else None,
        "curve_mode": cfg.curve_mode,
        "costs_included": cfg.cost_model is not None,
        "cost_model": cfg.cost_model.to_dict() if cfg.cost_model is not None else None,
        "cost_basis": (
            "relative_bps"
            if cfg.cost_model is not None and cfg.cost_model.round_trip_bps is not None
            else ("pips" if cfg.cost_model is not None else None)
        ),
        "round_trip_bps": (
            cfg.cost_model.round_trip_bps if cfg.cost_model is not None else None
        ),
        "units": spec_units,
        "point_value": point_value,
        "point_value_status": point_value_status,
        "annualization_days": spec_annualization,
        "session_timezone": spec_session,
        "protocol_split": protocol,
        "fidelity_caveats": (
            list(universe_spec.fidelity_caveats) if universe_spec is not None else None
        ),
        "candidate_set": {
            "selection": "discovery leaders + pinned journal windows",
            "full_grid_size": int(len(family)),
            "n_candidates_used": int(len(matrix_keys)),
            "n_candidates_incomplete": int(len(family) - len(matrix_keys)),
            "n_periods": int(matrix.shape[0]) if matrix.size else 0,
            "pinned_journal_windows": [list(k) for k in pinned],
            "discovery_filters": {
                "min_trades": DISCOVERY_MIN_TRADES,
                "min_win_rate": DISCOVERY_MIN_WIN_RATE,
                "entry_day_range": list(DISCOVERY_ENTRY_DAY_RANGE),
                "exit_day_range": list(DISCOVERY_EXIT_DAY_RANGE),
                "min_len_days": DISCOVERY_MIN_LEN_DAYS,
                "max_len_days": DISCOVERY_MAX_LEN_DAYS,
            },
        },
        "data_snooping": {
            "full_grid_size": int(len(family)),
            "n_candidates_used": int(len(matrix_keys)),
            "n_candidates_incomplete": int(len(family) - len(matrix_keys)),
            "incomplete_reason": (
                "columns with a missing per-year return (e.g. an exit that snaps "
                "into the next calendar year) are dropped from the bootstrap matrix"
            ),
            "n_periods": int(matrix.shape[0]) if matrix.size else 0,
            "hansen_spa": spa,
            "reality_check": rc,
            "family_pinned_before_run": True,
            "single_test_p_value_is_data_snooping_corrected": False,
        },
        "windows": windows_payload,
    }


def _resolve_direction(
    df: pd.DataFrame,
    symbol: str,
    entry_mmdd: str,
    exit_mmdd: str,
    curve_mode: str,
) -> str:
    curve = build_seasonal_pattern_curve(
        df,
        lookback_years=max(len(_available_years(df)), 1),
        smooth=0,
        exclude_incomplete_last_year=False,
        curve_mode=curve_mode,
    )
    mean_lr = dict(zip(curve["calendar_day"], curve["mean_log_return"]))
    total = 0.0
    for mmdd in _calendar_mmdd_range(entry_mmdd, exit_mmdd):
        r = mean_lr.get(mmdd)
        if r is None or not np.isfinite(r):
            continue
        total += float(r)
    return "long" if total >= 0.0 else "short"


def build_accuracy_index(
    artifacts: Sequence[Dict[str, Any]],
    *,
    as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the symbol index for the insight API."""
    rows = []
    for art in artifacts:
        windows = []
        for w in art.get("windows") or []:
            windows.append(
                {
                    "entry_mmdd": w.get("entry_mmdd"),
                    "exit_mmdd": w.get("exit_mmdd"),
                    "direction": w.get("direction"),
                    "sample_adequate": w.get("sample_adequate"),
                    "oos_avg_return": (w.get("oos") or {}).get("avg_return"),
                    "net_avg_return": (w.get("net") or {}).get("avg_return"),
                }
            )
        rows.append(
            {
                "symbol": art.get("symbol"),
                "windows": windows,
                "generated_at": art.get("generated_at"),
                "freshness": "ok",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of_date": as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "kind": "seasonality_accuracy_index",
        "symbols": rows,
    }
