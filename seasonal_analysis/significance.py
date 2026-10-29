"""Data-snooping significance tests and robustness helpers for seasonal windows.

Implements the multiple-comparison / data-snooping machinery required by
``SEAS-001C``:

- :func:`whites_reality_check` — White's (2000) Reality Check on the maximum mean
  excess return across a family of candidate strategies/windows.
- :func:`hansen_spa` — Hansen's (2005) Superior Predictive Ability test, the
  headline test (recentres only strategies with non-negative mean performance).
- :func:`stationary_bootstrap_indices` — Politis–Romano (1994) stationary
  bootstrap used by both tests.
- :func:`jitter_return_distribution` — robustness of a window's return to
  ``+/- n`` trading-day shifts of its entry/exit.
- :func:`subperiod_consistency` — early/late subperiod consistency of a return
  series.

All functions are pure (no I/O) and return JSON-serializable dicts, so the
validation worker and the promotion evidence report can consume them directly.
Criteria are **reported, not auto-gated**.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Stationary bootstrap (Politis & Romano, 1994)
# ---------------------------------------------------------------------------
def stationary_bootstrap_indices(
    n_periods: int,
    n_boot: int,
    expected_block: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate stationary-bootstrap resample index matrices.

    Args:
        n_periods: Number of time periods ``T`` (rows).
        n_boot: Number of bootstrap resamples.
        expected_block: Expected block length ``q >= 1`` (mean block size).
        rng: NumPy random generator.

    Returns:
        ``(n_boot, T)`` int array of time indices in ``[0, T)``.
    """
    T = int(n_periods)
    B = int(n_boot)
    if T <= 0:
        raise ValueError("n_periods must be positive")
    q = float(expected_block)
    if not np.isfinite(q) or q < 1.0:
        q = 1.0
    p = 1.0 / q

    out = np.empty((B, T), dtype=np.int64)
    for b in range(B):
        t = 0
        while t < T:
            start = int(rng.integers(0, T))
            length = int(rng.geometric(p))
            for j in range(length):
                if t >= T:
                    break
                out[b, t] = (start + j) % T
                t += 1
    return out


def _complete_columns(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop columns (strategies) with any NaN; return (clean_matrix, keep_mask)."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-D (n_periods x n_strategies)")
    mask = np.isfinite(arr).all(axis=0)
    return arr[:, mask], mask


def _spa_p_from_bootstrap(
    observed: float, boot_stats: np.ndarray, n_boot: int
) -> float:
    more_extreme = int(np.sum(boot_stats >= observed))
    return float((more_extreme + 1) / (n_boot + 1))


def whites_reality_check(
    returns: np.ndarray,
    benchmark: float = 0.0,
    n_boot: int = 2000,
    expected_block: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """White's Reality Check for the best strategy in a family.

    Args:
        returns: ``(T, M)`` array of per-period returns, one column per candidate
            strategy/window.
        benchmark: Benchmark return per period (scalar; default 0).
        n_boot: Number of stationary-bootstrap resamples.
        expected_block: Expected bootstrap block length (default ``min(5, T//3)+1``).
        seed: Optional RNG seed.

    Returns:
        Dict with ``test``, ``n_periods``, ``n_strategies``, ``observed_stat``,
        ``p_value``, ``n_boot``, ``expected_block``, and the best strategy index.
    """
    clean, keep = _complete_columns(returns)
    T = clean.shape[0]
    if T == 0 or clean.shape[1] == 0:
        return {
            "test": "white_reality_check",
            "n_periods": int(T),
            "n_strategies": int(clean.shape[1]),
            "observed_stat": None,
            "p_value": None,
            "n_boot": int(n_boot),
            "expected_block": None,
            "best_strategy_index": None,
        }

    d = clean - float(benchmark)  # (T, M) excess returns
    mean_d = d.mean(axis=0)  # (M,)
    f_bar = np.sqrt(T) * mean_d  # (M,) studentized mean
    observed = float(np.max(f_bar))
    best_k = int(np.argmax(f_bar))

    if expected_block is None:
        expected_block = min(5.0, max(2.0, T / 3.0))
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(T, int(n_boot), float(expected_block), rng)

    boot_stats = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        sample = d[idx[b], :]  # (T, M)
        boot_mean = sample.mean(axis=0)
        boot_stats[b] = np.max(np.sqrt(T) * (boot_mean - mean_d))

    p_value = _spa_p_from_bootstrap(observed, boot_stats, int(n_boot))

    return {
        "test": "white_reality_check",
        "n_periods": int(T),
        "n_strategies": int(clean.shape[1]),
        "observed_stat": observed,
        "p_value": p_value,
        "n_boot": int(n_boot),
        "expected_block": float(expected_block),
        "best_strategy_index": best_k,
        "keep_mask": keep.tolist(),
    }


def hansen_spa(
    returns: np.ndarray,
    benchmark: float = 0.0,
    n_boot: int = 2000,
    expected_block: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Hansen's Superior Predictive Ability (SPA) test (headline test).

    Uses the "consistent" recentering: only strategies with non-negative sample
    mean excess return are recentred by their sample mean; inferior strategies
    (negative mean) are recentred at zero, which removes their influence on the
    null distribution.

    Args:
        returns: ``(T, M)`` per-period returns (columns = strategies/windows).
        benchmark: Benchmark return per period (scalar; default 0).
        n_boot: Number of stationary-bootstrap resamples.
        expected_block: Expected bootstrap block length.
        seed: Optional RNG seed.

    Returns:
        Same shape as :func:`whites_reality_check` with ``test="hansen_spa"``.
    """
    clean, keep = _complete_columns(returns)
    T = clean.shape[0]
    if T == 0 or clean.shape[1] == 0:
        return {
            "test": "hansen_spa",
            "n_periods": int(T),
            "n_strategies": int(clean.shape[1]),
            "observed_stat": None,
            "p_value": None,
            "n_boot": int(n_boot),
            "expected_block": None,
            "best_strategy_index": None,
        }

    d = clean - float(benchmark)
    mean_d = d.mean(axis=0)  # (M,)
    f_bar = np.sqrt(T) * mean_d
    observed = float(max(0.0, np.max(f_bar)))
    best_k = int(np.argmax(f_bar))

    # Consistent recentering: recentre only strategies with non-negative mean.
    recenter = np.where(mean_d >= 0.0, mean_d, 0.0)

    if expected_block is None:
        expected_block = min(5.0, max(2.0, T / 3.0))
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(T, int(n_boot), float(expected_block), rng)

    boot_stats = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        sample = d[idx[b], :]
        boot_mean = sample.mean(axis=0)
        centered = np.sqrt(T) * (boot_mean - recenter)
        boot_stats[b] = max(0.0, float(np.max(centered)))

    p_value = _spa_p_from_bootstrap(observed, boot_stats, int(n_boot))

    return {
        "test": "hansen_spa",
        "n_periods": int(T),
        "n_strategies": int(clean.shape[1]),
        "observed_stat": observed,
        "p_value": p_value,
        "n_boot": int(n_boot),
        "expected_block": float(expected_block),
        "best_strategy_index": best_k,
        "keep_mask": keep.tolist(),
    }


# ---------------------------------------------------------------------------
# Robustness helpers
# ---------------------------------------------------------------------------
def subperiod_consistency(
    values: Sequence[float], n_subperiods: int = 2
) -> Dict[str, Any]:
    """Split an ordered return series into contiguous subperiods and summarise.

    Args:
        values: Ordered per-period returns (e.g. per-year).
        n_subperiods: Number of contiguous subperiods (default 2 = early/late).

    Returns:
        Dict with per-subperiod means/counts, ``all_positive`` (every subperiod
        mean > 0), and ``sign_consistency`` (fraction of subperiods with the same
        sign as the overall mean).
    """
    arr = np.asarray([float(v) for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n_subperiods": int(n_subperiods),
            "subperiod_means": [],
            "subperiod_counts": [],
            "overall_mean": None,
            "all_positive": None,
            "sign_consistency": None,
        }
    n = max(1, int(n_subperiods))
    chunks = np.array_split(arr, n)
    means = [float(c.mean()) if c.size else None for c in chunks]
    counts = [int(c.size) for c in chunks]
    overall = float(arr.mean())
    signs = [m for m in means if m is not None]
    if overall == 0.0 or not signs:
        sign_consistency = None
    else:
        same = sum(1 for m in signs if np.sign(m) == np.sign(overall))
        sign_consistency = float(same) / float(len(signs))
    return {
        "n_subperiods": n,
        "subperiod_means": means,
        "subperiod_counts": counts,
        "overall_mean": overall,
        "all_positive": bool(all(m is not None and m > 0 for m in means)),
        "sign_consistency": sign_consistency,
    }


def jitter_return_distribution(
    returns_by_shift: Sequence[float], jitter_days: int = 2
) -> Dict[str, Any]:
    """Summarise the distribution of a window's return under date shifts.

    Args:
        returns_by_shift: Mean returns for each jitter variant (including the
            unshifted variant).
        jitter_days: The ``+/- n`` trading-day shift magnitude used to generate
            the variants.

    Returns:
        Dict with ``n_variants``, ``min``, ``median``, ``max``, ``mean``, and
        ``fraction_positive`` (share of variants with a positive mean return).
    """
    arr = np.asarray([float(v) for v in returns_by_shift if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "jitter_days": int(jitter_days),
            "n_variants": 0,
            "min": None,
            "median": None,
            "max": None,
            "mean": None,
            "fraction_positive": None,
        }
    return {
        "jitter_days": int(jitter_days),
        "n_variants": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "fraction_positive": float(np.mean(arr > 0.0)),
    }


# ---------------------------------------------------------------------------
# Shared validation primitives (TradeSurge worker + promotion evidence)
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float], n: int = 1000, seed: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Bootstrap the mean of ``values`` and return a 95% percentile interval."""
    arr = np.asarray([float(v) for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    n_i = int(n)
    if n_i <= 0:
        return None
    rng = np.random.default_rng(None if seed is None else int(seed))
    idx = rng.integers(0, arr.size, size=(n_i, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "n": n_i,
        "mean": float(arr.mean()),
        "p2_5": float(np.quantile(means, 0.025)),
        "p97_5": float(np.quantile(means, 0.975)),
    }


def sign_permutation_p_value(
    values: Sequence[float], n: int = 2000, seed: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Two-sided sign-flip permutation test on the mean of ``values``."""
    arr = np.asarray([float(v) for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    n_i = int(n)
    if n_i <= 0:
        return None
    rng = np.random.default_rng(None if seed is None else int(seed))
    observed = float(arr.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_i, arr.size), replace=True)
    perm_means = np.mean(signs * arr, axis=1)
    more_extreme = int(np.sum(np.abs(perm_means) >= abs(observed)))
    return {
        "n": n_i,
        "seed": None if seed is None else int(seed),
        "alternative": "two_sided",
        "observed_mean": observed,
        "p_value": float((more_extreme + 1) / (n_i + 1)),
    }


def adjust_pvalues_bonferroni(
    p_values: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """Bonferroni-adjust a list of p-values (``None`` entries pass through)."""
    valid = [p for p in p_values if isinstance(p, (int, float))]
    m = int(len(valid))
    return [
        None if not isinstance(p, (int, float)) else float(min(1.0, float(p) * m))
        for p in p_values
    ]


def adjust_pvalues_benjamini_hochberg(
    p_values: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """Benjamini–Hochberg q-values with a monotone step-up correction."""
    indexed: List[Tuple[int, float]] = [
        (int(idx), float(p))
        for idx, p in enumerate(p_values)
        if isinstance(p, (int, float))
    ]
    m = int(len(indexed))
    if m == 0:
        return [None for _ in p_values]

    indexed_sorted = sorted(indexed, key=lambda t: t[1])
    q_by_sorted = [
        float(p) * m / float(rank)
        for rank, (_idx, p) in enumerate(indexed_sorted, start=1)
    ]

    running = 1.0
    q_mono: List[float] = []
    for q in reversed(q_by_sorted):
        running = float(min(running, q))
        q_mono.append(running)
    q_mono = list(reversed(q_mono))

    out: List[Optional[float]] = [None for _ in p_values]
    for (orig_idx, _p), q in zip(indexed_sorted, q_mono):
        out[int(orig_idx)] = float(min(1.0, q))
    return out
