from __future__ import annotations

import numpy as np

from seasonal_analysis.significance import (
    hansen_spa,
    jitter_return_distribution,
    stationary_bootstrap_indices,
    subperiod_consistency,
    whites_reality_check,
)


def test_stationary_bootstrap_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(20, 100, 5.0, rng)
    assert idx.shape == (100, 20)
    assert idx.min() >= 0 and idx.max() < 20


def test_reality_check_detects_dominant_strategy() -> None:
    rng = np.random.default_rng(1)
    T, M = 200, 40
    noise = rng.normal(0.0, 0.01, size=(T, M))
    noise[:, 0] += 0.02  # one strongly superior strategy
    rc = whites_reality_check(noise, n_boot=500, seed=7)
    assert rc["test"] == "white_reality_check"
    assert rc["best_strategy_index"] == 0
    assert rc["p_value"] < 0.05
    assert 0.0 <= rc["p_value"] <= 1.0


def test_reality_check_null_is_not_significant() -> None:
    rng = np.random.default_rng(2)
    noise = rng.normal(0.0, 0.01, size=(200, 40))
    rc = whites_reality_check(noise, n_boot=500, seed=11)
    assert rc["p_value"] > 0.10


def test_spa_detects_dominant_and_flags_inferior() -> None:
    rng = np.random.default_rng(3)
    T, M = 200, 40
    d = rng.normal(0.0, 0.01, size=(T, M))
    d[:, 0] += 0.03
    d[:, 1] -= 0.03  # inferior strategy should not affect the null distribution
    spa = hansen_spa(d, n_boot=500, seed=13)
    assert spa["test"] == "hansen_spa"
    assert spa["p_value"] < 0.05
    assert spa["observed_stat"] >= 0.0


def test_spa_null_is_not_significant() -> None:
    rng = np.random.default_rng(4)
    noise = rng.normal(0.0, 0.01, size=(200, 40))
    spa = hansen_spa(noise, n_boot=500, seed=17)
    assert spa["p_value"] > 0.10


def test_tests_handle_empty_and_nan_columns() -> None:
    rc = whites_reality_check(np.empty((0, 0)))
    assert rc["p_value"] is None
    spa = hansen_spa(np.empty((0, 0)))
    assert spa["p_value"] is None

    mat = np.array(
        [
            [0.01, np.nan, 0.02],
            [0.02, np.nan, 0.01],
            [0.03, np.nan, 0.015],
        ]
    )
    rc2 = whites_reality_check(mat, n_boot=50, seed=1)
    assert rc2["n_strategies"] == 2  # NaN column dropped


def test_subperiod_consistency() -> None:
    assert subperiod_consistency([0.01, 0.02, -0.01, -0.02])["all_positive"] is False
    consistent = subperiod_consistency([0.01, 0.02, 0.005, 0.03])
    assert consistent["all_positive"] is True
    assert consistent["sign_consistency"] == 1.0
    empty = subperiod_consistency([])
    assert empty["all_positive"] is None


def test_jitter_return_distribution() -> None:
    dist = jitter_return_distribution([0.01, 0.02, -0.005, 0.015], jitter_days=2)
    assert dist["n_variants"] == 4
    assert dist["fraction_positive"] == 0.75
    assert dist["min"] == -0.005
    assert jitter_return_distribution([])["n_variants"] == 0
