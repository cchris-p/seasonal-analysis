import numpy as np
import pandas as pd

from seasonal_analysis import accuracy
from seasonal_analysis.accuracy import (
    AccuracyConfig,
    build_accuracy_index,
    evaluate_symbol_accuracy,
)
from seasonal_analysis.costs import CostModel


def _synthetic_df(years=range(2010, 2025)):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2010-01-01", "2024-12-31")
    idx = idx[idx.year.isin(list(years))]
    seasonal = np.zeros(len(idx))
    mmdd = idx.strftime("%m-%d")
    seasonal[(mmdd >= "10-07") & (mmdd <= "10-31")] = 0.0015
    noise = rng.normal(0.0, 0.004, len(idx))
    close = 1.0 * np.exp(np.cumsum(seasonal + noise))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )
    df.attrs["data_provenance"] = {
        "data_source": "dukascopy",
        "time_base": "GMT+0_US-DST",
    }
    return df


def test_evaluate_symbol_accuracy_schema(monkeypatch):
    df = _synthetic_df()
    monkeypatch.setattr(
        accuracy,
        "full_candidate_family",
        lambda: [("10-07", "10-31"), ("10-21", "10-31"), ("03-01", "03-20")],
    )
    art = evaluate_symbol_accuracy(
        df,
        "EURUSD",
        config=AccuracyConfig(
            curve_mode="link_relatives",
            data_source="dukascopy_gmt+0_us-dst",
            cost_model=CostModel(spread_pips=0.5),
            include_context=True,
            spa_n_boot=50,
        ),
        as_of_date="2026-10-26",
    )

    assert art["schema_version"] == 1
    assert art["symbol"] == "EURUSD"
    assert art["data_source"] == "dukascopy"
    assert art["curve_mode"] == "link_relatives"
    assert art["costs_included"] is True
    assert art["candidate_set"]["full_grid_size"] == 3
    assert art["data_snooping"]["n_candidates_used"] > 0
    assert art["data_snooping"]["hansen_spa"]["n_strategies"] >= 1
    assert art["data_snooping"]["single_test_p_value_is_data_snooping_corrected"] is False

    windows = {f"{w['entry_mmdd']}->{w['exit_mmdd']}": w for w in art["windows"]}
    assert "10-07->10-31" in windows
    w = windows["10-07->10-31"]
    for key in ("full_window", "net", "oos", "robustness", "calibration", "path_profile"):
        assert key in w
    assert w["metric_scope"]["discovery"] == "in_sample_discovery"
    assert isinstance(w["sample_adequate"], bool)
    assert w["path_profile"]
    assert w["context"]["status"] == "available"
    assert w["context"]["calendar_baseline"]["attribution_label"] in {
        "calendar_explained",
        "partly_explained",
        "distinct",
    }
    assert w["context"]["cycle_diagnostic"]["status"] == "diagnostic_only"


def test_build_accuracy_index():
    art = {
        "symbol": "EURUSD",
        "generated_at": "2026-10-26T00:00:00Z",
        "windows": [
            {
                "entry_mmdd": "10-07",
                "exit_mmdd": "10-31",
                "direction": "long",
                "sample_adequate": True,
                "oos": {"avg_return": 0.01},
                "net": {"avg_return": 0.008},
            }
        ],
    }
    idx = build_accuracy_index([art], as_of_date="2026-10-26")
    assert idx["kind"] == "seasonality_accuracy_index"
    assert idx["symbols"][0]["symbol"] == "EURUSD"
    assert idx["symbols"][0]["windows"][0]["entry_mmdd"] == "10-07"
