from __future__ import annotations

from seasonal_analysis.promotion import (
    build_promotion_report,
    render_promotion_markdown,
)


def _evidence(strong: bool = True) -> dict:
    return {
        "discovery": {
            "costs_included": True,
            "net_avg_profit_points": 30.0 if strong else -2.0,
            "net_win_rate": 0.75 if strong else 0.4,
        },
        "validation": {
            "split_mode": "holdout",
            "test_metrics": {"avg_return": 0.01 if strong else -0.01, "win_rate": 0.7, "num_trades": 10},
        },
        "stats": {
            "data_snooping": {
                "full_grid_size": 7216,
                "n_candidates_used": 600,
                "n_periods": 14,
                "spa": {"p_value": 0.01 if strong else 0.6, "n_strategies": 600},
                "reality_check": {"p_value": 0.02 if strong else 0.7},
            },
            "robustness": {
                "jitter": {"jitter_days": 2, "n_variants": 25, "median": 0.01 if strong else -0.01, "fraction_positive": 0.9 if strong else 0.2},
                "subperiod_consistency": {"all_positive": strong, "sign_consistency": 1.0 if strong else 0.0},
            },
            "min_sample": {"independent_years": 12, "test_trades": 12},
        },
        "assumptions": {
            "data_source": "dukascopy",
            "data_provenance": {"time_base": "GMT+0_US-DST", "quality": "canonical"},
            "costs_included": True,
            "cost_model": {"round_trip_pips": 2.6},
        },
    }


def test_report_structure_and_strong_verdict() -> None:
    report = build_promotion_report(
        window={"symbol": "EURUSD", "entry_mmdd": "02-01", "exit_mmdd": "03-15", "direction": "long", "direction_mode": "auto_curve", "curve_mode": "link_relatives"},
        evidence=_evidence(strong=True),
        generated_at="2026-10-16T00:00:00Z",
    )
    assert report["schema_version"] == 1
    assert report["promotion_kind"] == "human_decided"
    assert report["summary"]["overall_evidence"] == "strong"
    keys = {c["key"] for c in report["criteria"]}
    assert {"data_snooping", "out_of_sample", "sample_adequacy", "robustness_jitter", "cost", "data_provenance"} <= keys


def test_weak_evidence_is_insufficient() -> None:
    report = build_promotion_report(
        window={"symbol": "EURUSD", "entry_mmdd": "02-01", "exit_mmdd": "03-15", "direction": "long"},
        evidence=_evidence(strong=False),
    )
    assert report["summary"]["overall_evidence"] in {"mixed", "insufficient"}


def test_markdown_is_self_contained() -> None:
    report = build_promotion_report(
        window={"symbol": "EURUSD", "entry_mmdd": "02-01", "exit_mmdd": "03-15", "direction": "long", "direction_mode": "auto_curve", "curve_mode": "link_relatives"},
        evidence=_evidence(strong=True),
        generated_at="2026-10-16T00:00:00Z",
    )
    md = render_promotion_markdown(report)
    assert md.startswith("# Seasonal window promotion evidence")
    assert "Hansen SPA" in md
    assert "Data provenance" in md
    assert "dukascopy" in md
    assert "human" in md.lower()
