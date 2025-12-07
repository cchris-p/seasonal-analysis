from __future__ import annotations

from .analysis import (
    WindowStats,
    SeasonalAnalysisResult,
    compute_year_range_normalized_index,
    build_seasonal_pattern_curve,
    generate_calendar_grid,
    score_seasonal_windows,
    run_seasonal_analysis,
    enrich_per_year_results_with_returns,
    build_cumulative_profit_series,
    summarize_window_kpis,
    compute_pattern_vs_rest_vs_buy_and_hold,
    plot_seasonal_curve_with_windows,
    plot_per_year_pnl,
    plot_seasonal_stacks_by_lookback,
)
from .seasonal_filters import build_seasonal_filter

__all__ = [
    "WindowStats",
    "SeasonalAnalysisResult",
    "compute_year_range_normalized_index",
    "build_seasonal_pattern_curve",
    "generate_calendar_grid",
    "score_seasonal_windows",
    "run_seasonal_analysis",
    "enrich_per_year_results_with_returns",
    "build_cumulative_profit_series",
    "summarize_window_kpis",
    "compute_pattern_vs_rest_vs_buy_and_hold",
    "plot_seasonal_curve_with_windows",
    "plot_per_year_pnl",
    "plot_seasonal_stacks_by_lookback",
    "build_seasonal_filter",
]
