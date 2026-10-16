from __future__ import annotations

from .analysis import (
    WindowStats,
    SeasonalAnalysisResult,
    CURVE_MODE_LOG_RETURN_COMPOUNDED,
    CURVE_MODE_LINK_RELATIVES,
    CURVE_MODE_YEARLY_AVERAGE,
    CURVE_MODES,
    compute_year_range_normalized_index,
    build_seasonal_pattern_curve,
    generate_calendar_grid,
    score_seasonal_windows,
    run_seasonal_analysis,
    enrich_per_year_results_with_returns,
    build_cumulative_profit_series,
    summarize_window_kpis,
    compute_pattern_vs_rest_vs_buy_and_hold,
    build_window_return_matrix,
)
from .plotting import (
    plot_seasonal_curve_with_windows,
    plot_per_year_pnl,
    plot_seasonal_stacks_by_lookback,
)
from .seasonal_filters import build_seasonal_filter
from .poi_charts import compute_poi_charts_payload
from .costs import CostModel, lookup_spread, trade_cost_frac
from .spread_estimates import (
    SPREAD_ESTIMATES_PIPS,
    SPREAD_ESTIMATES_SOURCE,
    default_cost_model,
    estimated_spread_pips,
)
from .promotion import build_promotion_report, render_promotion_markdown
from .significance import (
    hansen_spa,
    jitter_return_distribution,
    stationary_bootstrap_indices,
    subperiod_consistency,
    whites_reality_check,
)

__all__ = [
    "WindowStats",
    "SeasonalAnalysisResult",
    "CURVE_MODE_LOG_RETURN_COMPOUNDED",
    "CURVE_MODE_LINK_RELATIVES",
    "CURVE_MODE_YEARLY_AVERAGE",
    "CURVE_MODES",
    "compute_year_range_normalized_index",
    "build_seasonal_pattern_curve",
    "generate_calendar_grid",
    "score_seasonal_windows",
    "run_seasonal_analysis",
    "enrich_per_year_results_with_returns",
    "build_cumulative_profit_series",
    "summarize_window_kpis",
    "compute_pattern_vs_rest_vs_buy_and_hold",
    "build_window_return_matrix",
    "plot_seasonal_curve_with_windows",
    "plot_per_year_pnl",
    "plot_seasonal_stacks_by_lookback",
    "build_seasonal_filter",
    "compute_poi_charts_payload",
    "hansen_spa",
    "whites_reality_check",
    "stationary_bootstrap_indices",
    "subperiod_consistency",
    "jitter_return_distribution",
    "CostModel",
    "lookup_spread",
    "trade_cost_frac",
    "SPREAD_ESTIMATES_PIPS",
    "SPREAD_ESTIMATES_SOURCE",
    "estimated_spread_pips",
    "default_cost_model",
    "build_promotion_report",
    "render_promotion_markdown",
]
