"""Plotting utilities for seasonal analysis."""

from __future__ import annotations
import pandas as pd


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
    from seasonal_analysis.analysis import build_seasonal_pattern_curve

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
