from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List
import shutil
import pandas as pd
import sys

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seasonal_analysis.constants import (
    DEFAULT_SYMBOL,
    DEFAULT_GRANULARITY,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    LOOKBACK_YEARS,
    MONTHS,
    ENTRY_DAY_RANGE,
    EXIT_DAY_RANGE,
    MIN_LEN_DAYS,
    MAX_LEN_DAYS,
    DIRECTION,
    SMOOTH,
    EXCLUDE_INCOMPLETE_LAST_YEAR,
    results_dir_for,
)
from seasonal_analysis.load_data import load_price_data
from seasonal_analysis.analysis import (
    run_seasonal_analysis,
)


def save_csv_reports(out_dir: Path, res) -> None:
    sc: pd.DataFrame = res.seasonal_curve
    sc.to_csv(out_dir / "seasonal_curve.csv", index=False)

    per_year: pd.DataFrame = res.per_year_results
    per_year.to_csv(out_dir / "per_year_results.csv", index=True)

    # top windows list of dataclasses
    tw = res.top_windows
    if tw:
        df_tw = pd.DataFrame([asdict(w) for w in tw])
        df_tw.to_csv(out_dir / "top_windows.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "top_windows.csv", index=False)


def save_plot_seasonal_curve_with_windows(
    out_dir: Path, seasonal_curve: pd.DataFrame, windows: List
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Skipping seasonal curve plot (matplotlib unavailable): {e}")
        return
    sc = seasonal_curve.copy().sort_values("calendar_day")
    sc["ts"] = pd.to_datetime("2024-" + sc["calendar_day"])  # common plotting year
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sc["ts"], sc["seasonal_index"], lw=2)
    ax.set_title("Seasonal index with top windows")
    ax.set_ylabel("Seasonal index (0–100)")
    ax.set_xlim(sc["ts"].min(), sc["ts"].max())
    ax.grid(alpha=0.3)

    for w in windows:
        entry = w.entry_mmdd if hasattr(w, "entry_mmdd") else w[0]
        exit_ = w.exit_mmdd if hasattr(w, "exit_mmdd") else w[1]
        side = (
            w.direction if hasattr(w, "direction") else (w[2] if len(w) > 2 else "auto")
        )
        x0 = pd.to_datetime("2024-" + entry)
        x1 = pd.to_datetime("2024-" + exit_)
        color = "tab:green" if side == "long" else "tab:red"
        if x1 < x0:
            ax.axvspan(pd.to_datetime("2024-01-01"), x1, color=color, alpha=0.2)
            ax.axvspan(x0, pd.to_datetime("2024-12-31"), color=color, alpha=0.2)
        else:
            ax.axvspan(x0, x1, color=color, alpha=0.2)

    ax.set_xticks(pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 13)]))
    ax.set_xticklabels(
        [pd.to_datetime(f"2024-{m:02d}-01").strftime("%b") for m in range(1, 13)]
    )
    plt.tight_layout()
    fig.savefig(out_dir / "seasonal_curve.png", dpi=150)
    plt.close(fig)


def save_plot_per_year_pnl(
    out_dir: Path, per_year: pd.DataFrame, entry_mmdd: str, exit_mmdd: str
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Skipping per-year PnL plot (matplotlib unavailable): {e}")
        return
    dfw = per_year[
        (per_year["entry_mmdd"] == entry_mmdd) & (per_year["exit_mmdd"] == exit_mmdd)
    ].copy()
    if dfw.empty:
        return
    dfw = dfw.sort_values("year")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(dfw["year"].astype(str), dfw["pnl_points"])
    ax.axhline(0, lw=1, color="k")
    ax.set_title(f"P&L by year: {entry_mmdd} → {exit_mmdd}")
    ax.set_ylabel("Points")
    plt.tight_layout()
    safe_entry = entry_mmdd.replace("-", "")
    safe_exit = exit_mmdd.replace("-", "")
    fig.savefig(out_dir / f"per_year_pnl_{safe_entry}_{safe_exit}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("Deleting previous results...")
    results_path = Path("results")
    if results_path.exists():
        shutil.rmtree(results_path)
    print("Beginning analysis...")
    symbol = DEFAULT_SYMBOL

    out_dir = results_dir_for(symbol)

    df = load_price_data(
        symbol,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
        granularity=DEFAULT_GRANULARITY,
    )

    res = run_seasonal_analysis(
        symbol,
        df=df,
        lookback_years=LOOKBACK_YEARS,
        months=MONTHS,
        entry_day_range=ENTRY_DAY_RANGE,
        exit_day_range=EXIT_DAY_RANGE,
        min_len_days=MIN_LEN_DAYS,
        max_len_days=MAX_LEN_DAYS,
        direction=DIRECTION,
        smooth=SMOOTH,
        exclude_incomplete_last_year=EXCLUDE_INCOMPLETE_LAST_YEAR,
    )

    save_csv_reports(out_dir, res)

    top = res.top_windows
    if top:
        save_plot_seasonal_curve_with_windows(out_dir, res.seasonal_curve, top[:5])
        w0 = top[0]
        save_plot_per_year_pnl(
            out_dir, res.per_year_results, w0.entry_mmdd, w0.exit_mmdd
        )

    print(f"Saved reports to: {out_dir}")


if __name__ == "__main__":
    main()
