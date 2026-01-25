from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd


def compute_poi_charts_payload(symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
    if symbol is None or not str(symbol).strip():
        raise ValueError("symbol is required")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df is required")

    symbol_str = str(symbol).strip().upper()

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    if "Open" not in df.columns or "Close" not in df.columns:
        raise ValueError("df must contain Open and Close columns")

    open_series = df["Open"]
    close_series = df["Close"]

    records: List[Dict[str, Any]] = []
    for ts, open_val, close_val in zip(df.index, open_series, close_series):
        if pd.isna(open_val) or pd.isna(close_val):
            continue

        ts_parsed = pd.Timestamp(ts)
        open_float = float(open_val)
        close_float = float(close_val)

        records.append(
            {
                "date": ts_parsed.strftime("%Y-%m-%d"),
                "year": int(ts_parsed.year),
                "month": int(ts_parsed.month),
                "day_of_month": int(ts_parsed.day),
                "day_of_week": int(ts_parsed.dayofweek),
                "open": open_float,
                "close": close_float,
                "poi_value": close_float - open_float,
            }
        )

    if not records:
        raise ValueError("No valid records to compute POI charts")

    years = sorted({int(r["year"]) for r in records})
    months = sorted({int(r["month"]) for r in records})

    decades: List[str] = []
    if years:
        min_year = min(years)
        max_year = max(years)
        min_decade = (min_year // 10) * 10
        max_decade = (max_year // 10) * 10
        for decade_start in range(min_decade, max_decade + 10, 10):
            decades.append(f"{decade_start}-{decade_start + 9}")

    generated_at = datetime.now(timezone.utc).isoformat()
    data_start_date = str(records[0]["date"])
    data_end_date = str(records[-1]["date"])

    return {
        "symbol": symbol_str,
        "kind": "poi-charts",
        "granularity": "D",
        "generated_at": generated_at,
        "data_start_date": data_start_date,
        "data_end_date": data_end_date,
        "filters": {
            "years": years,
            "months": months,
            "decades": decades,
        },
        "records": records,
    }
