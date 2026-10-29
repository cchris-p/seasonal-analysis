"""Calendar helpers shared by the seasonal engine and validation workers."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def shift_mmdd_trading_days(mmdd: str, k: int) -> Optional[str]:
    """Shift an ``MM-DD`` slot by ``k`` business days (2024 reference calendar).

    Returns ``None`` when the shift leaves the year or lands on leap day.
    """
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
