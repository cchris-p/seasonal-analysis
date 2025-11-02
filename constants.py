from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

from config_assets import get_config

# ---------------------------------------------------------------------------
# Core selector (you can change just this for quick runs)
# ---------------------------------------------------------------------------
DEFAULT_SYMBOL: str = "ES"  # e.g., "EURUSD" (forex) or "ES" (futures)
DEFAULT_GRANULARITY: str = "D"
DEFAULT_START_DATE: str = "2010-01-01"
DEFAULT_END_DATE: str = "2024-10-31"

# Asset-aware configuration for the default symbol
CFG = get_config(DEFAULT_SYMBOL)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# Optional data directory if you want to place CSVs for non-forex assets
DATA_DIR = (PROJECT_ROOT / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)


def results_dir_for(symbol: str) -> Path:
    cfg = get_config(symbol)
    asset_class = cfg.asset_class.value
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_ROOT / asset_class / symbol / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Seasonal pipeline defaults (can be overridden per asset if desired)
# ---------------------------------------------------------------------------
LOOKBACK_YEARS: int = 15
MONTHS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
ENTRY_DAY_RANGE: Tuple[int, int] = (8, 22)
EXIT_DAY_RANGE: Tuple[int, int] = (12, 31)
MIN_LEN_DAYS: int = 5
MAX_LEN_DAYS: int = 35
DIRECTION: str = "auto"
SMOOTH: int = 3
EXCLUDE_INCOMPLETE_LAST_YEAR: bool = True

# If you later want to customize per asset, you can implement a mapping here
# or infer from get_config(symbol). For now the above are shared defaults.
