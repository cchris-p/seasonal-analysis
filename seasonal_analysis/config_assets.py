from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, Optional
from zoneinfo import ZoneInfo

from trading_utils.constants import (
    forex_majors_minors as FOREX_ASSETS,
    firstratedata_futures_tickers as FUTURES_ASSETS,
)


class AssetClass(Enum):
    FOREX = "forex"
    FUTURES = "futures"
    STOCK = "stock"
    CRYPTO = "crypto"


@dataclass(frozen=True)
class AssetConfig:
    symbol: str
    asset_class: AssetClass
    timezone: str
    pip_size: Optional[float] = None  # FX convenience
    tick_size: Optional[float] = None  # futures convenience
    point_value: Optional[float] = None  # futures convenience

    def validate(self) -> None:
        _ = ZoneInfo(self.timezone)
        # basic timezone validation only
        _ = ZoneInfo(self.timezone)


DEFAULTS: Dict[AssetClass, AssetConfig] = {
    AssetClass.FOREX: AssetConfig(
        symbol="*",
        asset_class=AssetClass.FOREX,
        timezone="America/New_York",
        pip_size=0.0001,
    ),
    AssetClass.FUTURES: AssetConfig(
        symbol="*",
        asset_class=AssetClass.FUTURES,
        timezone="America/Chicago",
    ),
    AssetClass.STOCK: AssetConfig(
        symbol="*",
        asset_class=AssetClass.STOCK,
        timezone="America/New_York",
    ),
    AssetClass.CRYPTO: AssetConfig(
        symbol="*",
        asset_class=AssetClass.CRYPTO,
        timezone="UTC",
    ),
}

REGISTRY: Dict[str, AssetConfig] = {
    "USDJPY": replace(DEFAULTS[AssetClass.FOREX], symbol="USDJPY", pip_size=0.01),
    "EURUSD": replace(DEFAULTS[AssetClass.FOREX], symbol="EURUSD", pip_size=0.0001),
    "GBPUSD": replace(DEFAULTS[AssetClass.FOREX], symbol="GBPUSD", pip_size=0.0001),
    "ES": replace(
        DEFAULTS[AssetClass.FUTURES],
        symbol="ES",
        timezone="America/Chicago",
    ),
}

FOREX_SET = frozenset(FOREX_ASSETS)
FUTURES_SET = frozenset(FUTURES_ASSETS)


def is_forex_asset(symbol: str) -> bool:
    return symbol in FOREX_SET


def _is_jpy_pair(symbol: str) -> bool:
    s = symbol.upper()
    return (len(s) == 6 and s.endswith("JPY")) or ("JPY" in s)


def is_futures_asset(symbol: str) -> bool:
    return symbol in FUTURES_SET


def get_config(s: str) -> AssetConfig:
    if s in REGISTRY:
        cfg = REGISTRY[s]
    elif s in FOREX_SET:
        base = DEFAULTS[AssetClass.FOREX]
        pip = 0.01 if _is_jpy_pair(s) else 0.0001
        cfg = replace(base, symbol=s, pip_size=pip)
    elif s in FUTURES_SET:
        cfg = replace(DEFAULTS[AssetClass.FUTURES], symbol=s)
    else:
        # default to stock-style config for unknowns
        cfg = replace(DEFAULTS[AssetClass.STOCK], symbol=s)
    cfg.validate()
    return cfg


def pip_size(symbol: str) -> float:
    return get_config(symbol).pip_size or 0.0001
