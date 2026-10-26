"""Frozen asset-class universes for the seasonal engine (``SEAS-004``).

This module is the single source of truth for *which* symbols a seasonal job may
use and *how* their data is interpreted. It deliberately replaces the
``symbol.isalpha() and len(symbol) == 6`` heuristic with an explicit
asset-class lookup over the committed symbol sets.

The futures universes are frozen. A membership change is a **new** universe
(version + hash), never an in-place edit: accuracy/discovery artifacts record
the universe token so a later run can never be compared against a silently
different symbol set.

Crypto classification is present so the shared abstraction is generic, but there
is **no crypto data routing or universe freeze here**; that is ``H-006``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .config_assets import AssetClass, CRYPTO_SET, FOREX_SET, FUTURES_SET

# ---------------------------------------------------------------------------
# Frozen futures universes (``GATE-AUX-FUT-ROOTS`` / aa-studies
# ``shared/min_backtest/futures.py``). Pinned here rather than re-derived at
# runtime; the hash is asserted in tests.
# ---------------------------------------------------------------------------
FUTURES_FULL_VERSION = "frd-futures-audited-73-v1"
FUTURES_CROSS_SECTOR_VERSION = "frd-futures-cross-sector-24-v1"

FUTURES_FULL_ROOTS: Tuple[str, ...] = (
    "A6", "AD", "B", "B6", "C", "CB", "CC", "CL", "CT", "DC", "DX", "E1",
    "E6", "E7", "ED", "ER", "ES", "EW", "FBTP", "FCE", "FDAX", "FESX",
    "FGBL", "FGBM", "FGBS", "FGBX", "FTDX", "FTI", "FTUK", "G", "GC", "GF",
    "HE", "HG", "HO", "J1", "J7", "KC", "L", "LBS", "MFS", "MME", "MP", "N6",
    "NG", "NIY", "NKD", "NQ", "OJ", "PA", "PL", "QG", "RB", "RP", "RS", "RTY",
    "SB", "SI", "US", "VX", "XC", "YM", "ZC", "ZF", "ZL", "ZM", "ZN", "ZO",
    "ZQ", "ZR", "ZS", "ZT", "ZW",
)

FUTURES_CROSS_SECTOR_ROOTS: Tuple[str, ...] = (
    "ES", "NQ", "YM", "RTY",
    "ZN", "UB", "ZF", "ZT",
    "CL", "NG", "RB", "HO",
    "GC", "SI", "HG", "PL",
    "ZC", "ZS", "ZW", "ZM",
    "KC", "SB", "CC", "CT",
)


def universe_hash(version: str, symbols: Tuple[str, ...]) -> str:
    """Stable short hash used by every artifact/universe token."""
    payload = json.dumps({"version": version, "symbols": list(symbols)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


FUTURES_FULL_HASH = universe_hash(FUTURES_FULL_VERSION, FUTURES_FULL_ROOTS)
FUTURES_CROSS_SECTOR_HASH = universe_hash(
    FUTURES_CROSS_SECTOR_VERSION, FUTURES_CROSS_SECTOR_ROOTS
)

# Membership lookup used for classification. Uses the frozen audited roots (not
# the broader provider ticker list) so provider-only tickers such as ``BTC`` are
# not silently classified as futures.
FUTURES_FULL_SET = frozenset(FUTURES_FULL_ROOTS)
FUTURES_CROSS_SECTOR_SET = frozenset(FUTURES_CROSS_SECTOR_ROOTS)

# CME root spellings that are unambiguously futures but are not present in the
# frozen FRD membership (the audited corpus uses ``E6`` for the EUR future).
# Classification only: it does not widen either frozen universe.
FUTURES_CLASS_ALIASES = frozenset({"6E"})

FUTURES_CLASS_SET = (
    FUTURES_FULL_SET | FUTURES_CROSS_SECTOR_SET | FUTURES_CLASS_ALIASES
)


@dataclass(frozen=True)
class UniverseSpec:
    """Frozen interpretation contract for one symbol universe."""

    name: str
    asset_class: AssetClass
    symbols: Tuple[str, ...]
    version: str
    hash: str
    data_source: str
    adjustment: str
    curve_mode: str
    annualization_days: int
    session_timezone: str
    cost_basis: str
    round_trip_bps: Optional[float]
    stress_round_trip_bps: Tuple[float, ...] = ()
    units: str = "fractional_return"
    profiles: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    is_start: str = "2010-01-01"
    is_end: str = "2023-12-31"
    oos_years: Tuple[str, ...] = ("2024", "2025")
    oos_end: Optional[str] = "2025-02-24"
    fidelity_caveats: Tuple[str, ...] = ()

    def as_metadata(self) -> Dict[str, object]:
        """JSON-safe provenance block recorded on artifacts."""
        return {
            "universe": self.name,
            "universe_version": self.version,
            "universe_hash": self.hash,
            "universe_size": len(self.symbols),
            "asset_class": self.asset_class.value,
            "data_source": self.data_source,
            "adjustment": self.adjustment,
            "curve_mode": self.curve_mode,
            "annualization_days": self.annualization_days,
            "session_timezone": self.session_timezone,
            "cost_basis": self.cost_basis,
            "round_trip_bps": self.round_trip_bps,
            "stress_round_trip_bps": list(self.stress_round_trip_bps),
            "units": self.units,
            "is_start": self.is_start,
            "is_end": self.is_end,
            "oos_years": list(self.oos_years),
            "oos_end": self.oos_end,
        }


FUTURES_FULL = UniverseSpec(
    name="futures_full",
    asset_class=AssetClass.FUTURES,
    symbols=FUTURES_FULL_ROOTS,
    version=FUTURES_FULL_VERSION,
    hash=FUTURES_FULL_HASH,
    data_source="firstratedata",
    adjustment="frd_continuous_ratio_adjusted",
    curve_mode="link_relatives",
    annualization_days=252,
    session_timezone="America/Chicago",
    cost_basis="relative_bps",
    round_trip_bps=5.0,
    stress_round_trip_bps=(7.5, 10.0),
    units="points",
    profiles={
        "all": FUTURES_FULL_ROOTS,
        "cross_sector": FUTURES_CROSS_SECTOR_ROOTS,
    },
    fidelity_caveats=(
        "ratio-adjusted continuous series; levels are arbitrary and returns are "
        "continuous across rolls",
        "roll/carry is excluded from the 5 bps thin cost",
        "FRD continuous futures end at 2025-02-24; OOS 2025 is partial",
    ),
)

FUTURES_CROSS_SECTOR = UniverseSpec(
    name="futures_cross_sector",
    asset_class=AssetClass.FUTURES,
    symbols=FUTURES_CROSS_SECTOR_ROOTS,
    version=FUTURES_CROSS_SECTOR_VERSION,
    hash=FUTURES_CROSS_SECTOR_HASH,
    data_source="firstratedata",
    adjustment="frd_continuous_ratio_adjusted",
    curve_mode="link_relatives",
    annualization_days=252,
    session_timezone="America/Chicago",
    cost_basis="relative_bps",
    round_trip_bps=5.0,
    stress_round_trip_bps=(7.5, 10.0),
    units="points",
    profiles={"cross_sector": FUTURES_CROSS_SECTOR_ROOTS},
    fidelity_caveats=FUTURES_FULL.fidelity_caveats,
)

UNIVERSES: Dict[str, UniverseSpec] = {
    FUTURES_FULL.name: FUTURES_FULL,
    FUTURES_CROSS_SECTOR.name: FUTURES_CROSS_SECTOR,
}


def get_universe(name: str) -> UniverseSpec:
    key = str(name or "").strip().lower()
    if key not in UNIVERSES:
        raise ValueError(
            f"Unknown universe {name!r}; known universes are {sorted(UNIVERSES)}"
        )
    return UNIVERSES[key]


def futures_universe_for_profile(profile: str) -> UniverseSpec:
    """Resolve a futures insight profile to its frozen universe."""
    p = str(profile or "all").strip().lower()
    if p == "cross_sector":
        return FUTURES_CROSS_SECTOR
    if p in {"all", "futures"}:
        return FUTURES_FULL
    raise ValueError(
        f"Unknown futures profile {profile!r}; expected 'all' or 'cross_sector'"
    )


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    if ":" in s:
        s = s.split(":", 1)[1]
    return s


def asset_class_for_symbol(symbol: str) -> AssetClass:
    """Classify a symbol without relying on string length/alphabetic heuristics.

    Order matters: forex, then the frozen futures roots, then crypto, then the
    broader provider futures ticker list. ``BTC`` is a provider futures ticker
    *and* a crypto asset; it is classified as crypto so the futures universe is
    never silently widened.
    """
    s = _normalize_symbol(symbol)
    if s in FOREX_SET:
        return AssetClass.FOREX
    if s in FUTURES_CLASS_SET:
        return AssetClass.FUTURES
    if s in CRYPTO_SET:
        return AssetClass.CRYPTO
    if s in FUTURES_SET:
        return AssetClass.FUTURES
    return AssetClass.STOCK


def default_cost_model_for_symbol(
    symbol: str,
    *,
    round_trip_bps: Optional[float] = None,
):
    """Asset-aware default execution cost.

    Futures use the frozen relative-bps basis. Forex keeps the tick-derived pip
    model. Crypto has no frozen cost basis yet (``H-006``); it requires an
    explicit ``round_trip_bps`` rather than inventing one here.
    """
    from .costs import CostModel

    asset_class = asset_class_for_symbol(symbol)
    if asset_class == AssetClass.FUTURES:
        bps = FUTURES_FULL.round_trip_bps if round_trip_bps is None else round_trip_bps
        return CostModel(round_trip_bps=float(bps))
    if asset_class == AssetClass.FOREX:
        from .spread_estimates import default_cost_model

        return default_cost_model(symbol, round_trip_bps=round_trip_bps)
    if asset_class == AssetClass.CRYPTO:
        if round_trip_bps is None:
            raise ValueError(
                "Crypto seasonality has no frozen cost basis yet (H-006); "
                "pass an explicit round_trip_bps."
            )
        return CostModel(round_trip_bps=float(round_trip_bps))
    raise ValueError(f"Unsupported asset class for cost model: {asset_class.value}")


__all__ = [
    "AssetClass",
    "UniverseSpec",
    "FUTURES_FULL",
    "FUTURES_CROSS_SECTOR",
    "FUTURES_FULL_ROOTS",
    "FUTURES_CROSS_SECTOR_ROOTS",
    "FUTURES_FULL_HASH",
    "FUTURES_CROSS_SECTOR_HASH",
    "FUTURES_FULL_SET",
    "FUTURES_CROSS_SECTOR_SET",
    "FUTURES_CLASS_SET",
    "UNIVERSES",
    "asset_class_for_symbol",
    "default_cost_model_for_symbol",
    "futures_universe_for_profile",
    "get_universe",
    "universe_hash",
]
