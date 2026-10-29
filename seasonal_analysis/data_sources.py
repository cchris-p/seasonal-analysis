"""Canonical data-source tokens for the seasonal engine.

The web UI historically offered a *display* token
(``dukascopy_gmt+0_us-dst``) while the loader only accepted the *loader* token
``dukascopy``. The two concepts are different: the loader token selects the
provider adapter, while the display token describes the corpus time base that
the adapter is expected to return.

This module is the single normalization point. It maps every accepted spelling
to the canonical provider token and rejects anything else, so a job can never
be recorded under a token the loader silently treats as another source.

Decision-use default (``SEAS-003A``): forex analysis uses Dukascopy
(``GMT+0_US-DST``) with the drift-free ``link_relatives`` curve and a
cost-aware evaluation. The retired ``log_return_compounded`` curve mode is not
accepted.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

DUKASCOPY = "dukascopy"
FIRSTRATEDATA = "firstratedata"

# Accepted spellings mapped to the canonical provider/loader token.
_SOURCE_ALIASES = {
    "dukascopy": DUKASCOPY,
    "dukascopy_gmt+0_us-dst": DUKASCOPY,
    "dukascopy_gmt0_us_dst": DUKASCOPY,
    "dukascopy_gmt0_usdst": DUKASCOPY,
    "dukascopy-gmt+0-us-dst": DUKASCOPY,
    "dukascopy gmt+0 us-dst": DUKASCOPY,
    "firstratedata": FIRSTRATEDATA,
    "firstrate": FIRSTRATEDATA,
    "frd": FIRSTRATEDATA,
}

# Tokens that are decision-use for forex (all map to Dukascopy today).
DECISION_USE_SOURCES = frozenset({DUKASCOPY})
LEGACY_SOURCES = frozenset({FIRSTRATEDATA})

# Default display token for the Dukascopy corpus.
DUKASCOPY_DISPLAY_TOKEN = "dukascopy_gmt+0_us-dst"


class UnsupportedDataSource(ValueError):
    """Raised when a caller supplies a data-source token we cannot map."""


def normalize_data_source(value: Optional[str], *, default: str = DUKASCOPY) -> str:
    """Normalize a requested source token to the canonical loader token.

    Args:
        value: Raw token from the API/UI/config. ``None``/blank -> ``default``.
        default: Canonical default when ``value`` is blank.

    Raises:
        UnsupportedDataSource: if the token is not a known alias.

    Returns:
        ``"dukascopy"`` or ``"firstratedata"``.
    """
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    key = " ".join(raw.replace("_", " ").split())
    # Preserve the original separators for the alias table lookup too.
    for candidate in (raw, key, key.replace(" ", "_")):
        mapped = _SOURCE_ALIASES.get(candidate)
        if mapped is not None:
            return mapped
    raise UnsupportedDataSource(
        f"Unsupported data_source {value!r}; accepted tokens are "
        "dukascopy, dukascopy_gmt+0_us-dst, firstratedata"
    )


def normalize_requested_source(value: Optional[str], *, default: str = DUKASCOPY) -> str:
    """Return the *requested* canonical token (same as normalize)."""
    return normalize_data_source(value, default=default)


def is_decision_use_source(value: Optional[str]) -> bool:
    """True when the token maps to a decision-use data source."""
    try:
        return normalize_data_source(value) in DECISION_USE_SOURCES
    except UnsupportedDataSource:
        return False


def is_legacy_source(value: Optional[str]) -> bool:
    """True when the token maps to a legacy/diagnostic data source."""
    try:
        return normalize_data_source(value) in LEGACY_SOURCES
    except UnsupportedDataSource:
        return False


DECISION_CURVE_MODE = "link_relatives"


def dukascopy_available() -> bool:
    """Best-effort check that the canonical Dukascopy corpus is readable.

    A stale NFS handle raises ``OSError`` even when ``os.path.isdir`` returns
    True, so probe with ``listdir``. Returns ``False`` when ``DUKASCOPY_ROOT`` is
    unset or unreadable. This is a *availability* check, never a data-quality
    check.
    """
    root = str(os.environ.get("DUKASCOPY_ROOT") or "").strip()
    if not root:
        return False
    try:
        os.listdir(root)
    except OSError:
        return False
    return True


def dukascopy_m5_cache_dir() -> Optional[str]:
    """Explicit cache directory for daily bars derived from Dukascopy M5."""
    root = str(os.environ.get("DUKASCOPY_M5_CACHE_DIR") or "").strip()
    if not root:
        return None
    try:
        if not os.listdir(root):
            return None
    except OSError:
        return None
    return root


def dukascopy_m5_cache_available() -> bool:
    return dukascopy_m5_cache_dir() is not None


def resolve_decision_source(
    value: Optional[str],
    *,
    default: str = DUKASCOPY,
    allow_fallback: bool = True,
) -> Tuple[str, Optional[str]]:
    """Resolve the source actually used, with an explicit, recorded fallback.

    The decision-use default is Dukascopy. If the canonical corpus is not
    readable and ``allow_fallback`` is true, fall back to the legacy FRD source
    and return a reason. Callers must record both the requested and the actual
    source; a fallback is never silent.
    """
    requested = normalize_data_source(value, default=default)
    if requested == DUKASCOPY and allow_fallback:
        if dukascopy_available() or dukascopy_m5_cache_available():
            return DUKASCOPY, None
        return FIRSTRATEDATA, "dukascopy_corpus_unavailable"
    return requested, None
