import glob
import os
import pandas as pd
from typing import Optional

import sys

sys.path.append("../jupyter-notebooks")

# Local data loaders
from trading_utils.get_forex_data import get_forex_data_by_pair
from trading_utils.get_futures_data import get_futures_data_by_ticker

# Minimal asset helpers
from seasonal_analysis.config_assets import get_config, AssetClass
from seasonal_analysis.config_assets import is_forex_asset, is_futures_asset
from seasonal_analysis.data_sources import (
    dukascopy_available,
    dukascopy_m5_cache_available,
    dukascopy_m5_cache_dir,
    normalize_data_source,
    resolve_decision_source,
)


def apply_data_hygiene(df: pd.DataFrame) -> pd.DataFrame:
    """Basic hygiene suitable for seasonal daily pipeline.

    - lower-case column names
    - keep OHLCV if present
    - ensure DatetimeIndex, sort, drop duplicate timestamps
    - drop rows with missing close

    Declared data provenance (``df.attrs["data_provenance"]``) is preserved.
    """
    provenance = None
    try:
        provenance = df.attrs.get("data_provenance")
    except Exception:
        provenance = None

    df = df.copy()
    # standardize columns
    df.rename(columns={c: str(c).lower() for c in df.columns}, inplace=True)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if keep:
        df = df[keep]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if "close" in df.columns:
        df = df.dropna(subset=["close"])
    if provenance is not None:
        df.attrs["data_provenance"] = provenance
    return df


def _load_dukascopy_m5_cache_daily(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate cached Dukascopy M5 bars to daily OHLCV.

    Used only when ``DUKASCOPY_ROOT`` is unreachable *and* the caller has
    explicitly configured ``DUKASCOPY_M5_CACHE_DIR``. The daily bars are a
    representation of the same provider corpus, not a different data source;
    the provenance records that they were derived from the M5 cache.
    """
    root = dukascopy_m5_cache_dir()
    if not root:
        raise ValueError("DUKASCOPY_M5_CACHE_DIR is not configured")
    pattern = os.path.join(
        root,
        f"{symbol.lower()}_processed_*_M5_dukascopy_GMT+0_US-DST*_hygiene.pkl",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"No Dukascopy M5 cache files for {symbol}")
    frames = []
    for path in files:
        try:
            frame = pd.read_pickle(path)
            frame.attrs = {}
            frames.append(frame)
        except Exception:
            continue
    if not frames:
        raise ValueError(f"Could not read Dukascopy M5 cache for {symbol}")
    df = pd.concat(frames, axis=0)
    if "session_date" not in df.columns:
        raise ValueError("Dukascopy M5 cache is missing session_date")
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    session = pd.to_datetime(df["session_date"], errors="coerce")
    df = df.loc[~session.isna()].copy()
    session = session.loc[df.index]
    grouped = df.groupby(session)
    out = pd.DataFrame(
        {
            "open": grouped["Open"].first(),
            "high": grouped["High"].max(),
            "low": grouped["Low"].min(),
            "close": grouped["Close"].last(),
            "volume": grouped["Volume"].sum()
            if "Volume" in df.columns
            else 0.0,
        }
    )
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    if start_date:
        out = out.loc[out.index >= pd.to_datetime(start_date)]
    if end_date:
        out = out.loc[out.index <= pd.to_datetime(end_date)]
    out.attrs["data_provenance"] = {
        "data_source": "dukascopy",
        "corpus": "dukascopy_m5_cache",
        "derived_from": "dukascopy GMT+0_US-DST M5",
        "time_base": "GMT+0_US-DST",
        "quality": "derived_daily_from_m5_cache",
        "cache_files": [os.path.basename(p) for p in files],
        "fallback_reason": "canonical_dukascopy_root_unavailable",
    }
    return out


def load_price_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = "D",
    source: str = "firstratedata",
) -> pd.DataFrame:
    """Centralized loader for seasonal pipeline.

    Selects the appropriate local data loader by asset class and applies basic hygiene.
    Returns a DataFrame indexed by datetime with at least a 'close' column.

    Args:
        symbol: Trading symbol (e.g. 'EURUSD', 'ES').
        start_date: Optional inclusive start date ('YYYY-MM-DD').
        end_date: Optional inclusive end date ('YYYY-MM-DD').
        granularity: Bar granularity (default daily 'D').
        source: Spot-FX data source. ``"dukascopy"`` selects the canonical
            Dukascopy ``GMT+0_US-DST`` corpus (provenance-tagged); the default
            ``"firstratedata"`` is the legacy FRD source. Futures ignore this
            (FRD continuous futures remain the futures source).

    Returns:
        DataFrame indexed by datetime with at least a 'close' column; a
        provenance mapping (``df.attrs["data_provenance"]``) is attached when the
        provider adapter supplies one (Dukascopy does).
    """
    cfg = get_config(symbol)
    requested_source = source
    fallback_reason = None
    if is_forex_asset(symbol):
        # Normalize at the loader boundary: the UI display token
        # (``dukascopy_gmt+0_us-dst``) and the loader token (``dukascopy``)
        # are different concepts. Never pass an unnormalized token through.
        canonical_source, fallback_reason = resolve_decision_source(source)
        if canonical_source == "dukascopy" and not dukascopy_available():
            # The canonical corpus root is unreachable. Use the explicitly
            # configured Dukascopy M5 cache as the same-provider representation
            # (never silently substitute a different provider).
            df = _load_dukascopy_m5_cache_daily(
                symbol, start_date=start_date, end_date=end_date
            )
            fallback_reason = "canonical_corpus_unavailable_using_m5_cache"
        else:
            df = get_forex_data_by_pair(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                granularity=granularity,
                source=canonical_source,
            )
    elif is_futures_asset(symbol):
        canonical_source = "firstratedata"
        df = get_futures_data_by_ticker(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
        )
    else:
        raise ValueError(
            f"Unsupported or unknown asset class '{cfg.asset_class}' for symbol {symbol}."
        )

    if df is None or len(df) == 0:
        raise ValueError(f"Loader returned no data for {symbol}")

    df = apply_data_hygiene(df)
    # Record both the requested token and the canonical token the adapter used.
    provenance = None
    try:
        provenance = df.attrs.get("data_provenance")
    except Exception:
        provenance = None
    if provenance is None:
        provenance = {}
    elif not isinstance(provenance, dict):
        provenance = {"raw": provenance}
    provenance = dict(provenance)
    provenance.setdefault("data_source", canonical_source)
    provenance.setdefault("requested_data_source", str(requested_source or ""))
    provenance.setdefault("canonical_data_source", canonical_source)
    if fallback_reason:
        provenance.setdefault("fallback_reason", fallback_reason)
    df.attrs["data_provenance"] = provenance
    return df


def load_symbol_data(
    symbol: str,
    start_year: int,
    end_year: int,
    use_cache: bool = True,
    apply_hygiene: bool = False,
) -> pd.DataFrame:
    """Load symbol M5 data with FX day assignments and optional data hygiene.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'ES', etc.)
        start_year: Start year for data
        end_year: End year for data
        use_cache: Whether to use cached processed data
        apply_hygiene: Whether to apply data hygiene filters (also cached)

    Returns:
        DataFrame with OHLCV data, session_date column, and optional hygiene filtering
    """

    print(f"\n{'='*70}")
    print(f"LOADING DATA: {start_year}-{end_year}")
    print(f"{'='*70}")

    # Cache file for processed data (includes FX day assignments and optional hygiene)
    cache_suffix = "_hygiene" if apply_hygiene else ""
    cache_file = (
        CACHE_DIR
        / f"{symbol.lower()}_processed_{start_year}_{end_year}_M5{cache_suffix}.pkl"
    )

    # Try to load from cache first
    if use_cache and cache_file.exists():
        try:
            print(f"Loading processed data from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                df = pickle.load(f)

            # Validate cached data
            expected_hygiene_cols = ["time_ny_str"] if apply_hygiene else []
            hygiene_cols_present = all(
                col in df.columns for col in expected_hygiene_cols
            )

            # Check for pre-computed session gaps
            has_session_gaps = (
                "session_gaps" in df.attrs and df.attrs["session_gaps"] is not None
            )

            if (
                "session_date" in df.columns
                and len(df) > 0
                and df.index[0].year >= start_year
                and df.index[-1].year <= end_year
                and (not apply_hygiene or hygiene_cols_present)
                and has_session_gaps
            ):

                cache_type = (
                    "with hygiene filters + session gaps"
                    if apply_hygiene
                    else "with session day assignments + session gaps"
                )
                print(f"✓ Loaded {len(df)} bars {cache_type} from cache")
                print(f"  Data range: {df.index[0]} to {df.index[-1]}")
                try:
                    print(f"  Session days: {df['session_date'].nunique()}")
                except Exception:
                    pass
                if has_session_gaps:
                    print(
                        f"  Session gaps: {len(df.attrs['session_gaps'])} pre-computed"
                    )
                if apply_hygiene and "time_ny_str" in df.columns:
                    print(f"  NY time strings: pre-computed")
                return df
            else:
                print("  Cache validation failed - reprocessing data")
        except Exception as e:
            print(f"  Cache loading failed: {e} - reprocessing data")

    # Load raw data if cache miss or disabled
    cfg_local = get_config(symbol)
    if cfg_local.asset_class == AssetClass.FOREX:
        print(f"Loading raw OHLCV data for {symbol} (forex)...")
        data = get_forex_data_by_list_of_pairs(
            pairs=[symbol],
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year+1}-01-01",
            granularity="M5",
        )
        df = data[symbol]
    elif cfg_local.asset_class == AssetClass.FUTURES:
        print(f"Loading raw OHLCV data for {symbol} (futures)...")
        df = get_futures_data_by_ticker(
            symbol=symbol,
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year+1}-01-01",
            granularity="M5",
        )
        if isinstance(df, dict):
            # Defensive: some loaders may return dict
            df = df.get(symbol, pd.DataFrame())
    else:
        raise ValueError(
            f"Unsupported or unknown asset class '{cfg_local.asset_class}' for symbol {symbol}. Configure in config_assets.REGISTRY."
        )

    # Normalize column names to expected schema
    rename_map = {}
    for old, new in {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }.items():
        if hasattr(df, "columns") and old in df.columns:
            rename_map[old] = new
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Apply FX day boundaries (the expensive processing step)
    if "session_date" not in df.columns:
        print(
            f"    Creating IntervalIndex for {(end_year - start_year + 1) * 365} FX day boundaries..."
        )
        print(f"    Assigning {len(df)} timestamps to FX days...")

        try:
            df = assign_session_days(df, DATA_TIMEZONE)
            print(
                f"✓ Session day assignment completed - {df['session_date'].nunique()} days"
            )
        except Exception as e:
            print(f"✗ Session day assignment failed: {e}")
            raise

    # Pre-compute gap analysis (expensive but needed for threshold estimation)
    print(f"    Pre-computing session gap analysis...")
    try:
        from session_utils import get_session_gaps

        gaps_df = get_session_gaps(df, DATA_TIMEZONE)
        print(f"✓ Gap analysis completed - {len(gaps_df)} gaps calculated")

        # Store gaps as metadata for later use (attached to DataFrame)
        df.attrs["session_gaps"] = gaps_df
    except Exception as e:
        print(f"✗ Gap analysis failed: {e}")
        raise

    # Apply data hygiene if requested (also expensive)
    if apply_hygiene:
        print(f"    Applying data hygiene filters...")
        try:
            df = apply_data_hygiene(df)
            print(f"✓ Data hygiene completed - {len(df)} bars retained")

            # Re-compute gaps after hygiene filtering if data changed
            if len(gaps_df) > 0:
                print(f"    Re-computing gaps after hygiene filtering...")
                gaps_df = get_session_gaps(df, DATA_TIMEZONE)
                df.attrs["session_gaps"] = gaps_df
                print(f"✓ Updated gap analysis - {len(gaps_df)} gaps after filtering")
        except Exception as e:
            print(f"✗ Data hygiene failed: {e}")
            raise

    # Save processed data to cache
    if use_cache:
        try:
            cache_desc = (
                "with hygiene filters + session gaps"
                if apply_hygiene
                else "with session day assignments + session gaps"
            )
            print(f"Saving processed data {cache_desc} to cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
            print(f"✓ Cache saved - future loads will be faster")
        except Exception as e:
            print(f"⚠ Cache save failed: {e} - will reprocess next time")

    return df
