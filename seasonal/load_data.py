import pandas as pd
from typing import Optional

import sys

sys.path.append("../jupyter-notebooks")

# Local data loaders
from trading_utils.get_forex_data import get_forex_data_by_pair
from trading_utils.get_futures_data import get_futures_data_by_ticker

# Minimal asset helpers
from seasonal.config_assets import get_config, AssetClass
from seasonal.config_assets import is_forex_asset, is_futures_asset


def apply_data_hygiene(df: pd.DataFrame) -> pd.DataFrame:
    """Basic hygiene suitable for seasonal daily pipeline.

    - lower-case column names
    - keep OHLCV if present
    - ensure DatetimeIndex, sort, drop duplicate timestamps
    - drop rows with missing close
    """
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
    return df


def load_price_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = "D",
) -> pd.DataFrame:
    """Centralized loader for seasonal pipeline.

    Selects the appropriate local data loader by asset class and applies basic hygiene.
    Returns a DataFrame indexed by datetime with at least a 'close' column.
    """
    cfg = get_config(symbol)
    if is_forex_asset(symbol):
        df = get_forex_data_by_pair(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
        )
    elif is_futures_asset(symbol):
        df = get_futures_data_by_ticker(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            print_candles_retrieved=False,
            use_cache=False,
        )
    else:
        raise ValueError(
            f"Unsupported or unknown asset class '{cfg.asset_class}' for symbol {symbol}."
        )

    if df is None or len(df) == 0:
        raise ValueError(f"Loader returned no data for {symbol}")
    return apply_data_hygiene(df)


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
            print_candles_retrieved=False,
            use_cache=True,
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
