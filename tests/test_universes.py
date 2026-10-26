import numpy as np
import pandas as pd

from seasonal_analysis.config_assets import AssetClass
from seasonal_analysis.load_data import load_price_data
from seasonal_analysis.universes import (
    CRYPTO,
    CRYPTO_HASH,
    CRYPTO_SYMBOLS,
    CRYPTO_UNIVERSE_SET,
    FUTURES_CROSS_SECTOR,
    FUTURES_CROSS_SECTOR_HASH,
    FUTURES_CROSS_SECTOR_ROOTS,
    FUTURES_FULL,
    FUTURES_FULL_HASH,
    FUTURES_FULL_ROOTS,
    asset_class_for_symbol,
    crypto_universe_for_profile,
    default_cost_model_for_symbol,
    futures_universe_for_profile,
    get_universe,
    is_crypto_universe_member,
    universe_hash,
)


def test_futures_universe_frozen_count_and_hash():
    assert len(FUTURES_FULL_ROOTS) == 73
    assert len(set(FUTURES_FULL_ROOTS)) == 73
    assert FUTURES_FULL_HASH == "28e09a52f18ed416"
    assert universe_hash(FUTURES_FULL.version, FUTURES_FULL_ROOTS) == FUTURES_FULL_HASH
    assert FUTURES_FULL.version == "frd-futures-audited-73-v1"

    assert len(FUTURES_CROSS_SECTOR_ROOTS) == 24
    assert len(set(FUTURES_CROSS_SECTOR_ROOTS)) == 24
    assert FUTURES_CROSS_SECTOR_HASH == "960fb26590fbc959"
    assert FUTURES_CROSS_SECTOR_HASH == universe_hash(
        FUTURES_CROSS_SECTOR.version, FUTURES_CROSS_SECTOR_ROOTS
    )


def test_futures_profiles_are_pinned_subsets():
    assert FUTURES_FULL.profiles["all"] == FUTURES_FULL_ROOTS
    assert FUTURES_FULL.profiles["cross_sector"] == FUTURES_CROSS_SECTOR_ROOTS
    # The frozen cross-sector profile is pinned independently; UB is not a
    # member of the audited 73-root full universe.
    assert set(FUTURES_CROSS_SECTOR_ROOTS) - set(FUTURES_FULL_ROOTS) == {"UB"}
    assert futures_universe_for_profile("all") is FUTURES_FULL
    assert futures_universe_for_profile("cross_sector") is FUTURES_CROSS_SECTOR
    assert get_universe("futures_full") is FUTURES_FULL


def test_futures_universe_metadata():
    meta = FUTURES_FULL.as_metadata()
    assert meta["data_source"] == "firstratedata"
    assert meta["adjustment"] == "frd_continuous_ratio_adjusted"
    assert meta["curve_mode"] == "link_relatives"
    assert meta["cost_basis"] == "relative_bps"
    assert meta["round_trip_bps"] == 5.0
    assert meta["stress_round_trip_bps"] == [7.5, 10.0]
    assert meta["units"] == "points"
    assert meta["annualization_days"] == 252
    assert meta["session_timezone"] == "America/Chicago"
    assert meta["universe_hash"] == "28e09a52f18ed416"


def test_asset_class_for_symbol():
    assert asset_class_for_symbol("EURUSD") == AssetClass.FOREX
    assert asset_class_for_symbol("usdjpy") == AssetClass.FOREX
    assert asset_class_for_symbol("ES") == AssetClass.FUTURES
    assert asset_class_for_symbol("NQ") == AssetClass.FUTURES
    assert asset_class_for_symbol("6E") == AssetClass.FUTURES
    assert asset_class_for_symbol("BTC") == AssetClass.CRYPTO
    assert asset_class_for_symbol("BTC-EUR") == AssetClass.CRYPTO
    assert asset_class_for_symbol("OANDA:EUR_USD") != AssetClass.FUTURES


def test_cost_model_uses_bps_for_futures_and_pips_for_forex():
    fut = default_cost_model_for_symbol("ES")
    assert fut.round_trip_bps == 5.0
    assert fut.round_trip_pips == 2.6  # untouched pip fields, unused when bps set

    fx = default_cost_model_for_symbol("EURUSD")
    assert fx.round_trip_bps is None
    assert fx.round_trip_pips > 0.0

    crypto = default_cost_model_for_symbol("BTC")
    assert crypto.round_trip_bps == 20.0

    stressed = default_cost_model_for_symbol("BTC", round_trip_bps=30.0)
    assert stressed.round_trip_bps == 30.0


def test_crypto_universe_frozen_count_and_hash():
    assert len(CRYPTO_SYMBOLS) == 25
    assert len(set(CRYPTO_SYMBOLS)) == 25
    assert CRYPTO_HASH == "1005791f9bcc96b2"
    assert universe_hash(CRYPTO.version, CRYPTO_SYMBOLS) == CRYPTO_HASH
    assert CRYPTO.version == "frd-crypto-usd-liquid-25-v1"
    assert CRYPTO_UNIVERSE_SET == frozenset(CRYPTO_SYMBOLS)
    assert crypto_universe_for_profile("all") is CRYPTO
    assert get_universe("crypto") is CRYPTO


def test_crypto_universe_metadata():
    meta = CRYPTO.as_metadata()
    assert meta["asset_class"] == "crypto"
    assert meta["data_source"] == "firstratedata"
    assert meta["adjustment"] == "crypto_spot"
    assert meta["curve_mode"] == "link_relatives"
    assert meta["cost_basis"] == "relative_bps"
    assert meta["round_trip_bps"] == 20.0
    assert meta["stress_round_trip_bps"] == [30.0, 40.0]
    assert meta["annualization_days"] == 365
    assert meta["session_timezone"] == "UTC"
    assert meta["is_start"] == "2018-01-01"
    assert meta["is_end"] == "2023-12-31"
    assert meta["universe_hash"] == "1005791f9bcc96b2"
    assert meta["universe_size"] == 25


def test_crypto_classification_and_membership_are_separate():
    assert asset_class_for_symbol("BTC") == AssetClass.CRYPTO
    assert asset_class_for_symbol("ETH") == AssetClass.CRYPTO
    assert asset_class_for_symbol("FUN") == AssetClass.CRYPTO
    assert is_crypto_universe_member("BTC") is True
    assert is_crypto_universe_member("FUN") is True
    # A provider crypto ticker outside the frozen universe classifies as crypto
    # but is not a universe member.
    assert asset_class_for_symbol("ADA") == AssetClass.CRYPTO
    assert is_crypto_universe_member("ADA") is False


def test_crypto_load_price_data_provenance(monkeypatch):
    idx = pd.bdate_range("2018-01-01", "2020-03-31")
    close = np.linspace(100.0, 220.0, len(idx))

    def fake_crypto_loader(**kwargs):
        return pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 10.0,
            },
            index=idx,
        )

    import seasonal_analysis.load_data as load_data

    monkeypatch.setattr(load_data, "get_crypto_data_by_ticker", fake_crypto_loader)
    df = load_price_data("BTC", start_date="2018-01-01", end_date="2020-03-31")
    prov = df.attrs["data_provenance"]
    assert prov["data_source"] == "firstratedata"
    assert prov["adjustment"] == "crypto_spot"
    assert prov["time_base"] == "UTC_date_only"
    assert prov["cost_basis"] == "relative_bps"
    assert prov["round_trip_bps"] == 20.0
    assert prov["units"] == "quote_price"
    assert prov["annualization_days"] == 365
    assert prov["session_timezone"] == "UTC"
    assert prov["universe"] == "crypto"
    assert prov["universe_hash"] == "1005791f9bcc96b2"


def test_crypto_provider_symbol_outside_universe_has_no_universe_token(monkeypatch):
    idx = pd.bdate_range("2018-01-01", "2019-12-31")
    close = np.linspace(1.0, 2.0, len(idx))

    def fake_crypto_loader(**kwargs):
        return pd.DataFrame({"close": close}, index=idx)

    import seasonal_analysis.load_data as load_data

    monkeypatch.setattr(load_data, "get_crypto_data_by_ticker", fake_crypto_loader)
    df = load_price_data("ADA", start_date="2018-01-01", end_date="2019-12-31")
    prov = df.attrs["data_provenance"]
    assert "universe_hash" not in prov
    assert prov["universe_membership"] == "outside_frozen_universe"


def test_futures_load_price_data_provenance(monkeypatch):
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    close = np.linspace(3000.0, 3100.0, len(idx))

    def fake_futures_loader(**kwargs):
        return pd.DataFrame(
            {
                "Open": close,
                "High": close + 5,
                "Low": close - 5,
                "Close": close,
                "Volume": 1000.0,
            },
            index=idx,
        )

    import seasonal_analysis.load_data as load_data

    monkeypatch.setattr(load_data, "get_futures_data_by_ticker", fake_futures_loader)
    df = load_price_data("ES", start_date="2020-01-01", end_date="2020-03-31")
    prov = df.attrs["data_provenance"]
    assert prov["data_source"] == "firstratedata"
    assert prov["adjustment"] == "frd_continuous_ratio_adjusted"
    assert prov["cost_basis"] == "relative_bps"
    assert prov["round_trip_bps"] == 5.0
    assert prov["units"] == "points"
    assert prov["session_timezone"] == "America/Chicago"
    assert prov["universe"] == "futures_full"
    assert prov["universe_hash"] == "28e09a52f18ed416"
