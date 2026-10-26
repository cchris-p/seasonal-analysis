import numpy as np
import pandas as pd

from seasonal_analysis.config_assets import AssetClass
from seasonal_analysis.load_data import load_price_data
from seasonal_analysis.universes import (
    FUTURES_CROSS_SECTOR,
    FUTURES_CROSS_SECTOR_HASH,
    FUTURES_CROSS_SECTOR_ROOTS,
    FUTURES_FULL,
    FUTURES_FULL_HASH,
    FUTURES_FULL_ROOTS,
    asset_class_for_symbol,
    default_cost_model_for_symbol,
    futures_universe_for_profile,
    get_universe,
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

    try:
        default_cost_model_for_symbol("BTC")
    except ValueError:
        pass
    else:  # pragma: no cover - crypto has no frozen cost basis yet
        raise AssertionError("crypto cost model must require an explicit bps")

    crypto = default_cost_model_for_symbol("BTC", round_trip_bps=12.0)
    assert crypto.round_trip_bps == 12.0


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
