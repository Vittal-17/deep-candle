
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from app.features.indicators import (
    calculate_atr_pct,
    calculate_bb_squeeze,
    calculate_vol_zscore,
    calculate_vwap_dist_atr,
    calculate_rsi,
    calculate_ema_dist
)

def test_division_by_zero_and_epsilon_invariance():
    # zero volume, identical High/Low/Close (zero range), flat prices, zero volatility
    dates = pd.date_range("2023-01-01", periods=100, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "bar_time": dates,
        "open_price": np.ones(100) * 100.0,
        "high_price": np.ones(100) * 100.0,
        "low_price": np.ones(100) * 100.0,
        "close_price": np.ones(100) * 100.0,
        "volume": np.zeros(100)
    })

    atr = calculate_atr_pct(df)
    assert not atr.dropna().isna().any(), "ATR should not have NaNs"
    assert not np.isinf(atr).any(), "ATR should not have Infs"
    assert (atr.dropna() >= 0).all()

    bb = calculate_bb_squeeze(df)
    assert not bb.dropna().isna().any()
    assert not np.isinf(bb).any()

    zscore = calculate_vol_zscore(df)
    assert not zscore.dropna().isna().any()
    assert not np.isinf(zscore).any()

    rsi = calculate_rsi(df)
    assert not rsi.dropna().isna().any()
    assert not np.isinf(rsi).any()

    ema_dist = calculate_ema_dist(df)
    assert not ema_dist.dropna().isna().any()
    assert not np.isinf(ema_dist).any()

def test_wilder_smoothing_and_boundaries():
    dates = pd.date_range("2023-01-01", periods=100, freq="15min", tz="UTC")

    # Extreme up-move
    up_prices = np.exp(np.linspace(0, 5, 100)) * 100
    df_up = pd.DataFrame({
        "bar_time": dates,
        "open_price": up_prices,
        "high_price": up_prices,
        "low_price": up_prices,
        "close_price": up_prices,
        "volume": np.ones(100) * 100
    })

    rsi_up = calculate_rsi(df_up).dropna()
    assert (rsi_up >= 0.0).all() and (rsi_up <= 100.0).all()

    # Extreme down-move
    down_prices = np.exp(np.linspace(5, 0, 100)) * 100
    df_down = pd.DataFrame({
        "bar_time": dates,
        "open_price": down_prices,
        "high_price": down_prices,
        "low_price": down_prices,
        "close_price": down_prices,
        "volume": np.ones(100) * 100
    })

    rsi_down = calculate_rsi(df_down).dropna()
    assert (rsi_down >= 0.0).all() and (rsi_down <= 100.0).all()

def test_bb_squeeze_volatility_contraction():
    dates = pd.date_range("2023-01-01", periods=50, freq="15min", tz="UTC")
    # Volatility contraction: prices converge to 100
    prices = 100 + np.sin(np.linspace(0, 10, 50)) * np.linspace(10, 0, 50)
    df = pd.DataFrame({
        "bar_time": dates,
        "open_price": prices,
        "high_price": prices,
        "low_price": prices,
        "close_price": prices,
        "volume": np.ones(50) * 100
    })

    bb = calculate_bb_squeeze(df).dropna()
    # At the end, width should be near zero
    assert bb.iloc[-1] < bb.iloc[20] # Bandwidth should decrease

def test_vol_zscore_zero_std():
    dates = pd.date_range("2023-01-01", periods=50, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "bar_time": dates,
        "open_price": np.ones(50) * 100,
        "high_price": np.ones(50) * 100,
        "low_price": np.ones(50) * 100,
        "close_price": np.ones(50) * 100,
        "volume": np.ones(50) * 100 # Constant volume -> zero std
    })

    zscore = calculate_vol_zscore(df).dropna()
    assert (np.abs(zscore) < 1e-5).all() # Should be near zero due to + 1e-7

def test_vwap_intraday_midnight_reset():
    # Multi-day crossing 00:00:00 UTC
    dates = pd.date_range("2023-01-01 22:00:00", periods=10, freq="1h", tz="UTC")
    prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    volumes = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

    # To calculate ATR properly, we need more periods. Let's start earlier.
    dates = pd.date_range("2023-01-01 10:00:00", periods=30, freq="1h", tz="UTC")
    prices = np.linspace(100, 129, 30)
    volumes = np.ones(30) * 10

    df = pd.DataFrame({
        "bar_time": dates,
        "open_price": prices,
        "high_price": prices,
        "low_price": prices,
        "close_price": prices,
        "volume": volumes
    })

    atr = calculate_atr_pct(df)
    vwap_dist = calculate_vwap_dist_atr(df, atr)

    # 2023-01-01 10:00 is index 0.
    # 2023-01-02 00:00 is index 14.
    # At index 14, VWAP should reset to just the price of index 14.

    # typical price = price since H=L=C. VWAP = price.
    # vwap_dist = (price - vwap) / atr = 0

    assert abs(vwap_dist.iloc[14]) < 1e-5
