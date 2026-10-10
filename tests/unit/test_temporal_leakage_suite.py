
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from app.features.temporal_aligner import (
    align_lagged_1h_features,
    compute_5m_micro_features,
    align_btc_anchor
)
from app.features.indicators import calculate_rsi, calculate_ema_dist
from app.features.matrix_builder import build_features_for_symbol

def test_point_in_time_shuffling_masking():
    # Take bar at index T. Modify future bars T+1 ... T+N with extreme price spikes (+500%).
    # Assert feature calculation for bar T remains identical down to 8 decimal places.

    dates_15m = pd.date_range("2023-01-01", periods=1000, freq="15min", tz="UTC")
    df_15m = pd.DataFrame({
        "symbol": ["BTCUSDT"] * 1000,
        "bar_time": dates_15m,
        "open_price": np.ones(1000) * 100,
        "high_price": np.ones(1000) * 105,
        "low_price": np.ones(1000) * 95,
        "close_price": np.ones(1000) * 100,
        "volume": np.ones(1000) * 1000
    })

    dates_1h = pd.date_range("2023-01-01", periods=250, freq="1h", tz="UTC")
    df_1h = pd.DataFrame({
        "symbol": ["BTCUSDT"] * 250,
        "bar_time": dates_1h,
        "open_price": np.ones(250) * 100,
        "high_price": np.ones(250) * 105,
        "low_price": np.ones(250) * 95,
        "close_price": np.ones(250) * 100,
        "volume": np.ones(250) * 4000
    })

    dates_5m = pd.date_range("2023-01-01", periods=3000, freq="5min", tz="UTC")
    df_5m = pd.DataFrame({
        "symbol": ["BTCUSDT"] * 3000,
        "bar_time": dates_5m,
        "open_price": np.ones(3000) * 100,
        "high_price": np.ones(3000) * 102,
        "low_price": np.ones(3000) * 98,
        "close_price": np.ones(3000) * 100,
        "volume": np.ones(3000) * 333
    })

    df_btc = df_15m.copy()

    # Calculate baseline
    df_base = build_features_for_symbol(df_15m, df_1h, df_5m, df_btc)

    # Choose index T (e.g., index 900 in 15m)
    T_15m = 900
    T_time = df_15m.iloc[T_15m]['bar_time']

    # Corrupt future
    df_15m_corrupted = df_15m.copy()
    df_15m_corrupted.loc[T_15m+1:, ['open_price', 'high_price', 'low_price', 'close_price']] *= 5.0

    df_1h_corrupted = df_1h.copy()
    df_1h_corrupted.loc[df_1h_corrupted['bar_time'] > T_time, ['open_price', 'high_price', 'low_price', 'close_price']] *= 5.0

    df_5m_corrupted = df_5m.copy()
    df_5m_corrupted.loc[df_5m_corrupted['bar_time'] > T_time, ['open_price', 'high_price', 'low_price', 'close_price']] *= 5.0

    df_btc_corrupted = df_btc.copy()
    df_btc_corrupted.loc[df_btc_corrupted['bar_time'] > T_time, ['open_price', 'high_price', 'low_price', 'close_price']] *= 5.0

    df_corrupted = build_features_for_symbol(df_15m_corrupted, df_1h_corrupted, df_5m_corrupted, df_btc_corrupted)

    # Find row corresponding to T_time in both
    row_base = df_base[df_base['bar_time'] == T_time].iloc[0]
    row_corrupted = df_corrupted[df_corrupted['bar_time'] == T_time].iloc[0]

    for col in df_base.columns:
        if col not in ['symbol', 'bar_time']:
            np.testing.assert_almost_equal(row_base[col], row_corrupted[col], decimal=8)

def test_strict_macro_lagging_invariance():
    # For a 15m candle ending at 14:15 UTC, assert the 1h indicators strictly match the 1h candle closed at 14:00 UTC (lagged by 1 full bar).
    # Assert that if the unclosed 14:00-15:00 UTC bar changes, the features for the 14:15 UTC candle do not shift.

    dates_15m = pd.date_range("2023-01-01 13:00:00", periods=8, freq="15min", tz="UTC")
    df_15m = pd.DataFrame({"bar_time": dates_15m})

    dates_1h = pd.date_range("2023-01-01 13:00:00", periods=3, freq="1h", tz="UTC")
    df_1h_features = pd.DataFrame({
        "bar_time": dates_1h,
        "rsi_1h": [30, 40, 50],
        "ema_200_dist_1h": [0.1, 0.2, 0.3]
    })

    merged = align_lagged_1h_features(df_15m, df_1h_features)

    # 14:15 UTC is index 5
    row_1415 = merged[merged['bar_time'] == pd.Timestamp("2023-01-01 14:15:00", tz="UTC")].iloc[0]

    # It should match the 14:00 UTC bar from 1h features (rsi_1h = 40)
    assert row_1415['rsi_1h'] == 40
    assert row_1415['ema_200_dist_1h'] == 0.2

    # If unclosed 14:00-15:00 (which would be timestamped 15:00 when closed) changes, it shouldn't affect 14:15
    # The merge_asof direction='backward', allow_exact_matches=False handles this.
    # At 14:00 exactly, it matches 13:00.
    row_1400 = merged[merged['bar_time'] == pd.Timestamp("2023-01-01 14:00:00", tz="UTC")].iloc[0]
    assert row_1400['rsi_1h'] == 30 # strictly lagged! exact match at boundary gets previous closed 1h bar

def test_micro_sub_bar_timestamp_alignment():
    # Assert 5m micro metrics strictly ingest the three 5m intervals matching the 15m candle close (:05, :10, :15).
    # compute_5m_micro_features

    dates_15m = pd.date_range("2023-01-01 14:00:00", periods=2, freq="15min", tz="UTC")
    df_15m = pd.DataFrame({
        "bar_time": dates_15m,
        "open_price": [100, 100],
        "close_price": [105, 110]
    })

    dates_5m = pd.date_range("2023-01-01 14:00:00", periods=6, freq="5min", tz="UTC")
    df_5m = pd.DataFrame({
        "bar_time": dates_5m,
        "open_price": [100, 101, 103, 105, 106, 108],
        "close_price": [101, 103, 105, 106, 108, 110],
        "volume": [10, 20, 30, 40, 50, 60]
    })

    merged = compute_5m_micro_features(df_5m, df_15m)

    # For 14:15, the 5m bar at 14:15 has:
    # open=108, close=110, ret_5m = 2/108
    # 15m bar at 14:15 has:
    # open=100, close=110, ret_15m = 10/100
    # micro_momentum_ratio = (2/108) / (10/100)

    # micro_volume_acceleration for 14:15:
    # volume = 60
    # prev 2 vol avg = (40 + 50) / 2 = 45
    # acc = 60 / 45 = 1.3333

    row_1415 = merged[merged['bar_time'] == pd.Timestamp("2023-01-01 14:15:00", tz="UTC")].iloc[0]

    expected_mom = (1/105) / (10/100)
    assert abs(row_1415['micro_momentum_ratio'] - expected_mom) < 1e-5
    assert abs(row_1415['micro_volume_acceleration'] - (40/25)) < 1e-5
