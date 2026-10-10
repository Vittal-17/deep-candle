import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from statsmodels.tsa.stattools import adfuller

from app.features.indicators import (
    calculate_atr_pct, calculate_bb_squeeze, calculate_vol_zscore,
    calculate_vwap_dist_atr, calculate_rsi, calculate_ema_dist
)
from app.features.temporal_aligner import align_lagged_1h_features, compute_5m_micro_features, align_btc_anchor
from app.features.matrix_builder import build_features_for_symbol

def generate_synthetic_ohlcv(symbol: str, timeframe: str, start_time: datetime, periods: int) -> pd.DataFrame:
    np.random.seed(42)
    freq = "15min" if timeframe == "15m" else ("1h" if timeframe == "1h" else "5min")
    times = pd.date_range(start=start_time, periods=periods, freq=freq, tz=timezone.utc)

    price = 50000.0 if symbol == "BTCUSDT" else 3000.0
    returns = np.random.normal(0.0, 0.005, size=periods)
    close_prices = price * np.exp(returns)

    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.002, size=periods)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.002, size=periods)))
    open_prices = close_prices * (1 + np.random.normal(0, 0.001, size=periods))
    volumes = np.random.uniform(100, 1000, size=periods)

    return pd.DataFrame({
        'symbol': symbol,
        'timeframe': timeframe,
        'bar_time': times,
        'open_price': open_prices,
        'high_price': high_prices,
        'low_price': low_prices,
        'close_price': close_prices,
        'volume': volumes,
        'quote_volume': volumes * close_prices,
        'trades_count': np.random.randint(50, 500, size=periods)
    })

def test_zero_lookahead_bias():
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    df_15m = generate_synthetic_ohlcv("ETHUSDT", "15m", start_time, 1500)
    df_1h = generate_synthetic_ohlcv("ETHUSDT", "1h", start_time, 400)
    df_5m = generate_synthetic_ohlcv("ETHUSDT", "5m", start_time, 4500)
    df_btc_15m = generate_synthetic_ohlcv("BTCUSDT", "15m", start_time, 1500)

    # Full batch calculation
    full_matrix = build_features_for_symbol(df_15m, df_1h, df_5m, df_btc_15m)

    # Pick a point-in-time index T
    cut_idx_15m = 1200
    target_time = df_15m.iloc[cut_idx_15m]['bar_time']

    # Slice strictly up to target_time
    df_15m_slice = df_15m[df_15m['bar_time'] <= target_time].copy()
    df_1h_slice = df_1h[df_1h['bar_time'] <= target_time].copy()
    df_5m_slice = df_5m[df_5m['bar_time'] <= target_time].copy()
    df_btc_slice = df_btc_15m[df_btc_15m['bar_time'] <= target_time].copy()

    sliced_matrix = build_features_for_symbol(df_15m_slice, df_1h_slice, df_5m_slice, df_btc_slice)

    # Compare features at target_time
    full_row = full_matrix[full_matrix['bar_time'] == target_time].iloc[0]
    sliced_row = sliced_matrix[sliced_matrix['bar_time'] == target_time].iloc[0]

    feature_cols = [
        'atr_pct_15m', 'bb_squeeze_15m', 'vol_zscore_15m', 'vwap_dist_atr_15m',
        'rsi_15m', 'rsi_1h', 'ema_200_dist_1h', 'micro_momentum_ratio',
        'micro_volume_acceleration', 'btc_relative_delta', 'return_15m'
    ]

    for col in feature_cols:
        diff = abs(full_row[col] - sliced_row[col])
        assert diff < 1e-6, f"Lookahead bias detected in {col}: full={full_row[col]}, sliced={sliced_row[col]}"

def test_feature_stationarity():
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    df_15m = generate_synthetic_ohlcv("ETHUSDT", "15m", start_time, 1500)
    df_1h = generate_synthetic_ohlcv("ETHUSDT", "1h", start_time, 400)
    df_5m = generate_synthetic_ohlcv("ETHUSDT", "5m", start_time, 4500)
    df_btc_15m = generate_synthetic_ohlcv("BTCUSDT", "15m", start_time, 1500)

    matrix = build_features_for_symbol(df_15m, df_1h, df_5m, df_btc_15m)

    feature_cols = [
        'atr_pct_15m', 'bb_squeeze_15m', 'vol_zscore_15m', 'vwap_dist_atr_15m',
        'rsi_15m', 'rsi_1h', 'ema_200_dist_1h', 'micro_momentum_ratio',
        'micro_volume_acceleration', 'btc_relative_delta', 'return_15m'
    ]

    for col in feature_cols:
        series = matrix[col].values
        res = adfuller(series)
        p_value = res[1]
        assert p_value < 0.01, f"Feature {col} is non-stationary: ADF p-value={p_value}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
