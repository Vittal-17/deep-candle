import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from app.pipeline.rolling_buffer import RollingFeatureBuffer
from app.features.matrix_builder import build_features_for_symbol

def test_buffer_feature_parity():
    # Synthesize bars for ETHUSDT and BTCUSDT
    dates_15m = pd.date_range("2023-01-01", periods=1000, freq="15min", tz="UTC")
    dates_1h = pd.date_range("2023-01-01", periods=250, freq="1h", tz="UTC")
    dates_5m = pd.date_range("2023-01-01", periods=3000, freq="5min", tz="UTC")

    # Let's create varying prices to ensure indicators compute non-zero values
    np.random.seed(42)

    def create_df(dates, symbol):
        n = len(dates)
        base = np.linspace(100, 200, n)
        noise = np.random.normal(0, 1, n)
        close_p = base + noise
        open_p = np.roll(close_p, 1)
        open_p[0] = close_p[0]
        high_p = np.maximum(open_p, close_p) + np.abs(np.random.normal(0, 0.5, n))
        low_p = np.minimum(open_p, close_p) - np.abs(np.random.normal(0, 0.5, n))
        vol = np.random.uniform(100, 1000, n)

        return pd.DataFrame({
            "symbol": [symbol] * n,
            "bar_time": dates,
            "open_price": open_p,
            "high_price": high_p,
            "low_price": low_p,
            "close_price": close_p,
            "volume": vol
        })

    df_15m_eth = create_df(dates_15m, "ETHUSDT")
    df_1h_eth = create_df(dates_1h, "ETHUSDT")
    df_5m_eth = create_df(dates_5m, "ETHUSDT")

    df_15m_btc = create_df(dates_15m, "BTCUSDT")

    # 1. Compute baseline vector using matrix_builder
    # matrix_builder drops first 880 15m bars, we need to bypass that for direct comparison,
    # or just use the last row after generating.
    # Actually, we can just call build_features_for_symbol and get the last row.
    df_features = build_features_for_symbol(df_15m_eth, df_1h_eth, df_5m_eth, df_15m_btc)

    # The last row of df_features
    baseline_features = df_features.iloc[-1].to_dict()

    # 2. Stream exact same bars into RollingFeatureBuffer
    buffer = RollingFeatureBuffer()

    # Stream events in temporal order to simulate live engine
    # Combine all bars into a single event stream
    events = []
    for _, row in df_15m_btc.iterrows():
        events.append((row['bar_time'], 'BTCUSDT', '15m', row.to_dict()))
    for _, row in df_1h_eth.iterrows():
        events.append((row['bar_time'], 'ETHUSDT', '1h', row.to_dict()))
    for _, row in df_5m_eth.iterrows():
        events.append((row['bar_time'], 'ETHUSDT', '5m', row.to_dict()))
    for _, row in df_15m_eth.iterrows():
        events.append((row['bar_time'], 'ETHUSDT', '15m', row.to_dict()))


    # Sort by time. For same time, process 5m first, then 15m, then 1h.
    order = {'5m': 0, '15m': 1, '1h': 2}
    events.sort(key=lambda x: (x[0], order[x[2]]))


    buffer_features = None
    for dt, sym, tf, bar in events:
        feat = buffer.on_candle_closed(sym, tf, bar)
        if sym == 'ETHUSDT' and tf == '15m' and dt == dates_15m[-1]:
            buffer_features = feat

    # 3. Assert parity
    assert buffer_features is not None

    keys_to_compare = [
        'atr_pct_15m', 'bb_squeeze_15m', 'vol_zscore_15m', 'vwap_dist_atr_15m',
        'rsi_15m', 'rsi_1h', 'ema_200_dist_1h', 'micro_momentum_ratio',
        'micro_volume_acceleration', 'btc_relative_delta', 'return_15m'
    ]

    for key in keys_to_compare:
        val_baseline = baseline_features[key]
        val_buffer = buffer_features[key]
        assert abs(val_baseline - val_buffer) < 1e-5, f"Mismatch in {key}: baseline {val_baseline} != buffer {val_buffer}"
