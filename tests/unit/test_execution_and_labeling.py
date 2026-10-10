
import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timezone
from app.ml.labeler import apply_triple_barrier
from app.strategy.paper_trader import PaperTrader

def test_wick_collision_precedence():
    # Test candle where both Take Profit and Stop Loss barriers are breached within the same bar.
    # Assert lower barrier (SL) takes strict precedence -> y = 0.

    df = pd.DataFrame({
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "bar_time": [pd.Timestamp("2023-01-01 14:00:00", tz="UTC"), pd.Timestamp("2023-01-01 14:15:00", tz="UTC")],
        "close_price": [100.0, 100.0],
        "high_price": [100.0, 110.0], # Breaches TP (target is ~101.5)
        "low_price": [100.0, 90.0],   # Breaches SL (stop is ~99.0)
        "atr_pct_15m": [0.01, 0.01]   # ATR = 1.0
    })

    # ATR_abs = 1.0
    # P_entry = 100
    # P_target = 100 + 1.5 - 0.1 = 101.4
    # P_stop = 100 - 1.0 + 0.1 = 99.1
    # At index 1, high=110, low=90. Both breached.
    # labeler.py checks lower barrier first:
    # if lows[eval_idx] <= p_stop: hit = 0; break

    labeled = apply_triple_barrier(df)
    assert labeled.iloc[0]['target'] == 0

def test_fee_friction_exactness():
    # Test exact breakeven scenarios to verify 0.10% round-trip fee friction turns marginal wins into net-zero or negative outcomes.
    # P_target = P_entry + (1.5 * ATR) - (0.0010 * P_entry)
    # If high exactly hits P_entry + 1.5 * ATR without subtracting fee, but doesn't hit P_target.

    df = pd.DataFrame({
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "bar_time": [pd.Timestamp("2023-01-01 14:00:00", tz="UTC"), pd.Timestamp("2023-01-01 14:15:00", tz="UTC")],
        "close_price": [100.0, 100.0],
        "high_price": [100.0, 101.45], # 100 + 1.5 - 0.1 = 101.4. If high = 101.45, it hits.
        "low_price": [100.0, 100.0],
        "atr_pct_15m": [0.01, 0.01]
    })

    labeled = apply_triple_barrier(df)
    assert labeled.iloc[0]['target'] == 1

    df_no_hit = pd.DataFrame({
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "bar_time": [pd.Timestamp("2023-01-01 14:00:00", tz="UTC"), pd.Timestamp("2023-01-01 14:15:00", tz="UTC")],
        "close_price": [100.0, 100.0],
        "high_price": [100.0, 101.35], # Doesn't hit 101.4
        "low_price": [100.0, 100.0],
        "atr_pct_15m": [0.01, 0.01]
    })

    labeled_no_hit = apply_triple_barrier(df_no_hit)
    assert labeled_no_hit.iloc[0]['target'] == 0

def test_horizon_boundary_bounds():
    # Assert trades that breach barriers on bar 6 are evaluated, while trades surviving past bar 6 trigger timeout closure on the close of bar 6.

    df = pd.DataFrame({
        "symbol": ["BTCUSDT"] * 8,
        "bar_time": pd.date_range("2023-01-01 14:00:00", periods=8, freq="15min", tz="UTC"),
        "close_price": [100.0] * 8,
        "high_price": [100.0] * 8,
        "low_price": [100.0] * 8,
        "atr_pct_15m": [0.01] * 8
    })

    # Breach on bar 6 (index 6, which is i+6)
    df.loc[6, 'high_price'] = 110.0
    labeled_6 = apply_triple_barrier(df)
    assert labeled_6.iloc[0]['target'] == 1

    # Breach on bar 7 (index 7, which is i+7)
    df.loc[6, 'high_price'] = 100.0
    df.loc[7, 'high_price'] = 110.0
    labeled_7 = apply_triple_barrier(df)
    assert labeled_7.iloc[0]['target'] == 0 # Out of bounds, hit=0

@pytest.mark.asyncio
async def test_paper_trader_exposure_limits():
    trader = PaperTrader(db_pool=None)

    # Max 1 concurrent open position per symbol
    await trader.open_position({
        'symbol': 'BTCUSDT', 'entry_time': datetime.now(timezone.utc),
        'entry_price': 100, 'target_price': 105, 'stop_price': 95, 'probability': 0.7, 'horizon_bars': 6
    })

    await trader.open_position({
        'symbol': 'BTCUSDT', 'entry_time': datetime.now(timezone.utc),
        'entry_price': 101, 'target_price': 106, 'stop_price': 96, 'probability': 0.8, 'horizon_bars': 6
    })

    assert len(trader.open_positions) == 1
    assert trader.open_positions['BTCUSDT']['entry_price'] == 100

    # Max 3 concurrent positions across the entire portfolio
    await trader.open_position({
        'symbol': 'ETHUSDT', 'entry_time': datetime.now(timezone.utc),
        'entry_price': 100, 'target_price': 105, 'stop_price': 95, 'probability': 0.7, 'horizon_bars': 6
    })

    await trader.open_position({
        'symbol': 'SOLUSDT', 'entry_time': datetime.now(timezone.utc),
        'entry_price': 100, 'target_price': 105, 'stop_price': 95, 'probability': 0.7, 'horizon_bars': 6
    })

    assert len(trader.open_positions) == 3

    # 4th entry rejected
    await trader.open_position({
        'symbol': 'BNBUSDT', 'entry_time': datetime.now(timezone.utc),
        'entry_price': 100, 'target_price': 105, 'stop_price': 95, 'probability': 0.9, 'horizon_bars': 6
    })

    assert len(trader.open_positions) == 3
    assert 'BNBUSDT' not in trader.open_positions
