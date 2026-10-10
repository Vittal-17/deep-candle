import pytest
import asyncio
import asyncpg
import os
import math
from unittest.mock import patch
from app.pipeline.rolling_buffer import RollingFeatureBuffer
from app.strategy.signal_engine import SignalEngine
from app.strategy.paper_trader import PaperTrader

@pytest.mark.asyncio
@patch('app.ml.inference.LGBMInferenceEngine.predict_proba', return_value=0.75)
async def test_live_signal_loop(mock_predict):
    # Connect to db
    conn = await asyncpg.connect(
        user=os.getenv("POSTGRES_USER", "deepcandle"),
        password=os.getenv("POSTGRES_PASSWORD", "deepcandle_password"),
        database=os.getenv("POSTGRES_DB", "deepcandle"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5440")
    )

    symbol = 'BTCUSDT'

    # We need 300 15m bars, 300 1h bars, 900 5m bars to have enough for buffer warmup + 50 historical bars for test loop
    query_1h = f"SELECT * FROM market_bars WHERE symbol='{symbol}' AND timeframe='1h' ORDER BY bar_time DESC LIMIT 300"
    query_15m = f"SELECT * FROM market_bars WHERE symbol='{symbol}' AND timeframe='15m' ORDER BY bar_time DESC LIMIT 300"
    query_5m = f"SELECT * FROM market_bars WHERE symbol='{symbol}' AND timeframe='5m' ORDER BY bar_time DESC LIMIT 900"

    bars_1h = [dict(r) for r in reversed(await conn.fetch(query_1h))]
    bars_15m = [dict(r) for r in reversed(await conn.fetch(query_15m))]
    bars_5m = [dict(r) for r in reversed(await conn.fetch(query_5m))]

    await conn.close()

    buffer = RollingFeatureBuffer()
    signal_engine = SignalEngine()
    paper_trader = PaperTrader()

    # We test on the last 50 15m bars
    # This means the "warmup" is the first 250 bars

    # Warmup buffer
    for b in bars_1h[:-50]:
        buffer.on_candle_closed(symbol, '1h', b)
    for b in bars_5m[:-150]:
        buffer.on_candle_closed(symbol, '5m', b)
    for b in bars_15m[:-50]:
        buffer.on_candle_closed(symbol, '15m', b)

    # Test Loop: Push the last 50 15m bars sequentially (with their corresponding 5m/1h bars)
    # For simplicity of alignment in this test, we push 3 5m bars, then 1 15m bar.
    # Every 4th 15m bar, we push a 1h bar.
    for i in range(50):
        # Push 3 5m bars
        for j in range(3):
            idx = -150 + i*3 + j
            buffer.on_candle_closed(symbol, '5m', bars_5m[idx])

        # Push 15m bar
        bar_15m = bars_15m[-50 + i]
        features = buffer.on_candle_closed(symbol, '15m', bar_15m)

        # Every 4 15m bars, a 1h bar closes. We will approximate this by pushing 1h bar every 4 steps
        if i % 4 == 0 and i // 4 < 50:
            idx = -50 + i // 4
            if idx < 0:
                buffer.on_candle_closed(symbol, '1h', bars_1h[idx])

        # Assert features compute with zero NaNs
        assert features is not None
        assert not any(math.isnan(v) or math.isinf(v) for v in features.values())

        # Evaluate paper trader on the new incoming bar
        await paper_trader.evaluate_tick_or_bar(symbol, bar_15m['high_price'], bar_15m['low_price'], bar_15m['close_price'], current_time=bar_15m['bar_time'], is_15m_close=True)

        # Assert inference runs cleanly and outputs bounded probability
        signal = signal_engine.process_features(symbol, features, bar_15m['close_price'], bar_15m['bar_time'])

        if signal:
            assert 0.0 <= signal['probability'] <= 1.0
            await paper_trader.open_position(signal)

    print(f"Smoke test completed. Total trades taken: {len(paper_trader.trade_history)}")
    for t in paper_trader.trade_history:
        print(f"Trade: {t['trigger_reason']}, PnL: {t['net_pnl']:.4%}, Duration: {t['hold_duration_bars']} bars")
