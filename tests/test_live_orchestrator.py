import pytest
import asyncio
import asyncpg
import os
import psutil
from datetime import datetime, timezone, timedelta
from app.engine.live_orchestrator import LiveOrchestrator

@pytest.mark.asyncio
async def test_live_orchestrator():
    db_pool = await asyncpg.create_pool(
        user=os.getenv("POSTGRES_USER", "deepcandle"),
        password=os.getenv("POSTGRES_PASSWORD", "deepcandle_password"),
        database=os.getenv("POSTGRES_DB", "deepcandle"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5440")
    )

    # Ensure tables exist
    with open("backend/app/pipeline/schema.sql", "r") as f:
        schema = f.read()
    async with db_pool.acquire() as conn:
        await conn.execute(schema)
        await conn.execute("TRUNCATE TABLE paper_trades, system_telemetry RESTART IDENTITY")

    orchestrator = LiveOrchestrator(db_pool=db_pool)
    await orchestrator.initialize()

    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss / (1024 * 1024)

    # Base timestamp
    base_time = datetime.now(timezone.utc)

    symbol = 'BTCUSDT'
    base_price = 50000.0

    # We will simulate 500 WebSocket events
    # We want to force a trade. The model gating is P >= 0.68.
    # Since it's an ML model, we'll just inject a mock into the signal engine to guarantee a signal for testing.
    original_process_features = orchestrator.signal_engine.process_features

    async def mock_process_features(sym, features, close, current_time):
        return {
            'symbol': sym,
            'entry_time': current_time,
            'entry_price': close,
            'target_price': close + 500.0,
            'stop_price': close - 200.0,
            'probability': 0.85,
            'horizon_bars': 6
        }

    orchestrator.signal_engine.process_features = mock_process_features

    for i in range(500):
        # Every 10 events is a closed 15m candle
        is_15m_close = (i % 10 == 0)

        # Simulate price moving up to hit target
        current_price = base_price + (i * 10)

        event_time = base_time + timedelta(minutes=i*1.5)

        kline = {
            'e': 'kline',
            's': symbol,
            'k': {
                't': int(event_time.timestamp() * 1000),
                'T': int(event_time.timestamp() * 1000) + 1000,
                's': symbol,
                'i': '15m',
                'f': 100,
                'L': 200,
                'o': str(current_price - 10),
                'c': str(current_price),
                'h': str(current_price + 10),
                'l': str(current_price - 20),
                'v': '100',
                'n': 10,
                'x': is_15m_close,
                'q': '1000',
                'V': '50',
                'Q': '500',
                'B': '0'
            }
        }

        await orchestrator._handle_ws_message(kline)

    # Check DB
    async with db_pool.acquire() as conn:
        trades = await conn.fetch("SELECT * FROM paper_trades")
        assert len(trades) > 0

        # At least one trade should be CLOSED (TP) because price moved up 5000 points
        closed_trades = [t for t in trades if t['status'] == 'CLOSED']
        assert len(closed_trades) > 0
        assert closed_trades[0]['exit_reason'] == 'TP'

    end_rss = process.memory_info().rss / (1024 * 1024)
    rss_delta = end_rss - start_rss

    print(f"Start RSS: {start_rss:.2f}MB, End RSS: {end_rss:.2f}MB, Delta: {rss_delta:.2f}MB")
    assert rss_delta < 5.0, f"Memory leak detected! RSS grew by {rss_delta:.2f}MB"

    await db_pool.close()
