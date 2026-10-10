
import pytest
from app.pipeline.database import Database
from app.pipeline.schemas import NormalizedBar
from unittest.mock import patch, AsyncMock
from datetime import datetime, timezone

@pytest.mark.asyncio
async def test_database_methods():
    Database._pool = None
    with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_create:
        mock_pool = AsyncMock()
        mock_create.return_value = mock_pool

        # Connect
        await Database.connect()
        assert Database._pool is not None

        # Disconnect
        await Database.disconnect()
        mock_pool.close.assert_awaited_once()
        Database._pool = None

@pytest.mark.asyncio
async def test_database_upsert_and_get():
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_pool = AsyncMock()
    mock_pool.acquire = lambda: MockAcquire()
    Database._pool = mock_pool

    bar = NormalizedBar(
        symbol="BTCUSDT",
        timeframe="15m",
        bar_time=datetime.now(timezone.utc),
        open_price=100.0,
        high_price=105.0,
        low_price=95.0,
        close_price=100.0,
        volume=1000.0,
        quote_volume=100000.0,
        trades_count=100
    )

    await Database.upsert_bar(bar)
    mock_conn.execute.assert_called_once()

    mock_conn.fetchval.return_value = datetime.now(timezone.utc)
    res = await Database.get_last_bar_time("BTCUSDT", "15m")
    assert res is not None
    mock_conn.fetchval.assert_called_once()

    Database._pool = None
