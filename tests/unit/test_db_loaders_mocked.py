
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
import pandas as pd
from app.features.matrix_builder import load_bars_from_db, generate_matrix
from app.ml.labeler import fetch_market_bars

@pytest.mark.asyncio
async def test_matrix_builder_load_bars():
    # Mock Database._pool.acquire

    class MockConn:
        async def fetch(self, query, *args):
            return [
                {'symbol': 'BTCUSDT', 'timeframe': '15m', 'bar_time': '2023-01-01', 'open_price': 100, 'high_price': 105, 'low_price': 95, 'close_price': 100, 'volume': 1000, 'quote_volume': 100000, 'trades_count': 100}
            ]

    class MockAcquire:
        async def __aenter__(self):
            return MockConn()
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch('app.pipeline.database.Database._pool') as mock_pool:
        mock_pool.acquire = lambda: MockAcquire()
        df = await load_bars_from_db("BTCUSDT", "15m")
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]['symbol'] == 'BTCUSDT'

@pytest.mark.asyncio
async def test_labeler_fetch_market_bars():
    class MockConn:
        async def fetch(self, query, *args):
            return [
                {'symbol': 'BTCUSDT', 'bar_time': '2023-01-01', 'high_price': 105, 'low_price': 95, 'close_price': 100}
            ]
        async def close(self):
            pass

    with patch('asyncpg.connect', new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = MockConn()
        df = await fetch_market_bars()
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]['symbol'] == 'BTCUSDT'
