
import pytest
import pandas as pd
from unittest.mock import patch, AsyncMock
from app.features.matrix_builder import generate_matrix
from app.ml.labeler import main as labeler_main

@pytest.mark.asyncio
async def test_generate_matrix_and_labeler():
    # Mock load_bars_from_db
    async def mock_load(symbol, timeframe):
        if timeframe == '15m':
            return pd.DataFrame({
                'symbol': [symbol]*1000,
                'bar_time': pd.date_range("2023-01-01", periods=1000, freq="15min", tz="UTC"),
                'open_price': [100.0]*1000,
                'high_price': [105.0]*1000,
                'low_price': [95.0]*1000,
                'close_price': [100.0]*1000,
                'volume': [1000.0]*1000
            })
        elif timeframe == '1h':
            return pd.DataFrame({
                'symbol': [symbol]*250,
                'bar_time': pd.date_range("2023-01-01", periods=250, freq="1h", tz="UTC"),
                'open_price': [100.0]*250,
                'high_price': [105.0]*250,
                'low_price': [95.0]*250,
                'close_price': [100.0]*250,
                'volume': [1000.0]*250
            })
        elif timeframe == '5m':
            return pd.DataFrame({
                'symbol': [symbol]*3000,
                'bar_time': pd.date_range("2023-01-01", periods=3000, freq="5min", tz="UTC"),
                'open_price': [100.0]*3000,
                'high_price': [105.0]*3000,
                'low_price': [95.0]*3000,
                'close_price': [100.0]*3000,
                'volume': [1000.0]*3000
            })
        return pd.DataFrame()

    with patch('app.features.matrix_builder.load_bars_from_db', side_effect=mock_load):
        with patch('app.features.matrix_builder.Database.connect', new_callable=AsyncMock):
            with patch('app.features.matrix_builder.Database.disconnect', new_callable=AsyncMock):
                with patch('pandas.DataFrame.to_parquet'):
                    # Test successful generation
                    df = await generate_matrix("dummy/dummy.parquet")
                    assert not df.empty

    # Test empty generation
    async def mock_load_empty(*args, **kwargs):
        return pd.DataFrame()

    with patch('app.features.matrix_builder.load_bars_from_db', side_effect=mock_load_empty):
        with patch('app.features.matrix_builder.Database.connect', new_callable=AsyncMock):
            with patch('app.features.matrix_builder.Database.disconnect', new_callable=AsyncMock):
                df = await generate_matrix("dummy/dummy.parquet")
                assert df.empty

def test_labeler_main():
    with patch('pandas.read_parquet') as mock_read:
        mock_read.return_value = pd.DataFrame({
            'symbol': ['BTCUSDT'],
            'bar_time': [pd.Timestamp('2023-01-01', tz='UTC')],
            'atr_pct_15m': [0.01]
        })

        with patch('app.ml.labeler.fetch_market_bars', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                'symbol': ['BTCUSDT'],
                'bar_time': [pd.Timestamp('2023-01-01', tz='UTC')],
                'high_price': [105.0],
                'low_price': [95.0],
                'close_price': [100.0]
            })

            with patch('pandas.DataFrame.to_parquet'):
                labeler_main()
