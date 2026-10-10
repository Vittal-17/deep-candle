import asyncpg
import logging
from app.config import settings
from app.pipeline.schemas import NormalizedBar

logger = logging.getLogger(__name__)

class Database:
    _pool = None

    @classmethod
    async def connect(cls):
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(
                dsn=settings.DATABASE_URL,
                min_size=5,
                max_size=20,
                command_timeout=30.0
            )
            logger.info("Database connection pool created (min_size=5, max_size=20).")

    @classmethod
    async def disconnect(cls):
        if cls._pool:
            await cls._pool.close()
            logger.info("Database connection pool closed.")

    @classmethod
    async def upsert_bar(cls, bar: NormalizedBar):
        query = """
            INSERT INTO market_bars (
                symbol, timeframe, bar_time, open_price, high_price,
                low_price, close_price, volume, quote_volume, trades_count
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
            ) ON CONFLICT (symbol, timeframe, bar_time) DO UPDATE SET
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                volume = EXCLUDED.volume,
                quote_volume = EXCLUDED.quote_volume,
                trades_count = EXCLUDED.trades_count;
        """
        async with cls._pool.acquire() as conn:
            await conn.execute(
                query,
                bar.symbol, bar.timeframe, bar.bar_time, bar.open_price,
                bar.high_price, bar.low_price, bar.close_price, bar.volume,
                bar.quote_volume, bar.trades_count
            )

    @classmethod
    async def get_last_bar_time(cls, symbol: str, timeframe: str):
        query = """
            SELECT MAX(bar_time) FROM market_bars
            WHERE symbol = $1 AND timeframe = $2
        """
        async with cls._pool.acquire() as conn:
            result = await conn.fetchval(query, symbol, timeframe)
            return result
