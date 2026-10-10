import argparse
import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
import httpx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app.config import settings
from app.pipeline.database import Database
from app.pipeline.schemas import NormalizedBar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

async def backfill_klines(symbol: str, timeframe: str, days: int):
    client = httpx.AsyncClient(base_url="https://api.binance.com/api/v3", timeout=15.0)
    target_start_time = datetime.now(timezone.utc) - timedelta(days=days)

    while True:
        query = """
            SELECT MIN(bar_time) FROM market_bars
            WHERE symbol = $1 AND timeframe = $2
        """
        async with Database._pool.acquire() as conn:
            earliest_time = await conn.fetchval(query, symbol, timeframe)

        if earliest_time:
            end_ts = int(earliest_time.timestamp() * 1000) - 1
        else:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        if earliest_time and earliest_time <= target_start_time:
            logger.info(f"[{symbol} {timeframe}] Reached target start time: {earliest_time}")
            break

        params = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": 1000,
            "endTime": end_ts
        }

        success = False
        for attempt in range(1, 5):
            try:
                response = await client.get("/klines", params=params)
                if response.status_code in (429, 418):
                    logger.warning(f"[{symbol} {timeframe}] Rate limit hit! Sleeping for 60s (Attempt {attempt}/4)...")
                    await asyncio.sleep(60)
                    continue

                response.raise_for_status()
                data = response.json()

                if not data:
                    logger.info(f"[{symbol} {timeframe}] No more data returned from Binance.")
                    success = True
                    break

                bars = []
                for row in data:
                    bar = NormalizedBar(
                        symbol=symbol.upper(),
                        timeframe=timeframe,
                        bar_time=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                        open_price=float(row[1]),
                        high_price=float(row[2]),
                        low_price=float(row[3]),
                        close_price=float(row[4]),
                        volume=float(row[5]),
                        quote_volume=float(row[7]),
                        trades_count=int(row[8])
                    )
                    bars.append(bar)

                for bar in bars:
                    await Database.upsert_bar(bar)

                first_bar_time = bars[0].bar_time
                logger.info(f"[{symbol} {timeframe}] Ingested {len(bars)} bars down to {first_bar_time}")

                success = True
                if first_bar_time <= target_start_time:
                    logger.info(f"[{symbol} {timeframe}] Target reach satisfied.")
                break

            except Exception as e:
                logger.warning(
                    f"[{symbol} {timeframe}] Exception during backfill (Attempt {attempt}/4, {type(e).__name__}): {repr(e)}",
                    exc_info=True
                )
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt)

        if not success:
            logger.error(f"[{symbol} {timeframe}] Failed batch after 4 consecutive attempts. Aborting timeframe backfill.")
            break

        await asyncio.sleep(0.4)

    await client.aclose()

async def main():
    parser = argparse.ArgumentParser(description="Deep Candle Resumable Historical Backfill Script")
    parser.add_argument("--days", type=int, default=240, help="Number of days to backfill (default: 240)")
    parser.add_argument("--test-mode", action="store_true", help="Fetch only 2 days for verification")
    args = parser.parse_args()

    days_to_fetch = 2 if args.test_mode else args.days
    logger.info(f"Starting backfill for {days_to_fetch} days...")

    await Database.connect()

    for symbol in settings.TRACKED_SYMBOLS:
        for tf in settings.TIMEFRAMES:
            tf_days = min(days_to_fetch, 30) if tf == "5m" else days_to_fetch
            await backfill_klines(symbol, tf, tf_days)

    await Database.disconnect()
    logger.info("Backfill process complete.")

if __name__ == "__main__":
    asyncio.run(main())
