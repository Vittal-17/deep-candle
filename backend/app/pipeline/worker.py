import asyncio
import logging
import signal
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.pipeline.database import Database
from app.pipeline.adapters.binance_ws import BinanceWSClient
from app.pipeline.adapters.binance_rest import BinanceRESTClient
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

async def heal_gaps():
    rest_client = BinanceRESTClient()
    for symbol in settings.TRACKED_SYMBOLS:
        for tf in settings.TIMEFRAMES:
            last_time = await Database.get_last_bar_time(symbol, tf)
            if last_time:
                logger.info(f"Last bar time for {symbol} {tf} is {last_time}")
                start_time = int(last_time.timestamp() * 1000)
                bars = await rest_client.fetch_klines(symbol, tf, start_time=start_time)
                for bar in bars:
                    if bar.bar_time > last_time:
                        await Database.upsert_bar(bar)
                        logger.info(f"Backfilled bar: {bar.symbol} {bar.timeframe} at {bar.bar_time}")
            else:
                logger.info(f"No existing data for {symbol} {tf}. Fetching initial batch...")
                bars = await rest_client.fetch_klines(symbol, tf)
                for bar in bars:
                    await Database.upsert_bar(bar)
                if bars:
                    logger.info(f"Fetched {len(bars)} initial bars for {symbol} {tf}")
    await rest_client.close()

async def main():
    logger.info("Starting Data Ingestion Worker...")
    await Database.connect()

    # Heal gaps on startup
    await heal_gaps()

    ws_client = BinanceWSClient()

    # Run WS client
    ws_task = asyncio.create_task(ws_client.connect_and_listen())

    def handle_sigint():
        logger.info("Received exit signal, shutting down...")
        ws_task.cancel()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_sigint)

    try:
        await ws_task
    except asyncio.CancelledError:
        pass
    finally:
        await Database.disconnect()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
