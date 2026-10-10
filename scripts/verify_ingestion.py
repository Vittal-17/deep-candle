import asyncio
import psutil
import os
import sys
import logging

# Add the backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app.pipeline.database import Database
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify():
    await Database.connect()

    logger.info("Checking database records...")

    records_found = False
    for symbol in settings.TRACKED_SYMBOLS:
        for tf in settings.TIMEFRAMES:
            last_time = await Database.get_last_bar_time(symbol, tf)
            if last_time:
                logger.info(f"Found record for {symbol} {tf}: {last_time}")
                records_found = True

    if not records_found:
        logger.warning("No records found in the database. Ensure the worker has run and ingested data.")
    else:
        logger.info("Verification SUCCESS: Records found in PostgreSQL.")

    await Database.disconnect()

    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Process RSS Memory: {rss_mb:.2f} MB")
    if rss_mb <= 120:
        logger.info("Memory usage is well under the 120MB ceiling.")
    else:
        logger.warning("Memory usage exceeds the 120MB ceiling!")

if __name__ == "__main__":
    asyncio.run(verify())
