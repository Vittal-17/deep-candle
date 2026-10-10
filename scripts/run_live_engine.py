import asyncio
import asyncpg
import argparse
import signal
import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from app.engine.live_orchestrator import LiveOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_engine(paper: bool, duration: int):
    logger.info(f"Starting Live Engine. Paper Mode: {paper}, Duration: {duration}s")

    db_pool = await asyncpg.create_pool(
        user=os.getenv("POSTGRES_USER", "deepcandle"),
        password=os.getenv("POSTGRES_PASSWORD", "deepcandle_password"),
        database=os.getenv("POSTGRES_DB", "deepcandle"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5440")
    )

    orchestrator = LiveOrchestrator(db_pool=db_pool if paper else None)

    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received exit signal, initiating graceful shutdown...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await orchestrator.initialize()

    orchestrator_task = asyncio.create_task(orchestrator.run())

    try:
        if duration > 0:
            await asyncio.wait_for(stop_event.wait(), timeout=duration)
        else:
            await stop_event.wait()
    except asyncio.TimeoutError:
        logger.info(f"Test duration of {duration}s reached.")
    finally:
        await orchestrator.stop()
        await orchestrator_task
        await db_pool.close()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Candle Live Engine")
    parser.add_argument("--paper", type=bool, default=True, help="Run live paper trading")
    parser.add_argument("--test-duration", type=int, default=0, help="Timeout in seconds for testing (0 = run indefinitely)")
    args = parser.parse_args()

    asyncio.run(run_engine(args.paper, args.test_duration))
