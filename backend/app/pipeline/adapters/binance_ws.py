import asyncio
import json
import logging
import websockets
from datetime import datetime, timezone
from app.config import settings
from app.pipeline.schemas import NormalizedBar
from app.pipeline.database import Database

logger = logging.getLogger(__name__)

class BinanceWSClient:
    def __init__(self):
        self.streams = []
        for symbol in settings.TRACKED_SYMBOLS:
            for tf in settings.TIMEFRAMES:
                self.streams.append(f"{symbol.lower()}@kline_{tf}")

        self.ws_url = f"{settings.BINANCE_WS_URL}?streams={'/'.join(self.streams)}"

    async def connect_and_listen(self):
        backoff = 1
        max_backoff = 60

        while True:
            try:
                logger.info(f"Connecting to Binance WS: {self.ws_url}")
                async with websockets.connect(self.ws_url, ping_interval=180, ping_timeout=10) as ws:
                    logger.info("Connected to Binance WS.")
                    backoff = 1  # Reset backoff on successful connection

                    async for message in ws:
                        await self._process_message(message)
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed.")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            logger.info(f"Reconnecting in {backoff} seconds...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    async def _process_message(self, message: str):
        try:
            data = json.loads(message)
            if 'data' not in data:
                return

            kline_data = data['data']['k']
            is_closed = kline_data['x']

            if not is_closed:
                return  # Discard immediately if k.x == false

            bar = NormalizedBar(
                symbol=kline_data['s'],
                timeframe=kline_data['i'],
                bar_time=datetime.fromtimestamp(kline_data['t'] / 1000, tz=timezone.utc),
                open_price=float(kline_data['o']),
                high_price=float(kline_data['h']),
                low_price=float(kline_data['l']),
                close_price=float(kline_data['c']),
                volume=float(kline_data['v']),
                quote_volume=float(kline_data['q']),
                trades_count=int(kline_data['n'])
            )

            await Database.upsert_bar(bar)
            logger.info(f"Persisted closed bar: {bar.symbol} {bar.timeframe} at {bar.bar_time}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
