import httpx
import logging
import asyncio
from datetime import datetime, timezone
from typing import List
from app.config import settings
from app.pipeline.schemas import NormalizedBar

logger = logging.getLogger(__name__)

class BinanceRESTClient:
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=settings.BINANCE_REST_URL)
        self.last_request_time = 0

    async def _rate_limit(self):
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_request_time
        if elapsed < 0.5:
            await asyncio.sleep(0.5 - elapsed)
        self.last_request_time = asyncio.get_event_loop().time()

    async def fetch_klines(self, symbol: str, timeframe: str, start_time: int = None, end_time: int = None) -> List[NormalizedBar]:
        await self._rate_limit()

        params = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": 1000
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        try:
            response = await self.client.get("", params=params)
            response.raise_for_status()
            data = response.json()

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
            return bars
        except Exception as e:
            logger.error(f"Error fetching REST data for {symbol} {timeframe}: {e}")
            return []

    async def close(self):
        await self.client.aclose()
