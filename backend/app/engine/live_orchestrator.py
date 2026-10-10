import asyncio
import asyncpg
import logging
import psutil
import os
import json
from datetime import datetime, timezone
from app.pipeline.rolling_buffer import RollingFeatureBuffer
from app.strategy.signal_engine import SignalEngine
from app.strategy.paper_trader import PaperTrader
import websockets

logger = logging.getLogger(__name__)

class LiveOrchestrator:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.buffer = RollingFeatureBuffer()
        self.signal_engine = SignalEngine()
        self.paper_trader = PaperTrader(db_pool=self.db_pool)
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']
        self.running = False
        self.ws = None
        self.telemetry_task = None

    async def initialize(self):
        logger.info("Initializing Live Orchestrator...")
        await self.buffer.initialize_from_db(self.db_pool)

    async def run(self):
        self.running = True
        self.telemetry_task = asyncio.create_task(self._telemetry_loop())
        await self._websocket_loop()

    async def stop(self):
        logger.info("Stopping Live Orchestrator...")
        self.running = False
        if self.ws:
            await self.ws.close()
        if self.telemetry_task:
            self.telemetry_task.cancel()

    async def _telemetry_loop(self):
        while self.running:
            try:
                await asyncio.sleep(60)

                process = psutil.Process(os.getpid())
                rss_mb = process.memory_info().rss / (1024 * 1024)

                open_pos = len(self.paper_trader.open_positions)
                total_trades = len(self.paper_trader.trade_history)

                wins = sum(1 for t in self.paper_trader.trade_history if t['net_pnl'] > 0)
                win_rate = wins / total_trades if total_trades > 0 else 0.0

                cum_pnl = sum(t['net_pnl'] for t in self.paper_trader.trade_history)

                logger.info(f"[TELEMETRY] RSS: {rss_mb:.2f}MB | Open Positions: {open_pos} | Closed Trades: {total_trades} | Win Rate: {win_rate:.1%}")

                if rss_mb >= 120:
                    logger.error(f"CRITICAL: RSS Memory {rss_mb:.2f}MB exceeds 120MB ceiling. Shutting down.")
                    self.running = False
                    break
                elif rss_mb >= 100:
                    logger.warning(f"WARNING: RSS Memory {rss_mb:.2f}MB exceeds 100MB warning threshold.")

                query = """
                INSERT INTO system_telemetry
                (rss_memory_mb, open_positions, total_trades, win_rate, cumulative_pnl)
                VALUES ($1, $2, $3, $4, $5)
                """
                async with self.db_pool.acquire() as conn:
                    await conn.execute(query, rss_mb, open_pos, total_trades, win_rate, cum_pnl)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in telemetry loop: {e}")

    async def _websocket_loop(self):
        streams = []
        for s in self.symbols:
            s_lower = s.lower()
            streams.extend([f"{s_lower}@kline_5m", f"{s_lower}@kline_15m", f"{s_lower}@kline_1h"])

        stream_str = "/".join(streams)
        from app.config import settings
        url = f"{settings.BINANCE_WS_URL}/{stream_str}"

        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    self.ws = ws
                    logger.info(f"Connected to Binance WebSocket for {len(self.symbols)} symbols")

                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        await self._handle_ws_message(data)

            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                if self.running:
                    logger.error(f"WebSocket error: {e}. Reconnecting in 5s...")
                    await asyncio.sleep(5)

    async def _handle_ws_message(self, data: dict):
        if 'e' not in data or data['e'] != 'kline':
            return

        kline = data['k']
        symbol = data['s']
        interval = kline['i']
        is_closed = kline['x']

        high = float(kline['h'])
        low = float(kline['l'])
        close = float(kline['c'])
        current_time = datetime.fromtimestamp(kline['T'] / 1000.0, tz=timezone.utc)

        # 1. Feed tick to paper trader for barrier checks
        is_15m_close = (interval == '15m' and is_closed)
        await self.paper_trader.evaluate_tick_or_bar(symbol, high, low, close, current_time, is_15m_close)

        # 2. If candle closed, feed to buffer
        if is_closed:
            bar = {
                'bar_time': datetime.fromtimestamp(kline['t'] / 1000.0, tz=timezone.utc),
                'open_price': float(kline['o']),
                'high_price': high,
                'low_price': low,
                'close_price': close,
                'volume': float(kline['v'])
            }

            features = self.buffer.on_candle_closed(symbol, interval, bar)

            # 3. If 15m candle closed and we got features back, run inference
            if interval == '15m' and features is not None:
                signal = self.signal_engine.process_features(symbol, features, close, current_time)
                if signal:
                    await self.paper_trader.open_position(signal)
