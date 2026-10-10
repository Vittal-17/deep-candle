import logging
import asyncpg
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperTrader:
    def __init__(self, db_pool=None):
        self.open_positions = {}
        self.trade_history = []
        self.db_pool = db_pool

    async def open_position(self, trade_payload: dict):
        symbol = trade_payload['symbol']

        if symbol in self.open_positions:
            return # Max 1 per symbol

        if len(self.open_positions) >= 3:
            return # Max 3 across all assets

        trade_payload['bars_held'] = 0
        self.open_positions[symbol] = trade_payload
        logger.info(f"Opened position for {symbol} at {trade_payload['entry_price']:.4f}")

        if self.db_pool:
            query = """
            INSERT INTO paper_trades
            (symbol, entry_time, entry_price, target_price, stop_price, probability, status)
            VALUES ($1, $2, $3, $4, $5, $6, 'OPEN')
            RETURNING id
            """
            async with self.db_pool.acquire() as conn:
                trade_id = await conn.fetchval(
                    query, symbol, trade_payload['entry_time'], trade_payload['entry_price'],
                    trade_payload['target_price'], trade_payload['stop_price'], trade_payload['probability']
                )
                self.open_positions[symbol]['db_id'] = trade_id

    async def evaluate_tick_or_bar(self, symbol: str, high: float, low: float, close: float, current_time=None, is_15m_close: bool = False):
        if symbol not in self.open_positions:
            return

        pos = self.open_positions[symbol]

        trigger_reason = None
        exit_price = 0.0

        if low <= pos['stop_price']:
            trigger_reason = 'SL'
            exit_price = pos['stop_price']
        elif high >= pos['target_price']:
            trigger_reason = 'TP'
            exit_price = pos['target_price']
        elif is_15m_close:
            pos['bars_held'] += 1
            if pos['bars_held'] >= pos['horizon_bars']:
                trigger_reason = 'TIMEOUT'
                exit_price = close

        if trigger_reason:
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
            net_pnl = pnl_pct - 0.0010

            trade_record = {
                'symbol': symbol,
                'entry_time': pos['entry_time'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'net_pnl': net_pnl,
                'hold_duration_bars': pos['bars_held'],
                'trigger_reason': trigger_reason
            }
            self.trade_history.append(trade_record)

            if self.db_pool and 'db_id' in pos:
                query = """
                UPDATE paper_trades
                SET exit_time = $1, exit_price = $2, status = 'CLOSED',
                    exit_reason = $3, gross_pnl_pct = $4, net_pnl_pct = $5
                WHERE id = $6
                """
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        query, current_time, exit_price, trigger_reason, pnl_pct, net_pnl, pos['db_id']
                    )

            del self.open_positions[symbol]
            logger.info(f"Closed position {symbol} via {trigger_reason}. Net PnL: {net_pnl:.4%}")
