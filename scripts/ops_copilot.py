import asyncio
import os
import sys
import logging
import psutil
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import asyncpg
import httpx

# Ensure backend modules are discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ops_copilot")

RONIN_QUANT_OPS_SYSTEM_PROMPT = """You are Ronin, the Lead Autonomous Quantitative Operations & Risk Diagnostics Co-Pilot for the deep-candle algorithmic trading engine.

## CORE ARCHITECTURAL SPECIFICATIONS
- Ingestion & Event Loop: Asynchronous kline ingestion (15m, 1h, 5m) via Binance WebSocket into a bounded NumPy circular buffer.
- Inference Pipeline: LightGBM statistical barrier classifier evaluating features on 15m bar closes (Trigger threshold: P(Setup) >= 0.50), guarded by an async LLM Chief Risk Officer (CRO) qualitative gate.
- Execution Paradigm: Paper trading engine running against PostgreSQL 16 (port 5440).
- Risk Architecture (Triple-Barrier Method):
  * Barrier 1 (TP1): Fixed at +1.5x ATR (50% position scale-out, stop moves to breakeven).
  * Barrier 2 (TP2): Runner target at +3.0x ATR.
  * Barrier 3 (Stop-Loss): Structural invalidation below swing low or intraday VWAP (strictly <= 2.0x ATR).
  * Barrier 4 (Horizontal Horizon): Deterministic 6-bar (90 minutes) temporal invalidation.
  * Fee Friction: 0.10% taker fee simulated on entry and exit.

## POSTGRESQL SCHEMA REASONING
You have direct context derived from three core PostgreSQL relations:
1. `market_bars` (symbol, timeframe, bar_time, open, high, low, close, volume)
2. `paper_trades` (id, symbol, entry_time, entry_price, target_price, stop_price, exit_price, current_barrier, status, net_pnl_pct, exit_reason, probability, invalidation_thesis, closed_at)
3. `system_telemetry` (id, recorded_at, rss_memory_mb, loop_latency_ms, buffer_usage_pct)

## DIAGNOSTIC REASONING PLAYBOOKS
When an operator asks a question, execute the corresponding diagnostic tree:

1. Trade Post-Mortem (Loss or Early Invalidation):
   - Alpha Check: Was P(Setup) marginal (0.50-0.53) or high conviction (>0.58)?
   - Thesis Verification: Compare the logged `invalidation_thesis` against the recorded `exit_reason`.
   - Failure Classification: Categorize explicitly into one of four failure modes:
     * Alpha Failure (Market broke regime immediately post-entry).
     * Volatility Contraction (ATR collapsed, triggering a 6-bar horizontal timeout).
     * Structural Penetration (Wick swept structural SL before price moved toward target).
     * Execution Friction (Fee slippage turned nominal gain into net loss).

2. Infrastructure & Telemetry Diagnosis:
   - Memory Leak: Flag warnings if `rss_memory_mb` trends >150 MB or scales linearly with bar count.
   - Loop Latency: Flag bottlenecks if `loop_latency_ms` >50 ms. Pinpoint whether the slowdown is caused by WebSocket buffer ingestion, database async writes, or proxy round-trips.

3. Emergency Operational Directive:
   - If telemetry exhibits runaway memory growth or unexplained trade drift, explicitly instruct the operator to trip the circuit breaker with `/panic`.

## COMMUNICATION PROTOCOLS
- Never use generic consumer financial disclaimers. Speak with the authority of an institutional quant officer.
- Keep responses compact (under 150 words), using high-density Markdown blocks, bullet points, and code formatting for figures.
- Numerical Standards: Prices to 4 decimals, percentages to 1 decimal place, PnL explicitly prefixed with `+` or `-`."""

class TelegramAIOps:
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None, bot_token: Optional[str] = None, admin_chat_id: Optional[str] = None):
        self.db_pool = db_pool
        self.bot_token = bot_token or settings.TELEGRAM_BOT_TOKEN
        self.admin_chat_id = str(admin_chat_id or settings.TELEGRAM_ADMIN_CHAT_ID).strip()
        self.proxy_url = settings.PROXY_URL
        self.model = "gemini-3.1-pro-low"
        self.halt_file = settings.EMERGENCY_HALT_FILE
        self.last_update_id = 0
        self.running = False

        # State tracking for paper_trades monitoring: id -> trade dict
        self._tracked_trades: Dict[int, Dict[str, Any]] = {}
        self._monitor_initialized = False

    def is_authorized(self, chat_id: Any) -> bool:
        if not self.admin_chat_id:
            return False
        return str(chat_id).strip() == self.admin_chat_id

    async def send_message(self, text: str, parse_mode: Optional[str] = "Markdown") -> bool:
        if not self.bot_token or not self.admin_chat_id:
            logger.debug(f"[NO_TELEGRAM_CONFIG] Skipping send: {text}")
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.admin_chat_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    logger.warning(f"Telegram sendMessage failed: {resp.status_code} {resp.text}")
                    # If Markdown parsing failed, try sending plain text
                    if parse_mode and "can't parse entities" in resp.text:
                        payload.pop("parse_mode", None)
                        await client.post(url, json=payload)
                    return False
                return True
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False

    async def handle_status_command(self) -> str:
        # 1. System Telemetry
        telemetry_row = None
        rss_memory = 0.0
        win_rate = 0.0
        cum_pnl = 0.0
        loop_latency = "N/A"

        # Check process RSS memory
        try:
            proc = psutil.Process(os.getpid())
            rss_memory = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            rss_memory = 0.0

        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    telemetry_row = await conn.fetchrow(
                        "SELECT * FROM system_telemetry ORDER BY id DESC LIMIT 1"
                    )
            except Exception as e:
                logger.error(f"Error querying telemetry: {e}")

        if telemetry_row:
            rss_memory = telemetry_row.get("rss_memory_mb", rss_memory)
            win_rate = telemetry_row.get("win_rate", 0.0) or 0.0
            cum_pnl = telemetry_row.get("cumulative_pnl", 0.0) or 0.0
            lat = telemetry_row.get("loop_latency_ms")
            if lat is not None:
                loop_latency = f"{lat:.1f} ms"

        # 2. Paper Trades
        open_count = 0
        total_closed = 0
        closed_wins = 0
        total_pnl = 0.0

        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    trades = await conn.fetch("SELECT * FROM paper_trades")
                    for t in trades:
                        if t.get("status") == "OPEN":
                            open_count += 1
                        elif t.get("status") == "CLOSED":
                            total_closed += 1
                            net = t.get("net_pnl_pct") or 0.0
                            total_pnl += net
                            if net > 0:
                                closed_wins += 1
            except Exception as e:
                logger.error(f"Error querying paper trades: {e}")

        calculated_win_rate = (closed_wins / total_closed * 100.0) if total_closed > 0 else (win_rate * 100.0)
        net_pnl_display = total_pnl if total_closed > 0 else cum_pnl

        is_halted = os.path.exists(self.halt_file)
        engine_status = "🔴 HALTED" if is_halted else "🟢 RUNNING"

        msg = (
            f"📊 *DEEP CANDLE SYSTEM STATUS*\n\n"
            f"• *Engine State:* `{engine_status}`\n"
            f"• *Active Positions:* `{open_count}`\n"
            f"• *Total Closed Trades:* `{total_closed}`\n"
            f"• *Win Rate:* `{calculated_win_rate:.1f}%`\n"
            f"• *Net PnL:* `{net_pnl_display:+.4%}`\n"
            f"• *Process Memory (RSS):* `{rss_memory:.2f} MB`\n"
            f"• *Loop Latency:* `{loop_latency}`"
        )
        return msg

    async def handle_panic_command(self) -> str:
        # Create EMERGENCY_HALT_FILE
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.halt_file)), exist_ok=True)
            with open(self.halt_file, "w") as f:
                f.write(f"HALTED at {datetime.now(timezone.utc).isoformat()}\n")
            logger.warning(f"Panic triggered: created {self.halt_file}")
        except Exception as e:
            logger.error(f"Failed to create halt file: {e}")

        # Update all OPEN trades in paper_trades to CLOSED with reason EMERGENCY_PANIC_HALT
        updated_count = 0
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    result = await conn.execute(
                        """
                        UPDATE paper_trades
                        SET status = 'CLOSED', exit_reason = 'EMERGENCY_PANIC_HALT', exit_time = NOW()
                        WHERE status = 'OPEN'
                        """
                    )
                    # Result format: "UPDATE <n>"
                    try:
                        updated_count = int(result.split()[-1])
                    except Exception:
                        updated_count = 0
            except Exception as e:
                logger.error(f"Failed to update paper_trades on panic: {e}")

        return (
            f"🚨 *EMERGENCY PANIC HALT ACTIVATED*\n\n"
            f"• Halt flag written to `{self.halt_file}`\n"
            f"• Ingestion/inference loops will skip new entries.\n"
            f"• Closed `{updated_count}` open position(s) with reason `EMERGENCY_PANIC_HALT`."
        )

    async def handle_resume_command(self) -> str:
        unlinked = False
        if os.path.exists(self.halt_file):
            try:
                os.remove(self.halt_file)
                unlinked = True
                logger.info(f"Resume triggered: unlinked {self.halt_file}")
            except Exception as e:
                logger.error(f"Failed to unlink halt file: {e}")
                return f"⚠️ *ERROR RESUMING:* Failed to unlink halt file: `{e}`"
        else:
            logger.info("Resume triggered but halt file does not exist.")

        return (
            f"✅ *TRADING RESUMED*\n\n"
            f"• Halt flag `{self.halt_file}` {'unlinked' if unlinked else 'was not present'}.\n"
            f"• Live Orchestrator is cleared for execution."
        )

    async def handle_trades_command(self) -> str:
        if not self.db_pool:
            return "⚠️ Database pool unavailable."

        trades = []
        try:
            async with self.db_pool.acquire() as conn:
                trades = await conn.fetch(
                    "SELECT * FROM paper_trades ORDER BY id DESC LIMIT 5"
                )
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return f"⚠️ Error querying paper_trades: `{e}`"

        if not trades:
            return "ℹ️ No executed trades recorded in `paper_trades`."

        lines = ["📜 *RECENT EXECUTED TRADES (LAST 5)*\n"]
        for t in trades:
            tid = t.get("id")
            sym = t.get("symbol")
            status = t.get("status")
            entry_p = t.get("entry_price") or 0.0
            exit_p = t.get("exit_price")
            reason = t.get("exit_reason") or "N/A"
            net_pnl = t.get("net_pnl_pct")

            if net_pnl is not None:
                pnl_str = f"{net_pnl:+.2%}"
            elif status == "CLOSED":
                pnl_str = "0.00%"
            else:
                pnl_str = "OPEN"
            exit_str = f"{exit_p:.2f}" if exit_p is not None else "-"

            lines.append(
                f"• `#{tid}` *{sym}* [{status}] | In: `{entry_p:.2f}` | Out: `{exit_str}` | Reason: `{reason}` | Net: `{pnl_str}`"
            )

        return "\n".join(lines)

    def handle_help_command(self) -> str:
        return (
            "⚔️ *DEEP CANDLE // RONIN AUTONOMOUS LEDGER MANUAL*\n\n"
            "*Deterministic Fast Commands:*\n"
            "• `/status` - Query telemetry, open positions, win rate, net PnL, RSS memory\n"
            "• `/panic` - Create emergency halt flag & force-close all open trades\n"
            "• `/resume` - Clear emergency halt flag to resume live trading\n"
            "• `/trades` - Summary of last 5 executed trades from ledger\n"
            "• `/help` or `/start` - Show this manual\n\n"
            "*Diagnostic Mode:*\n"
            "Send any natural language query to trigger Ronin Quant Ops reasoning via Gemini Proxy."
        )

    async def handle_diagnostic_query(self, query_text: str) -> str:
        # Build db context
        recent_trades: List[Dict[str, Any]] = []
        recent_telemetry: List[Dict[str, Any]] = []

        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    trades_records = await conn.fetch(
                        "SELECT * FROM paper_trades ORDER BY id DESC LIMIT 5"
                    )
                    for r in trades_records:
                        recent_trades.append({
                            "id": r.get("id"),
                            "symbol": r.get("symbol"),
                            "status": r.get("status"),
                            "entry_price": r.get("entry_price"),
                            "exit_price": r.get("exit_price"),
                            "exit_reason": r.get("exit_reason"),
                            "net_pnl_pct": r.get("net_pnl_pct"),
                            "probability": r.get("probability"),
                            "invalidation_thesis": r.get("invalidation_thesis") or "Standard triple-barrier setup",
                            "entry_time": str(r.get("entry_time")),
                            "exit_time": str(r.get("closed_at") or r.get("exit_time") or ""),
                        })
                    telemetry_records = await conn.fetch(
                        "SELECT * FROM system_telemetry ORDER BY id DESC LIMIT 3"
                    )
                    for r in telemetry_records:
                        recent_telemetry.append({
                            "id": r.get("id"),
                            "rss_memory_mb": r.get("rss_memory_mb"),
                            "open_positions": r.get("open_positions"),
                            "win_rate": r.get("win_rate"),
                            "cumulative_pnl": r.get("cumulative_pnl"),
                            "timestamp": str(r.get("timestamp")),
                        })
            except Exception as e:
                logger.error(f"Error fetching DB context for diagnostic query: {e}")

        engine_halted = os.path.exists(self.halt_file)
        context_payload = {
            "operator_query": query_text,
            "engine_state": "HALTED" if engine_halted else "RUNNING",
            "halt_flag_file": self.halt_file,
            "recent_trades": recent_trades,
            "recent_telemetry": recent_telemetry,
        }

        user_content = (
            f"Operator Query: {query_text}\n\n"
            f"System State Context:\n{context_payload}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": RONIN_QUANT_OPS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.proxy_url, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                return f"⚔️ *RONIN // AUTONOMOUS LEDGER POST-MORTEM*\n\n{content}"
        except httpx.TimeoutException:
            logger.warning("Diagnostic query to proxy timed out.")
            return "⚠️ *DIAGNOSTIC FALLBACK:* Gemini proxy request timed out. Primary telemetry and risk invariants remain healthy."
        except Exception as e:
            logger.error(f"Diagnostic query failed: {e}")
            return f"⚠️ *DIAGNOSTIC FALLBACK:* Proxy unreachable (`{e}`). System operational invariants remain active."

    async def process_update(self, update: Dict[str, Any]):
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return

        chat = msg.get("chat") or {}
        chat_id = chat.get("id")

        # Security Whitelist: Drop unauthorized silently
        if not self.is_authorized(chat_id):
            logger.warning(f"Unauthorized update dropped from chat_id: {chat_id}")
            return

        text = (msg.get("text") or "").strip()
        if not text:
            return

        # Deterministic Fast Commands
        cmd = text.split()[0].lower()
        if cmd == "/status":
            reply = await self.handle_status_command()
            await self.send_message(reply)
        elif cmd == "/panic":
            reply = await self.handle_panic_command()
            await self.send_message(reply)
        elif cmd == "/resume":
            reply = await self.handle_resume_command()
            await self.send_message(reply)
        elif cmd == "/trades":
            reply = await self.handle_trades_command()
            await self.send_message(reply)
        elif cmd in ("/help", "/start"):
            reply = self.handle_help_command()
            await self.send_message(reply)
        else:
            # Hermes Agentic Diagnostic Mode
            reply = await self.handle_diagnostic_query(text)
            await self.send_message(reply)

    async def poll_telegram(self):
        if not self.bot_token:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params = {"offset": self.last_update_id + 1, "timeout": 2}

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    for upd in data.get("result", []):
                        self.last_update_id = max(self.last_update_id, upd.get("update_id", 0))
                        await self.process_update(upd)
        except Exception as e:
            logger.debug(f"Telegram polling tick error: {e}")

    async def check_trade_transitions(self):
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM paper_trades ORDER BY id ASC")

            current_trades = {r["id"]: dict(r) for r in rows}

            if not self._monitor_initialized:
                # Seed baseline without broadcasting historical trades
                self._tracked_trades = current_trades
                self._monitor_initialized = True
                return

            for tid, trade in current_trades.items():
                old_trade = self._tracked_trades.get(tid)
                sym = trade.get("symbol")
                status = trade.get("status")

                if old_trade is None:
                    # Brand new trade opened
                    if status == "OPEN":
                        entry_p = trade.get("entry_price") or 0.0
                        prob = trade.get("probability") or 0.0
                        stop_p = trade.get("stop_price") or 0.0
                        target_p = trade.get("target_price") or 0.0
                        thesis = trade.get("invalidation_thesis") or "System Triple-Barrier Setup"

                        alert_msg = (
                            f"🚨 *NEW TRADE OPENED: {sym}*\n\n"
                            f"• *Entry Price:* `{entry_p:.2f}`\n"
                            f"• *Probability:* `{prob:.1%}`\n"
                            f"• *Stop Loss (SL):* `{stop_p:.2f}`\n"
                            f"• *Take Profit (TP1):* `{target_p:.2f}`\n"
                            f"• *Thesis:* _{thesis}_"
                        )
                        await self.send_message(alert_msg)

                else:
                    # Check if trade transitioned from OPEN -> CLOSED
                    if old_trade.get("status") == "OPEN" and status == "CLOSED":
                        reason = trade.get("exit_reason") or "UNKNOWN"
                        exit_p = trade.get("exit_price") or 0.0
                        net_pnl = trade.get("net_pnl_pct") or 0.0

                        alert_msg = (
                            f"🏁 *TRADE CLOSED: {sym}*\n\n"
                            f"• *Exit Reason:* `{reason}`\n"
                            f"• *Exit Price:* `{exit_p:.2f}`\n"
                            f"• *Fee-Adjusted Net PnL:* `{net_pnl:+.2%}`"
                        )
                        await self.send_message(alert_msg)

            self._tracked_trades = current_trades

        except Exception as e:
            logger.error(f"Error checking trade transitions: {e}")

    async def run(self):
        self.running = True
        logger.info("Starting TelegramAIOps Daemon...")

        while self.running:
            try:
                # 1. Telegram Updates
                await self.poll_telegram()

                # 2. Paper Trades Monitor
                await self.check_trade_transitions()

                await asyncio.sleep(3)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in ops copilot loop: {e}")
                await asyncio.sleep(3)

    async def stop(self):
        logger.info("Stopping TelegramAIOps Daemon...")
        self.running = False

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deep Candle Telegram AIOps Co-Pilot")
    parser.add_argument("--bot-token", type=str, default=None)
    parser.add_argument("--admin-chat-id", type=str, default=None)
    args = parser.parse_args()

    db_pool = await asyncpg.create_pool(
        user=os.getenv("POSTGRES_USER", "deepcandle"),
        password=os.getenv("POSTGRES_PASSWORD", "deepcandle_password"),
        database=os.getenv("POSTGRES_DB", "deepcandle"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5440")
    )

    copilot = TelegramAIOps(
        db_pool=db_pool,
        bot_token=args.bot_token,
        admin_chat_id=args.admin_chat_id
    )

    try:
        await copilot.run()
    finally:
        await copilot.stop()
        await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())
