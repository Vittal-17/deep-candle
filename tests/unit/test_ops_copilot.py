import os
import sys
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

# Ensure scripts and backend are in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))

from scripts.ops_copilot import TelegramAIOps, RONIN_QUANT_OPS_SYSTEM_PROMPT
from app.config import settings

@pytest.fixture
def mock_db_pool():
    pool = MagicMock()
    conn = AsyncMock()
    # Support async with pool.acquire() as conn:
    pool.acquire.return_value.__aenter__.return_value = conn
    return pool, conn

@pytest.fixture
def copilot(tmp_path, mock_db_pool):
    pool, conn = mock_db_pool
    halt_file = str(tmp_path / "deep_candle_halt.flag")
    bot = TelegramAIOps(
        db_pool=pool,
        bot_token="test_token_123",
        admin_chat_id="12345678"
    )
    bot.halt_file = halt_file
    return bot

def test_security_whitelist(copilot):
    # Authorized ID matches
    assert copilot.is_authorized("12345678") is True
    assert copilot.is_authorized(12345678) is True
    assert copilot.is_authorized(" 12345678 ") is True

    # Unauthorized IDs dropped
    assert copilot.is_authorized("99999999") is False
    assert copilot.is_authorized(None) is False
    assert copilot.is_authorized("") is False

@pytest.mark.asyncio
async def test_security_filtering_drops_unauthorized(copilot):
    with patch.object(copilot, "handle_status_command", new_callable=AsyncMock) as mock_status:
        update = {
            "update_id": 1,
            "message": {
                "chat": {"id": 99999999},  # Unauthorized
                "text": "/status"
            }
        }
        await copilot.process_update(update)
        mock_status.assert_not_called()

@pytest.mark.asyncio
async def test_panic_creates_halt_file_and_updates_db(copilot, mock_db_pool):
    pool, conn = mock_db_pool
    conn.execute.return_value = "UPDATE 2"

    assert not os.path.exists(copilot.halt_file)

    msg = await copilot.handle_panic_command()

    # Verify halt file created
    assert os.path.exists(copilot.halt_file)
    assert "EMERGENCY PANIC HALT ACTIVATED" in msg
    assert "Closed `2` open position(s)" in msg

    # Verify DB query was executed
    conn.execute.assert_called_once()
    sql_called = conn.execute.call_args[0][0]
    assert "UPDATE paper_trades" in sql_called
    assert "EMERGENCY_PANIC_HALT" in sql_called
    assert "exit_time = NOW()" in sql_called

@pytest.mark.asyncio
async def test_resume_clears_halt_file(copilot):
    # Pre-create halt file
    with open(copilot.halt_file, "w") as f:
        f.write("HALTED\n")
    assert os.path.exists(copilot.halt_file)

    msg = await copilot.handle_resume_command()

    # Verify halt file removed
    assert not os.path.exists(copilot.halt_file)
    assert "TRADING RESUMED" in msg

@pytest.mark.asyncio
async def test_deterministic_status_format(copilot, mock_db_pool):
    pool, conn = mock_db_pool
    conn.fetchrow.return_value = {
        "rss_memory_mb": 85.5,
        "win_rate": 0.65,
        "cumulative_pnl": 0.045,
        "loop_latency_ms": 18.5
    }
    conn.fetch.return_value = [
        {"status": "OPEN", "net_pnl_pct": None},
        {"status": "CLOSED", "net_pnl_pct": 0.02},
        {"status": "CLOSED", "net_pnl_pct": -0.01},
    ]

    msg = await copilot.handle_status_command()

    assert "DEEP CANDLE SYSTEM STATUS" in msg
    assert "Active Positions:" in msg
    assert "`1`" in msg
    assert "Total Closed Trades:" in msg
    assert "`2`" in msg
    assert "Win Rate:" in msg
    assert "Net PnL:" in msg
    assert "Process Memory (RSS):" in msg
    assert "Engine State:" in msg
    assert "18.5 ms" in msg

@pytest.mark.asyncio
async def test_deterministic_trades_command(copilot, mock_db_pool):
    pool, conn = mock_db_pool
    conn.fetch.return_value = [
        {
            "id": 101,
            "symbol": "BTCUSDT",
            "status": "CLOSED",
            "entry_price": 50000.0,
            "exit_price": 50800.0,
            "exit_reason": "TP",
            "net_pnl_pct": 0.015
        },
        {
            "id": 102,
            "symbol": "ETHUSDT",
            "status": "CLOSED",
            "entry_price": 3000.0,
            "exit_price": 3000.0,
            "exit_reason": "EMERGENCY_PANIC_HALT",
            "net_pnl_pct": None
        }
    ]

    msg = await copilot.handle_trades_command()
    assert "RECENT EXECUTED TRADES" in msg
    assert "#101" in msg
    assert "BTCUSDT" in msg
    assert "TP" in msg
    assert "+1.50%" in msg
    assert "#102" in msg
    assert "ETHUSDT" in msg
    assert "EMERGENCY_PANIC_HALT" in msg
    assert "Net: `0.00%`" in msg
    assert "Net: `OPEN`" not in msg

@pytest.mark.asyncio
async def test_diagnostic_proxy_payload_dispatch(copilot, mock_db_pool):
    pool, conn = mock_db_pool
    conn.fetch.side_effect = [
        # recent trades
        [{"id": 1, "symbol": "BTCUSDT", "status": "CLOSED", "entry_price": 50000.0, "exit_price": 50500.0, "exit_reason": "TP", "net_pnl_pct": 0.009, "probability": 0.54, "invalidation_thesis": "Breakout", "entry_time": None, "exit_time": None}],
        # recent telemetry
        [{"id": 1, "rss_memory_mb": 80.0, "open_positions": 0, "win_rate": 0.7, "cumulative_pnl": 0.05, "timestamp": None}]
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Risk levels nominal. TP1 captured at 1.5x ATR. No systemic anomalies detected."
                }
            }
        ]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        reply = await copilot.handle_diagnostic_query("What happened to the last BTC trade?")

        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json")

        assert payload["model"] == "gemini-3.1-pro-low"
        assert payload["messages"][0]["role"] == "system"
        assert RONIN_QUANT_OPS_SYSTEM_PROMPT in payload["messages"][0]["content"]
        assert "BTCUSDT" in payload["messages"][1]["content"]
        assert "0.54" in payload["messages"][1]["content"]
        assert "Breakout" in payload["messages"][1]["content"]
        assert "RONIN // AUTONOMOUS LEDGER POST-MORTEM" in reply
        assert "Risk levels nominal" in reply

@pytest.mark.asyncio
async def test_diagnostic_timeout_fallback(copilot):
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Read timeout")):
        reply = await copilot.handle_diagnostic_query("Analyze recent volatility spikes")
        assert "DIAGNOSTIC FALLBACK" in reply
        assert "timed out" in reply

@pytest.mark.asyncio
async def test_push_monitor_loop_detects_transitions(copilot, mock_db_pool):
    pool, conn = mock_db_pool

    # Initial snapshot
    conn.fetch.return_value = [
        {"id": 1, "symbol": "BTCUSDT", "status": "OPEN", "entry_price": 50000.0, "stop_price": 49000.0, "target_price": 51500.0, "probability": 0.8, "invalidation_thesis": "Test thesis"}
    ]

    with patch.object(copilot, "send_message", new_callable=AsyncMock) as mock_send:
        # 1. Initialize baseline (no alert should be sent for existing rows)
        await copilot.check_trade_transitions()
        mock_send.assert_not_called()

        # 2. Next tick: a new trade id=2 is opened, and trade id=1 closed
        conn.fetch.return_value = [
            {"id": 1, "symbol": "BTCUSDT", "status": "CLOSED", "exit_reason": "TP", "exit_price": 51500.0, "net_pnl_pct": 0.029},
            {"id": 2, "symbol": "ETHUSDT", "status": "OPEN", "entry_price": 3000.0, "stop_price": 2900.0, "target_price": 3150.0, "probability": 0.75, "invalidation_thesis": "Breakout"}
        ]
        await copilot.check_trade_transitions()

        # Should send 2 alerts: one for closed trade #1, one for opened trade #2
        assert mock_send.call_count == 2
        messages = [call.args[0] for call in mock_send.call_args_list]

        # Verify new trade alert content
        new_trade_alert = [m for m in messages if "NEW TRADE OPENED" in m][0]
        assert "ETHUSDT" in new_trade_alert
        assert "3000.00" in new_trade_alert
        assert "Breakout" in new_trade_alert

        # Verify closed trade alert content
        closed_alert = [m for m in messages if "TRADE CLOSED" in m][0]
        assert "BTCUSDT" in closed_alert
        assert "TP" in closed_alert
        assert "+2.90%" in closed_alert
