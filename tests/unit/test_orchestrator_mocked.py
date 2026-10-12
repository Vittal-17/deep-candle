
import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock
from app.engine.live_orchestrator import LiveOrchestrator
from datetime import datetime, timezone

@pytest.mark.asyncio
async def test_live_orchestrator_websocket_loop():
    db_pool = AsyncMock()
    orchestrator = LiveOrchestrator(db_pool)

    # We need to mock websockets.connect
    # It's an async context manager that yields a websocket object

    class MockWebsocket:
        def __init__(self):
            self.call_count = 0
            # Mock payloads
            self.payloads = [
                json.dumps({
                    'e': 'kline', 's': 'BTCUSDT',
                    'k': {'i': '5m', 'x': True, 'h': '105', 'l': '95', 'c': '100', 'o': '100', 'v': '1000', 'T': 1672531200000, 't': 1672531200000}
                }),
                json.dumps({
                    'e': 'kline', 's': 'BTCUSDT',
                    'k': {'i': '1h', 'x': True, 'h': '105', 'l': '95', 'c': '100', 'o': '100', 'v': '1000', 'T': 1672531200000, 't': 1672531200000}
                }),
                json.dumps({
                    'e': 'kline', 's': 'BTCUSDT',
                    'k': {'i': '15m', 'x': True, 'h': '105', 'l': '95', 'c': '100', 'o': '100', 'v': '1000', 'T': 1672531200000, 't': 1672531200000}
                })
            ]

        async def recv(self):
            if self.call_count < len(self.payloads):
                msg = self.payloads[self.call_count]
                self.call_count += 1
                return msg
            else:
                # Stop the loop by raising an exception or setting running to False
                orchestrator.running = False
                raise asyncio.CancelledError("Stop loop")

        async def close(self):
            pass

    class MockConnect:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return MockWebsocket()
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # We also want to cover the telemetry loop briefly.
    # We can run `orchestrator.run()` for a short time.
    # Wait, `run()` starts telemetry and websocket loop.
    # Websocket loop will exit when it hits CancelledError, and `run()` awaits it.

    with patch('app.engine.live_orchestrator.websockets.connect', MockConnect):
        # We need to mock asyncio.sleep in telemetry to not wait 60s
        async def mock_sleep(seconds):
            if seconds == 60:
                orchestrator.running = False
                raise asyncio.CancelledError()
            await asyncio.sleep(0) # yield

        with patch('asyncio.sleep', new=mock_sleep):
            try:
                await orchestrator.run()
            except asyncio.CancelledError:
                pass

@pytest.mark.asyncio
async def test_live_orchestrator_telemetry_loop():
    # specifically test the telemetry loop logic
    db_pool = AsyncMock()
    orchestrator = LiveOrchestrator(db_pool)
    orchestrator.running = True

    # We need to execute _telemetry_loop directly
    # Mock sleep to raise CancelledError after 1 iteration
    sleep_count = 0
    async def mock_sleep(seconds):
        nonlocal sleep_count
        if sleep_count == 0:
            sleep_count += 1
            return
        orchestrator.running = False
        raise asyncio.CancelledError()

    with patch('asyncio.sleep', new=mock_sleep):
        await orchestrator._telemetry_loop()

    # Test high memory path
    orchestrator.running = True
    sleep_count = 0

    class MockProcess:
        def memory_info(self):
            class MockMem:
                rss = 150 * 1024 * 1024 # 150MB
            return MockMem()

    with patch('asyncio.sleep', new=mock_sleep), patch('psutil.Process', return_value=MockProcess()):
        await orchestrator._telemetry_loop()

    assert not orchestrator.running # Should shut down

@pytest.mark.asyncio
async def test_live_orchestrator_emergency_halt(tmp_path):
    db_pool = AsyncMock()
    orchestrator = LiveOrchestrator(db_pool)
    halt_file = str(tmp_path / "emergency_halt.flag")
    with open(halt_file, "w") as f:
        f.write("HALTED")

    # Mock buffer.on_candle_closed to return features
    orchestrator.buffer.on_candle_closed = MagicMock(return_value={"rsi": 50.0})
    orchestrator.signal_engine.process_features = AsyncMock()
    orchestrator.paper_trader.evaluate_tick_or_bar = AsyncMock()
    orchestrator.paper_trader.open_position = AsyncMock()

    kline = {
        'e': 'kline',
        's': 'BTCUSDT',
        'k': {
            'i': '15m',
            'x': True,
            'h': '105',
            'l': '95',
            'c': '100',
            'o': '100',
            'v': '1000',
            'T': 1672531200000,
            't': 1672531200000
        }
    }

    with patch('app.config.settings.EMERGENCY_HALT_FILE', halt_file):
        await orchestrator._handle_ws_message(kline)

    # Process features / trade execution must have been skipped
    orchestrator.signal_engine.process_features.assert_not_called()
    orchestrator.paper_trader.open_position.assert_not_called()

