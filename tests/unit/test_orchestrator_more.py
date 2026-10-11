
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.engine.live_orchestrator import LiveOrchestrator

@pytest.mark.asyncio
async def test_live_orchestrator_stop_and_exceptions():
    db_pool = AsyncMock()
    orchestrator = LiveOrchestrator(db_pool)

    # Test stop
    orchestrator.ws = AsyncMock()
    orchestrator.telemetry_task = MagicMock()
    await orchestrator.stop()
    assert not orchestrator.running
    orchestrator.ws.close.assert_called_once()
    orchestrator.telemetry_task.cancel.assert_called_once()

    # Test websocket exception paths
    import websockets

    class MockWebsocketConnClosed:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            raise websockets.ConnectionClosed(None, None)
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockWebsocketGenericError:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            raise Exception("Generic WS Error")
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    orchestrator.running = True

    # We need a mock sleep that stops the loop on the first sleep
    async def mock_sleep(*args, **kwargs):
        orchestrator.running = False

    with patch('app.engine.live_orchestrator.websockets.connect', MockWebsocketConnClosed):
        with patch('asyncio.sleep', new=mock_sleep):
            await orchestrator._websocket_loop()

    orchestrator.running = True
    with patch('app.engine.live_orchestrator.websockets.connect', MockWebsocketGenericError):
        with patch('asyncio.sleep', new=mock_sleep):
            await orchestrator._websocket_loop()
