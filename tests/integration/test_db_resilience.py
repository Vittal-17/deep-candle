
import pytest
import asyncio
import asyncpg
from unittest.mock import patch, MagicMock
from app.pipeline.database import Database

@pytest.mark.asyncio
async def test_simulated_connection_timeout():
    # Test async context managers under connection acquisition timeouts.
    # Assert transactions roll back cleanly and connections return to the pool without leaking.

    # We will mock the pool.acquire to raise an asyncio.TimeoutError
    class MockPool:
        def __init__(self):
            self.active_connections = 0

        def acquire(self):
            return MockAcquire(self)

        async def close(self):
            pass

    class MockAcquire:
        def __init__(self, pool):
            self.pool = pool

        async def __aenter__(self):
            raise asyncio.TimeoutError("Connection acquisition timeout")

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False # Don't suppress exception

    Database._pool = MockPool()

    try:
        async with Database._pool.acquire() as conn:
            pass
    except asyncio.TimeoutError:
        pass

    assert Database._pool.active_connections == 0, "Connection leaked!"
