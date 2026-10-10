
import pytest
import asyncio
import psutil
import os
import numpy as np
from datetime import datetime, timezone
from app.pipeline.rolling_buffer import RollingFeatureBuffer
from app.ml.inference import LGBMInferenceEngine

@pytest.mark.asyncio
async def test_ring_buffer_overwrite_stability():
    buffer = RollingFeatureBuffer()

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024 * 1024)


    for i in range(1000):

        bar_15m = {
            'bar_time': datetime.now(timezone.utc),
            'open_price': 100.0,
            'high_price': 105.0,
            'low_price': 95.0,
            'close_price': 100.0,
            'volume': 1000.0
        }
        buffer.on_candle_closed('BTCUSDT', '15m', bar_15m)

        if i % 4 == 0: # 1h
            buffer.on_candle_closed('BTCUSDT', '1h', bar_15m)
        if i % 3 == 0: # 5m
            buffer.on_candle_closed('BTCUSDT', '5m', bar_15m)

    rss_after = process.memory_info().rss / (1024 * 1024)

    assert len(buffer.bars['BTCUSDT']['15m']) <= 250
    assert len(buffer.bars['BTCUSDT']['1h']) <= 250
    assert len(buffer.bars['BTCUSDT']['5m']) <= 10


    delta_rss = rss_after - rss_before
    assert delta_rss < 15.0, f"Memory leak detected: {delta_rss:.2f}MB"


@pytest.mark.asyncio
async def test_booster_concurrent_forward_pass_stress():
    engine = LGBMInferenceEngine()

    # Needs a mock model if the real one isn't there, but let's assume it is or we mock it.
    # Actually, the requirement asks to execute 5000 predictions across concurrent async tasks.
    # We can mock the booster if it's not present, but let's see if we can just patch it.

    # We will mock the lgb.Booster to avoid needing the real model file for the test
    import lightgbm as lgb
    from unittest.mock import MagicMock

    original_booster = lgb.Booster

    class MockBooster:
        def __init__(self, **kwargs):
            pass
        def feature_name(self):
            return ['f1', 'f2']
        def predict(self, features):
            return [0.75]

    lgb.Booster = MockBooster

    # Re-initialize engine to use mock
    LGBMInferenceEngine._instance = None
    engine = LGBMInferenceEngine()

    async def predict_task():
        features = {'f1': 1.0, 'f2': 2.0}
        # In real code LGBM predict is synchronous, so we run it in executor or just call it directly
        # The engine is sync, so we just call it
        return engine.predict_proba(features)

    tasks = [predict_task() for _ in range(5000)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 5000
    assert all(r == 0.75 for r in results)

    # Restore
    lgb.Booster = original_booster
