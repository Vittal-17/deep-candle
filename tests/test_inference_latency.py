import time
import psutil
import os
import random

def test_inference_latency_and_memory():
    # Import everything first to establish a true baseline
    from app.ml.inference import LGBMInferenceEngine

    process = psutil.Process(os.getpid())
    baseline_mb = process.memory_info().rss / (1024 * 1024)

    # Initialize the engine (loads the booster)
    engine = LGBMInferenceEngine()

    loaded_mb = process.memory_info().rss / (1024 * 1024)
    booster_rss_mb = loaded_mb - baseline_mb

    # Generate 1000 random feature dictionaries
    features_list = []
    for _ in range(1000):
        features = {name: random.random() for name in engine.feature_names}
        features_list.append(features)

    # Benchmark latency
    start_time = time.time()
    for features in features_list:
        engine.predict_proba(features)
    end_time = time.time()

    total_time = end_time - start_time
    mean_latency_ms = (total_time / 1000) * 1000

    print(f"Mean latency: {mean_latency_ms:.4f} ms")
    assert mean_latency_ms < 2.0, f"Latency {mean_latency_ms:.4f}ms exceeds 2.0ms limit"

    print(f"Booster RSS Memory: {booster_rss_mb:.2f} MB")
    assert booster_rss_mb < 30.0, f"Memory {booster_rss_mb:.2f}MB exceeds 30MB limit"
