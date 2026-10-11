import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://deepcandle:deepcandle_password@localhost:5440/deepcandle")
    TRACKED_SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]
    TIMEFRAMES: List[str] = ["15m", "1h", "5m"]
    BINANCE_WS_URL: str = "wss://stream.binance.com:9443/stream"
    BINANCE_REST_URL: str = "https://api.binance.com/api/v3/klines"
    PROXY_URL: str = os.getenv("PROXY_URL", "http://127.0.0.1:3009/v1/chat/completions")

settings = Settings()
