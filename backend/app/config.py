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
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_ADMIN_CHAT_ID: str = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
    EMERGENCY_HALT_FILE: str = os.getenv("EMERGENCY_HALT_FILE", "/tmp/deep_candle_halt.flag")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "backend/app/ml/artifacts/deepcandle_lgbm_v1.txt")
    SIGNAL_THRESHOLD: float = float(os.getenv("SIGNAL_THRESHOLD", "0.50"))

settings = Settings()
