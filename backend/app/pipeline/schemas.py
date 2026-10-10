from pydantic import BaseModel
from datetime import datetime

class NormalizedBar(BaseModel):
    symbol: str
    timeframe: str
    bar_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_volume: float
    trades_count: int
