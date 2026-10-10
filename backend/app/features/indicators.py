import pandas as pd
import numpy as np

def calculate_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high_price']
    low = df['low_price']
    close = df['close_price']
    prev_close = close.shift(1)

    tr = np.maximum(
        high - low,
        np.maximum(
            (high - prev_close).abs(),
            (low - prev_close).abs()
        )
    )

    # Wilder's smoothing
    atr = pd.Series(index=df.index, dtype='float64')
    atr.iloc[period - 1] = tr.iloc[:period].mean()
    for i in range(period, len(df)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period

    return atr / (close + 1e-7)

def calculate_bb_squeeze(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    close = df['close_price']
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)
    width = 2.0 * std_mult * std
    return width / (sma + 1e-7)

def calculate_vol_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    vol = df['volume']
    mean = vol.rolling(window=period).mean()
    std = vol.rolling(window=period).std(ddof=0)
    return (vol - mean) / (std + 1e-7)

def calculate_vwap_dist_atr(df: pd.DataFrame, atr_pct_15m: pd.Series) -> pd.Series:
    # Intraday VWAP resetting daily at 00:00:00 UTC
    typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3.0
    tp_vol = typical_price * df['volume']

    dates = df['bar_time'].dt.date
    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = df['volume'].groupby(dates).cumsum()

    vwap = cum_tp_vol / (cum_vol + 1e-7)
    atr = atr_pct_15m * df['close_price']
    return (df['close_price'] - vwap) / (atr + 1e-7)

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df['close_price']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = pd.Series(index=df.index, dtype='float64')
    avg_loss = pd.Series(index=df.index, dtype='float64')

    avg_gain.iloc[period] = gain.iloc[1:period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1:period + 1].mean()

    for i in range(period + 1, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / (avg_loss + 1e-7)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.clip(lower=0.0, upper=100.0)

def calculate_ema_dist(df: pd.DataFrame, period: int = 200, atr_period: int = 14) -> pd.Series:
    close = df['close_price']
    ema = close.ewm(span=period, adjust=False).mean()

    high = df['high_price']
    low = df['low_price']
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))

    atr = pd.Series(index=df.index, dtype='float64')
    atr.iloc[atr_period - 1] = tr.iloc[:atr_period].mean()
    for i in range(atr_period, len(df)):
        atr.iloc[i] = (atr.iloc[i - 1] * (atr_period - 1) + tr.iloc[i]) / atr_period

    return (close - ema) / (atr + 1e-7)
