import pandas as pd
import numpy as np
from collections import deque, defaultdict
import asyncpg
from app.features.indicators import (
    calculate_atr_pct,
    calculate_bb_squeeze,
    calculate_vol_zscore,
    calculate_vwap_dist_atr,
    calculate_rsi,
    calculate_ema_dist
)

class RollingFeatureBuffer:
    def __init__(self):
        self.bars = defaultdict(lambda: {
            '15m': deque(maxlen=250),
            '1h': deque(maxlen=250),
            '5m': deque(maxlen=10)
        })

    async def initialize_from_db(self, db_pool):
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT DISTINCT symbol FROM market_bars")
            symbols = [r['symbol'] for r in rows]

            for symbol in symbols:
                for tf, limit in [('15m', 250), ('1h', 250), ('5m', 10)]:
                    query = """
                        SELECT bar_time, open_price, high_price, low_price, close_price, volume
                        FROM market_bars
                        WHERE symbol = $1 AND timeframe = $2
                        ORDER BY bar_time DESC LIMIT $3
                    """
                    data = await conn.fetch(query, symbol, tf, limit)
                    for row in reversed(data):
                        self.bars[symbol][tf].append(dict(row))

    def on_candle_closed(self, symbol: str, timeframe: str, bar: dict) -> dict | None:
        self.bars[symbol][timeframe].append(bar)
        if timeframe == '15m':
            return self._compute_features(symbol)
        return None

    def _compute_features(self, symbol: str) -> dict | None:
        bars_15m = self.bars[symbol]['15m']
        bars_1h = self.bars[symbol]['1h']
        bars_5m = self.bars[symbol]['5m']
        btc_bars_15m = self.bars['BTCUSDT']['15m']

        if len(bars_15m) < 30 or len(bars_1h) < 200 or len(bars_5m) < 3 or len(btc_bars_15m) < 1:
            return None

        # 1. Base 15m Indicators
        df_15 = pd.DataFrame(list(bars_15m))
        df_15['bar_time'] = pd.to_datetime(df_15['bar_time'], utc=True)

        atr_series = calculate_atr_pct(df_15)
        bb_squeeze_series = calculate_bb_squeeze(df_15)
        vol_zscore_series = calculate_vol_zscore(df_15)
        vwap_dist_series = calculate_vwap_dist_atr(df_15, atr_series)
        rsi_15m_series = calculate_rsi(df_15)

        # 2. 1h Indicators (strictly lagged by 1 bar as per temporal_aligner)
        df_1h = pd.DataFrame(list(bars_1h))
        df_1h['bar_time'] = pd.to_datetime(df_1h['bar_time'], utc=True)
        rsi_1h_series = calculate_rsi(df_1h)
        ema_200_dist_1h_series = calculate_ema_dist(df_1h)

        # For the latest 15m bar, retrieve the latest 1h bar closed strictly prior to its close
        current_15m_time = df_15.iloc[-1]['bar_time']
        prior_1h_mask = df_1h['bar_time'] < current_15m_time
        if not prior_1h_mask.any():
            return None

        last_prior_1h_idx = df_1h[prior_1h_mask].index[-1]
        rsi_1h = float(rsi_1h_series.iloc[last_prior_1h_idx])
        ema_200_dist_1h = float(ema_200_dist_1h_series.iloc[last_prior_1h_idx])

        # 3. 5m Micro Metrics
        m5_latest = bars_5m[-1]
        m5_prev1 = bars_5m[-2]
        m5_prev2 = bars_5m[-3]

        ret_5m = (m5_latest['close_price'] - m5_latest['open_price']) / (m5_latest['open_price'] + 1e-7)
        ret_15m = (df_15.iloc[-1]['close_price'] - df_15.iloc[-1]['open_price']) / (df_15.iloc[-1]['open_price'] + 1e-7)

        micro_momentum_ratio = float(ret_5m / (ret_15m + 1e-7))
        prev_2_vol_avg = (m5_prev1['volume'] + m5_prev2['volume']) / 2.0
        micro_volume_acceleration = float(m5_latest['volume'] / (prev_2_vol_avg + 1e-7))

        # 4. BTC Relative Delta
        btc_latest = btc_bars_15m[-1]
        btc_ret_15m = (btc_latest['close_price'] - btc_latest['open_price']) / (btc_latest['open_price'] + 1e-7)
        btc_relative_delta = float(ret_15m - btc_ret_15m)

        features = {
            'atr_pct_15m': float(atr_series.iloc[-1]),
            'bb_squeeze_15m': float(bb_squeeze_series.iloc[-1]),
            'vol_zscore_15m': float(vol_zscore_series.iloc[-1]),
            'vwap_dist_atr_15m': float(vwap_dist_series.iloc[-1]),
            'rsi_15m': float(rsi_15m_series.iloc[-1]),
            'rsi_1h': rsi_1h,
            'ema_200_dist_1h': ema_200_dist_1h,
            'micro_momentum_ratio': micro_momentum_ratio,
            'micro_volume_acceleration': micro_volume_acceleration,
            'btc_relative_delta': btc_relative_delta,
            'return_15m': float(ret_15m)
        }

        # Sanitize NaNs/Infs
        for k, v in features.items():
            if np.isnan(v) or np.isinf(v):
                features[k] = 0.0

        return features
