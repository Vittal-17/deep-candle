import os
import sys
import pandas as pd
import numpy as np
import logging
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.pipeline.database import Database
from app.features.indicators import (
    calculate_atr_pct, calculate_bb_squeeze, calculate_vol_zscore,
    calculate_vwap_dist_atr, calculate_rsi, calculate_ema_dist
)
from app.features.temporal_aligner import (
    align_lagged_1h_features, compute_5m_micro_features, align_btc_anchor
)

logger = logging.getLogger(__name__)

async def load_bars_from_db(symbol: str, timeframe: str) -> pd.DataFrame:
    query = """
        SELECT symbol, timeframe, bar_time, open_price, high_price, low_price, close_price, volume, quote_volume, trades_count
        FROM market_bars
        WHERE symbol = $1 AND timeframe = $2
        ORDER BY bar_time ASC
    """
    async with Database._pool.acquire() as conn:
        rows = await conn.fetch(query, symbol, timeframe)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    return df

def build_features_for_symbol(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_btc_15m: pd.DataFrame
) -> pd.DataFrame:
    if df_15m.empty or df_1h.empty:
        return pd.DataFrame()

    # 1. Base 15m Indicators
    df_15m = df_15m.sort_values('bar_time').reset_index(drop=True)
    df_15m['atr_pct_15m'] = calculate_atr_pct(df_15m)
    df_15m['bb_squeeze_15m'] = calculate_bb_squeeze(df_15m)
    df_15m['vol_zscore_15m'] = calculate_vol_zscore(df_15m)
    df_15m['vwap_dist_atr_15m'] = calculate_vwap_dist_atr(df_15m, df_15m['atr_pct_15m'])
    df_15m['rsi_15m'] = calculate_rsi(df_15m)

    # 2. 1h Indicators
    df_1h = df_1h.sort_values('bar_time').reset_index(drop=True)
    df_1h['rsi_1h'] = calculate_rsi(df_1h)
    df_1h['ema_200_dist_1h'] = calculate_ema_dist(df_1h)

    # 3. Align 1h macro indicators
    df_merged = align_lagged_1h_features(df_15m, df_1h)

    # 4. Align 5m micro sub-bars
    df_merged = compute_5m_micro_features(df_5m, df_merged)

    # 5. Align BTC relative delta
    df_merged = align_btc_anchor(df_merged, df_btc_15m)

    # Select 11 features + metadata
    feature_cols = [
        'symbol', 'bar_time',
        'atr_pct_15m', 'bb_squeeze_15m', 'vol_zscore_15m', 'vwap_dist_atr_15m',
        'rsi_15m', 'rsi_1h', 'ema_200_dist_1h', 'micro_momentum_ratio',
        'micro_volume_acceleration', 'btc_relative_delta', 'return_15m'
    ]

    df_final = df_merged[feature_cols].copy()

    # Warmup data requirement: drop first 220 1h bars equivalent (~880 15m bars)
    df_final = df_final.iloc[880:].reset_index(drop=True)

    # Quality Gating: drop NaN / Inf
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df_final

async def generate_matrix(output_path: str = "data/processed/features_matrix.parquet"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    await Database.connect()

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]
    df_btc_15m = await load_bars_from_db("BTCUSDT", "15m")

    dfs = []
    for sym in symbols:
        logger.info(f"Processing features for {sym}...")
        df_15m = await load_bars_from_db(sym, "15m")
        df_1h = await load_bars_from_db(sym, "1h")
        df_5m = await load_bars_from_db(sym, "5m")

        df_feat = build_features_for_symbol(df_15m, df_1h, df_5m, df_btc_15m)
        if not df_feat.empty:
            dfs.append(df_feat)

    await Database.disconnect()

    if dfs:
        matrix_df = pd.concat(dfs, ignore_index=True)
        matrix_df.to_parquet(output_path, index=False)
        logger.info(f"Saved feature matrix to {output_path} with shape {matrix_df.shape}")
        return matrix_df
    else:
        logger.warning("No data found to build feature matrix.")
        return pd.DataFrame()

if __name__ == "__main__":  # pragma: no cover
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(generate_matrix())
