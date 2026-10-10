import asyncio
import asyncpg
import pandas as pd
import numpy as np
import os

DB_USER = os.getenv("POSTGRES_USER", "deepcandle")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "deepcandle_password")
DB_NAME = os.getenv("POSTGRES_DB", "deepcandle")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5440")

async def fetch_market_bars():
    conn = await asyncpg.connect(
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT
    )

    # We need symbol, bar_time, high_price, low_price, close_price
    query = """
    SELECT symbol, bar_time, high_price, low_price, close_price
    FROM market_bars
    WHERE timeframe = '15m'
    """
    rows = await conn.fetch(query)
    await conn.close()

    df = pd.DataFrame([dict(r) for r in rows])
    return df

def apply_triple_barrier(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure sorted by symbol and bar_time
    df = df.sort_values(['symbol', 'bar_time']).reset_index(drop=True)

    # Pre-calculate barriers
    # P_entry = Close_t0
    # ATR = atr_pct_15m_t0 * P_entry
    # P_target = P_entry + (1.5 * ATR) - (0.0010 * P_entry)
    # P_stop = P_entry - (1.0 * ATR) + (0.0010 * P_entry)

    df['P_entry'] = df['close_price']
    df['ATR_abs'] = df['atr_pct_15m'] * df['P_entry']
    df['P_target'] = df['P_entry'] + (1.5 * df['ATR_abs']) - (0.0010 * df['P_entry'])
    df['P_stop'] = df['P_entry'] - (1.0 * df['ATR_abs']) + (0.0010 * df['P_entry'])

    # Group by symbol to avoid forward-looking into other symbols
    targets = np.zeros(len(df), dtype=int)

    for symbol, group in df.groupby('symbol'):
        idx = group.index.values
        highs = df['high_price'].values
        lows = df['low_price'].values
        p_targets = df['P_target'].values
        p_stops = df['P_stop'].values

        for i in range(len(idx)):
            curr_idx = idx[i]
            p_target = p_targets[curr_idx]
            p_stop = p_stops[curr_idx]

            # Look ahead up to 6 bars
            hit = 0
            for k in range(1, 7):
                if i + k >= len(idx):
                    break

                eval_idx = idx[i + k]

                # Check lower barrier first
                if lows[eval_idx] <= p_stop:
                    hit = 0
                    break

                # Check upper barrier
                if highs[eval_idx] >= p_target:
                    hit = 1
                    break

            targets[curr_idx] = hit

    df['target'] = targets

    # Clean up intermediate columns
    df = df.drop(columns=['P_entry', 'ATR_abs', 'P_target', 'P_stop', 'high_price', 'low_price', 'close_price'])
    return df

def main():
    print("Loading features matrix...")
    features_df = pd.read_parquet("data/processed/features_matrix.parquet")

    print("Fetching market bars from PostgreSQL...")
    bars_df = asyncio.run(fetch_market_bars())
    if 'bar_time' in bars_df.columns:
        bars_df['bar_time'] = pd.to_datetime(bars_df['bar_time'], utc=True)
    if 'bar_time' in features_df.columns:
        features_df['bar_time'] = pd.to_datetime(features_df['bar_time'], utc=True)

    print("Merging data...")
    df = pd.merge(features_df, bars_df, on=['symbol', 'bar_time'], how='inner')

    print(f"Applying triple-barrier labeling on {len(df)} rows...")
    labeled_df = apply_triple_barrier(df)

    out_path = "data/processed/labeled_features_matrix.parquet"
    labeled_df.to_parquet(out_path, index=False)
    print(f"Labeled data saved to {out_path}")

if __name__ == "__main__":  # pragma: no cover
    main()
