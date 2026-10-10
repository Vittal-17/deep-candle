import pandas as pd

def align_lagged_1h_features(df_15m: pd.DataFrame, df_1h_features: pd.DataFrame) -> pd.DataFrame:
    """
    Align 15m base bars with strictly lagged 1h macro indicators.
    For a 15m candle ending at 14:15 UTC, the closed 1h bar ending at 14:00 UTC (lagged by 1 bar) must be used.
    """
    df_15m = df_15m.sort_values('bar_time').copy()
    df_1h_features = df_1h_features.sort_values('bar_time').copy()

    # Shift 1h bar times by 1 bar to strictly enforce lagging
    # A 1h bar timestamped at 14:00 ends at 14:00. Its features become visible ONLY after 14:00.
    merged = pd.merge_asof(
        df_15m,
        df_1h_features[['bar_time', 'rsi_1h', 'ema_200_dist_1h']],
        on='bar_time',
        direction='backward',
        allow_exact_matches=False  # strictly lagged, exact match at boundary gets previous closed 1h bar
    )
    return merged

def compute_5m_micro_features(df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute micro momentum and micro volume acceleration from 5m sub-bars aligned to 15m bar close.
    For historical periods where 5m data is unavailable, fill with baseline neutral values.
    """
    df_15m = df_15m.sort_values('bar_time').copy()
    df_15m['ret_15m'] = (df_15m['close_price'] - df_15m['open_price']) / (df_15m['open_price'] + 1e-7)

    if df_5m.empty:
        df_15m['micro_momentum_ratio'] = 0.333
        df_15m['micro_volume_acceleration'] = 1.0
        df_15m.drop(columns=['ret_15m'], inplace=True, errors='ignore')
        return df_15m

    df_5m = df_5m.sort_values('bar_time').copy()
    df_5m['ret_5m'] = (df_5m['close_price'] - df_5m['open_price']) / (df_5m['open_price'] + 1e-7)
    df_5m['prev_2_vol_avg'] = df_5m['volume'].shift(1).rolling(2).mean()
    df_5m['micro_volume_acceleration'] = df_5m['volume'] / (df_5m['prev_2_vol_avg'] + 1e-7)

    # Filter 5m bars that match 15m bar close timestamps
    df_5m_aligned = df_5m[df_5m['bar_time'].isin(df_15m['bar_time'])].copy()

    merged = pd.merge(
        df_15m,
        df_5m_aligned[['bar_time', 'ret_5m', 'micro_volume_acceleration']],
        on='bar_time',
        how='left'
    )

    merged['micro_momentum_ratio'] = merged['ret_5m'] / (merged['ret_15m'] + 1e-7)
    merged.drop(columns=['ret_5m', 'ret_15m'], inplace=True, errors='ignore')

    # Fill neutral baseline values for historical bars older than 5m data availability
    merged['micro_momentum_ratio'] = merged['micro_momentum_ratio'].fillna(0.333)
    merged['micro_volume_acceleration'] = merged['micro_volume_acceleration'].fillna(1.0)

    return merged

def align_btc_anchor(df_asset_15m: pd.DataFrame, df_btc_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute btc_relative_delta = R_asset_15m - R_btc_15m.
    """
    df_btc = df_btc_15m.sort_values('bar_time').copy()
    df_btc['btc_ret_15m'] = (df_btc['close_price'] - df_btc['open_price']) / (df_btc['open_price'] + 1e-7)

    merged = pd.merge(
        df_asset_15m,
        df_btc[['bar_time', 'btc_ret_15m']],
        on='bar_time',
        how='left'
    )

    merged['return_15m'] = (merged['close_price'] - merged['open_price']) / (merged['open_price'] + 1e-7)
    merged['btc_relative_delta'] = merged['return_15m'] - merged['btc_ret_15m']
    merged.drop(columns=['btc_ret_15m'], inplace=True)
    return merged
