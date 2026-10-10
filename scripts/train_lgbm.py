import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score
import os

def main():
    print("Loading labeled features matrix...")
    df = pd.read_parquet("data/processed/labeled_features_matrix.parquet")

    # Sort strictly by bar_time
    df = df.sort_values('bar_time').reset_index(drop=True)

    features = [
        'atr_pct_15m', 'bb_squeeze_15m', 'vol_zscore_15m', 'vwap_dist_atr_15m',
        'rsi_15m', 'rsi_1h', 'ema_200_dist_1h', 'micro_momentum_ratio',
        'micro_volume_acceleration', 'btc_relative_delta', 'return_15m'
    ]
    target = 'target'

    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 1.5,
        "reg_lambda": 3.0,
        "verbose": -1,
        "n_jobs": -1
    }

    min_time = df['bar_time'].min()

    folds = [
        {"train": (1, 120), "val": (121, 150)},
        {"train": (30, 150), "val": (151, 180)},
        {"train": (60, 180), "val": (181, 210)},
        {"train": (90, 210), "val": (211, 240)},
    ]

    # 6-bar purge window = 6 * 15 mins = 90 mins
    purge_td = pd.Timedelta(minutes=90)

    last_booster = None

    for i, fold in enumerate(folds):
        train_start = min_time + pd.Timedelta(days=fold["train"][0] - 1)
        train_end = min_time + pd.Timedelta(days=fold["train"][1])

        val_start = min_time + pd.Timedelta(days=fold["val"][0] - 1)
        val_end = min_time + pd.Timedelta(days=fold["val"][1])

        # Apply purge window
        train_end_purged = train_end - purge_td

        train_mask = (df['bar_time'] >= train_start) & (df['bar_time'] < train_end_purged)
        val_mask = (df['bar_time'] >= val_start) & (df['bar_time'] < val_end)

        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, target]

        X_val = df.loc[val_mask, features]
        y_val = df.loc[val_mask, target]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        y_pred_prob = booster.predict(X_val, num_iteration=booster.best_iteration)
        auc = roc_auc_score(y_val, y_pred_prob)

        gate_threshold = 0.68
        y_pred_bin = (y_pred_prob >= gate_threshold).astype(int)

        if sum(y_pred_bin) > 0:
            precision = precision_score(y_val, y_pred_bin)
        else:
            precision = 0.0

        trade_count = sum(y_pred_bin)

        print(f"Fold {i+1}: Train {fold['train']} days, Val {fold['val']} days")
        print(f"  OOS AUC: {auc:.4f}")
        print(f"  OOS Precision (@ > 0.68): {precision:.4f}")
        print(f"  Sample Trade Count (@ > 0.68): {trade_count}")

        last_booster = booster

    print("Exporting model artifact...")
    os.makedirs("backend/app/ml/artifacts", exist_ok=True)
    last_booster.save_model("backend/app/ml/artifacts/deepcandle_lgbm_v1.txt")
    print("Model saved to backend/app/ml/artifacts/deepcandle_lgbm_v1.txt")

if __name__ == "__main__":
    main()
