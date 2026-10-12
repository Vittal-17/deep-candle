import os
import sys
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    precision_score
)

def evaluate_split(y_true, y_prob, threshold: float = 0.50, round_trip_fee: float = 0.0010, atr_pct = None) -> dict:
    auc_val = float(roc_auc_score(y_true, y_prob))
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall_arr, precision_arr))
    brier = float(brier_score_loss(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob))

    y_pred = (y_prob >= threshold).astype(int)
    n_signals = int(y_pred.sum())
    total_bars = len(y_true)

    if n_signals > 0:
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    else:
        prec = 0.0
        tp = 0
        fp = 0
        fn = int(y_true.sum())
        tn = int((1 - y_true).sum())

    # Triple Barrier Expectancy Calculation:
    # TP1 target = +1.5x ATR, SL target = -1.0x ATR
    # Fee friction: 0.10% (0.0010) round-trip deducted from each trade.
    # Win payoff: (1.5 * ATR) - fee
    # Loss payoff: (-1.0 * ATR) - fee
    # If atr_pct is provided, use per-trade ATR; else fallback to empirical mean (0.37% = 0.0037).
    if n_signals > 0:
        sel_idx = np.where(y_pred == 1)[0]
        sel_targets = y_true[sel_idx]
        sel_atr = atr_pct[sel_idx] if atr_pct is not None else np.full(n_signals, 0.0037)
        payoffs = np.where(
            sel_targets == 1,
            (1.5 * sel_atr) - round_trip_fee,
            (-1.0 * sel_atr) - round_trip_fee
        )
        expected_payoff = float(np.mean(payoffs))
    else:
        expected_payoff = 0.0

    return {
        "total_bars": total_bars,
        "auc": auc_val,
        "pr_auc": pr_auc,
        "brier": brier,
        "log_loss": loss,
        "threshold": threshold,
        "n_signals": n_signals,
        "signal_rate": (n_signals / total_bars) * 100.0 if total_bars > 0 else 0.0,
        "precision": prec,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "expected_payoff": expected_payoff
    }

def print_metrics_table(title: str, m: dict):
    print("=" * 72)
    print(f" {title.upper()} ")
    print("=" * 72)
    print(f" Total Bars Evaluated       : {m['total_bars']:,}")
    print(f" Global ROC-AUC             : {m['auc']:.4f}")
    print(f" Global PR-AUC (Avg Prec)   : {m['pr_auc']:.4f}")
    print(f" Brier Score (Calibration)  : {m['brier']:.4f}")
    print(f" Binary Log Loss            : {m['log_loss']:.4f}")
    print("-" * 72)
    print(f" Threshold Trigger Gate     : P(Setup) >= {m['threshold']:.2f}")
    print(f" Total Signals Triggered    : {m['n_signals']:,} ({m['signal_rate']:.3f}% of bars)")
    print(f" Precision / Win Rate       : {m['precision']:.1%}")
    print(f" Confusion Matrix           : TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}")
    print(f" Expected Net Payoff/Trade  : {m['expected_payoff']:+.4%} (after 0.10% friction)")
    print("=" * 72)
    print()

def main():
    parser = argparse.ArgumentParser(description="Deep Candle LightGBM Model Statistical Evaluator")
    parser.add_argument("--data", type=str, default="data/processed/labeled_features_matrix.parquet", help="Path to labeled parquet dataset")
    parser.add_argument("--model", type=str, default="backend/app/ml/artifacts/deepcandle_lgbm_v1.txt", help="Path to LightGBM model artifact")
    parser.add_argument("--threshold", type=float, default=0.50, help="Signal gating probability threshold")
    parser.add_argument("--fee", type=float, default=0.0010, help="Round-trip fee friction (e.g. 0.0010 = 0.10%%)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model artifact not found at {args.model}")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at {args.data}")
        sys.exit(1)

    print(f"Loading LightGBM model artifact from '{args.model}'...")
    booster = lgb.Booster(model_file=args.model)
    features = booster.feature_name()
    print(f"Model loaded successfully with {booster.num_trees()} trees and {len(features)} features: {features}\n")

    print(f"Loading dataset from '{args.data}'...")
    df = pd.read_parquet(args.data)
    df = df.sort_values("bar_time").reset_index(drop=True)
    min_time = df["bar_time"].min()
    max_time = df["bar_time"].max()
    print(f"Dataset loaded: {len(df):,} bars from {min_time} to {max_time}\n")

    # Temporal holdout split definition (Chronological out-of-sample validation: days 211 to 240)
    # Matching Fold 4 in train_lgbm.py
    val_start = min_time + pd.Timedelta(days=210)
    val_end = min_time + pd.Timedelta(days=240)

    val_mask = (df["bar_time"] >= val_start) & (df["bar_time"] < val_end)
    val_df = df[val_mask].copy()

    # Predict probabilities
    val_y_true = val_df["target"].values
    val_y_prob = booster.predict(val_df[features].values)
    val_atr = val_df["atr_pct_15m"].values if "atr_pct_15m" in val_df.columns else None

    # Full Dataset evaluation for reference
    full_y_true = df["target"].values
    full_y_prob = booster.predict(df[features].values)
    full_atr = df["atr_pct_15m"].values if "atr_pct_15m" in df.columns else None

    # Evaluate Holdout
    holdout_metrics = evaluate_split(val_y_true, val_y_prob, threshold=args.threshold, round_trip_fee=args.fee, atr_pct=val_atr)
    print_metrics_table(f"Out-of-Sample Holdout (Days 211-240: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')})", holdout_metrics)

    # Multi-threshold distribution scan on Holdout
    print("HOLD-OUT PROBABILITY THRESHOLD SENSITIVITY SCAN:")
    print("-" * 72)
    print(f"{'Threshold':<12} | {'Signals':<10} | {'Rate':<8} | {'Precision':<11} | {'Net Payoff/Trade':<18}")
    print("-" * 72)
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68]:
        scan_m = evaluate_split(val_y_true, val_y_prob, threshold=t, round_trip_fee=args.fee, atr_pct=val_atr)
        print(f"P >= {t:<7.2f} | {scan_m['n_signals']:<10,d} | {scan_m['signal_rate']:<7.2f}% | {scan_m['precision']:<11.1%} | {scan_m['expected_payoff']:<+18.4%}")
    print("-" * 72)
    print()

    # Multi-threshold distribution scan on Full Dataset
    print("FULL DATASET PROBABILITY THRESHOLD SENSITIVITY SCAN:")
    print("-" * 72)
    print(f"{'Threshold':<12} | {'Signals':<10} | {'Rate':<8} | {'Precision':<11} | {'Net Payoff/Trade':<18}")
    print("-" * 72)
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68]:
        scan_m = evaluate_split(full_y_true, full_y_prob, threshold=t, round_trip_fee=args.fee, atr_pct=full_atr)
        print(f"P >= {t:<7.2f} | {scan_m['n_signals']:<10,d} | {scan_m['signal_rate']:<7.2f}% | {scan_m['precision']:<11.1%} | {scan_m['expected_payoff']:<+18.4%}")
    print("-" * 72)
    print()

    # Evaluate Global Dataset
    full_metrics = evaluate_split(full_y_true, full_y_prob, threshold=args.threshold, round_trip_fee=args.fee, atr_pct=full_atr)
    print_metrics_table(f"Full Dataset Benchmark (240 Days: {min_time.strftime('%Y-%m-%d')} to {max_time.strftime('%Y-%m-%d')})", full_metrics)

if __name__ == "__main__":
    main()
