# Crypto ML Inference Skill

## 1. Workload Isolation Architecture
- **Training Environment:** Strictly executed on the local desktop workstation to leverage high-performance compute.
- **Inference Environment:** Strictly in-process inside a FastAPI application running on the Oracle Cloud VPS.
- **Resource Constraints:**
  - Memory allocation limit: $\le 30$MB in the C++ booster heap.
  - Latency limit: $\le 2.0$ms per coin per inference cycle.

## 2. Triple-Barrier Labeling Protocol
For a given 15m entry candle closing at time $t$:
- **Upper Barrier:** $P_{upper} = C_t + (1.5 \times ATR_{14,t})$
- **Lower Barrier:** $P_{lower} = C_t - (1.0 \times ATR_{14,t})$
- **Time Barrier:** $t_{max} = t + 6$ periods (90 minutes max).
- **Evaluation Precedence:**
  Iterate through subsequent 15m candles $i \in [1, 6]$:
  1. If $L_{t+i} \le P_{lower}$, the lower barrier is hit first $\rightarrow y = 0$ (Failure).
  2. Else if $H_{t+i} \ge P_{upper}$, the upper barrier is hit first $\rightarrow y = 1$ (Success).
  3. If neither barrier is hit by $i = 6$, the time barrier expires $\rightarrow y = 0$ (Failure).
- **Exchange Fee:** Include a 0.10% round-trip deduction (0.05% entry, 0.05% exit) when computing actual profitability.

## 3. Walk-Forward Cross-Validation & Hyperparameters
- **Validation Scheme:** Purged Walk-Forward Time-Series Split.
  - Dataset: 8 months of historical data ($\sim 140,000$ rows).
  - Purge Window: 6-bar purge window between train and test folds to prevent data leakage from overlapping barriers.
- **Exact LightGBM Hyperparameters:**
  ```json
  {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 50,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 1.5,
    "reg_lambda": 3.0
  }
  ```

## 4. Artifact Delivery & Runtime Execution
- **Artifact Export Format:** Standard LightGBM text export format (`deepcandle_lgbm_v1.txt`).
- **Loading:** The model is loaded into the in-memory C++ booster on FastAPI startup, requiring no disk I/O during execution.
- **Gating Condition:**
  - Let $P(\text{Setup})$ be the model's output probability.
  - If $P(\text{Setup}) \ge 0.68$: Forward the feature vector and probability to the LLM proxy for risk evaluation.
  - If $P(\text{Setup}) < 0.68$: Drop the setup and log the event implicitly.
