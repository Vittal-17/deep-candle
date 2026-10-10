# Crypto Alert Ledger Skill

## 1. Immutability Invariant
- Signals committed to the PostgreSQL ledger are permanently read-only regarding `entry_price`, `stop_loss`, `target_1`, and `target_2`. No retroactive modifications are permitted.

## 2. PostgreSQL Schema (Port 5440)
- **Table Name:** `signal_ledger`
- **Exact DDL:**
```sql
CREATE TYPE signal_status AS ENUM (
    'PENDING', 'ACTIVE', 'HIT_TP1', 'HIT_TP2', 'HIT_SL', 'STOPPED_BE', 'EXPIRED'
);

CREATE TABLE IF NOT EXISTS signal_ledger (
    signal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(16) NOT NULL,
    direction VARCHAR(4) NOT NULL,
    ml_confidence DOUBLE PRECISION NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    stop_loss DOUBLE PRECISION NOT NULL,
    target_1 DOUBLE PRECISION NOT NULL,
    target_2 DOUBLE PRECISION NOT NULL,
    status signal_status NOT NULL DEFAULT 'PENDING',
    realized_pnl_pct DOUBLE PRECISION DEFAULT 0.0,
    thesis TEXT NOT NULL,
    market_regime VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);
```

## 3. Real-Time Position Tracking Rules (Every 15m Bar)
Evaluations occur sequentially on every newly closed 15m bar:
1. **`PENDING` $\rightarrow$ `ACTIVE`**: If $L_t \le \text{entry\_price} \le H_t$.
   - **`PENDING` $\rightarrow$ `EXPIRED`**: If $H_t \ge \text{target\_1}$ before being filled, or if 6 bars pass without a fill.
2. **`ACTIVE` $\rightarrow$ `HIT_SL`**: If $L_t \le \text{stop\_loss}$.
3. **`ACTIVE` $\rightarrow$ `HIT_TP1`**: If $H_t \ge \text{target\_1}$. (Automatically trails stop loss to `entry_price`).
4. **`HIT_TP1` $\rightarrow$ `STOPPED_BE`**: If $L_t \le \text{entry\_price}$ (locks in 50% gain at TP1).
5. **`HIT_TP1` $\rightarrow$ `HIT_TP2`**: If $H_t \ge \text{target\_2}$ (full position closed).
- **Mathematical PnL Formulas (deducting 0.10% round-trip fee):**
  - `HIT_SL`: $\text{PnL} = \frac{\text{stop\_loss} - \text{entry\_price}}{\text{entry\_price}} - 0.0010$
  - `STOPPED_BE`: $\text{PnL} = 0.5 \times \frac{\text{target\_1} - \text{entry\_price}}{\text{entry\_price}} - 0.0010$
  - `HIT_TP2`: $\text{PnL} = 0.5 \times \frac{\text{target\_1} - \text{entry\_price}}{\text{entry\_price}} + 0.5 \times \frac{\text{target\_2} - \text{entry\_price}}{\text{entry\_price}} - 0.0010$

## 4. Telegram Broadcast Templates
- **New Setup:**
  ```markdown
  🚨 **NEW SIGNAL: {symbol} {direction}**
  **Entry:** {entry_price}
  **Stop Loss:** {stop_loss}
  **TP1 (1.5R):** {target_1}
  **TP2 (3.0R):** {target_2}
  **ML Confidence:** {ml_confidence}
  **Thesis:** {thesis}
  ```
- **Take Profit 1 Hit:**
  ```markdown
  ✅ **TP1 HIT: {symbol}**
  Target 1 reached at {target_1}.
  Stop loss trailed to breakeven ({entry_price}). 50% profits secured.
  ```
- **Stop Loss Hit:**
  ```markdown
  ❌ **STOP LOSS: {symbol}**
  Position closed at {stop_loss}. Realized PnL: {realized_pnl_pct}%.
  ```
- **Trade Closed:**
  ```markdown
  🏁 **TRADE CLOSED: {symbol}**
  Final Status: {status}. Total PnL: {realized_pnl_pct}%.
  ```

## 5. Rolling Public Ledger Metrics
Calculated on a rolling 30-day window:
- **Win Rate:** $\frac{\text{Count}(\text{HIT\_TP1} \cup \text{HIT\_TP2} \cup \text{STOPPED\_BE})}{\text{Total Closed Trades}}$
- **Profit Factor:** $\frac{\sum \text{Gross Profits}}{\sum |\text{Gross Losses}|}$
- **Max Drawdown:** $\frac{\text{Peak Equity} - \text{Trough Equity}}{\text{Peak Equity}}$
- **Sharpe Ratio:** $\frac{\text{Mean}(\text{Daily Returns})}{\text{StdDev}(\text{Daily Returns})} \times \sqrt{365}$
