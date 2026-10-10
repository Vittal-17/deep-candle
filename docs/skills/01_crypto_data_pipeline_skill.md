# Crypto Data Pipeline Skill

## 1. Operational Boundaries
- **Target Assets:** `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `DOGEUSDT`
- **Timeframes:**
  - Primary execution clock: `15m`
  - Macro trend filter: `1h`
  - Micro momentum: `5m`
- **Host Environment:** Oracle Cloud Always Free (2 OCPUs, 12GB RAM, Ubuntu 24.04)
- **Resource Ceilings:**
  - Resident Memory (RSS): $\le$ 120MB
  - CPU Core Load: $\le$ 2.5%
  - Storage Growth: $\le$ 250MB/month

## 2. Upstream Network Interfaces & Gating
- **Real-Time WebSocket Multiplex Endpoint:**
  `wss://stream.binance.com:9443/stream?streams=btcusdt@kline_15m/btcusdt@kline_1h/btcusdt@kline_5m/ethusdt@kline_15m/...` (combining all 5 coins across 15m, 1h, and 5m timeframes).
- **Frame Gating Invariant:**
  Extract the `k.x` boolean flag from the incoming JSON payload.
  - If `k.x == false` (candle is open): Discard immediately.
  - If `k.x == true` (candle is closed): Cast string numbers to IEEE 754 float64 and persist to the database.
- **REST Fallback & Backfill Endpoint:**
  `https://api.binance.com/api/v3/klines`
  - Rate limit constraints: 1,200 weight/min, max 1 request per 500ms.

## 3. Database Architecture (PostgreSQL on Port 5440)
- **Table Name:** `market_bars`
- **Exact DDL:**
```sql
CREATE TABLE IF NOT EXISTS market_bars (
    symbol VARCHAR(16) NOT NULL,
    timeframe VARCHAR(8) NOT NULL,
    bar_time TIMESTAMPTZ NOT NULL,
    open_price DOUBLE PRECISION NOT NULL,
    high_price DOUBLE PRECISION NOT NULL,
    low_price DOUBLE PRECISION NOT NULL,
    close_price DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    quote_volume DOUBLE PRECISION NOT NULL,
    trades_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, bar_time)
);

CREATE INDEX IF NOT EXISTS idx_market_bars_symbol_tf_time
ON market_bars USING btree (symbol, timeframe, bar_time DESC);
```
- **Idempotent Upsert SQL:**
```sql
INSERT INTO market_bars (
    symbol, timeframe, bar_time, open_price, high_price, low_price, close_price, volume, quote_volume, trades_count
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
) ON CONFLICT (symbol, timeframe, bar_time) DO UPDATE SET
    open_price = EXCLUDED.open_price,
    high_price = EXCLUDED.high_price,
    low_price = EXCLUDED.low_price,
    close_price = EXCLUDED.close_price,
    volume = EXCLUDED.volume,
    quote_volume = EXCLUDED.quote_volume,
    trades_count = EXCLUDED.trades_count;
```

## 4. Gap-Healing & Resilience State Machine
On WebSocket disconnect or missed heartbeat:
1. **Backoff Mechanism:** Initiate exponential backoff: $1s, 2s, 4s, 8s, 16s, 32s, 60s$.
2. **Buffer:** Queue any successfully received incoming frames in-memory during reconnection.
3. **Gap Detection:** Upon successful reconnection, query PostgreSQL for $T_{\text{last\_persisted}} = \max(\text{bar\_time})$.
4. **Comparison:** Compare $T_{\text{last\_persisted}}$ with the timestamp of the first new incoming frame $T_{\text{incoming}}$.
5. **Backfill:** If $T_{\text{incoming}} - T_{\text{last\_persisted}} > \text{timeframe\_interval}$:
   - Trigger automated REST chunked pagination using `startTime` and `endTime`.
   - Process the fetched data, applying the `k.x == true` invariant logic implicitly (REST returns closed bars except the last).
6. **Commit:** Execute atomic transaction commit for the backfilled data to prevent partial states.

## 5. Storage Retention & Cron Pruning
- **Retention Policy:**
  - `15m` and `1h`: 8 months.
  - `5m`: 14 days.
- **Exact SQL Pruning Command:**
```sql
DELETE FROM market_bars
WHERE (timeframe IN ('15m', '1h') AND bar_time < NOW() - INTERVAL '8 months')
   OR (timeframe = '5m' AND bar_time < NOW() - INTERVAL '14 days');
```
- **Vacuum Scheduling:**
  Run `VACUUM ANALYZE market_bars;` weekly via cron (e.g., Sunday at 02:00 UTC) to reclaim storage and update query planner statistics.
