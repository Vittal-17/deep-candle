# PROJECT: DEEP CANDLE

**TYPE:** Autonomous Zero-Budget Quantitative Crypto Intelligence Platform
**BUDGET:** $0.00 Total Operating Expenditure
**INFRASTRUCTURE:** Oracle Cloud Always Free (2 OCPUs, 12GB RAM) + Local Workstation

---

## 1. PROJECT SUMMARY AND CORE OBJECTIVE

**Deep Candle** is an autonomous quantitative financial intelligence platform and signal engine for high-liquidity cryptocurrency assets (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `DOGEUSDT`). The system pairs high-speed statistical machine learning (LightGBM) with deep qualitative LLM reasoning (Gemini 3.1 Pro) at zero marginal cost.

The platform continuously processes real-time multi-timeframe market data, transforms prices into stationary mathematical features, runs sub-2ms directional inference on candle closes, validates approved setups through an AI Chief Risk Officer proxy cluster, broadcasts trade theses to Telegram, and autonomously audits all open positions inside an immutable PostgreSQL ledger.

## 2. ASSETS AND TIMEFRAME ARCHITECTURE

- **Tracked Assets:** `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `DOGEUSDT`.
- **Primary Execution Clock:** 15-Minute (`15m`) closed bars. All prediction and trade execution decisions anchor strictly to the 15m close to eliminate high-frequency fee drag and execution latency penalties.
- **Macro Trend Filter:** 1-Hour (`1h`) closed bars. Strictly lagged to prevent lookahead bias.
- **Micro Momentum Layer:** 5-Minute (`5m`) closed bars for measuring candle close velocity and volume acceleration.
- **Market Anchor:** Bitcoin (`BTCUSDT`) concurrent returns and 1h EMA distance injected into all altcoin vectors.

## 3. ZERO-COST INFRASTRUCTURE STACK

- **Cloud Host:** Oracle Cloud Infrastructure Always Free Tier (Ampere A1, 2 OCPUs, 12GB RAM, Ubuntu 24.04).
- Database: PostgreSQL 16 running on isolated port 5440 with Timescale-lite indexing.
- **Machine Learning Engine:** LightGBM binary classifier running local C++ inference in memory (< 2ms per coin, < 30MB RAM footprint). Offline training executed locally on development desktop.
- **Qualitative AI Reasoning:** `antigravity-proxy` running on port 3009 routing upstream to `gemini-3.1-pro-low` across 5 pooled Google AI Pro accounts via round-robin. Zero API token expenditure.
- **Backend API and Ingestion:** FastAPI, Asyncpg, WebSockets, Uvicorn, and Caddy reverse proxy.
- **Alert and Notification:** Telegram Bot API pushing directly to VIP and public channels.
- **Mesh Networking:** Tailscale WireGuard connecting development desktop and cloud VPS.

## 4. SYSTEM MODULES AND EXECUTION WORKFLOW

### MODULE 1: DATA INGESTION AND STORAGE ENGINE
- **Upstream:** Binance Public Combined WebSocket Streams (`wss://stream.binance.com:9443/ws`).
- **Stream Multiplexing:** Single socket connection handling all 5 coins across 15m, 1h, and 5m intervals.
- **Invariant:** Closed bars only. Inspect candle close parameter `k.x`. If false, discard immediately. If true, cast strings to float64 and upsert to database.
- Database Table: market_bars on port 5440. Composite primary key on (symbol, timeframe, bar_time).
- **Self-Healing Gap Engine:** Upon network disconnect, use exponential backoff (1s to 60s). On reconnect, compare last stored timestamp against incoming stream. Pull missing candles via Binance REST API `/v3/klines` in 1000-bar chunks and commit before resuming live socket stream.
- **Data Retention:** Retain 8 months of 15m and 1h bars; automatically prune 5m bars older than 14 days.

### MODULE 2: MULTI-TIMEFRAME FEATURE ENGINE
- **Trigger:** Fired immediately on confirmed 15m candle close.
- **Anti-Leakage Temporal Barrier:** Indicators for 1h candles must strictly use the last fully completed 1h bar (lagged by 1 bar). The ongoing 1h bar is strictly hidden.
- **Vector Dimensions (11 Scale-Invariant Features):**
  1. `atr_pct_15m`: 14-period Wilder smoothed ATR divided by close price.
  2. `bb_squeeze_15m`: Bollinger Band width (20 SMA, 2.0 std) normalized by 20 SMA.
  3. `vol_zscore_15m`: 20-period volume Gaussian Z-Score anomaly metric.
  4. `vwap_dist_atr_15m`: Distance from intraday session VWAP (resetting daily at 00:00 UTC) normalized by ATR.
  5. `rsi_15m`: 14-period Wilder Relative Strength Index.
  6. `rsi_1h`: Lagged 1h 14-period RSI.
  7. `ema_200_dist_1h`: Distance from lagged 1h 200 EMA normalized by 1h ATR.
  8. `micro_momentum_ratio`: Return of latest 5m bar divided by full 15m bar return.
  9. `micro_volume_acceleration`: Volume of latest 5m bar divided by average of prior two 5m bars.
  10. `btc_relative_delta`: Asset 15m return minus Bitcoin 15m return.
  11. `return_15m`: Percentage return of current 15m candle.
- **Gating Rule:** Any vector containing `NaN` or `Inf` is rejected and logged.

### MODULE 3: LIGHTGBM PROBABILISTIC INFERENCE ENGINE
- **Target Labeling:** Triple-Barrier Method across 8 months of historical data (~140,000 rows).
  * **Upper Barrier:** Close + (1.5 * ATR_14).
  * **Lower Barrier:** Close - (1.0 * ATR_14).
  * **Vertical Barrier:** 6 candles (90 minutes max).
  * Target $y = 1$ if upper barrier hit first; $y = 0$ if lower barrier hit first or 90m expires. Includes 0.10% round-trip exchange fee deduction.
- **Model Training:** Purged Walk-Forward Time-Series Split (4 rolling folds, 6-bar purge buffer). Exported as plain-text model `deepcandle_lgbm_v1.txt`.
- **Inference Protocol:** Model loaded into memory on FastAPI boot. On 15m close, calculates 11 features and runs `predict()`.
  * If probability < 0.68: Log telemetry and terminate.
  * If probability >= 0.68: Construct context payload and dispatch to Proxy Reasoning Engine.

### MODULE 4: PROXY REASONING AND RISK OFFICER
- **Gateway:** `antigravity-proxy` at `http://127.0.0.1:3009/v1/chat/completions`.
- **Model:** `gemini-3.1-pro-low` with low thinking tokens for fast response (< 4.5s timeout).
- **Role:** Chief Quantitative Risk Officer tasked with actively rejecting flawed breakout candidates.
- **Evaluation Directives:**
  1. **Structural Invalidation:** Stop loss must sit below recent 15m swing low or intraday VWAP. If stop distance > 2.0x ATR, emit `REJECT_SETUP`.
  2. **Reward-to-Risk:** Take Profit 1 must offer >= 1.5x risk. Combined setup must offer >= 1.8x R:R.
  3. **Volume Confirmation:** Breakout candle volume Z-score must be >= 1.5.
  4. **Macro Alignment:** Reject altcoin longs if BTC is trading below its 1h 200 EMA with negative 1h return.
  5. **Overbought Filter:** Reject if 15m RSI > 75.0.
- **Contract:** Emits strict JSON with action (`EXECUTE_LONG` or `REJECT_SETUP`), `entry_price`, `stop_loss`, `take_profit_1`, `take_profit_2`, `risk_reward_ratio`, `invalidation_thesis`, and `market_regime`.

### MODULE 5: TELEGRAM DISPATCHER AND REAL-TIME AUDIT LEDGER
- **Immutability Rule:** Once a setup is assigned a UUID and written to PostgreSQL table `signal_ledger`, entry, stop, and targets are permanently read-only.
- **Lifecycle State Machine:**
  * **PENDING:** Awaiting price fill within entry bounds. Transitions to `ACTIVE` on fill, or `EXPIRED` if target 1 hits before entry.
  * **ACTIVE:** Tracks position on every 15m close. Transitions to `HIT_SL` if low <= stop loss, or `HIT_TP1` if high >= target 1.
  * **HIT_TP1:** Locks 50% position profit. Trailing stop automatically moves to Entry Price (Breakeven). Transitions to `STOPPED_BE` if low <= entry, or `HIT_TP2` if high >= target 2.
  * **HIT_TP2:** Closes remaining 50% position. Records full win.
- **Dispatch Format:** Structured Telegram markdown messages for new setups, breakeven moves, and trade closures.
- **Public Performance Tracking:** Continuous rolling 30-day calculations for Win Rate, Profit Factor, Maximum Drawdown, and Sharpe Ratio.

## 5. IMPLEMENTATION ROADMAP
- Phase 1: Database initialization and PostgreSQL schema migrations on port 5440.
- **Phase 2:** Binance multiplex WebSocket ingestion worker and gap-healing REST client.
- **Phase 3:** Historical data fetcher (8 months) and multi-timeframe feature calculation pipeline.
- **Phase 4:** Local offline LightGBM model training, purged walk-forward cross-validation, and model text export.
- **Phase 5:** FastAPI inference router and `antigravity-proxy` integration with Pydantic validation.
- **Phase 6:** Telegram bot alerts and real-time trade position tracking engine.
- **Phase 7:** 7-day live paper test against WebSocket feeds before deploying to production VPS.
