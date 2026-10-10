# Crypto LLM Reasoning Skill

## 1. Infrastructure & Routing
- **Endpoint:** `http://127.0.0.1:3009/v1/chat/completions` (routed via `antigravity-proxy`).
- **Upstream Model:** `gemini-3.1-pro-low`.
- **Account Pooling:** 5 Google AI Pro accounts cycling via round-robin to maintain rate limits.
- **Cost:** $0.

## 2. Chief Risk Officer (CRO) Evaluation Rules
The LLM evaluates the setup based on the following strict mandates:
- **Rule 1:** Structural stop-loss must sit below the 15m swing low or intraday VWAP. If the distance from entry to stop-loss is $> 2.0 \times ATR$, emit `REJECT_SETUP`.
- **Rule 2:** Take Profit 1 (TP1) must be at $1.5 \times ATR$. Take Profit 2 (TP2) must be at $3.0 \times ATR$. The minimum Risk:Reward ratio to TP1 must be $\ge 1.5$, and the combined average $R:R$ must be $\ge 1.8$.
- **Rule 3:** The breakout candle's volume Z-score must be $\ge 1.5$.
- **Rule 4:** If an altcoin long is flagged while BTC is below its 1h 200 EMA with a negative 1h return ($< -0.5\%$), emit `REJECT_SETUP`.
- **Rule 5:** If the 15m RSI is $> 75.0$, emit `REJECT_SETUP`.

## 3. Wire Protocol & Schemas
- **System Prompt:**
  ```text
  You are the Chief Risk Officer for a quantitative cryptocurrency fund. Evaluate the incoming trade setup against your strict risk rules. Reject any setup that violates the rules. Return ONLY valid JSON matching the exact schema.
  ```
- **User Prompt Template:**
  ```json
  {
    "symbol": "{symbol}",
    "price": {close_price},
    "features": {feature_dict},
    "ml_probability": {p_setup},
    "market_context": {btc_context}
  }
  ```
- **Strict JSON Response Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "action": { "type": "string", "enum": ["EXECUTE_LONG", "REJECT_SETUP"] },
      "rejection_reason": { "type": "string" },
      "entry_price": { "type": "number" },
      "stop_loss": { "type": "number" },
      "take_profit_1": { "type": "number" },
      "take_profit_2": { "type": "number" },
      "risk_reward_ratio": { "type": "number" },
      "invalidation_thesis": { "type": "string" },
      "market_regime": { "type": "string" }
    },
    "required": ["action", "entry_price", "stop_loss", "take_profit_1", "take_profit_2", "risk_reward_ratio", "invalidation_thesis", "market_regime"]
  }
  ```

## 4. Resilience & Timeouts
- **Timeout:** 4.5-second hard timeout on the HTTP request.
- **Retries:** Maximum of 2 retries across different account slots on HTTP `429`, `502`, or `503` errors.
- **Failsafe:** If all retries fail or a parsing error occurs, default to `REJECT_SETUP` (failsafe rejection).
