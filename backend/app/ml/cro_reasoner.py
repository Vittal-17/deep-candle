import json
import logging
import asyncio
import httpx
from pydantic import BaseModel, Field
from typing import Optional, Literal
from app.config import settings

logger = logging.getLogger(__name__)

class CROVerdict(BaseModel):
    action: Literal["EXECUTE_LONG", "REJECT_SETUP"]
    rejection_reason: Optional[str] = None
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    invalidation_thesis: str
    market_regime: str

class CROReasoner:
    def __init__(self, proxy_url: Optional[str] = None):
        self.proxy_url = proxy_url or settings.PROXY_URL
        self.model = "gemini-3.1-pro-low"

    def _build_failsafe_verdict(self, current_close: float, reason: str = "Failsafe: Proxy timeout or unparseable response") -> CROVerdict:
        return CROVerdict(
            action="REJECT_SETUP",
            rejection_reason=reason,
            entry_price=current_close,
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward_ratio=0.0,
            invalidation_thesis="CRO service unavailable",
            market_regime="UNKNOWN"
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are the Chief Risk Officer for a quantitative cryptocurrency fund. Evaluate the incoming trade setup against your strict risk rules. Reject any setup that violates the rules. Return ONLY valid JSON matching the exact schema.\n\n"
            "Mandatory Risk Evaluation Rules:\n"
            "Rule 1: Structural SL distance <= 2.0x ATR. Structural stop-loss must sit below the 15m swing low or intraday VWAP.\n"
            "Rule 2: Minimum R:R to TP1 >= 1.5, combined setup average R:R >= 1.8. TP1 must be at 1.5x ATR, TP2 must be at 3.0x ATR.\n"
            "Rule 3: Breakout candle volume Z-score >= 1.5.\n"
            "Rule 4: Reject altcoin longs if BTC is below 1h 200 EMA with negative 1h return (< -0.5%).\n"
            "Rule 5: Reject if 15m RSI > 75.0."
        )

    async def evaluate(self, symbol: str, features: dict, current_close: float, p_setup: float = 0.68, market_context: Optional[dict] = None) -> CROVerdict:
        if market_context is None:
            market_context = {}

        user_content = {
            "symbol": symbol,
            "price": current_close,
            "features": features,
            "ml_probability": p_setup,
            "market_context": market_context
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": json.dumps(user_content)}
            ],
            "temperature": 0.1
        }

        backoff_delays = [0.5, 1.0]
        max_attempts = 3  # Initial try + up to 2 retries

        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=4.5) as client:
                    response = await client.post(self.proxy_url, json=payload)

                if response.status_code in (429, 502, 503):
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(backoff_delays[attempt])
                        continue
                    else:
                        logger.warning(f"CRO Reasoner exhausted retries on status {response.status_code}")
                        return self._build_failsafe_verdict(current_close)

                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Strip potential markdown code fence blocks if returned by proxy/LLM
                content_str = content.strip()
                if content_str.startswith("```"):
                    lines = content_str.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    content_str = "\n".join(lines).strip()

                verdict = CROVerdict.model_validate_json(content_str)
                return verdict

            except (httpx.TimeoutException, httpx.HTTPError, KeyError, IndexError, Exception) as e:
                logger.warning(f"CRO evaluation attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1 and isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (429, 502, 503):
                    await asyncio.sleep(backoff_delays[attempt])
                    continue
                else:
                    return self._build_failsafe_verdict(current_close)

        return self._build_failsafe_verdict(current_close)
