import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from app.ml.cro_reasoner import CROReasoner, CROVerdict

@pytest.mark.asyncio
async def test_valid_setup_approval():
    cro = CROReasoner()
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"action": "EXECUTE_LONG", "rejection_reason": null, "entry_price": 50000.0, "stop_loss": 49500.0, "take_profit_1": 50750.0, "take_profit_2": 51500.0, "risk_reward_ratio": 1.5, "invalidation_thesis": "Bullish momentum supported by volume", "market_regime": "TRENDING_UP"}'
            }
        }]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        verdict = await cro.evaluate("BTCUSDT", {"atr_pct_15m": 0.01}, 50000.0)

        assert isinstance(verdict, CROVerdict)
        assert verdict.action == "EXECUTE_LONG"
        assert verdict.entry_price == 50000.0
        assert verdict.stop_loss == 49500.0
        assert verdict.take_profit_1 == 50750.0
        assert verdict.risk_reward_ratio == 1.5


@pytest.mark.asyncio
async def test_setup_rejection_rule_5():
    cro = CROReasoner()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"action": "REJECT_SETUP", "rejection_reason": "Rule 5 Violation: 15m RSI is 78.5 (> 75.0)", "entry_price": 50000.0, "stop_loss": 0.0, "take_profit_1": 0.0, "take_profit_2": 0.0, "risk_reward_ratio": 0.0, "invalidation_thesis": "Overbought conditions", "market_regime": "EXTREME_BULLISH"}'
            }
        }]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        verdict = await cro.evaluate("BTCUSDT", {"rsi_15m": 78.5}, 50000.0)

        assert isinstance(verdict, CROVerdict)
        assert verdict.action == "REJECT_SETUP"
        assert "Rule 5 Violation" in verdict.rejection_reason


@pytest.mark.asyncio
async def test_http_429_retry_and_recovery():
    cro = CROReasoner()

    resp_429 = MagicMock()
    resp_429.status_code = 429

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"action": "EXECUTE_LONG", "rejection_reason": null, "entry_price": 50000.0, "stop_loss": 49500.0, "take_profit_1": 50750.0, "take_profit_2": 51500.0, "risk_reward_ratio": 1.5, "invalidation_thesis": "Valid setup after retry", "market_regime": "TRENDING"}'
            }
        }]
    }
    resp_200.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_post.side_effect = [resp_429, resp_200]

        verdict = await cro.evaluate("BTCUSDT", {}, 50000.0)

        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(0.5)
        assert verdict.action == "EXECUTE_LONG"


@pytest.mark.asyncio
async def test_timeout_failsafe():
    cro = CROReasoner()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Connection timed out")

        verdict = await cro.evaluate("BTCUSDT", {}, 50000.0)

        assert isinstance(verdict, CROVerdict)
        assert verdict.action == "REJECT_SETUP"
        assert verdict.rejection_reason == "Failsafe: Proxy timeout or unparseable response"
        assert verdict.entry_price == 50000.0
        assert verdict.stop_loss == 0.0
        assert verdict.take_profit_1 == 0.0
        assert verdict.take_profit_2 == 0.0
        assert verdict.risk_reward_ratio == 0.0
        assert verdict.invalidation_thesis == "CRO service unavailable"
        assert verdict.market_regime == "UNKNOWN"


@pytest.mark.asyncio
async def test_json_malformation():
    cro = CROReasoner()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": "Not valid JSON output from model! HTML 500 Error!"
            }
        }]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        verdict = await cro.evaluate("BTCUSDT", {}, 50000.0)

        assert isinstance(verdict, CROVerdict)
        assert verdict.action == "REJECT_SETUP"
        assert verdict.rejection_reason == "Failsafe: Proxy timeout or unparseable response"
