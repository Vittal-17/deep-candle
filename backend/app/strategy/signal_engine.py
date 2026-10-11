import logging
from typing import Optional
from app.ml.inference import LGBMInferenceEngine
from app.ml.cro_reasoner import CROReasoner, CROVerdict

logger = logging.getLogger(__name__)

class SignalEngine:
    def __init__(self, cro_reasoner: Optional[CROReasoner] = None):
        self.inference_engine = LGBMInferenceEngine()
        self.cro_reasoner = cro_reasoner or CROReasoner()

    async def process_features(self, symbol: str, features: dict, current_close: float, current_time) -> dict | None:
        prob = self.inference_engine.predict_proba(features)

        if prob >= 0.68:
            verdict: CROVerdict = await self.cro_reasoner.evaluate(
                symbol=symbol,
                features=features,
                current_close=current_close,
                p_setup=prob
            )

            if verdict.action == "EXECUTE_LONG":
                return {
                    'symbol': symbol,
                    'entry_time': current_time,
                    'entry_price': verdict.entry_price,
                    'target_price': verdict.take_profit_1,
                    'stop_price': verdict.stop_loss,
                    'probability': prob,
                    'horizon_bars': 6,
                    'invalidation_thesis': verdict.invalidation_thesis
                }
            elif verdict.action == "REJECT_SETUP":
                logger.info(f"CRO Rejected Setup for {symbol}: {verdict.rejection_reason}")
                return None

        return None
