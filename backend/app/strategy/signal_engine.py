from app.ml.inference import LGBMInferenceEngine

class SignalEngine:
    def __init__(self):
        self.inference_engine = LGBMInferenceEngine()

    def process_features(self, symbol: str, features: dict, current_close: float, current_time) -> dict | None:
        prob = self.inference_engine.predict_proba(features)

        if prob >= 0.68:
            p_entry = current_close
            atr_abs = features['atr_pct_15m'] * p_entry

            p_target = p_entry + (1.5 * atr_abs) - (0.0010 * p_entry)
            p_stop = p_entry - (1.0 * atr_abs) + (0.0010 * p_entry)

            return {
                'symbol': symbol,
                'entry_time': current_time,
                'entry_price': p_entry,
                'target_price': p_target,
                'stop_price': p_stop,
                'probability': prob,
                'horizon_bars': 6
            }
        return None
