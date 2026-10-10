import lightgbm as lgb
import numpy as np

class LGBMInferenceEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LGBMInferenceEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.booster = lgb.Booster(model_file="backend/app/ml/artifacts/deepcandle_lgbm_v1.txt")
        self.feature_names = self.booster.feature_name()

    def predict_proba(self, features: dict) -> float:
        feature_values = np.array([features[name] for name in self.feature_names], dtype=np.float32).reshape(1, -1)
        prob = self.booster.predict(feature_values)[0]
        return float(prob)
