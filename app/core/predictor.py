from __future__ import annotations

from pathlib import Path
from threading import RLock

import joblib
import numpy as np

from app.core.features import FEATURE_NAMES
from app.core.scoring import TARGET_NAMES


class ModelNotReadyError(RuntimeError):
    pass


class Predictor:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._bundle: dict | None = None
        self._lock = RLock()

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    @property
    def version(self) -> str | None:
        return None if self._bundle is None else str(self._bundle.get("version", "unknown"))

    def load(self) -> None:
        if not self.model_path.exists():
            self._bundle = None
            return
        bundle = joblib.load(self.model_path)
        required = {"model", "feature_names", "target_names", "thresholds"}
        if not isinstance(bundle, dict) or not required.issubset(bundle):
            raise ValueError("Model artifact has an unsupported format")
        if list(bundle["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Model feature order does not match application feature order")
        if list(bundle["target_names"]) != TARGET_NAMES:
            raise ValueError("Model target order does not match application target order")
        self._bundle = bundle

    @staticmethod
    def _positive_probability(probability_matrix: np.ndarray, classes: np.ndarray) -> float:
        positive = np.flatnonzero(np.asarray(classes) == 1)
        return 0.0 if positive.size == 0 else float(probability_matrix[0, positive[0]])

    def predict(self, features: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        if self._bundle is None:
            raise ModelNotReadyError("The trained model artifact is not available")
        vector = np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float64)
        model = self._bundle["model"]
        with self._lock:
            probability_outputs = model.predict_proba(vector)
            classes = [estimator.classes_ for estimator in model.estimators_]
            probabilities = {
                target: self._positive_probability(matrix, class_values)
                for target, matrix, class_values in zip(TARGET_NAMES, probability_outputs, classes, strict=True)
            }
            anomaly_model = self._bundle.get("anomaly_model")
            if anomaly_model is not None:
                raw_score = float(-anomaly_model.decision_function(vector)[0])
                center = float(self._bundle.get("anomaly_center", 0.0))
                scale = max(float(self._bundle.get("anomaly_scale", 1.0)), 1e-6)
                z_score = float(np.clip((raw_score - center) / scale, -40.0, 40.0))
                probabilities["potential_visual_defect"] = float(1.0 / (1.0 + np.exp(-z_score)))
        return probabilities, dict(self._bundle["thresholds"])
