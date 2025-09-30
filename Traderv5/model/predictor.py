"""Model loading and inference helpers for Trader V5."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "trade_classifier.joblib"
DEFAULT_METADATA_PATH = Path(__file__).resolve().parents[1] / "models" / "trade_classifier_meta.json"


@dataclass
class ModelBundle:
    estimator: any
    features: List[str]
    label_mapping: List[int]


class ModelPredictor:
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    @classmethod
    def load_default(cls) -> "ModelPredictor":
        bundle = load_model_bundle(DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH)
        return cls(bundle)

    def predict_proba(self, features: pd.Series | pd.DataFrame | np.ndarray) -> float:
        vector = self._prepare_vector(features)
        estimator = self.bundle.estimator
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(vector)[0]
            if len(proba) == 1:
                return float(proba[0])
            # assume binary classification with label_mapping ordering
            positive_index = int(self.bundle.label_mapping.index(1)) if 1 in self.bundle.label_mapping else 1
            return float(proba[positive_index])
        if hasattr(estimator, "decision_function"):
            score = estimator.decision_function(vector)
            return float(1.0 / (1.0 + np.exp(-score)))[0]
        raise RuntimeError("Estimator does not support probability outputs")

    def predict_class(self, features: pd.Series | pd.DataFrame | np.ndarray) -> int:
        vector = self._prepare_vector(features)
        estimator = self.bundle.estimator
        prediction = estimator.predict(vector)
        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()
        if isinstance(prediction, list):
            return int(prediction[0])
        return int(prediction)

    def _prepare_vector(self, features: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            frame = features[self.bundle.features]
            return frame.values.astype(float)
        if isinstance(features, pd.Series):
            values = [features.get(name, 0.0) for name in self.bundle.features]
            return np.asarray([values], dtype=float)
        if isinstance(features, np.ndarray):
            if features.ndim == 1:
                return features.reshape(1, -1)
            return features
        raise TypeError(f"Unsupported feature type: {type(features)!r}")


def load_model_bundle(model_path: Path, metadata_path: Path) -> ModelBundle:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact missing at {model_path}")
    estimator = joblib.load(model_path)
    metadata = {
        "features": [],
        "label_mapping": [0, 1],
    }
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata.update(json.load(handle))
    features = list(metadata.get("features", []))
    labels = list(metadata.get("label_mapping", [0, 1]))
    return ModelBundle(estimator=estimator, features=features, label_mapping=labels)


__all__ = ["ModelPredictor", "load_model_bundle", "ModelBundle", "DEFAULT_MODEL_PATH"]
