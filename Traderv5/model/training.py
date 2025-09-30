"""Model training CLI for Trader V5."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Traderv5.data_sources import YahooFinanceDataFetcher, collect_gdelt_window
from Traderv5.features import FeatureEngineer, FeatureEngineerConfig

_LOG = logging.getLogger("traderv5.training")


@dataclass
class TrainingConfig:
    tickers: Tuple[str, ...] = (
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA",
        "AMD",
        "AMZN",
        "GOOGL",
        "META",
    )
    lookback_days: int = 30
    price_interval: str = "1h"
    timeline_minutes: int = 60
    label_horizon_minutes: int = 90
    positive_threshold: float = 0.0035
    negative_threshold: float = -0.0035
    min_samples: int = 200
    model_dir: Path = Path(__file__).resolve().parents[1] / "models"
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"


@dataclass
class TrainingResult:
    estimator_name: str
    roc_auc: float
    metrics: Dict[str, Dict[str, float]]
    model_path: Path
    metadata_path: Path
    samples: int
    features: List[str]


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.feature_engineer = FeatureEngineer(
            FeatureEngineerConfig(
                label_horizon_minutes=config.label_horizon_minutes,
                positive_threshold=config.positive_threshold,
                negative_threshold=config.negative_threshold,
            )
        )
        self.yahoo = YahooFinanceDataFetcher()

    def collect_dataset(self) -> pd.DataFrame:
        end = datetime.utcnow().replace(tzinfo=timezone.utc)
        start = end - timedelta(days=self.config.lookback_days)
        frames: List[pd.DataFrame] = []
        for ticker in self.config.tickers:
            _LOG.info("Collecting data for %s", ticker)
            price = self.yahoo.fetch_price_history(
                ticker,
                range_=f"{self.config.lookback_days}d",
                interval=self.config.price_interval,
            )
            if price.empty:
                _LOG.warning("No price data for %s", ticker)
                continue
            gdelt_window = collect_gdelt_window(
                ticker,
                start=start.replace(tzinfo=None),
                end=end.replace(tzinfo=None),
                timeline_minutes=self.config.timeline_minutes,
            )
            frame = self.feature_engineer.build_training_frame(ticker, price, gdelt_window)
            if frame.empty:
                _LOG.warning("No features produced for %s", ticker)
                continue
            frames.append(frame)
        if not frames:
            raise RuntimeError("No training data collected; verify tickers and API access")
        dataset = pd.concat(frames, axis=0, ignore_index=False).sort_index()
        dataset = dataset.dropna()
        dataset = dataset[dataset["label"] != 0]
        dataset = dataset.replace({np.inf: 0.0, -np.inf: 0.0})
        return dataset

    def train(self, dataset: pd.DataFrame) -> TrainingResult:
        features = self.feature_engineer.feature_columns(dataset)
        X = dataset[features].values.astype(float)
        y = dataset["label"].values.astype(int)
        if len(dataset) < self.config.min_samples:
            _LOG.warning("Dataset smaller (%s) than min_samples (%s)", len(dataset), self.config.min_samples)
        skf = StratifiedKFold(n_splits=min(5, max(2, len(dataset) // 200)), shuffle=True, random_state=42)
        candidates: List[Tuple[str, any]] = [
            (
                "histgb",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=8,
                    max_iter=400,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=10,
                    min_samples_leaf=2,
                    min_samples_split=4,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
            (
                "logreg",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=2000,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            ),
        ]
        scores: Dict[str, float] = {}
        for name, estimator in candidates:
            try:
                cv_scores = cross_val_score(estimator, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
                scores[name] = float(cv_scores.mean())
                _LOG.info("Model %s ROC-AUC: %.4f ± %.4f", name, cv_scores.mean(), cv_scores.std())
            except Exception as exc:  # pragma: no cover - sklearn error
                _LOG.warning("Model %s failed during CV: %s", name, exc)
        if not scores:
            raise RuntimeError("All candidate models failed during cross validation")
        best_name = max(scores.items(), key=lambda item: item[1])[0]
        best_estimator = dict(candidates)[best_name]
        best_estimator.fit(X, y)
        predictions = best_estimator.predict(X)
        if hasattr(best_estimator, "predict_proba"):
            proba = best_estimator.predict_proba(X)[:, 1]
            roc_auc = float(roc_auc_score(y, proba))
        else:
            roc_auc = float(scores[best_name])
        report = classification_report(y, predictions, output_dict=True)
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.config.model_dir / "trade_classifier.joblib"
        metadata_path = self.config.model_dir / "trade_classifier_meta.json"
        joblib.dump(best_estimator, model_path)
        metadata = {
            "features": features,
            "label_mapping": sorted(pd.unique(dataset["label"]).tolist()),
            "trained_at": datetime.utcnow().isoformat(),
            "estimator": best_name,
            "scores": scores,
            "roc_auc": roc_auc,
            "samples": int(len(dataset)),
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        dataset_path = self.config.data_dir / "training_samples.parquet"
        try:
            dataset.to_parquet(dataset_path)
        except Exception:  # pragma: no cover - parquet optional
            dataset_path = self.config.data_dir / "training_samples.csv"
            dataset.to_csv(dataset_path)
        _LOG.info("Persisted model to %s", model_path)
        return TrainingResult(
            estimator_name=best_name,
            roc_auc=roc_auc,
            metrics={k: v for k, v in report.items() if k in {"-1", "1", "macro avg", "weighted avg"}},
            model_path=model_path,
            metadata_path=metadata_path,
            samples=len(dataset),
            features=features,
        )


def run_cli() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    config = TrainingConfig()
    trainer = Trainer(config)
    dataset = trainer.collect_dataset()
    result = trainer.train(dataset)
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    run_cli()
