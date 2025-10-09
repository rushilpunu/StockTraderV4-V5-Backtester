"""Train the TraderV5 classifier from a local OHLCV CSV dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backtester.data_sources import _load_multi_csv

from Traderv5.data_sources import GDELTWindow
from Traderv5.features import FeatureEngineer
from Traderv5.model.predictor import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH


def _load_price_frames(csv_path: Path, tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
    data = _load_multi_csv(csv_path)
    frames: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = data.get(ticker.upper())
        if frame is None or frame.empty:
            continue
        frames[ticker.upper()] = frame.sort_index()
    return frames


def _build_feature_frame(
    ticker: str,
    price_df: pd.DataFrame,
    engineer: FeatureEngineer,
) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    # The offline dataset has no sentiment, so pass an empty GDELT window.
    gdelt = GDELTWindow(
        ticker=ticker,
        start=price_df.index.min().to_pydatetime(),
        end=price_df.index.max().to_pydatetime(),
        timeline_minutes=1440,
        timeline=[],
        articles=[],
        summaries=[],
    )
    frame = engineer.build_training_frame(
        ticker,
        price_df,
        gdelt,
        include_labels=True,
    )
    if frame.empty:
        return frame
    frame = frame[frame["label"].isin({-1, 1})]
    return frame


def train_from_csv(
    csv_path: Path,
    tickers: List[str],
    split_date: pd.Timestamp,
    model_path: Path,
    metadata_path: Path,
) -> Dict[str, float]:
    frames = _load_price_frames(csv_path, tickers)
    if not frames:
        raise RuntimeError(f"No price data found in {csv_path} for tickers {tickers}")

    engineer = FeatureEngineer()
    dataset_parts: List[pd.DataFrame] = []
    for ticker, frame in frames.items():
        engineered = _build_feature_frame(ticker, frame, engineer)
        if engineered.empty:
            continue
        dataset_parts.append(engineered)
    if not dataset_parts:
        raise RuntimeError("Feature engineering produced no labeled samples")

    dataset = pd.concat(dataset_parts).sort_index()
    feature_columns = engineer.feature_columns(dataset)
    if not feature_columns:
        raise RuntimeError("No feature columns available for training")

    X = dataset[feature_columns].values.astype(float)
    y = dataset["label"].values.astype(int)

    time_index = dataset.index
    train_mask = time_index < split_date
    test_mask = time_index >= split_date

    if train_mask.sum() < 10 or test_mask.sum() < 5:
        raise RuntimeError("Not enough samples to train/test the classifier")

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, pos_label=1))
    recall = float(recall_score(y_test, y_pred, pos_label=1))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(pipeline, model_path)
    metadata = {
        "features": feature_columns,
        "label_mapping": sorted(int(label) for label in np.unique(y)),
        "trained_at": time.time(),
        "estimator": "logreg_csv",
        "split_date": split_date.isoformat(),
        "tickers": tickers,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "train_size": float(train_mask.sum()),
        "test_size": float(test_mask.sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TraderV5 model from CSV data")
    default_csv = Path(__file__).resolve().parents[2] / "data" / "stocks.csv"
    parser.add_argument("--csv", type=Path, default=default_csv, help="Path to multi-ticker CSV")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "AMZN", "GOOG"], help="Tickers to include")
    parser.add_argument(
        "--split-date",
        default="2007-01-01",
        help="ISO date string used as the train/test split boundary",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Output model path")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH, help="Output metadata path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_date = pd.Timestamp(args.split_date).tz_localize(None)
    metrics = train_from_csv(args.csv, args.tickers, split_date, args.model, args.metadata)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

