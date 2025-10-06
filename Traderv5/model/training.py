"""Model training CLI for Trader V5."""

from __future__ import annotations

import json
import logging
import time
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

from Traderv5.configuration import load_trading_config
from Traderv5.data_sources import YahooFinanceDataFetcher, collect_gdelt_window
from Traderv5.features import FeatureEngineer, FeatureEngineerConfig
from Traderv5.http_client import RateLimitedHttpClient
from Traderv5.performance_optimizer import ParallelDataFetcher, FeatureCache, OptimizedFeatureEngineer, time_operation

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
    max_tickers_per_batch: int = 2
    model_dir: Path = Path(__file__).resolve().parents[1] / "models"
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"
    
    @classmethod
    def from_config_dict(cls, config: Dict) -> "TrainingConfig":
        """Create TrainingConfig from loaded configuration dictionary."""
        training_config = config.get("training", {})
        return cls(
            tickers=tuple(training_config.get("tickers", cls.tickers)),
            lookback_days=training_config.get("lookback_days", cls.lookback_days),
            price_interval=training_config.get("price_interval", cls.price_interval),
            timeline_minutes=training_config.get("timeline_minutes", cls.timeline_minutes),
            label_horizon_minutes=training_config.get("label_horizon_minutes", cls.label_horizon_minutes),
            positive_threshold=training_config.get("positive_threshold", cls.positive_threshold),
            negative_threshold=training_config.get("negative_threshold", cls.negative_threshold),
            min_samples=training_config.get("min_samples", cls.min_samples),
            max_tickers_per_batch=training_config.get("max_tickers_per_batch", cls.max_tickers_per_batch),
        )


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
    def __init__(self, config: TrainingConfig, trading_config: Dict) -> None:
        self.config = config
        self.trading_config = trading_config
        self.feature_engineer = FeatureEngineer(
            FeatureEngineerConfig(
                label_horizon_minutes=config.label_horizon_minutes,
                positive_threshold=config.positive_threshold,
                negative_threshold=config.negative_threshold,
            )
        )
        
        # Initialize rate-limited HTTP client
        data_config = trading_config.get("data", {})
        self.http_client = RateLimitedHttpClient(
            queries_per_second=data_config.get("queries_per_second", 0.5),
            cache_dir=Path(data_config.get("cache_dir", "cache/http")),
            cache_ttl=timedelta(hours=data_config.get("cache_ttl_hours", 24)),
            max_retries=data_config.get("max_retries", 4),
            backoff_factor=data_config.get("backoff_factor", 1.8),
        )
        self.yahoo = YahooFinanceDataFetcher()
        
        # Initialize performance optimizations
        self.parallel_fetcher = ParallelDataFetcher(
            max_workers=min(4, len(config.tickers)),
            batch_size=config.max_tickers_per_batch,
        )
        
        # Initialize feature cache
        cache_dir = Path(data_config.get("cache_dir", "cache/http")) / "features"
        self.feature_cache = FeatureCache(cache_dir, ttl_hours=24)
        
        # Initialize optimized feature engineer
        self.optimized_feature_engineer = OptimizedFeatureEngineer(
            self.feature_cache, 
            self.feature_engineer.config
        )

    @time_operation("collect_dataset")
    def collect_dataset(self) -> pd.DataFrame:
        end = datetime.utcnow().replace(tzinfo=timezone.utc)
        start = end - timedelta(days=self.config.lookback_days)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        
        _LOG.info("Collecting dataset for %d tickers with parallel processing", len(self.config.tickers))
        
        # Fetch price data in parallel
        _LOG.info("Fetching price data in parallel...")
        price_data = self.parallel_fetcher.fetch_price_data_parallel(
            list(self.config.tickers),
            self.yahoo,
            self.config.lookback_days,
            self.config.price_interval,
        )
        
        if not price_data:
            raise RuntimeError("No price data collected; verify tickers and API access")
        
        _LOG.info("Successfully fetched price data for %d/%d tickers", 
                 len(price_data), len(self.config.tickers))
        
        # Fetch GDELT data in parallel
        _LOG.info("Fetching GDELT data in parallel...")
        gdelt_data = self.parallel_fetcher.fetch_gdelt_data_parallel(
            list(price_data.keys()),
            None,  # Will use default GDELT client
            start.replace(tzinfo=None),
            end.replace(tzinfo=None),
            self.config.timeline_minutes,
        )
        
        _LOG.info("Successfully fetched GDELT data for %d/%d tickers", 
                 len(gdelt_data), len(price_data))
        
        # Build features with caching
        _LOG.info("Building features with caching...")
        frames: List[pd.DataFrame] = []
        
        for ticker in price_data.keys():
            try:
                # Use optimized feature engineering with caching
                frame = self.optimized_feature_engineer.build_features_optimized(
                    ticker,
                    price_data[ticker],
                    gdelt_data.get(ticker),
                    start_str,
                    end_str,
                )
                
                if frame.empty:
                    _LOG.warning("No features produced for %s", ticker)
                    continue
                
                frames.append(frame)
                
            except Exception as e:
                _LOG.warning("Failed to build features for %s: %s", ticker, e)
                continue
        
        if not frames:
            raise RuntimeError("No training data collected; verify tickers and API access")
        
        # Combine all frames
        _LOG.info("Combining %d feature frames...", len(frames))
        dataset = pd.concat(frames, axis=0, ignore_index=False).sort_index()
        dataset = dataset.dropna()
        dataset = dataset[dataset["label"] != 0]
        dataset = dataset.replace({np.inf: 0.0, -np.inf: 0.0})
        
        _LOG.info("Dataset collection complete: %d samples, %d features", 
                 len(dataset), len(dataset.columns))
        
        return dataset

    @time_operation("train_models")
    def train(self, dataset: pd.DataFrame) -> TrainingResult:
        _LOG.info("Starting model training with %d samples", len(dataset))
        
        features = self.feature_engineer.feature_columns(dataset)
        X = dataset[features].values.astype(float)
        y = dataset["label"].values.astype(int)
        
        if len(dataset) < self.config.min_samples:
            _LOG.warning("Dataset smaller (%s) than min_samples (%s)", len(dataset), self.config.min_samples)
        
        # Optimize cross-validation splits
        n_splits = min(5, max(2, len(dataset) // 200))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Optimized model candidates with better performance
        candidates: List[Tuple[str, any]] = [
            (
                "histgb",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=8,
                    max_iter=400,
                    l2_regularization=0.1,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,  # Reduced for faster training
                    max_depth=10,
                    min_samples_leaf=2,
                    min_samples_split=4,
                    n_jobs=-1,
                    random_state=42,
                    max_features='sqrt',  # Optimize for performance
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
                                max_iter=1000,  # Reduced for faster training
                                class_weight="balanced",
                                random_state=42,
                                solver='liblinear',  # Faster solver
                            ),
                        ),
                    ]
                ),
            ),
        ]
        
        scores: Dict[str, float] = {}
        _LOG.info("Training %d models with %d-fold CV", len(candidates), n_splits)
        
        for name, estimator in candidates:
            try:
                cv_scores = cross_val_score(estimator, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
                scores[name] = float(cv_scores.mean())
                _LOG.info("Model %s ROC-AUC: %.4f ± %.4f", name, cv_scores.mean(), cv_scores.std())
            except Exception as exc:
                _LOG.warning("Model %s failed during CV: %s", name, exc)
        
        if not scores:
            raise RuntimeError("All candidate models failed during cross validation")
        
        # Select best model
        best_name = max(scores.items(), key=lambda item: item[1])[0]
        best_estimator = dict(candidates)[best_name]
        
        _LOG.info("Training final model: %s", best_name)
        best_estimator.fit(X, y)
        
        # Evaluate final model
        predictions = best_estimator.predict(X)
        if hasattr(best_estimator, "predict_proba"):
            proba = best_estimator.predict_proba(X)[:, 1]
            roc_auc = float(roc_auc_score(y, proba))
        else:
            roc_auc = float(scores[best_name])
        
        report = classification_report(y, predictions, output_dict=True)
        
        # Save model and metadata
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = self.config.model_dir / "trade_classifier.joblib"
        metadata_path = self.config.model_dir / "trade_classifier_meta.json"
        
        # Save with compression for faster loading
        joblib.dump(best_estimator, model_path, compress=3)
        
        metadata = {
            "features": features,
            "label_mapping": sorted(pd.unique(dataset["label"]).tolist()),
            "trained_at": datetime.utcnow().isoformat(),
            "estimator": best_name,
            "scores": scores,
            "roc_auc": roc_auc,
            "samples": int(len(dataset)),
            "feature_count": len(features),
            "cv_splits": n_splits,
        }
        
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        
        # Save dataset with compression
        dataset_path = self.config.data_dir / "training_samples.parquet"
        try:
            dataset.to_parquet(dataset_path, compression='snappy')
        except Exception:
            dataset_path = self.config.data_dir / "training_samples.csv"
            dataset.to_csv(dataset_path, index=False)
        
        _LOG.info("Model training complete: %s (ROC-AUC: %.4f)", best_name, roc_auc)
        _LOG.info("Model saved to %s", model_path)
        
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
    
    # Load trading configuration
    try:
        trading_config = load_trading_config()
        _LOG.info("Loaded trading configuration from %s", trading_config.get("config_path"))
    except Exception as e:
        _LOG.warning("Failed to load trading configuration: %s. Using defaults.", e)
        trading_config = {}
    
    # Create training config from trading config
    config = TrainingConfig.from_config_dict(trading_config)
    _LOG.info("Training configuration: %d tickers, %d lookback days, %d max per batch", 
             len(config.tickers), config.lookback_days, config.max_tickers_per_batch)
    
    trainer = Trainer(config, trading_config)
    
    # Collect dataset with performance monitoring
    start_time = time.time()
    dataset = trainer.collect_dataset()
    collection_time = time.time() - start_time
    
    # Train models with performance monitoring
    start_time = time.time()
    result = trainer.train(dataset)
    training_time = time.time() - start_time
    
    # Print performance summary
    from Traderv5.performance_optimizer import performance_monitor
    perf_summary = performance_monitor.get_summary()
    
    print("\n" + "="*60)
    print("TRAINING PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Dataset collection: {collection_time:.1f}s")
    print(f"Model training: {training_time:.1f}s")
    print(f"Total time: {collection_time + training_time:.1f}s")
    print(f"Dataset size: {len(dataset)} samples, {len(dataset.columns)} features")
    
    if perf_summary["total_operations"] > 0:
        print(f"\nPerformance metrics:")
        for op, metrics in perf_summary["by_operation"].items():
            print(f"  {op}: {metrics['avg_time_ms']:.1f}ms avg ({metrics['count']} ops)")
    
    # Print cache performance
    cache_metrics = trainer.feature_cache.get_metrics()
    if cache_metrics["hits"] + cache_metrics["misses"] > 0:
        print(f"\nFeature cache: {cache_metrics['hit_rate']:.1%} hit rate ({cache_metrics['hits']} hits, {cache_metrics['misses']} misses)")
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(json.dumps(asdict(result), indent=2, default=str))
    
    # Cleanup
    trainer.parallel_fetcher.shutdown()


if __name__ == "__main__":  # pragma: no cover
    run_cli()
