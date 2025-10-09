# TraderV5 Design Overview

## Goals
- Introduce a machine learning driven decision layer that fuses GDELT sentiment with Yahoo Finance market state.
- Provide data collection and training tooling to build and refresh the model in-house.
- Maintain compatibility with existing risk controls and backtester interfaces.

## High-Level Architecture
- `Traderv5/data_sources.py` – wrappers around GDELT and Yahoo APIs with batch utilities for historical pulls.
- `Traderv5/features.py` – feature engineering logic shared by training and live inference, producing aligned market/sentiment feature vectors.
- `Traderv5/model/training.py` – dataset builder, model selection (RandomForest, HistGradientBoosting, Logistic baseline), persistence of best model.
- `Traderv5/model/predictor.py` – lightweight loader/inference helper returning trade confidence scores.
- `Traderv5/decision.py` – model-aware decision engine that wraps existing risk machinery but sizes trades by model confidence.
- `Traderv5/trader.py` – orchestrates live cycle execution, wiring together data fetchers, feature builder, model, and execution stack.
- `Traderv5/backtest_adapter.py` – adapter implementing the backtester `BaseBot` interface so TraderV5 can be simulated alongside prior versions.
- `Traderv5/README.md` – usage notes and training/backtest instructions.

## Feature Set (initial)
- GDELT tone statistics over multiple windows (15m/60m/240m) and deltas.
- Article density & source-weighted tone aggregates.
- Yahoo Finance price levels, returns (1h/4h/1d), volatility proxies, and volume z-scores.
- Interaction features between sentiment strength and price momentum.

## Labeling Strategy
- Frame as binary classification: positive class when forward 90-minute return exceeds +0.35%, negative when below -0.35%; other samples dropped.
- Confidence derived from predicted probability of positive outcome.

## Model Strategy
- Compare multiple estimators with sizable capacity: `HistGradientBoostingClassifier`, `RandomForestClassifier` (400 trees), `LogisticRegression` (baseline scaler pipeline).
- Cross-validated ROC-AUC scoring; best model persisted via `joblib` at `Traderv5/models/trade_classifier.joblib` with accompanying metadata JSON.

## Backtester Integration
- New bot to translate sentiment snapshots and price bars produced by backtester into TraderV5 feature vectors, query the model, and emit `TradeDecision` objects.
- Reuse existing `TradeDecision`, `RiskConfig`, `TradeFrequencyTracker` to keep downstream tooling unchanged.

## Data Refresh Workflow
1. Define tickers/time range in `training.py` CLI entrypoint.
2. Pull GDELT timeline/articles and Yahoo price history concurrently with request pacing.
3. Build feature/label DataFrame, write to `Traderv5/data/training_samples.parquet` for audit.
4. Train/evaluate models, persist best to `Traderv5/models/`.
