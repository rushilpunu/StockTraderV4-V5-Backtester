"""Optimized model predictor with performance improvements."""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import load
from sklearn.base import BaseEstimator

from Traderv5.performance_optimizer import performance_monitor, time_operation

_LOG = logging.getLogger("traderv5.optimized_predictor")


class OptimizedModelPredictor:
    """High-performance model predictor with caching and optimization."""
    
    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        feature_columns: Optional[List[str]] = None,
        model_path: Optional[Path] = None,
        cache_predictions: bool = True,
        max_cache_size: int = 10000,
    ):
        self.model = model
        self.feature_columns = feature_columns or []
        self.model_path = model_path
        self.cache_predictions = cache_predictions
        self.max_cache_size = max_cache_size
        
        # Prediction cache
        self._prediction_cache: Dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance tracking
        self.performance_stats = {
            "predictions_made": 0,
            "total_prediction_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_predictions": 0,
            "single_predictions": 0,
        }
        
        # Load model if path provided
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: Path) -> None:
        """Load model from file with optimization."""
        try:
            _LOG.info(f"Loading model from {model_path}")
            start_time = time.time()
            
            # Load model with joblib for better performance
            self.model = load(model_path)
            
            # Load metadata if available
            metadata_path = model_path.parent / "trade_classifier_meta.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_columns = metadata.get("features", [])
                    _LOG.info(f"Loaded {len(self.feature_columns)} feature columns from metadata")
            
            load_time = time.time() - start_time
            _LOG.info(f"Model loaded in {load_time:.3f}s")
            
        except Exception as e:
            _LOG.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _cache_key(self, features: Union[Dict[str, float], pd.Series, np.ndarray]) -> str:
        """Generate cache key for features."""
        if isinstance(features, dict):
            # Sort keys for consistent hashing
            sorted_items = sorted(features.items())
            key_str = "|".join(f"{k}:{v:.6f}" for k, v in sorted_items)
        elif isinstance(features, pd.Series):
            key_str = "|".join(f"{k}:{v:.6f}" for k, v in features.items())
        elif isinstance(features, np.ndarray):
            key_str = "|".join(f"{i}:{v:.6f}" for i, v in enumerate(features))
        else:
            key_str = str(features)
        
        return hash(key_str) % (2**32)  # Use hash for faster lookup
    
    @time_operation("predict_proba")
    def predict_proba(self, features: Union[Dict[str, float], pd.Series, np.ndarray]) -> float:
        """Predict probability with caching and optimization."""
        if self.model is None:
            _LOG.warning("No model loaded, returning neutral prediction")
            return 0.5
        
        # Check cache first
        if self.cache_predictions:
            cache_key = str(self._cache_key(features))
            if cache_key in self._prediction_cache:
                self._cache_hits += 1
                self.performance_stats["cache_hits"] += 1
                return self._prediction_cache[cache_key]
        
        # Make prediction
        start_time = time.time()
        
        try:
            # Convert features to numpy array
            if isinstance(features, dict):
                # Ensure features are in the correct order
                feature_array = np.array([features.get(col, 0.0) for col in self.feature_columns])
            elif isinstance(features, pd.Series):
                feature_array = features.values
            elif isinstance(features, np.ndarray):
                feature_array = features
            else:
                raise ValueError(f"Unsupported feature type: {type(features)}")
            
            # Reshape for single prediction
            if feature_array.ndim == 1:
                feature_array = feature_array.reshape(1, -1)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(feature_array)[0, 1]  # Get positive class probability
            else:
                # Fallback to predict
                prediction = self.model.predict(feature_array)[0]
                proba = float(prediction)
            
            prediction_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats["predictions_made"] += 1
            self.performance_stats["total_prediction_time"] += prediction_time
            self.performance_stats["single_predictions"] += 1
            
            # Cache prediction
            if self.cache_predictions:
                cache_key = str(self._cache_key(features))
                if len(self._prediction_cache) < self.max_cache_size:
                    self._prediction_cache[cache_key] = proba
                else:
                    # Clear cache if it's too large
                    self._prediction_cache.clear()
                    self._prediction_cache[cache_key] = proba
                
                self._cache_misses += 1
                self.performance_stats["cache_misses"] += 1
            
            return float(proba)
            
        except Exception as e:
            _LOG.warning(f"Prediction failed: {e}")
            return 0.5
    
    @time_operation("batch_predict")
    def batch_predict(self, features_list: List[Union[Dict[str, float], pd.Series, np.ndarray]]) -> List[float]:
        """Batch prediction for better performance."""
        if self.model is None:
            _LOG.warning("No model loaded, returning neutral predictions")
            return [0.5] * len(features_list)
        
        if not features_list:
            return []
        
        start_time = time.time()
        
        try:
            # Convert all features to numpy array
            feature_arrays = []
            for features in features_list:
                if isinstance(features, dict):
                    feature_array = np.array([features.get(col, 0.0) for col in self.feature_columns])
                elif isinstance(features, pd.Series):
                    feature_array = features.values
                elif isinstance(features, np.ndarray):
                    feature_array = features
                else:
                    raise ValueError(f"Unsupported feature type: {type(features)}")
                
                feature_arrays.append(feature_array)
            
            # Stack into batch
            batch_features = np.vstack(feature_arrays)
            
            # Make batch prediction
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(batch_features)[:, 1]  # Get positive class probabilities
            else:
                # Fallback to predict
                predictions = self.model.predict(batch_features)
                probas = predictions.astype(float)
            
            prediction_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats["predictions_made"] += len(features_list)
            self.performance_stats["total_prediction_time"] += prediction_time
            self.performance_stats["batch_predictions"] += 1
            
            return probas.tolist()
            
        except Exception as e:
            _LOG.warning(f"Batch prediction failed: {e}")
            return [0.5] * len(features_list)
    
    def predict_with_confidence(
        self, 
        features: Union[Dict[str, float], pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Predict with confidence interval (if model supports it)."""
        prediction = self.predict_proba(features)
        
        # Simple confidence based on prediction distance from 0.5
        confidence = abs(prediction - 0.5) * 2  # Scale to 0-1
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "is_strong_signal": confidence > 0.3,  # Threshold for strong signals
        }
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if model supports it."""
        if self.model is None:
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                return dict(zip(self.feature_columns, importances))
            elif hasattr(self.model, 'coef_'):
                # For linear models, use absolute coefficients
                coef = np.abs(self.model.coef_[0])
                return dict(zip(self.feature_columns, coef))
            else:
                return None
        except Exception as e:
            _LOG.warning(f"Failed to get feature importance: {e}")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        total_predictions = self.performance_stats["predictions_made"]
        total_time = self.performance_stats["total_prediction_time"]
        
        cache_hits = self.performance_stats["cache_hits"]
        cache_misses = self.performance_stats["cache_misses"]
        total_cache_requests = cache_hits + cache_misses
        
        return {
            "performance_stats": self.performance_stats.copy(),
            "efficiency_metrics": {
                "avg_prediction_time_ms": (total_time / max(1, total_predictions)) * 1000,
                "predictions_per_second": total_predictions / max(0.001, total_time),
                "cache_hit_rate": cache_hits / max(1, total_cache_requests),
                "batch_efficiency": self.performance_stats["batch_predictions"] / max(1, total_predictions),
            },
            "model_info": {
                "model_loaded": self.model is not None,
                "feature_count": len(self.feature_columns),
                "cache_size": len(self._prediction_cache),
                "max_cache_size": self.max_cache_size,
            }
        }
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self._prediction_cache.clear()
        _LOG.info("Prediction cache cleared")
    
    def save_model(self, model_path: Path) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            metadata_path = model_path.parent / "trade_classifier_meta.json"
            import json
            metadata = {
                "features": self.feature_columns,
                "saved_at": time.time(),
                "feature_count": len(self.feature_columns),
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            _LOG.info(f"Model saved to {model_path}")
            
        except Exception as e:
            _LOG.error(f"Failed to save model: {e}")
            raise
    
    @classmethod
    def load_default(cls) -> "OptimizedModelPredictor":
        """Load default model from standard location."""
        model_path = Path("Traderv5/model/artifacts/trade_classifier.joblib")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Default model not found at {model_path}")
        
        return cls(model_path=model_path)


def run_predictor_demo() -> None:
    """Demo function to show optimized predictor performance."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    
    _LOG.info("Starting OptimizedModelPredictor demo...")
    
    try:
        # Initialize predictor
        predictor = OptimizedModelPredictor()
        
        # Create mock features
        mock_features = {
            "return_1": 0.02,
            "return_5": 0.05,
            "volatility_5": 0.15,
            "atr_14": 0.03,
            "sma_5": 100.0,
            "price_vs_sma_5": 0.01,
            "volume_z": 0.5,
            "sentiment_avg": 0.1,
            "article_count": 5,
        }
        
        # Test single prediction
        _LOG.info("Testing single prediction...")
        start_time = time.time()
        prediction = predictor.predict_proba(mock_features)
        single_time = time.time() - start_time
        
        print(f"Single prediction: {prediction:.4f} (took {single_time*1000:.2f}ms)")
        
        # Test batch prediction
        _LOG.info("Testing batch prediction...")
        batch_features = [mock_features.copy() for _ in range(100)]
        start_time = time.time()
        batch_predictions = predictor.batch_predict(batch_features)
        batch_time = time.time() - start_time
        
        print(f"Batch prediction (100 samples): {batch_time*1000:.2f}ms total, {batch_time*1000/100:.2f}ms per sample")
        
        # Test prediction with confidence
        confidence_result = predictor.predict_with_confidence(mock_features)
        print(f"Prediction with confidence: {confidence_result}")
        
        # Get performance summary
        perf_summary = predictor.get_performance_summary()
        
        print("\n" + "="*60)
        print("PREDICTOR PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total predictions: {perf_summary['performance_stats']['predictions_made']}")
        print(f"Average prediction time: {perf_summary['efficiency_metrics']['avg_prediction_time_ms']:.2f}ms")
        print(f"Predictions per second: {perf_summary['efficiency_metrics']['predictions_per_second']:.1f}")
        print(f"Cache hit rate: {perf_summary['efficiency_metrics']['cache_hit_rate']:.1%}")
        print(f"Batch efficiency: {perf_summary['efficiency_metrics']['batch_efficiency']:.1%}")
        print(f"Model loaded: {perf_summary['model_info']['model_loaded']}")
        print(f"Feature count: {perf_summary['model_info']['feature_count']}")
        print(f"Cache size: {perf_summary['model_info']['cache_size']}")
        print("="*60)
        
    except Exception as e:
        _LOG.error(f"Demo failed: {e}")
        print(f"Demo failed: {e}")
        print("Note: This demo requires a trained model. Run training first.")


if __name__ == "__main__":
    run_predictor_demo()
